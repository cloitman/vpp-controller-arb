from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, Tuple, cast

import cvxpy as cp
import numpy as np
import pandas as pd

from .optimization_minimization import formulate_vpp_problem_minimization
from .optimization_maximization import formulate_battery_arbitrage_problem


@dataclass(frozen=True)
class DayOptimizationResult:
    """Structured outputs for a solved day-level optimization."""

    status: str
    objective_value: float | None
    variables: Dict[str, np.ndarray]
    duals: Dict[str, list[np.ndarray]]
    diagnostics: Dict[str, Any]


# ---------------------------------------------------------------------------
# Two-stage entry point
# ---------------------------------------------------------------------------

def run_two_stage_optimization(
    topology_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    price_df_root_node: pd.DataFrame,
    total_battery_capacity: float,
) -> Tuple[np.ndarray, DayOptimizationResult, DayOptimizationResult]:
    """
    Run the full two-stage LMP → arbitrage pipeline.

    Stage 1: Solve the no-battery OPF and recover Locational Marginal Prices
             (LMPs) from the dual variables of the active power balance
             equality constraints.

    Stage 2: Given those fixed LMPs as market prices, allocate a limited
             battery energy budget across nodes to maximise predicted
             arbitrage profit.

    Returns
    -------
    lmp : np.ndarray, shape (n_nodes, n_time)
        Nodal marginal prices ($/MWh) from Stage 1.
    lmp_result : DayOptimizationResult
        Full solver output for Stage 1.
    arb_result : DayOptimizationResult
        Full solver output for Stage 2.
    """
    lmp, lmp_result = run_lmp_problem(topology_df, demand_df, price_df_root_node)
    arb_result = run_battery_arbitrage_problem(
        topology_df=topology_df,
        demand_df=demand_df,
        lmp=lmp,
        total_battery_capacity=total_battery_capacity,
    )
    return lmp, lmp_result, arb_result


# ---------------------------------------------------------------------------
# Stage 1: no-battery OPF → LMPs
# ---------------------------------------------------------------------------

def run_lmp_problem(
    topology_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    price_df_root_node: pd.DataFrame,
    voltage_slack_penalty: float = 0.0,
) -> Tuple[np.ndarray, DayOptimizationResult]:
    """
    Stage 1: solve the no-battery, no-slack OPF and extract LMPs.

    The LMP at node j and hour t is the shadow price of serving one additional
    MW of demand at (j, t).  In the branch-flow OPF it equals the negative of
    the dual variable attached to the active power balance equality for node j.

    LMP sign convention
    -------------------
    For the root node (generator):
        LMP[root, t] = c[root, t]  (marginal generation cost)
    For every other node j:
        LMP[j, t] = -dual_value of active_power_balance constraint for (j, t)

    Both expressions are unified by the formula  LMP = -dual_value  because
    the KKT stationarity condition at the root gives
    dual_value_root[t] = -c[root, t].

    Returns
    -------
    lmp : np.ndarray, shape (n_nodes, n_time)
    result : DayOptimizationResult
    """
    price_series = price_df_root_node["$/MW"]
    model_inputs = build_model_inputs_minimization(
        topology_df=topology_df,
        demand_df=demand_df,
        price_series_root_node=price_series,
        voltage_slack_penalty=voltage_slack_penalty,
    )

    formulation = formulate_vpp_problem_minimization(**model_inputs)
    _solve_for_lmp_duals(formulation.problem)
    
    n_nodes = formulation.dimensions["|N|"]
    n_time = formulation.dimensions["|T|"]
    node_order: list[int] = formulation.dimensions["active_power_balance_node_order"]
    balance_constraints = formulation.constraints["active_power_balance"]
    
    # LMP[j, t] = -dual_value of the active power balance constraint for node j
    lmp = np.zeros((n_nodes, n_time))
    for ci, j_idx in enumerate(node_order):
        raw_dual = balance_constraints[ci].dual_value
        if raw_dual is not None:
            lmp[j_idx, :] = -np.asarray(raw_dual, dtype=float).reshape(-1)

    duals = {
        group: [np.array(con.dual_value) for con in group_constraints]
        for group, group_constraints in formulation.constraints.items()
    }
    variables = {
        name: np.array(var.value) if hasattr(var, "value") else np.array(var)
        for name, var in formulation.variables.items()
    }
    # Store root-node generation cost per hour so callers can compute weighted cost.
    variables["c_root_t"] = model_inputs["c"][0, :]

    diagnostics = {
        "solver": formulation.problem.solver_stats.solver_name,
        "solve_time": formulation.problem.solver_stats.solve_time,
        "num_iters": formulation.problem.solver_stats.num_iters,
        "dimensions": formulation.dimensions,
        "lmp": lmp,
    }

    result = DayOptimizationResult(
        status=formulation.problem.status,
        objective_value=cast(float | None, formulation.problem.value),
        variables=variables,
        duals=duals,
        diagnostics=diagnostics,
    )
    return lmp, result


# ---------------------------------------------------------------------------
# Stage 2: battery arbitrage using fixed LMPs
# ---------------------------------------------------------------------------

def run_battery_arbitrage_problem(
    topology_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    lmp: np.ndarray,
    total_battery_capacity: float,
) -> DayOptimizationResult:
    """
    Stage 2: allocate batteries across nodes to maximise LMP arbitrage profit.

    The network is not re-solved here.  The LMPs from Stage 1 are treated as
    fixed market prices.  The problem is a pure LP over battery variables.

    Parameters
    ----------
    lmp : np.ndarray, shape (n_nodes, n_time)
        Nodal LMPs from Stage 1.
    total_battery_capacity : float
        Total energy capacity budget across all nodes (MWh).
    """
    model_inputs = build_model_inputs_maximization(
        topology_df=topology_df,
        demand_df=demand_df,
        lmp=lmp,
        total_battery_capacity=total_battery_capacity,
    )

    formulation = formulate_battery_arbitrage_problem(**model_inputs)
    solve_formulation_problem(formulation.problem)

    duals = {
        group: [np.array(con.dual_value) for con in group_constraints]
        for group, group_constraints in formulation.constraints.items()
    }
    variables = {
        name: np.array(var.value) if hasattr(var, "value") else np.array(var)
        for name, var in formulation.variables.items()
    }
    # Embed the LMP array so it is saved alongside battery variables and can
    # be loaded later without re-running Stage 1.
    variables["lmp"] = lmp

    diagnostics = {
        "solver": formulation.problem.solver_stats.solver_name,
        "solve_time": formulation.problem.solver_stats.solve_time,
        "num_iters": formulation.problem.solver_stats.num_iters,
        "dimensions": formulation.dimensions,
    }

    return DayOptimizationResult(
        status=formulation.problem.status,
        objective_value=cast(float | None, formulation.problem.value),
        variables=variables,
        duals=duals,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------

def build_model_inputs_minimization(
    *,
    topology_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    price_series_root_node: pd.Series | np.ndarray,
    voltage_slack_penalty: float = 0.0,
) -> Dict[str, Any]:
    """Build keyword-argument dict for formulate_vpp_problem_minimization."""
    node_col = _detect_node_column(topology_df)
    nodes = topology_df[node_col].to_numpy(dtype=int)
    n_nodes = len(nodes)

    r_matrix = _parse_vector_column(topology_df["r"], n_nodes)
    x_matrix = _parse_vector_column(topology_df["x"], n_nodes)
    i_max_matrix = _parse_vector_column(topology_df["I_max"], n_nodes)

    edges: list[Tuple[int, int]] = []
    r_list: list[float] = []
    x_list: list[float] = []
    i_max_list: list[float] = []

    for i in range(n_nodes):
        for j in range(n_nodes):
            if r_matrix[i, j] > 1e-3 or x_matrix[i, j] > 1e-3 or i_max_matrix[i, j] > 1e-3:
                edges.append((int(nodes[i]), int(nodes[j])))
                r_list.append(float(r_matrix[i, j]))
                x_list.append(float(x_matrix[i, j]))
                i_max_list.append(float(i_max_matrix[i, j]))

    rho = {int(i): 0 for i in nodes}
    for i, j in edges:
        rho[int(j)] = int(i)

    hourly_l_P, hourly_l_Q = _extract_hourly_demand(demand_df, n_nodes=n_nodes)

    price = np.asarray(price_series_root_node, dtype=float).reshape(-1)
    if price.shape[0] != hourly_l_P.shape[1]:
        raise ValueError("Price curve length must match number of time steps.")

    # Cost is non-zero only at root node (index 0)
    c = np.zeros_like(hourly_l_P)
    c[0, :] = price

    s_max = topology_df["s_max"].to_numpy(dtype=float)
    v_min = float(topology_df["v_min"].iloc[0])
    v_max = float(topology_df["v_max"].iloc[0])

    return {
        "N": list(nodes),
        "E": edges,
        "T": list(range(hourly_l_P.shape[1])),
        "rho": rho,
        "l_P": hourly_l_P,
        "l_Q": hourly_l_Q,
        "c": c,
        "s_max": s_max,
        "r": np.array(r_list),
        "x": np.array(x_list),
        "I_max": np.array(i_max_list),
        "v_min": v_min,
        "v_max": v_max,
        "v_0": 1.0,
        "voltage_slack_penalty": voltage_slack_penalty,
    }


def build_model_inputs_maximization(
    *,
    topology_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    lmp: np.ndarray,
    total_battery_capacity: float,
) -> Dict[str, Any]:
    """Build keyword-argument dict for formulate_battery_arbitrage_problem."""
    node_col = _detect_node_column(topology_df)
    nodes = topology_df[node_col].to_numpy(dtype=int)
    n_nodes = len(nodes)

    r_matrix = _parse_vector_column(topology_df["r"], n_nodes)
    x_matrix = _parse_vector_column(topology_df["x"], n_nodes)
    i_max_matrix = _parse_vector_column(topology_df["I_max"], n_nodes)

    edges: list[Tuple[int, int]] = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if r_matrix[i, j] > 1e-3 or x_matrix[i, j] > 1e-3 or i_max_matrix[i, j] > 1e-3:
                edges.append((int(nodes[i]), int(nodes[j])))

    rho = {int(i): 0 for i in nodes}
    for i, j in edges:
        rho[int(j)] = int(i)

    hourly_l_P, _ = _extract_hourly_demand(demand_df, n_nodes=n_nodes)
    n_time = hourly_l_P.shape[1]

    if lmp.shape != (n_nodes, n_time):
        raise ValueError(
            f"lmp shape {lmp.shape} does not match (n_nodes={n_nodes}, n_time={n_time})."
        )

    return {
        "N": list(nodes),
        "E": edges,
        "T": list(range(n_time)),
        "rho": rho,
        "lmp": lmp,
        "eta_ch": 0.95,
        "eta_dis": 0.95,
        "alpha": 2.0,
        "delta_t": 1.0,
        "e_0": 0.0,
        "e_batt_max": float(total_battery_capacity),
    }


# ---------------------------------------------------------------------------
# Stage 3: post-battery LMP (re-run Stage 1 with battery dispatch fixed)
# ---------------------------------------------------------------------------

def run_post_battery_lmp_problem(
    topology_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    price_df_root_node: pd.DataFrame,
    arb_result: DayOptimizationResult,
) -> Tuple[np.ndarray, DayOptimizationResult]:
    """
    Compute post-battery LMPs by re-running Stage 1 with battery net injections
    subtracted from nodal demand.

    Battery discharge (P_batt > 0) reduces local demand seen by the generator.
    Battery charging (P_batt < 0) increases it.
    Effective demand:  l_P_eff[j,t] = l_P[j,t] - P_batt[j,t]

    Returns
    -------
    post_lmp : np.ndarray, shape (n_nodes, n_time)
    post_result : DayOptimizationResult
        Full OPF result with p_{i,t} (generation) and V_{i,t} (squared voltage).
    """
    p_batt_net = np.array(arb_result.variables["P^{batt}_{j,t}"], dtype=float)
    modified_demand_df = _subtract_battery_from_demand(demand_df, p_batt_net)
    post_lmp, post_result = run_lmp_problem(
        topology_df, modified_demand_df, price_df_root_node,
        voltage_slack_penalty=1e4,
    )
    if post_result.status not in ("optimal", "optimal_inaccurate"):
        _diagnose_post_battery_infeasibility(modified_demand_df, p_batt_net, post_result)
    return post_lmp, post_result


def _diagnose_post_battery_infeasibility(
    modified_demand_df: pd.DataFrame,
    p_batt_net: np.ndarray,
    post_result: DayOptimizationResult,
) -> None:
    """Print diagnostic info when the post-battery OPF is infeasible."""
    n_nodes, n_time = p_batt_net.shape

    # Rebuild effective demand matrix from the modified demand dataframe.
    l_P_eff = np.zeros((n_nodes, n_time))
    for _, row in modified_demand_df.iterrows():
        n, h = int(row["node"]), int(row["hour"])
        if n < n_nodes and h < n_time:
            l_P_eff[n, h] = float(row["l_P"])

    print("\n  ── Stage 3 Infeasibility Diagnostic ─────────────────────────")
    print(f"  Solver status : {post_result.status}")

    # 1. Effective demand statistics
    neg = [(j, t, l_P_eff[j, t])
           for j in range(n_nodes) for t in range(n_time)
           if l_P_eff[j, t] < -1e-4]
    if neg:
        print(f"\n  Negative effective demand (battery exports exceed local load):")
        print(f"    Count : {len(neg)} (node, hour) pairs")
        worst = min(neg, key=lambda x: x[2])
        print(f"    Worst : node {worst[0]}, hour {worst[1]}, l_P_eff = {worst[2]:.4f} MW")
        # Per-node worst
        per_node_min = {j: min((v for nn, _, v in neg if nn == j), default=0.0)
                        for j in range(n_nodes)}
        for j, v in per_node_min.items():
            if v < -1e-4:
                print(f"    Node {j:2d}: min l_P_eff = {v:.4f} MW")
    else:
        print(f"\n  Effective demand: all non-negative "
              f"(range [{l_P_eff.min():.4f}, {l_P_eff.max():.4f}] MW)")

    # 2. Battery dispatch summary
    print(f"\n  Battery dispatch (P_batt = discharge − charge):")
    print(f"    Max discharge : {p_batt_net.max():.4f} MW")
    print(f"    Max charge    : {(-p_batt_net).clip(min=0).max():.4f} MW")
    net_per_node = p_batt_net.sum(axis=1)
    for j, net in enumerate(net_per_node):
        if abs(net) > 1e-4:
            print(f"    Node {j:2d} daily net : {net:+.4f} MWh "
                  f"({'discharge' if net > 0 else 'charge'})")

    # 3. Constraint violations — only meaningful if solver populated variable values.
    # ECOS returns no primal point on infeasible problems, so violations will be None.
    violated: Dict[str, Tuple[int, float]] = {}
    for group, duals_list in post_result.duals.items():
        # proxy: check if any dual is unusually large (indicates active constraint)
        big = [abs(float(np.asarray(d).flat[0]))
               for d in duals_list
               if d is not None and not np.any(np.isnan(np.asarray(d, dtype=float)))]
        if big and max(big) > 1e3:
            violated[group] = (len(big), max(big))

    if violated:
        print("\n  Constraints with very large duals (likely binding/violated):")
        for group, (cnt, mx) in sorted(violated.items(), key=lambda x: -x[1][1]):
            print(f"    {group}: {cnt} constraints, max |dual| = {mx:.2e}")
    else:
        print("\n  No primal point from solver — constraint violations cannot be computed "
              "directly.")
        print("  Likely causes:")
        if neg:
            print("    • Battery discharge causes reverse power flow → voltage may rise "
                  "above v_max at nodes receiving battery exports.")
        print("    • Check v_min / v_max bounds in the topology CSV.")
        print("    • Check s_max at the root node vs. total effective load.")

    print("  ─────────────────────────────────────────────────────────────\n")


def _subtract_battery_from_demand(demand_df: pd.DataFrame, p_batt_net: np.ndarray) -> pd.DataFrame:
    df = demand_df.copy()
    nodes = df["node"].astype(int).to_numpy()
    hours = df["hour"].astype(int).to_numpy()
    n_nodes, n_time = p_batt_net.shape
    adjustments = np.array([
        p_batt_net[n, h] if n < n_nodes and h < n_time else 0.0
        for n, h in zip(nodes, hours)
    ])
    df["l_P"] = df["l_P"].astype(float) - adjustments
    return df


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _solve_for_lmp_duals(problem: cp.Problem) -> None:
    """Solve the LMP OPF preferring solvers that reliably populate dual variables."""
    # ECOS is tried first because CLARABEL may return None for SOCP equality duals.
    preferred = ["ECOS", "CLARABEL", "SCS"]
    installed = set(cp.installed_solvers())

    for solver_name in preferred:
        if solver_name in installed:
            problem.solve(solver=solver_name, verbose=True)
            return

    installed_list = ", ".join(sorted(installed)) if installed else "none"
    raise cp.SolverError(
        "No suitable conic solver found. Install one of ECOS, CLARABEL, or SCS. "
        f"Installed solvers: {installed_list}."
    )


def solve_formulation_problem(problem: cp.Problem) -> None:
    preferred = ["CLARABEL", "ECOS", "SCS"]
    installed = set(cp.installed_solvers())

    for solver_name in preferred:
        if solver_name in installed:
            problem.solve(solver=solver_name, verbose=False)
            return

    installed_list = ", ".join(sorted(installed)) if installed else "none"
    raise cp.SolverError(
        "No suitable conic solver found. Install one of CLARABEL, ECOS, or SCS. "
        f"Installed solvers: {installed_list}."
    )


def _detect_node_column(topology_df: pd.DataFrame):
    candidate_names = ("node", "bus", "i", "Unnamed: 0")
    for candidate in candidate_names:
        if candidate in topology_df.columns:
            return candidate

    first_col = topology_df.columns[0]
    if pd.api.types.is_integer_dtype(topology_df[first_col]):
        return first_col

    raise ValueError(
        "Could not infer node index column in topology table. "
        "Expected one of: node, bus, i, Unnamed: 0."
    )


def _parse_vector_column(col: pd.Series, width: int) -> np.ndarray:
    rows = []
    for item in col:
        if isinstance(item, str):
            vector = ast.literal_eval(item)
        else:
            vector = item
        arr = np.asarray(vector, dtype=float)
        if arr.shape != (width,):
            raise ValueError("Topology vector column has invalid width.")
        rows.append(arr)
    return np.vstack(rows)


def _extract_hourly_demand(
    demand_df: pd.DataFrame, *, n_nodes: int
) -> Tuple[np.ndarray, np.ndarray]:
    required_cols = {"node", "hour", "l_P", "l_Q"}
    if not required_cols.issubset(set(demand_df.columns)):
        missing = required_cols.difference(set(demand_df.columns))
        raise ValueError(f"Demand table missing required columns: {sorted(missing)}")

    max_hour = int(demand_df["hour"].max())
    n_time = max_hour + 1

    l_P = np.zeros((n_nodes, n_time), dtype=float)
    l_Q = np.zeros((n_nodes, n_time), dtype=float)

    for _, row in demand_df.iterrows():
        node = int(row["node"])
        hour = int(row["hour"])
        l_P[node, hour] = float(row["l_P"])
        l_Q[node, hour] = float(row["l_Q"])

    return l_P, l_Q


# ---------------------------------------------------------------------------
# Backward-compatible alias: Stage 1 only (matches the old run_day_optimization
# signature used in __init__.py and existing tests)
# ---------------------------------------------------------------------------

def run_day_optimization(
    topology_df: pd.DataFrame,
    demand_df: pd.DataFrame,
    price_df_root_node: pd.DataFrame,
    total_battery_capacity: float,  # noqa: ARG001 — kept for API compatibility
) -> DayOptimizationResult:
    """Run Stage 1 only (LMP problem). Kept for backward compatibility."""
    return run_lmp_problem(topology_df, demand_df, price_df_root_node)[1]
