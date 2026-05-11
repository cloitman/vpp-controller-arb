from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cvxpy as cp
import numpy as np

from .optimization_minimization import VPPFormulation


def formulate_battery_opf_problem(
    N: Sequence[int],
    E: Sequence[Tuple[int, int]],
    T: np.ndarray,
    rho: Mapping[int, int],
    l_P: np.ndarray,
    l_Q: np.ndarray,
    c: np.ndarray,
    s_max: np.ndarray,
    r: np.ndarray,
    x: np.ndarray,
    I_max: np.ndarray,
    v_min: float,
    v_max: float,
    lmp: np.ndarray,
    eta_ch: float,
    eta_dis: float,
    alpha: float,
    delta_t: float,
    e_0: float,
    e_batt_max: float,
    v_0: float = 1.0,
    voltage_slack_penalty: float = 0.0,
    allow_battery_at_root: bool = False,
) -> VPPFormulation:
    """
    Stage 2: network-constrained battery OPF.

    Identical OPF structure to formulate_vpp_problem_minimization (Stage 1)
    with battery variables added to the power balance.  The objective maximises
    battery arbitrage profit using the fixed Stage-1 LMPs as market prices.

    Post-battery LMPs are read from the dual_value of the active_power_balance
    constraints after solving; their node ordering is stored in
    dimensions["active_power_balance_node_order"].

    Battery variables
    -----------------
    P_ch[j,t]            charge power drawn from grid at node j (MW), nonneg
    P_dis[j,t]           discharge power injected to grid (MW), nonneg
    Q_batt[j,t]          reactive power from inverter (MVAR), free
    e[j,t]               stored energy, shape (n_nodes, n_time+1) (MWh)
    S_batt[j,t]          inverter apparent power (MVA), nonneg
    P_batt_max[j]        installed power capacity (MW), decision variable
    e_batt_max_by_node[j] installed energy capacity (MWh), decision variable

    Power balance modifications (vs Stage 1)
    -----------------------------------------
    Active:   P_e[e_in,:] == l_P[j,:] - p[j,:] - P_batt_net[j,:] + r·L_e + children_P
    Reactive: Q_e[e_in,:] == l_Q[j,:] - q[j,:] - Q_batt[j,:]    + x·L_e + children_Q
    """
    node_ids = list(N)
    edge_ids = list(E)
    time_ids = list(T)

    n_nodes = len(node_ids)
    n_edges = len(edge_ids)
    n_time = len(time_ids)

    if lmp.shape != (n_nodes, n_time):
        raise ValueError(f"lmp must have shape ({n_nodes}, {n_time}), got {lmp.shape}.")

    node_to_idx = {node: idx for idx, node in enumerate(node_ids)}

    outgoing_edges_by_node: Dict[int, List[int]] = {node: [] for node in node_ids}
    parent_edge_by_node: Dict[int, int] = {}
    for edge_idx, (i, j) in enumerate(edge_ids):
        outgoing_edges_by_node[i].append(edge_idx)
        parent_edge_by_node[j] = edge_idx

    root_candidates = [node for node in node_ids if node not in parent_edge_by_node]
    if len(root_candidates) != 1:
        raise ValueError("Expected exactly one root node in radial feeder.")
    root_node = root_candidates[0]
    root_node_idx = node_to_idx[root_node]

    # -------------------------------------------------------------------------
    # Network variables (same as Stage 1)
    # -------------------------------------------------------------------------
    p = cp.Variable((n_nodes, n_time), nonneg=True, name="p_{i,t}")
    q = cp.Variable((n_nodes, n_time), nonneg=True, name="q_{i,t}")
    s = cp.Variable((n_nodes, n_time), nonneg=True, name="s_{i,t}")
    P_e = cp.Variable((n_edges, n_time), name="P_{ij,t}")
    Q_e = cp.Variable((n_edges, n_time), name="Q_{ij,t}")
    L_e = cp.Variable((n_edges, n_time), nonneg=True, name="L_{ij,t}")
    V = cp.Variable((n_nodes, n_time), name="V_{i,t}")

    # -------------------------------------------------------------------------
    # Battery variables
    # -------------------------------------------------------------------------
    P_ch = cp.Variable((n_nodes, n_time), nonneg=True, name="P^{ch}_{j,t}")
    P_dis = cp.Variable((n_nodes, n_time), nonneg=True, name="P^{dis}_{j,t}")
    Q_batt = cp.Variable((n_nodes, n_time), name="Q^{batt}_{j,t}")
    e_batt = cp.Variable((n_nodes, n_time + 1), name="e_{j,t}")
    S_batt = cp.Variable((n_nodes, n_time), nonneg=True, name="S^{batt}_{j,t}")
    P_batt_max = cp.Variable(n_nodes, nonneg=True, name="P^{batt}_{j,max}")
    e_batt_max_by_node = cp.Variable(n_nodes, nonneg=True, name="e^{batt}_{j,max}")

    # Net battery injection into the network (positive = discharge)
    P_batt_net = P_dis - P_ch

    # -------------------------------------------------------------------------
    # Voltage slack (optional soft bounds)
    # -------------------------------------------------------------------------
    use_voltage_slack = voltage_slack_penalty > 0.0
    if use_voltage_slack:
        V_slack_above = cp.Variable((n_nodes, n_time), nonneg=True, name="V_slack_above_{i,t}")
        V_slack_below = cp.Variable((n_nodes, n_time), nonneg=True, name="V_slack_below_{i,t}")

    # -------------------------------------------------------------------------
    # OPF constraints (mirror Stage 1, with battery in power balance)
    # -------------------------------------------------------------------------
    constraints: Dict[str, List[cp.Constraint]] = {}

    constraints["voltage_bounds"] = []
    constraints["generator_apparent_power"] = []
    constraints["generation_apparent_power_cap"] = []

    for node in node_ids:
        j_idx = node_to_idx[node]
        for t_idx in range(n_time):
            if use_voltage_slack:
                constraints["voltage_bounds"].append(
                    V[j_idx, t_idx] >= v_min**2 - V_slack_below[j_idx, t_idx]
                )
                constraints["voltage_bounds"].append(
                    V[j_idx, t_idx] <= v_max**2 + V_slack_above[j_idx, t_idx]
                )
            else:
                constraints["voltage_bounds"].append(V[j_idx, t_idx] >= v_min**2)
                constraints["voltage_bounds"].append(V[j_idx, t_idx] <= v_max**2)
            constraints["generator_apparent_power"].append(
                cp.norm(cp.hstack([p[j_idx, t_idx], q[j_idx, t_idx]]), 2)
                <= s[j_idx, t_idx]
            )
            constraints["generation_apparent_power_cap"].append(
                s[j_idx, t_idx] <= s_max[j_idx]
            )

    constraints["active_power_balance"] = []
    constraints["reactive_power_balance"] = []
    constraints["voltage_balance"] = [V[root_node_idx, :] == v_0**2]
    constraints["thermal_limits"] = []
    constraints["current_relation_relaxed"] = []

    lmp_node_order: List[int] = []

    for node in node_ids:
        j_idx = node_to_idx[node]

        if j_idx == root_node_idx:
            # Root has no parent edge. Generation + battery supplies all outgoing flow.
            # When allow_battery_at_root=False, e_batt_max_by_node[root]=0 forces
            # P_batt_net and Q_batt to zero, reducing this to p == children_P.
            children_P: Any = np.zeros(n_time)
            children_Q: Any = np.zeros(n_time)
            for e_idx in outgoing_edges_by_node[node]:
                children_P = children_P + P_e[e_idx, :]
                children_Q = children_Q + Q_e[e_idx, :]

            constraints["active_power_balance"].append(
                p[j_idx, :] + P_batt_net[j_idx, :] == children_P
            )
            constraints["reactive_power_balance"].append(
                q[j_idx, :] + Q_batt[j_idx, :] == children_Q
            )
            lmp_node_order.append(j_idx)
            continue

        e_in = parent_edge_by_node[node]
        i_idx = node_to_idx[rho[node]]
        rij = r[e_in]
        xij = x[e_in]
        I_max_ij = I_max[e_in]

        children_P = np.zeros(n_time)
        children_Q = np.zeros(n_time)
        for e_out in outgoing_edges_by_node[node]:
            children_P = children_P + P_e[e_out, :]
            children_Q = children_Q + Q_e[e_out, :]

        # Active power balance — battery net injection reduces apparent load (eq. 10)
        constraints["active_power_balance"].append(
            P_e[e_in, :]
            == (l_P[j_idx, :] - p[j_idx, :] - P_batt_net[j_idx, :] + rij * L_e[e_in, :] + children_P)
        )
        lmp_node_order.append(j_idx)

        # Reactive power balance — inverter reactive dispatch (eq. 11)
        constraints["reactive_power_balance"].append(
            Q_e[e_in, :]
            == (l_Q[j_idx, :] - q[j_idx, :] - Q_batt[j_idx, :] + xij * L_e[e_in, :] + children_Q)
        )

        # Voltage balance — KVL (eq. 12 / eq. 3)
        constraints["voltage_balance"].append(
            V[j_idx, :]
            == V[i_idx, :]
            + (rij**2 + xij**2) * L_e[e_in, :]
            - 2.0 * (rij * P_e[e_in, :] + xij * Q_e[e_in, :])
        )

        # Thermal limit (eq. 13 / eq. 9)
        constraints["thermal_limits"].append(L_e[e_in, :] <= I_max_ij**2)

        # Current relation, relaxed SOCP (eq. 14 / eq. 4)
        for t_idx in range(n_time):
            constraints["current_relation_relaxed"].append(
                L_e[e_in, t_idx]
                >= cp.quad_over_lin(
                    cp.hstack([P_e[e_in, t_idx], Q_e[e_in, t_idx]]),
                    V[i_idx, t_idx],
                )
            )

    # -------------------------------------------------------------------------
    # Battery constraints
    # -------------------------------------------------------------------------
    constraints["battery_inverter"] = []
    constraints["battery_energy_limits"] = []
    constraints["battery_cycle"] = []
    constraints["battery_energy_dynamics"] = []
    constraints["battery_charge_power_capacity"] = []
    constraints["battery_discharge_power_capacity"] = []
    constraints["battery_energy_power_link"] = []

    for j_idx in range(n_nodes):
        # Inverter cone: ||[P_batt_net, Q_batt]||_2 <= S_batt (eq. 15)
        for t_idx in range(n_time):
            constraints["battery_inverter"].append(
                cp.norm(
                    cp.hstack([P_batt_net[j_idx, t_idx], Q_batt[j_idx, t_idx]]), 2
                )
                <= S_batt[j_idx, t_idx]
            )
        # Inverter capacity (eq. 16): S_batt <= P_batt_max
        constraints["battery_inverter"].append(S_batt[j_idx, :] <= P_batt_max[j_idx])

        # Energy limits: 0 <= e <= e_batt_max_by_node
        constraints["battery_energy_limits"].append(e_batt[j_idx, :] >= 0.0)
        constraints["battery_energy_limits"].append(
            e_batt[j_idx, :] <= e_batt_max_by_node[j_idx]
        )

        # Cycle: return to initial SOC at end of horizon
        constraints["battery_cycle"].append(e_batt[j_idx, 0] == e_0)
        constraints["battery_cycle"].append(e_batt[j_idx, n_time] == e_0)

        # Energy dynamics: e[t+1] = e[t] + (eta_ch*P_ch - (1/eta_dis)*P_dis)*dt
        constraints["battery_energy_dynamics"].append(
            e_batt[j_idx, 1:]
            == e_batt[j_idx, :-1]
            + (eta_ch * P_ch[j_idx, :] - (1.0 / eta_dis) * P_dis[j_idx, :]) * delta_t
        )

        # Power capacity limits
        constraints["battery_charge_power_capacity"].append(
            P_ch[j_idx, :] <= P_batt_max[j_idx]
        )
        constraints["battery_discharge_power_capacity"].append(
            P_dis[j_idx, :] <= P_batt_max[j_idx]
        )

        # Fixed duration: e_max = alpha * P_max
        constraints["battery_energy_power_link"].append(
            e_batt_max_by_node[j_idx] == alpha * P_batt_max[j_idx]
        )

    # Total energy budget across all nodes
    constraints["battery_total_capacity"] = [cp.sum(e_batt_max_by_node) <= e_batt_max]

    # Root battery exclusion — omit when allow_battery_at_root=True
    if not allow_battery_at_root:
        constraints["no_battery_at_root"] = [e_batt_max_by_node[root_node_idx] == 0.0]
    else:
        constraints["no_battery_at_root"] = []

    # -------------------------------------------------------------------------
    # Objective: maximise battery arbitrage profit (LMPs are fixed Stage-1 prices)
    # When voltage slack is active, penalise violations so the solver still
    # prefers to stay within bounds.
    # -------------------------------------------------------------------------
    battery_profit = cp.sum(cp.multiply(lmp, P_batt_net)) * delta_t
    if use_voltage_slack:
        slack_penalty = voltage_slack_penalty * cp.sum(V_slack_above + V_slack_below)
        objective = cp.Maximize(battery_profit - slack_penalty)
    else:
        objective = cp.Maximize(battery_profit)

    all_constraints = [con for group in constraints.values() for con in group]
    problem = cp.Problem(objective, all_constraints)

    variables: Dict[str, Any] = {
        # Network
        "p_{i,t}": p,
        "q_{i,t}": q,
        "s_{i,t}": s,
        "P_{ij,t}": P_e,
        "Q_{ij,t}": Q_e,
        "L_{ij,t}": L_e,
        "V_{i,t}": V,
        # Battery
        "P^{ch}_{j,t}": P_ch,
        "P^{dis}_{j,t}": P_dis,
        "Q^{batt}_{j,t}": Q_batt,
        "e_{j,t}": e_batt,
        "S^{batt}_{j,t}": S_batt,
        "P^{batt}_{j,max}": P_batt_max,
        "e^{batt}_{j,max}": e_batt_max_by_node,
        "P^{batt}_{j,t}": P_batt_net,  # derived expression; .value populated after solve
    }
    if use_voltage_slack:
        variables["V_slack_above_{i,t}"] = V_slack_above
        variables["V_slack_below_{i,t}"] = V_slack_below

    dimensions: Dict[str, Any] = {
        "|N|": n_nodes,
        "|E|": n_edges,
        "|T|": n_time,
        "root_node": int(root_node),
        "active_power_balance_node_order": lmp_node_order,
    }

    return VPPFormulation(
        problem=problem,
        variables=variables,
        constraints=constraints,
        dimensions=dimensions,
    )


def formulate_battery_arbitrage_problem(
    N: Sequence[int],
    E: Sequence[Tuple[int, int]],
    T: np.ndarray,
    rho: Mapping[int, int],
    lmp: np.ndarray,
    eta_ch: float,
    eta_dis: float,
    alpha: float,
    delta_t: float,
    e_0: float,
    e_batt_max: float,
) -> VPPFormulation:
    """
    Legacy battery-only LP (no network constraints).

    Kept for reference.  Use formulate_battery_opf_problem for the
    network-constrained Stage 2 formulation.
    """

    node_ids = list(N)
    edge_ids = list(E)
    time_ids = list(T)

    n_nodes = len(node_ids)
    n_time = len(time_ids)

    if lmp.shape != (n_nodes, n_time):
        raise ValueError(f"lmp must have shape ({n_nodes}, {n_time}), got {lmp.shape}.")
    if eta_ch <= 0.0 or eta_ch > 1.0:
        raise ValueError("eta_ch must be in (0, 1].")
    if eta_dis <= 0.0 or eta_dis > 1.0:
        raise ValueError("eta_dis must be in (0, 1].")
    if alpha <= 0.0:
        raise ValueError("alpha (battery duration) must be positive.")
    if e_batt_max < 0.0:
        raise ValueError("e_batt_max must be non-negative.")

    node_to_idx = {node: idx for idx, node in enumerate(node_ids)}

    parent_edge_by_node: Dict[int, int] = {}
    for edge_idx, (i, j) in enumerate(edge_ids):
        parent_edge_by_node[j] = edge_idx
    root_candidates = [node for node in node_ids if node not in parent_edge_by_node]
    if len(root_candidates) != 1:
        raise ValueError("Expected exactly one root node in radial feeder.")
    root_node = root_candidates[0]
    root_node_idx = node_to_idx[root_node]

    P_ch = cp.Variable((n_nodes, n_time), nonneg=True, name="P^{ch}_{j,t}")
    P_dis = cp.Variable((n_nodes, n_time), nonneg=True, name="P^{dis}_{j,t}")
    e = cp.Variable((n_nodes, n_time + 1), name="e_{j,t}")
    P_batt_max = cp.Variable(n_nodes, nonneg=True, name="P^{batt}_{j,max}")
    e_batt_max_by_node = cp.Variable(n_nodes, nonneg=True, name="e^{batt}_{j,max}")

    constraints: Dict[str, List[cp.Constraint]] = {}
    constraints["battery_energy_limits"] = []
    constraints["battery_cycle"] = []
    constraints["battery_energy_dynamics"] = []
    constraints["battery_charge_power_capacity"] = []
    constraints["battery_discharge_power_capacity"] = []
    constraints["battery_energy_power_link"] = []

    for j_idx in range(n_nodes):
        constraints["battery_energy_limits"].append(e[j_idx, :] >= 0.0)
        constraints["battery_energy_limits"].append(
            e[j_idx, :] <= e_batt_max_by_node[j_idx]
        )
        constraints["battery_cycle"].append(e[j_idx, 0] == e_0)
        constraints["battery_cycle"].append(e[j_idx, n_time] == e_0)
        constraints["battery_energy_dynamics"].append(
            e[j_idx, 1:]
            == e[j_idx, :-1]
            + (eta_ch * P_ch[j_idx, :] - (1.0 / eta_dis) * P_dis[j_idx, :]) * delta_t
        )
        constraints["battery_charge_power_capacity"].append(
            P_ch[j_idx, :] <= P_batt_max[j_idx]
        )
        constraints["battery_discharge_power_capacity"].append(
            P_dis[j_idx, :] <= P_batt_max[j_idx]
        )
        constraints["battery_energy_power_link"].append(
            e_batt_max_by_node[j_idx] == alpha * P_batt_max[j_idx]
        )

    constraints["battery_total_capacity"] = [cp.sum(e_batt_max_by_node) <= e_batt_max]
    constraints["no_battery_at_root"] = [e_batt_max_by_node[root_node_idx] == 0.0]

    P_batt = P_dis - P_ch
    profit = cp.sum(cp.multiply(lmp, P_batt)) * delta_t
    objective = cp.Maximize(profit)

    all_constraints = [con for group in constraints.values() for con in group]
    problem = cp.Problem(objective, all_constraints)

    variables: Dict[str, Any] = {
        "P^{ch}_{j,t}": P_ch,
        "P^{dis}_{j,t}": P_dis,
        "P^{batt}_{j,t}": P_batt,
        "e_{j,t}": e,
        "P^{batt}_{j,max}": P_batt_max,
        "e^{batt}_{j,max}": e_batt_max_by_node,
    }

    dimensions: Dict[str, Any] = {
        "|N|": n_nodes,
        "|E|": len(edge_ids),
        "|T|": n_time,
        "root_node": int(root_node),
    }

    return VPPFormulation(
        problem=problem,
        variables=variables,
        constraints=constraints,
        dimensions=dimensions,
    )


# Keep the old name as an alias so any existing callers don't break immediately.
formulate_vpp_problem_maximization = formulate_battery_arbitrage_problem
