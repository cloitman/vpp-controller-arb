from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class VPPFormulation:
    """Container for a formulated VPP optimization problem."""

    problem: cp.Problem
    variables: Dict[str, Any]
    constraints: Dict[str, List[cp.Constraint]]
    dimensions: Dict[str, int]


def formulate_vpp_problem_minimization(
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
    v_0: float = 1.0,
) -> VPPFormulation:
    """
    Solve the no-battery OPF to compute Locational Marginal Prices (LMPs).

    The dual variables of the active power balance equality constraints give
    the LMPs.  Specifically, LMP[j, t] = -dual_value of the balance constraint
    for node j at time t.  The ordering of nodes in that constraint list is
    stored in dimensions["active_power_balance_node_order"] so callers can
    map constraint index → node index.

    Branch-flow variables P, Q, L are 2D (edge x time) to avoid the CVXPY
    canonicalization bugs that occur with 3-D (node x node x time) arrays.
    """

    node_ids = list(N)
    edge_ids = list(E)
    time_ids = list(T)

    n_nodes = len(node_ids)
    n_edges = len(edge_ids)
    n_time = len(time_ids)

    _validate_inputs(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_time=n_time,
        rho=rho,
        l_P=l_P,
        l_Q=l_Q,
        c=c,
        s_max=s_max,
        r=r,
        x=x,
        I_max=I_max,
        v_0=v_0,
    )

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
    if rho[root_node] != 0:
        raise ValueError("Root node must have rho value of 0.")

    # -------------------------------------------------------------------------
    # Decision variables (no battery variables in this formulation)
    # -------------------------------------------------------------------------
    p = cp.Variable((n_nodes, n_time), nonneg=True, name="p_{i,t}")
    q = cp.Variable((n_nodes, n_time), nonneg=True, name="q_{i,t}")
    s = cp.Variable((n_nodes, n_time), nonneg=True, name="s_{i,t}")

    # 2D (edge x time) branch-flow variables — avoids 3D CVXPY indexing bugs.
    P_e = cp.Variable((n_edges, n_time), name="P_{ij,t}")
    Q_e = cp.Variable((n_edges, n_time), name="Q_{ij,t}")
    L_e = cp.Variable((n_edges, n_time), nonneg=True, name="L_{ij,t}")
    V = cp.Variable((n_nodes, n_time), name="V_{i,t}")

    # -------------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------------
    constraints: Dict[str, List[cp.Constraint]] = {}

    constraints["voltage_bounds"] = []
    constraints["generator_apparent_power"] = []
    constraints["generation_apparent_power_cap"] = []

    for node in node_ids:
        j_idx = node_to_idx[node]
        for t_idx in range(n_time):
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

    # Track which node_idx corresponds to each entry in active_power_balance.
    # The i-th entry in constraints["active_power_balance"] is the balance
    # constraint for node active_power_balance_node_order[i].
    # LMP[j, t] = -dual_value of that constraint evaluated at time index t.
    lmp_node_order: List[int] = []

    for node in node_ids:
        j_idx = node_to_idx[node]

        if j_idx == root_node_idx:
            # Root: generation equals total outgoing flow (no incoming edge).
            children_P: Any = np.zeros(n_time)
            children_Q: Any = np.zeros(n_time)
            for e_idx in outgoing_edges_by_node[node]:
                children_P = children_P + P_e[e_idx, :]
                children_Q = children_Q + Q_e[e_idx, :]

            constraints["active_power_balance"].append(p[j_idx, :] == children_P)
            constraints["reactive_power_balance"].append(q[j_idx, :] == children_Q)
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

        # Active power balance — no batteries, no delta slack (1)
        constraints["active_power_balance"].append(
            P_e[e_in, :]
            == (l_P[j_idx, :] - p[j_idx, :] + rij * L_e[e_in, :] + children_P)
        )
        lmp_node_order.append(j_idx)

        # Reactive power balance (2)
        constraints["reactive_power_balance"].append(
            Q_e[e_in, :]
            == (l_Q[j_idx, :] - q[j_idx, :] + xij * L_e[e_in, :] + children_Q)
        )

        # Voltage balance (3)
        constraints["voltage_balance"].append(
            V[j_idx, :]
            == V[i_idx, :]
            + (rij**2 + xij**2) * L_e[e_in, :]
            - 2.0 * (rij * P_e[e_in, :] + xij * Q_e[e_in, :])
        )

        # Thermal limit (9)
        constraints["thermal_limits"].append(L_e[e_in, :] <= I_max_ij**2)

        # Current relation, relaxed (4)
        for t_idx in range(n_time):
            constraints["current_relation_relaxed"].append(
                L_e[e_in, t_idx]
                >= cp.quad_over_lin(
                    cp.hstack([P_e[e_in, t_idx], Q_e[e_in, t_idx]]),
                    V[i_idx, t_idx],
                )
            )

    # -------------------------------------------------------------------------
    # Objective: minimize generation cost at root node
    # -------------------------------------------------------------------------
    objective = cp.Minimize(cp.sum(cp.multiply(c, p)))

    all_constraints = [con for group in constraints.values() for con in group]
    problem = cp.Problem(objective, all_constraints)

    variables: Dict[str, Any] = {
        "p_{i,t}": p,
        "q_{i,t}": q,
        "s_{i,t}": s,
        "P_{ij,t}": P_e,
        "Q_{ij,t}": Q_e,
        "L_{ij,t}": L_e,
        "V_{i,t}": V,
    }

    dimensions: Dict[str, Any] = {
        "|N|": n_nodes,
        "|E|": n_edges,
        "|T|": n_time,
        "root_node": int(root_node),
        # i-th entry of active_power_balance list → constraint for node_idx i
        "active_power_balance_node_order": lmp_node_order,
    }

    return VPPFormulation(
        problem=problem,
        variables=variables,
        constraints=constraints,
        dimensions=dimensions,
    )


def _validate_inputs(
    n_nodes: int,
    n_edges: int,
    n_time: int,
    rho: Mapping[int, int],
    l_P: np.ndarray,
    l_Q: np.ndarray,
    c: np.ndarray,
    s_max: np.ndarray,
    r: np.ndarray,
    x: np.ndarray,
    I_max: np.ndarray,
    v_0: float,
) -> None:
    if n_nodes == 0:
        raise ValueError("N cannot be empty.")
    if n_edges == 0:
        raise ValueError("E cannot be empty.")
    if n_time == 0:
        raise ValueError("T cannot be empty.")
    if l_P.shape != (n_nodes, n_time):
        raise ValueError("l_P must have shape (|N|, |T|).")
    if l_Q.shape != (n_nodes, n_time):
        raise ValueError("l_Q must have shape (|N|, |T|).")
    if c.shape != (n_nodes, n_time):
        raise ValueError("c must have shape (|N|, |T|).")
    if s_max.shape != (n_nodes,):
        raise ValueError("s_max must have shape (|N|,).")
    if r.shape != (n_edges,):
        raise ValueError("r must have shape (|E|,).")
    if x.shape != (n_edges,):
        raise ValueError("x must have shape (|E|,).")
    if I_max.shape != (n_edges,):
        raise ValueError("I_max must have shape (|E|,).")
    if len(rho) != n_nodes:
        raise ValueError("rho must contain one entry per node.")
    if v_0 != 1.0:
        raise ValueError(
            "v_0 should most likely be 1.0 p.u.."
            "If it is not the case, think about why and review the code."
        )
