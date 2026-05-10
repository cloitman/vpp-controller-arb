from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cvxpy as cp
import numpy as np

from .optimization_minimization import VPPFormulation


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
    Allocate a fixed battery energy budget across nodes to maximize LMP arbitrage.

    This is Stage 2 of the two-stage approach:
      Stage 1  →  solve no-battery OPF  →  get LMP[j, t] at each node and time
      Stage 2  →  this function  →  optimally place and operate batteries using
                  the Stage-1 LMPs as fixed market prices

    The network is intentionally absent here.  LMPs from Stage 1 already encode
    the effect of transmission losses and congestion.  The battery variables are:

      P_ch[j, t]           charge power drawn from the grid at node j, time t (MW)
      P_dis[j, t]          discharge power injected to the grid at node j, time t (MW)
      e[j, t]              stored energy at node j, time t (MWh)
      P_batt_max[j]        installed power capacity at node j (MW)   — decision
      e_batt_max_by_node[j] installed energy capacity at node j (MWh) — decision

    Objective:
      Maximize  Σ_j Σ_t  lmp[j,t] · (P_dis[j,t] − P_ch[j,t]) · Δt

    This is a linear program (LP): the battery dynamics and capacity constraints
    are all linear, so it solves rapidly even for large grids.

    Parameters
    ----------
    lmp : np.ndarray, shape (n_nodes, n_time)
        Locational marginal prices from Stage 1.  lmp[j, t] is the marginal cost
        of supplying one additional MW at node j during hour t ($/MWh).
    e_batt_max : float
        Total installed energy capacity budget across all nodes (MWh).
    alpha : float
        Battery duration (hours).  Enforces e_batt_max_by_node = alpha · P_batt_max.
    eta_ch, eta_dis : float
        Round-trip charge / discharge efficiency (0 < η ≤ 1).
    e_0 : float
        Initial (and terminal) stored energy at every node (MWh).
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

    # Identify root node (no incoming edge)
    parent_edge_by_node: Dict[int, int] = {}
    for edge_idx, (i, j) in enumerate(edge_ids):
        parent_edge_by_node[j] = edge_idx
    root_candidates = [node for node in node_ids if node not in parent_edge_by_node]
    if len(root_candidates) != 1:
        raise ValueError("Expected exactly one root node in radial feeder.")
    root_node = root_candidates[0]
    root_node_idx = node_to_idx[root_node]

    # -------------------------------------------------------------------------
    # Decision variables — battery only, no network-flow variables
    # -------------------------------------------------------------------------
    P_ch = cp.Variable((n_nodes, n_time), nonneg=True, name="P^{ch}_{j,t}")
    P_dis = cp.Variable((n_nodes, n_time), nonneg=True, name="P^{dis}_{j,t}")
    # e has n_time+1 columns: indices 0..n_time (start of each interval + end of last)
    e = cp.Variable((n_nodes, n_time + 1), name="e_{j,t}")

    P_batt_max = cp.Variable(n_nodes, nonneg=True, name="P^{batt}_{j,max}")
    e_batt_max_by_node = cp.Variable(n_nodes, nonneg=True, name="e^{batt}_{j,max}")

    # -------------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------------
    constraints: Dict[str, List[cp.Constraint]] = {}

    constraints["battery_energy_limits"] = []
    constraints["battery_cycle"] = []
    constraints["battery_energy_dynamics"] = []
    constraints["battery_charge_power_capacity"] = []
    constraints["battery_discharge_power_capacity"] = []
    constraints["battery_energy_power_link"] = []

    for j_idx in range(n_nodes):
        # Energy limits: 0 ≤ e[j, t] ≤ e_batt_max_by_node[j]  for all t ∈ {0..T}
        constraints["battery_energy_limits"].append(e[j_idx, :] >= 0.0)
        constraints["battery_energy_limits"].append(
            e[j_idx, :] <= e_batt_max_by_node[j_idx]
        )

        # Cycle constraint: start and end at the same state of charge
        constraints["battery_cycle"].append(e[j_idx, 0] == e_0)
        constraints["battery_cycle"].append(e[j_idx, n_time] == e_0)

        # Energy dynamics (vector form over all time steps):
        #   e[j, t+1] = e[j, t] + (η_ch · P_ch[j,t] − (1/η_dis) · P_dis[j,t]) · Δt
        constraints["battery_energy_dynamics"].append(
            e[j_idx, 1:]
            == e[j_idx, :-1]
            + (eta_ch * P_ch[j_idx, :] - (1.0 / eta_dis) * P_dis[j_idx, :]) * delta_t
        )

        # Power capacity limits
        constraints["battery_charge_power_capacity"].append(
            P_ch[j_idx, :] <= P_batt_max[j_idx]
        )
        constraints["battery_discharge_power_capacity"].append(
            P_dis[j_idx, :] <= P_batt_max[j_idx]
        )

        # Energy–power link: e_batt_max = α · P_batt_max  (fixed duration)
        constraints["battery_energy_power_link"].append(
            e_batt_max_by_node[j_idx] == alpha * P_batt_max[j_idx]
        )

    # Total capacity budget across all nodes
    constraints["battery_total_capacity"] = [cp.sum(e_batt_max_by_node) <= e_batt_max]

    # No battery at root node (root is a generator, not a storage site)
    constraints["no_battery_at_root"] = [e_batt_max_by_node[root_node_idx] == 0.0]

    # -------------------------------------------------------------------------
    # Objective: maximize arbitrage profit using Stage-1 LMPs as fixed prices
    #
    # profit = Σ_j Σ_t  lmp[j,t] · (P_dis[j,t] − P_ch[j,t]) · Δt
    #
    # Positive when battery discharges at high-LMP hours and charges at low-LMP
    # hours.  Efficiency losses are captured in the energy dynamics constraints.
    # -------------------------------------------------------------------------
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
