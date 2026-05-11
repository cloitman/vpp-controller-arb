import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vpp_controller.paths import FIGURE_PATH, OUTPUT_DIR
from vpp_controller.results_format import load_day_optimization_result


def create_output_folder(date_str):
    OUTPATH = FIGURE_PATH / date_str
    OUTPATH.mkdir(parents=True, exist_ok=True)


def get_results_dict(json_path):
    results = load_day_optimization_result(json_path)
    return results


# ---------------------------------------------------------------------------
# Objective / capacity overview
# ---------------------------------------------------------------------------

def plot_objective_vs_capacity(results_list, OUT_PATH):
    """Line + marginal-benefit bar: arbitrage profit (y) vs total battery capacity (x).

    Args:
        results_list: list of results dicts, one per JSON file for a given date.
    """
    capacities = [sum(r["variables"]["e^{batt}_{j,max}"]) for r in results_list]
    objectives = [r["objective_value"] for r in results_list]

    pairs = sorted(zip(capacities, objectives))
    capacities_sorted, objectives_sorted = zip(*pairs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        capacities_sorted, objectives_sorted,
        marker='o', linewidth=1.5, color='steelblue', label='Arbitrage Profit'
    )
    ax.set_xlabel('Total Battery Capacity Budget (MWh)')
    ax.set_ylabel('Arbitrage Profit ($)', color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')

    caps = list(capacities_sorted)
    objs = list(objectives_sorted)
    if len(caps) > 1:
        ax2 = ax.twinx()
        mid_caps = [(caps[k] + caps[k + 1]) / 2 for k in range(len(caps) - 1)]
        bar_widths = [(caps[k + 1] - caps[k]) * 0.5 for k in range(len(caps) - 1)]
        marginals = [(objs[k + 1] - objs[k]) / (caps[k + 1] - caps[k]) for k in range(len(caps) - 1)]
        ax2.bar(mid_caps, marginals, width=bar_widths, color='darkorange', alpha=0.45, label='Marginal Profit')
        ax2.set_ylabel('Marginal Profit (Δ$ / ΔMWh)', color='darkorange')
        ax2.tick_params(axis='y', labelcolor='darkorange')
        ax2.axhline(0, color='darkorange', linewidth=0.6, linestyle='--')

    ax.set_title('Battery Arbitrage Profit vs Capacity Budget')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_PATH / "objective_vs_capacity.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_capacity_allocation(results_list, OUT_PATH):
    """Grouped bar: per-node share of total capacity, one group per node, one bar per scenario.

    Args:
        results_list: list of results dicts, one per JSON file for a given date.
    """
    sorted_results = sorted(
        results_list, key=lambda r: sum(r["variables"]["e^{batt}_{j,max}"])
    )
    total_caps = [sum(r["variables"]["e^{batt}_{j,max}"]) for r in sorted_results]
    n_scenarios = len(sorted_results)
    n_nodes = len(sorted_results[0]["variables"]["e^{batt}_{j,max}"])
    nodes = np.arange(n_nodes)

    bar_width = 0.8 / n_scenarios
    fig, ax = plt.subplots(figsize=(max(10, n_nodes * 1.5), 6))

    for i, (r, total) in enumerate(zip(sorted_results, total_caps)):
        fractions = [v / total if total > 0 else 0 for v in r["variables"]["e^{batt}_{j,max}"]]
        offsets = nodes + (i - n_scenarios / 2 + 0.5) * bar_width
        ax.bar(offsets, fractions, width=bar_width, label=f"Cap = {total:.1f}")

    ax.set_xlabel("Node")
    ax.set_ylabel("Fraction of Total Capacity")
    ax.set_title("Battery Capacity Allocation by Node")
    ax.set_xticks(nodes)
    ax.set_xticklabels([str(i) for i in range(n_nodes)])
    ax.legend(title="Total Capacity (MWh)", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_PATH / "capacity_allocation_by_node.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_node_profit_by_capacity(results_list, OUT_PATH):
    """Grouped bar: per-node LMP arbitrage profit for each capacity scenario.

    Profit at each node = Σ_t  lmp[node, t] · P^{batt}[node, t].
    Uses the LMP stored in result["variables"]["lmp"] (embedded by the runner).

    Args:
        results_list: list of results dicts, one per JSON file for a given date.
    """
    sorted_results = sorted(results_list, key=lambda r: sum(r['variables']['e^{batt}_{j,max}']))
    total_caps = [sum(r['variables']['e^{batt}_{j,max}']) for r in sorted_results]
    n_scenarios = len(sorted_results)
    n_nodes = len(sorted_results[0]['variables']['e^{batt}_{j,max}'])
    nodes = np.arange(n_nodes)

    bar_width = 0.8 / n_scenarios
    fig, ax = plt.subplots(figsize=(max(10, n_nodes * 1.5), 6))

    for i, (r, total) in enumerate(zip(sorted_results, total_caps)):
        lmp = np.array(r['variables']['lmp'], dtype=float)        # (n_nodes, n_time)
        dispatch = np.array(r['variables']['P^{batt}_{j,t}'], dtype=float)  # (n_nodes, n_time)
        profits = [float(np.dot(lmp[node, :], dispatch[node, :])) for node in range(n_nodes)]
        offsets = nodes + (i - n_scenarios / 2 + 0.5) * bar_width
        ax.bar(offsets, profits, width=bar_width, label=f'Cap = {total:.1f}')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Node')
    ax.set_ylabel('Arbitrage Profit ($)')
    ax.set_title('Battery Arbitrage Profit by Node and Capacity Scenario')
    ax.set_xticks(nodes)
    ax.set_xticklabels([str(i) for i in range(n_nodes)])
    ax.legend(title='Total Capacity (MWh)', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_PATH / 'node_profit_by_capacity.png', dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# LMP visualisation
# ---------------------------------------------------------------------------

def plot_lmp_heatmap(results_dict, OUT_PATH):
    """Heatmap of LMP[node, hour] — shows the price signal driving battery dispatch.

    Rows = nodes, columns = hours.  Warm colours = high LMP (discharge here);
    cool colours = low LMP (charge here).

    Args:
        results_dict: single results dict loaded from a Stage-2 JSON file.
    """
    if "lmp" not in results_dict["variables"]:
        print("plot_lmp_heatmap: 'lmp' not found in variables, skipping.")
        return

    lmp = np.array(results_dict["variables"]["lmp"], dtype=float)   # (n_nodes, n_time)
    total_cap = float(sum(results_dict["variables"]["e^{batt}_{j,max}"]))
    n_nodes, n_time = lmp.shape

    fig, ax = plt.subplots(figsize=(max(10, n_time // 2), max(4, n_nodes // 2)))
    im = ax.imshow(lmp, aspect='auto', origin='upper', cmap='RdYlGn_r')
    plt.colorbar(im, ax=ax, label='LMP ($/MWh)')

    ax.set_xlabel('Hour')
    ax.set_ylabel('Node')
    ax.set_title(f'Locational Marginal Prices — Capacity Budget {total_cap:.1f} MWh')
    ax.set_xticks(range(0, n_time, max(1, n_time // 12)))
    ax.set_yticks(range(n_nodes))
    ax.set_yticklabels([str(j) for j in range(n_nodes)])
    plt.tight_layout()
    plt.savefig(OUT_PATH / f'lmp_heatmap_cap{total_cap:.0f}MWh.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_node_lmp_by_capacity(results_list, node, OUT_PATH):
    """Line chart: LMP at a single node over time, one line per capacity scenario.

    Because Stage-1 LMPs are independent of battery capacity, the lines will
    overlap — this confirms the LMP is a fixed price signal.  Overlaid with
    the battery dispatch so you can see when the battery charges vs discharges
    relative to the price.

    Args:
        results_list: list of results dicts, one per JSON file.
        node: integer node index to plot.
        OUT_PATH: output folder.
    """
    sorted_results = sorted(results_list, key=lambda r: sum(r['variables']['e^{batt}_{j,max}']))
    n_scen = len(sorted_results)

    # Check LMP is present
    if "lmp" not in sorted_results[0]["variables"]:
        print("plot_node_lmp_by_capacity: 'lmp' not found in variables, skipping.")
        return

    # LMP is the same across all scenarios — extract from first
    lmp_ref = np.array(sorted_results[0]["variables"]["lmp"], dtype=float)
    lmp_node = lmp_ref[node, :]
    hours = np.arange(len(lmp_node))

    colors = [plt.cm.tab10(i / max(n_scen - 1, 1)) for i in range(n_scen)]

    fig, ax1 = plt.subplots(figsize=(11, 5))

    # LMP on primary axis (same for all scenarios)
    ax1.plot(hours, lmp_node, color='black', linewidth=2, linestyle='--', label='LMP ($/MWh)', zorder=5)
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('LMP ($/MWh)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Battery dispatch on secondary axis, one line per capacity
    ax2 = ax1.twinx()
    cap_handles = []
    for r, color in zip(sorted_results, colors):
        total_cap = sum(r['variables']['e^{batt}_{j,max}'])
        if total_cap == 0:
            continue
        dispatch = np.array(r['variables']['P^{batt}_{j,t}'], dtype=float)[node, :]
        ax2.plot(hours, dispatch, color=color, linewidth=1.5, alpha=0.8, marker='o', markersize=2)
        cap_handles.append(Line2D([0], [0], color=color, linewidth=2, label=f'{total_cap:.1f} MWh'))

    ax2.axhline(0, color='gray', linewidth=0.6, linestyle=':')
    ax2.set_ylabel('Battery Dispatch (MW)  [+ = discharge]')

    lmp_handle = Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='LMP ($/MWh)')
    ax1.legend(
        handles=[lmp_handle] + cap_handles,
        title='LMP / Capacity',
        bbox_to_anchor=(1.12, 1),
        loc='upper left',
        fontsize=8,
    )
    ax1.set_title(f'LMP and Battery Dispatch at Node {node}')
    ax1.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_PATH / f'node_{node}_lmp_by_capacity.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_lmp_3d(results_dict, OUT_PATH):
    """3D bar chart of LMP[node, hour] — shows the nodal price structure spatially.

    Args:
        results_dict: single results dict loaded from a Stage-2 JSON file.
    """
    if "lmp" not in results_dict["variables"]:
        print("plot_lmp_3d: 'lmp' not found in variables, skipping.")
        return

    lmp = np.array(results_dict["variables"]["lmp"], dtype=float)
    total_cap = float(sum(results_dict["variables"]["e^{batt}_{j,max}"]))
    n_nodes, n_hours = lmp.shape

    node_idx, hour_idx = np.meshgrid(np.arange(n_nodes), np.arange(n_hours), indexing='ij')
    node_flat = node_idx.ravel().astype(float)
    hour_flat = hour_idx.ravel().astype(float)
    values_flat = lmp.ravel()

    # Colour: high LMP = orange/red (discharge opportunity), low/negative = blue (charge)
    vmax = np.abs(values_flat).max() or 1.0
    norm_vals = values_flat / vmax
    colors = [plt.cm.RdYlGn_r(0.5 + 0.5 * v) for v in np.clip(norm_vals, -1, 1)]

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(
        node_flat - 0.3, hour_flat - 0.3, np.zeros_like(values_flat),
        0.6, 0.6, values_flat,
        color=colors, alpha=0.85, zsort='average',
    )
    ax.set_xlabel('Node')
    ax.set_ylabel('Hour')
    ax.set_zlabel('LMP ($/MWh)')
    ax.set_title(f'Locational Marginal Prices by Node and Hour | Cap {total_cap:.1f} MWh')
    ax.set_xticks(range(n_nodes))
    ax.set_xticklabels([str(j) for j in range(n_nodes)])
    ax.set_yticks(range(0, n_hours, max(1, n_hours // 12)))
    plt.tight_layout()
    plt.savefig(
        OUT_PATH / f'lmp_3d_cap{total_cap:.0f}MWh.png', dpi=150,
        bbox_inches='tight', pad_inches=0.5,
    )
    plt.close()


# ---------------------------------------------------------------------------
# Pre-battery vs post-battery LMP visualisation
# ---------------------------------------------------------------------------

def plot_input_lmp(results_list, OUT_PATH):
    """Heatmap of the Stage-1 LMP used as input to the battery arbitrage problem.

    Called once — the input LMP is capacity-independent, so any result in
    results_list gives the same values.  Rows = nodes, columns = hours.

    Args:
        results_list: list of results dicts (any entry is used for the LMP).
        OUT_PATH: output folder.
    """
    ref = results_list[0]
    if "lmp" not in ref["variables"]:
        print("plot_input_lmp: 'lmp' not found in variables, skipping.")
        return

    lmp = np.array(ref["variables"]["lmp"], dtype=float)
    n_nodes, n_time = lmp.shape

    fig, ax = plt.subplots(figsize=(max(10, n_time // 2), max(4, n_nodes // 2)))
    im = ax.imshow(lmp, aspect="auto", origin="upper", cmap="RdYlGn_r")
    plt.colorbar(im, ax=ax, label="LMP ($/MWh)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Node")
    ax.set_title("Stage 1 LMP — Input to Battery Arbitrage (capacity-independent)")
    ax.set_xticks(range(0, n_time, max(1, n_time // 12)))
    ax.set_yticks(range(n_nodes))
    ax.set_yticklabels([str(j) for j in range(n_nodes)])
    plt.tight_layout()
    plt.savefig(OUT_PATH / "input_lmp_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_post_battery_lmp_by_capacity(results_list, node, OUT_PATH):
    """Line chart: pre-battery LMP vs post-battery LMP at a single node.

    The dashed black line shows the Stage-1 LMP that drove battery placement.
    Each coloured solid line shows the LMP that results after the battery
    dispatch for that capacity scenario is incorporated into the power flow.
    Converging toward the dashed line would indicate arbitrage is flattening
    prices; diverging could indicate unintended congestion effects.

    Args:
        results_list: list of results dicts, one per capacity scenario.
        node: integer node index to plot.
        OUT_PATH: output folder.
    """
    sorted_results = sorted(results_list, key=lambda r: sum(r["variables"]["e^{batt}_{j,max}"]))
    n_scen = len(sorted_results)

    if "post_battery_lmp" not in sorted_results[0]["variables"]:
        print("plot_post_battery_lmp_by_capacity: 'post_battery_lmp' not in variables, skipping.")
        return
    if "lmp" not in sorted_results[0]["variables"]:
        print("plot_post_battery_lmp_by_capacity: 'lmp' not in variables, skipping.")
        return

    input_lmp = np.array(sorted_results[0]["variables"]["lmp"], dtype=float)[node, :]
    hours = np.arange(len(input_lmp))
    colors = [plt.cm.viridis(i / max(n_scen - 1, 1)) for i in range(n_scen)]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hours, input_lmp, color="black", linewidth=2, linestyle="--",
            label="Stage 1 LMP (pre-battery)", zorder=5)

    cap_handles = [Line2D([0], [0], color="black", linewidth=2, linestyle="--",
                          label="Stage 1 LMP (pre-battery)")]
    for r, color in zip(sorted_results, colors):
        total_cap = sum(r["variables"]["e^{batt}_{j,max}"])
        post_lmp_node = np.array(r["variables"]["post_battery_lmp"], dtype=float)[node, :]
        ax.plot(hours, post_lmp_node, color=color, linewidth=1.5, alpha=0.85)
        cap_handles.append(
            Line2D([0], [0], color=color, linewidth=2, label=f"Post-batt {total_cap:.1f} MWh")
        )

    ax.set_xlabel("Hour")
    ax.set_ylabel("LMP ($/MWh)")
    ax.set_title(f"Pre- vs Post-Battery LMP at Node {node}")
    ax.legend(handles=cap_handles, title="Scenario", bbox_to_anchor=(1.05, 1),
              loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_PATH / f"node_{node}_post_battery_lmp_by_capacity.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Battery dispatch visualisation
# ---------------------------------------------------------------------------

def plot_battery_dispatch_3d(
    results_dict, OUT_PATH, normalize=False, filter_small_nodes=False
):
    """3D bar chart of battery charge/discharge by node (x) and hour (y).

    Discharging bars (positive P_batt) are blue; charging bars (negative) are red.

    Args:
        results_dict: single results dict for the scenario to visualise.
        normalize: if True, each bar shows dispatch as % of that node's capacity.
        filter_small_nodes: if True, omit nodes whose capacity is <0.1% of total.
    """
    node_caps = np.array(results_dict["variables"]["e^{batt}_{j,max}"])
    total_cap = float(node_caps.sum())

    dispatch = np.array(results_dict["variables"]["P^{batt}_{j,t}"], dtype=float)
    n_nodes, n_hours = dispatch.shape

    if filter_small_nodes and total_cap > 0:
        kept_nodes = [j for j in range(n_nodes) if node_caps[j] / total_cap >= 0.001]
    else:
        kept_nodes = list(range(n_nodes))

    dispatch = dispatch[kept_nodes, :]
    node_caps_kept = node_caps[kept_nodes]

    if normalize:
        for i, cap in enumerate(node_caps_kept):
            if cap > 0:
                dispatch[i, :] = dispatch[i, :] / cap * 100.0

    max_dispatch = np.abs(dispatch).max()
    if max_dispatch > 0:
        dispatch[np.abs(dispatch) < 0.001 * max_dispatch] = 0.0

    n_kept = len(kept_nodes)
    x_positions = np.arange(n_kept, dtype=float)
    node_idx, hour_idx = np.meshgrid(x_positions, np.arange(n_hours), indexing="ij")
    node_flat = node_idx.ravel()
    hour_flat = hour_idx.ravel()
    values_flat = dispatch.ravel()

    dx, dy = 0.6, 0.6
    dz = np.abs(values_flat)
    z_bottoms = np.where(values_flat >= 0, 0.0, values_flat)
    colors = ["steelblue" if v >= 0 else "tomato" for v in values_flat]

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(
        node_flat - dx / 2, hour_flat - dy / 2, z_bottoms,
        dx, dy, dz, color=colors, alpha=0.85, zsort="average",
    )
    ax.set_xlabel("Node")
    ax.set_ylabel("Hour")
    zlabel = "% of Node Capacity Dispatched" if normalize else "Charge / Discharge (MW)"
    ax.set_zlabel(zlabel)
    title_parts = ["Battery Dispatch by Node and Hour"]
    if normalize:
        title_parts.append("Normalized by Node Capacity")
    if filter_small_nodes:
        title_parts.append("Nodes <0.1% Filtered")
    ax.set_title(" | ".join(title_parts))
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(j) for j in kept_nodes])
    ax.set_yticks(range(0, n_hours, max(1, n_hours // 12)))
    legend_elements = [
        Patch(facecolor="steelblue", label="Discharging (+)"),
        Patch(facecolor="tomato", label="Charging (−)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    suffix = ("_norm" if normalize else "") + ("_filtered" if filter_small_nodes else "")
    plt.savefig(
        OUT_PATH / f"battery_dispatch_3d_cap{total_cap:.0f}MWh{suffix}.png",
        dpi=150, bbox_inches="tight", pad_inches=0.5,
    )
    plt.close()


def plot_node_dispatch_by_capacity(results_list, node, OUT_PATH, normalize=False):
    """Line chart: charge/discharge at a single node over time, one line per capacity.

    Args:
        results_list: list of results dicts, one per JSON file for a given date.
        node: integer index of the node to plot.
        OUT_PATH: folder in which to save the figure.
        normalize: if True, dispatch is shown as % of that node's capacity.
    """
    sorted_results = sorted(
        results_list, key=lambda r: sum(r["variables"]["e^{batt}_{j,max}"])
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for r in sorted_results:
        total_cap = sum(r["variables"]["e^{batt}_{j,max}"])
        dispatch = np.array(r["variables"]["P^{batt}_{j,t}"], dtype=float)[node, :]
        if normalize:
            node_cap = r["variables"]["e^{batt}_{j,max}"][node]
            if node_cap > 0:
                dispatch = dispatch / node_cap * 100.0
        ax.plot(
            range(len(dispatch)), dispatch,
            marker="o", markersize=3, linewidth=1.5, label=f"Cap = {total_cap:.1f}",
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Hour")
    ax.set_ylabel("% of Node Capacity Dispatched" if normalize else "Charge / Discharge (MW)")
    title = f"Battery Dispatch at Node {node} Across Capacity Scenarios"
    if normalize:
        title += " | Normalized by Node Capacity"
    ax.set_title(title)
    ax.legend(title="Total Capacity (MWh)", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    norm_suffix = "_norm" if normalize else ""
    plt.savefig(
        OUT_PATH / f"node_{node}_dispatch_by_capacity{norm_suffix}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close()


def plot_node_dispatch_and_lmp(results_list, node, OUT_PATH, normalize=False):
    """Combined line chart: battery dispatch (solid) overlaid with LMP (dashed).

    One dispatch line per capacity scenario; a single LMP line (same for all
    scenarios since Stage-1 LMPs are capacity-independent).

    Args:
        results_list: list of results dicts, one per JSON file for a given date.
        node: integer index of the node to plot.
        OUT_PATH: folder in which to save the figure.
        normalize: if True, dispatch is shown as % of that node's capacity.
    """
    sorted_results = sorted(results_list, key=lambda r: sum(r["variables"]["e^{batt}_{j,max}"]))
    n_scen = len(sorted_results)

    if "lmp" not in sorted_results[0]["variables"]:
        print("plot_node_dispatch_and_lmp: 'lmp' not found in variables, skipping.")
        return

    lmp_node = np.array(sorted_results[0]["variables"]["lmp"], dtype=float)[node, :]
    hours = np.arange(len(lmp_node))
    colors = [plt.cm.tab10(i / max(n_scen - 1, 1)) for i in range(n_scen)]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax2 = ax1.twinx()
    ax2.plot(hours, lmp_node, color='black', linewidth=1.8, linestyle='--', zorder=5, label='LMP')
    ax2.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax2.set_ylabel('LMP ($/MWh)', color='dimgray')
    ax2.tick_params(axis='y', labelcolor='dimgray')

    cap_handles = []
    for r, color in zip(sorted_results, colors):
        total_cap = sum(r["variables"]["e^{batt}_{j,max}"])
        dispatch = np.array(r["variables"]["P^{batt}_{j,t}"], dtype=float)[node, :]
        if normalize:
            node_cap = r["variables"]["e^{batt}_{j,max}"][node]
            if node_cap > 0:
                dispatch = dispatch / node_cap * 100.0
        ax1.plot(hours, dispatch, marker="o", markersize=3, linewidth=1.5, color=color)
        cap_handles.append(Line2D([0], [0], color=color, linewidth=2, label=f"{total_cap:.1f} MWh"))

    ax1.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax1.set_xlabel("Hour")
    ylabel = "% of Node Capacity" if normalize else "Dispatch (MW)  [+ = discharge]"
    ax1.set_ylabel(ylabel)
    ax1.set_title(f"Dispatch (—) and LMP (--) at Node {node}")

    lmp_handle = Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='LMP ($/MWh)')
    ax1.legend(
        handles=cap_handles + [lmp_handle],
        title="Capacity / Type",
        bbox_to_anchor=(1.12, 1),
        loc="upper left",
        fontsize=8,
    )
    ax1.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    norm_suffix = "_norm" if normalize else ""
    fig.savefig(
        OUT_PATH / f"node_{node}_dispatch_and_lmp{norm_suffix}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generation and voltage impact of batteries
# ---------------------------------------------------------------------------

def plot_generation_and_cost(results_list, OUT_PATH):
    """Two-panel plot: total system generation and cost per hour, with vs without batteries.

    Top panel: total active power generation summed across all nodes (MW).
    Bottom panel: total generation cost ($/h = sum_j c[j,t] * p[j,t]).

    The 'no battery' baseline (Stage 1 OPF) is a single dashed black line.
    Each capacity scenario (Stage 2 post-battery OPF) is a coloured line.

    Args:
        results_list: list of results dicts.
        OUT_PATH: output folder.
    """
    required = {"p_no_batt", "p_post_batt", "c_{i,t}"}
    ref = results_list[0]
    if not required.issubset(ref["variables"]):
        missing = required - set(ref["variables"])
        print(f"plot_generation_and_cost: missing variables {missing}, skipping.")
        return

    sorted_results = sorted(results_list, key=lambda r: sum(r["variables"]["e^{batt}_{j,max}"]))
    n_scen = len(sorted_results)
    colors = [plt.cm.viridis(i / max(n_scen - 1, 1)) for i in range(n_scen)]

    p_no_batt = np.array(sorted_results[0]["variables"]["p_no_batt"], dtype=float)
    c_mat = np.array(sorted_results[0]["variables"]["c_{i,t}"], dtype=float)
    gen_no_batt = p_no_batt.sum(axis=0)          # sum over nodes → shape (n_time,)
    cost_no_batt = (c_mat * p_no_batt).sum(axis=0)
    hours = np.arange(len(gen_no_batt))

    fig, (ax_gen, ax_cost) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax_gen.plot(hours, gen_no_batt, color="black", linewidth=2, linestyle="--",
                label="No battery", zorder=5)
    ax_cost.plot(hours, cost_no_batt, color="black", linewidth=2, linestyle="--",
                 label="No battery", zorder=5)

    cap_handles = [Line2D([0], [0], color="black", linewidth=2, linestyle="--",
                          label="No battery")]
    for r, color in zip(sorted_results, colors):
        total_cap = sum(r["variables"]["e^{batt}_{j,max}"])
        p_post = np.array(r["variables"]["p_post_batt"], dtype=float)
        gen_post = p_post.sum(axis=0)
        cost_post = (c_mat * p_post).sum(axis=0)
        ax_gen.plot(hours, gen_post, color=color, linewidth=1.5, alpha=0.85)
        ax_cost.plot(hours, cost_post, color=color, linewidth=1.5, alpha=0.85)
        cap_handles.append(
            Line2D([0], [0], color=color, linewidth=2, label=f"{total_cap:.1f} MWh")
        )

    ax_gen.set_ylabel("Total Generation (MW)")
    ax_gen.set_title("System Generation With and Without Batteries")
    ax_gen.grid(True, linestyle="--", alpha=0.4)

    ax_cost.set_xlabel("Hour")
    ax_cost.set_ylabel("Total Generation Cost ($/h)")
    ax_cost.set_title("System Generation Cost With and Without Batteries")
    ax_cost.grid(True, linestyle="--", alpha=0.4)

    ax_gen.legend(handles=cap_handles, title="Battery Capacity",
                  bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PATH / "generation_and_cost.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_voltage_by_capacity(results_list, node, OUT_PATH):
    """Line chart: per-unit voltage at one node over time, with v_min/v_max bounds.

    The dashed black line is the no-battery voltage (Stage 1, hard bounds enforced).
    Each coloured line is the post-battery voltage from the voltage-slack OPF, which
    is allowed to exceed the bounds — showing where batteries naturally push voltage.
    Red shading marks the infeasible regions (outside the hard bounds).

    Note: V_{i,t} stored in results is |V|^2 in p.u., so this function takes sqrt.

    Args:
        results_list: list of results dicts.
        node: integer node index to plot.
        OUT_PATH: output folder.
    """
    required = {"V_no_batt", "V_post_batt", "v_min", "v_max"}
    ref = results_list[0]
    if not required.issubset(ref["variables"]):
        missing = required - set(ref["variables"])
        print(f"plot_voltage_by_capacity node {node}: missing variables {missing}, skipping.")
        return

    sorted_results = sorted(results_list, key=lambda r: sum(r["variables"]["e^{batt}_{j,max}"]))
    n_scen = len(sorted_results)
    colors = [plt.cm.viridis(i / max(n_scen - 1, 1)) for i in range(n_scen)]

    V_no_batt = np.array(sorted_results[0]["variables"]["V_no_batt"], dtype=float)
    v_no_batt_node = np.sqrt(np.maximum(V_no_batt[node, :], 0.0))
    hours = np.arange(len(v_no_batt_node))

    v_min = float(np.array(sorted_results[0]["variables"]["v_min"]).flat[0])
    v_max = float(np.array(sorted_results[0]["variables"]["v_max"]).flat[0])

    # Collect all post-battery voltages to set y-axis limits sensibly.
    all_post_v = []
    for r in sorted_results:
        V_post = np.array(r["variables"]["V_post_batt"], dtype=float)
        all_post_v.append(np.sqrt(np.maximum(V_post[node, :], 0.0)))

    y_lo = min(v_min - 0.02, min(v.min() for v in all_post_v) - 0.01)
    y_hi = max(v_max + 0.02, max(v.max() for v in all_post_v) + 0.01)

    fig, ax = plt.subplots(figsize=(11, 5))

    # Shade infeasible regions.
    ax.axhspan(y_lo, v_min, color="red", alpha=0.08, zorder=0)
    ax.axhspan(v_max, y_hi, color="red", alpha=0.08, zorder=0)

    # Hard-bound reference lines.
    ax.axhline(v_min, color="red", linewidth=1.3, linestyle="--")
    ax.axhline(v_max, color="red", linewidth=1.3, linestyle="--")

    # No-battery baseline.
    ax.plot(hours, v_no_batt_node, color="black", linewidth=2, linestyle="--", zorder=5)

    cap_handles = [Line2D([0], [0], color="black", linewidth=2, linestyle="--",
                          label="No battery (within bounds)")]
    for r, color, v_post_node in zip(sorted_results, colors, all_post_v):
        total_cap = sum(r["variables"]["e^{batt}_{j,max}"])
        ax.plot(hours, v_post_node, color=color, linewidth=1.5, alpha=0.85)
        cap_handles.append(
            Line2D([0], [0], color=color, linewidth=2, label=f"{total_cap:.1f} MWh")
        )

    bound_handles = [
        Line2D([0], [0], color="red", linewidth=1.5, linestyle="--",
               label=f"v_min / v_max = {v_min} / {v_max} p.u."),
        Patch(facecolor="red", alpha=0.15, label="Infeasible region"),
    ]
    ax.legend(handles=cap_handles + bound_handles, title="Scenario / Bounds",
              bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(
        f"Voltage at Node {node} — Post-battery voltages use relaxed bounds\n"
        f"(shaded = outside hard constraints)"
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_PATH / f"node_{node}_voltage_by_capacity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generic 3D / multi-version utilities (unchanged)
# ---------------------------------------------------------------------------

def plot_generic_value(
    results_dict, OUT_PATH, variable_key, dict_key="variables", filter_small_nodes=False
):
    """3D bar chart of any (n_nodes, n_hours) variable.

    Args:
        results_dict: single results dict.
        variable_key: key in results_dict[dict_key] to plot.
        filter_small_nodes: if True, omit nodes whose capacity is <0.1% of total.
    """
    node_caps = np.array(results_dict["variables"]["e^{batt}_{j,max}"])
    total_cap = float(node_caps.sum())
    val_array = np.array(results_dict[dict_key][variable_key], dtype=float)
    n_nodes, n_hours = val_array.shape

    if filter_small_nodes and total_cap > 0:
        kept_nodes = [j for j in range(n_nodes) if max(val_array[j]) >= 1e-3]
    else:
        kept_nodes = list(range(n_nodes))

    val_array = val_array[kept_nodes, :]
    max_val = np.abs(val_array).max()
    if max_val > 0:
        val_array[np.abs(val_array) < 0.001] = 0.0

    n_kept = len(kept_nodes)
    x_positions = np.arange(n_kept, dtype=float)
    node_idx, hour_idx = np.meshgrid(x_positions, np.arange(n_hours), indexing="ij")
    values_flat = val_array.ravel()

    dx, dy = 0.6, 0.6
    dz = np.abs(values_flat)
    z_bottoms = np.where(values_flat >= 0, 0.0, values_flat)
    colors = ["steelblue" if v >= 0 else "tomato" for v in values_flat]

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(
        node_idx.ravel() - dx / 2, hour_idx.ravel() - dy / 2, z_bottoms,
        dx, dy, dz, color=colors, alpha=0.85, zsort="average",
    )
    ax.set_xlabel("Node (#)")
    ax.set_ylabel("Time (h)")
    ax.set_zlabel(variable_key)
    ax.set_title(variable_key + " by Node and Hour" + (" | Nodes <0.1% Filtered" if filter_small_nodes else ""))
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(j) for j in kept_nodes])
    ax.set_yticks(range(0, n_hours, max(1, n_hours // 12)))
    suffix = "_filtered" if filter_small_nodes else ""
    plt.savefig(
        OUT_PATH / f"{variable_key}_{total_cap:.0f}MWh{suffix}.png",
        dpi=150, bbox_inches="tight", pad_inches=0.5,
    )
    plt.close()


def plot_multi_version_objectives(versions_results, OUT_PATH):
    """Objective vs capacity for multiple version runs on one plot.

    Args:
        versions_results: dict mapping version_label (str) -> list of results dicts.
        OUT_PATH: folder in which to save the figure.
    """
    n_vers = len(versions_results)
    colors = [plt.cm.tab10(i / max(n_vers - 1, 1)) for i in range(n_vers)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for (version, results_list), color in zip(versions_results.items(), colors):
        if not results_list:
            continue
        capacities = [sum(r["variables"]["e^{batt}_{j,max}"]) for r in results_list]
        objectives = [r["objective_value"] for r in results_list]
        pairs = sorted(zip(capacities, objectives))
        caps, objs = zip(*pairs)
        ax.plot(caps, objs, marker="o", linewidth=1.5, color=color, label=version)

    ax.set_xlabel("Total Battery Capacity Budget (MWh)")
    ax.set_ylabel("Arbitrage Profit ($)")
    ax.set_title("Arbitrage Profit vs Battery Capacity — Multi-Version Comparison")
    ax.legend(title="Version", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT_PATH / "objective_vs_capacity_multi_version.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def main(date_string):
    OUT_PATH = FIGURE_PATH / date_string
    create_output_folder(OUT_PATH)
    json_files = find_files_by_date(date_string)
    results_list = []
    
    for file in json_files:
        try:
            results_dict = get_results_dict(file)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            import pdb;pdb.set_trace()

        # Per-scenario 3D plots
        plot_battery_dispatch_3d(results_dict, OUT_PATH, normalize=False, filter_small_nodes=False)
        plot_lmp_heatmap(results_dict, OUT_PATH)
        try:
            plot_lmp_3d(results_dict, OUT_PATH)
        except Exception as e:
            print(f"plot_lmp_3d skipped: {e}")

        results_list.append(results_dict)

    # One-time: input LMP (capacity-independent Stage-1 prices)
    try:
        plot_input_lmp(results_list, OUT_PATH)
    except Exception as e:
        print(f"  plot_input_lmp: {e}")

    # Per-node, across-capacity plots
    n_nodes = len(results_list[0]["variables"]["e^{batt}_{j,max}"]) if results_list else 0
    for interestNode in range(n_nodes):
        try:
            plot_post_battery_lmp_by_capacity(results_list, interestNode, OUT_PATH)
        except Exception as e:
            print(f"  plot_post_battery_lmp_by_capacity node {interestNode}: {e}")
        try:
            plot_node_dispatch_by_capacity(results_list, interestNode, OUT_PATH)
        except Exception as e:
            print(f"  plot_node_dispatch_by_capacity node {interestNode}: {e}")
        try:
            plot_node_dispatch_and_lmp(results_list, interestNode, OUT_PATH)
        except Exception as e:
            print(f"  plot_node_dispatch_and_lmp node {interestNode}: {e}")

    # System-wide summary plots
    plot_objective_vs_capacity(results_list, OUT_PATH)
    plot_capacity_allocation(results_list, OUT_PATH)
    plot_node_profit_by_capacity(results_list, OUT_PATH)

    try:
        plot_generation_and_cost(results_list, OUT_PATH)
    except Exception as e:
        print(f"  plot_generation_and_cost: {e}")

    for interestNode in range(n_nodes):
        try:
            plot_voltage_by_capacity(results_list, interestNode, OUT_PATH)
        except Exception as e:
            print(f"  plot_voltage_by_capacity node {interestNode}: {e}")


def find_files_by_date(date_str):
    json_files = []
    for file_path in OUTPUT_DIR.rglob("*.json"):
        try:
            if date_str in file_path.read_text(encoding="utf-8"):
                json_files.append(file_path)
        except (UnicodeDecodeError, PermissionError):
            continue
    return json_files


VERSION_LABELS = {
    "v_1": "spring",
    "v_2": "fall",
    "v_3": "summer",
    "v_4": "winter",
}


def main_multi_version(versions=("v_1", "v_2", "v_3", "v_4")):
    """Load results for all specified versions and produce multi-version comparison plots."""
    OUT_PATH = FIGURE_PATH / "comparison"
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    versions_results = {}
    for v in versions:
        label = VERSION_LABELS.get(v, v)
        json_files = find_files_by_date(v)
        versions_results[label] = [get_results_dict(f) for f in json_files]

    plot_multi_version_objectives(versions_results, OUT_PATH)


if __name__ == "__main__":
    date_string = "spring_neg_costs_root_battery"
    main(date_string)
    # main_multi_version()
