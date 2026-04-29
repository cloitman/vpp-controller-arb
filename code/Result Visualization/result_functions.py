import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.patches import Patch

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
# from src.vpp_controller.demand_data import create_all_nodes_demand
from src.vpp_controller.paths import OUTPUT_DIR, FIGURE_PATH
# from src.vpp_controller.runner import run_day_optimization
from src.results_format import load_day_optimization_result


def create_output_folder(date_str):
    OUTPATH = FIGURE_PATH / date_str
    OUTPATH.mkdir(parents=True, exist_ok=True)


def get_results_dict(json_path):
    results = load_day_optimization_result(json_path)
    return results


def plot_objective_vs_capacity(results_list,OUT_PATH):
    """Scatter/line plot: objective value (y) vs total battery capacity constraint (x).

    Args:
        results_list: list of results dicts, one per JSON file for a given date.
    """
    capacities = [sum(r['variables']['e^{batt}_{j,max}']) for r in results_list]
    objectives = [r['objective_value'] for r in results_list]

    pairs = sorted(zip(capacities, objectives))
    capacities_sorted, objectives_sorted = zip(*pairs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(capacities_sorted, objectives_sorted, marker='o', linewidth=1.5, color='steelblue', label='Objective')
    ax.set_xlabel('Total Battery Capacity Constraint (MWh)')
    ax.set_ylabel('Objective Value', color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')

    caps = list(capacities_sorted)
    objs = list(objectives_sorted)
    if len(caps) > 1:
        ax2 = ax.twinx()
        mid_caps = [(caps[k] + caps[k + 1]) / 2 for k in range(len(caps) - 1)]
        bar_widths = [(caps[k + 1] - caps[k]) * 0.5 for k in range(len(caps) - 1)]
        marginals = [(objs[k + 1] - objs[k]) / (caps[k + 1] - caps[k]) for k in range(len(caps) - 1)]
        ax2.bar(mid_caps, marginals, width=bar_widths, color='darkorange', alpha=0.45, label='Marginal Benefit')
        ax2.set_ylabel('Marginal Benefit (ΔObjective / ΔCapacity)', color='darkorange')
        ax2.tick_params(axis='y', labelcolor='darkorange')
        ax2.axhline(0, color='darkorange', linewidth=0.6, linestyle='--')

    ax.set_title('Objective Value vs Battery Capacity Constraint')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_PATH / 'objective_vs_capacity.png', dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()
    return


def plot_node_profit_by_capacity(results_list, OUT_PATH):
    """Grouped bar chart: per-node profit for each capacity scenario.

    Profit at each node = dot product of P^{batt}_{j,t}[node] with p_{i,t}[0]
    (price time series). Positive values indicate net revenue; negative indicate net cost.

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
        prices = np.array(r['variables']['p_{i,t}'][0], dtype=float)
        profits = [
            -1*float(np.dot(np.array(r['variables']['P^{batt}_{j,t}'][node], dtype=float), prices))
            for node in range(n_nodes)
        ]
        offsets = nodes + (i - n_scenarios / 2 + 0.5) * bar_width
        ax.bar(offsets, profits, width=bar_width, label=f'Cap = {total:.1f}')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Node')
    ax.set_ylabel('Profit ($)')
    ax.set_title('Battery Profit by Node')
    ax.set_xticks(nodes)
    ax.set_xticklabels([str(i) for i in range(n_nodes)])
    ax.legend(title='Total Capacity (MWh)', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_PATH / 'node_profit_by_capacity.png', dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()
    return


def plot_capacity_allocation(results_list,OUT_PATH):
    """Grouped bar chart: per-node share of total capacity, one group per node, one bar per scenario.

    Each bar shows e^{batt}_{j,max}[node] / sum(e^{batt}_{j,max}) so bars within a
    scenario sum to 1, letting you compare how allocation shifts across capacity levels.

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
        fractions = [v / total for v in r['variables']['e^{batt}_{j,max}']]
        offsets = nodes + (i - n_scenarios / 2 + 0.5) * bar_width
        ax.bar(offsets, fractions, width=bar_width, label=f'Cap = {total:.1f}')

    ax.set_xlabel('Node')
    ax.set_ylabel('Fraction of Total Capacity')
    ax.set_title('Battery Capacity Allocation by Node')
    ax.set_xticks(nodes)
    ax.set_xticklabels([str(i) for i in range(n_nodes)])
    ax.legend(title='Total Capacity (MWh)', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_PATH / 'capacity_allocation_by_node.png', dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()
    return


def plot_battery_dispatch_3d(results_dict, OUT_PATH, normalize=False, filter_small_nodes=False):
    """3D bar chart of battery charge/discharge by node (x) and hour (y), value on z-axis.

    Charging bars (positive P) are blue; discharging bars (negative P) are red.
    Assumes results['variables']['P^{batt}_{j,t}'] is a list of n_nodes lists,
    each of length n_hours, indexed as [node][hour].

    Args:
        results_dict: single results dict for the scenario to visualise.
        normalize: if True, each bar shows dispatch as % of that node's capacity.
        filter_small_nodes: if True, omit nodes whose capacity is <0.1% of total.
    """
    node_caps = np.array(results_dict['variables']['e^{batt}_{j,max}'])
    total_cap = float(node_caps.sum())

    P_batt = results_dict['variables']['P^{batt}_{j,t}']
    dispatch = np.array(P_batt, dtype=float)  # shape: (n_nodes, n_hours)
    n_nodes, n_hours = dispatch.shape

    # Determine which nodes to keep
    if filter_small_nodes:
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
    node_idx, hour_idx = np.meshgrid(x_positions, np.arange(n_hours), indexing='ij')
    node_flat = node_idx.ravel()
    hour_flat = hour_idx.ravel()
    values_flat = dispatch.ravel()

    dx = 0.6
    dy = 0.6
    dz = np.abs(values_flat)
    z_bottoms = np.where(values_flat >= 0, 0.0, values_flat)
    colors = ['steelblue' if v >= 0 else 'tomato' for v in values_flat]

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(
        node_flat - dx / 2,
        hour_flat - dy / 2,
        z_bottoms,
        dx, dy, dz,
        color=colors,
        alpha=0.85,
        zsort='average',
    )

    ax.set_xlabel('Node')
    ax.set_ylabel('Hour')
    zlabel = '% of Node Capacity Dispatched' if normalize else 'Charge / Discharge (MW)'
    ax.set_zlabel(zlabel)
    title_parts = ['Battery Dispatch by Node and Hour']
    if normalize:
        title_parts.append('Normalized by Node Capacity')
    if filter_small_nodes:
        title_parts.append('Nodes <0.1% Filtered')
    ax.set_title(' | '.join(title_parts))
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(j) for j in kept_nodes])
    ax.set_yticks(range(0, n_hours, max(1, n_hours // 12)))

    legend_elements = [
        Patch(facecolor='steelblue', label='Discharging (+)'),
        Patch(facecolor='tomato', label='Charging (−)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    suffix = ('_norm' if normalize else '') + ('_filtered' if filter_small_nodes else '')
    plt.savefig(OUT_PATH / f'battery_dispatch_3d_cap{total_cap:.0f}MWh{suffix}.png', dpi=150, bbox_inches='tight', pad_inches=0.5)
    # plt.show()
    plt.close()
    return

def plot_transmission_congestion(results_dict, OUT_PATH, filter_small_nodes=False):
    """3D bar chart of line congestion by node (x) and hour (y), value on z-axis.
    Assumes results['duals']['thermal_limits'] is a list of n_nodes lists,
    each of length n_hours, indexed as [node][hour].

    Args:
        results_dict: single results dict for the scenario to visualise.
        normalize: if True, each bar shows dispatch as % of that node's capacity.
        filter_small_nodes: if True, omit nodes whose capacity is <0.1% of total.
    """
    node_caps = np.array(results_dict['variables']['e^{batt}_{j,max}'])
    total_cap = float(node_caps.sum())
    thermal_duals = results_dict['duals']['thermal_limits']
    thermal_shadows = np.array(thermal_duals, dtype=float)  # shape: (n_nodes, n_hours)
    n_nodes, n_hours = thermal_shadows.shape

    # Determine which nodes to keep
    if filter_small_nodes:
        kept_nodes = [j for j in range(n_nodes) if max(thermal_shadows[j]) >= 1e-3]
    else:
        kept_nodes = list(range(n_nodes))

    thermal_shadows = thermal_shadows[kept_nodes, :]

    max_shadow = np.abs(thermal_shadows).max()
    if max_shadow > 0:
        thermal_shadows[np.abs(thermal_shadows) < 0.001] = 0.0

    n_kept = len(kept_nodes)
    x_positions = np.arange(n_kept, dtype=float)
    node_idx, hour_idx = np.meshgrid(x_positions, np.arange(n_hours), indexing='ij')
    node_flat = node_idx.ravel()
    hour_flat = hour_idx.ravel()
    values_flat = thermal_shadows.ravel()

    dx = 0.6
    dy = 0.6
    dz = np.abs(values_flat)
    z_bottoms = np.where(values_flat >= 0, 0.0, values_flat)
    colors = ['steelblue' if v >= 0 else 'tomato' for v in values_flat]

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(
        node_flat - dx / 2,
        hour_flat - dy / 2,
        z_bottoms,
        dx, dy, dz,
        color=colors,
        alpha=0.85,
        zsort='average',
    )

    ax.set_xlabel('Node (#)')
    ax.set_ylabel('Time (h)')
    zlabel = 'Thermal Capacity Shadow Price (#)'
    ax.set_zlabel(zlabel)
    title_parts = ['Thermal Capacity Shadow Price by Node and Hour']
    if filter_small_nodes:
        title_parts.append('Nodes <0.1% Filtered')
    ax.set_title(' | '.join(title_parts))
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(j) for j in kept_nodes])
    ax.set_yticks(range(0, n_hours, max(1, n_hours // 12)))

    # legend_elements = [
    #     Patch(facecolor='steelblue', label='Discharging (+)'),
    #     Patch(facecolor='tomato', label='Charging (−)'),
    # ]
    # ax.legend(handles=legend_elements, loc='upper left')

    suffix = ('_filtered' if filter_small_nodes else '')
    plt.savefig(OUT_PATH / f'battery_thermal_duals_cap{total_cap:.0f}MWh{suffix}.png', dpi=150, bbox_inches='tight', pad_inches=0.5)
    # plt.show()
    plt.close()
    return


def plot_generic_value(results_dict, OUT_PATH,variable_key,dict_key='variables', filter_small_nodes=False):
    """3D bar chart of line congestion by node (x) and hour (y), value on z-axis.
    Assumes results[dict_key][variable_key] is a list of n_nodes lists,
    each of length n_hours, indexed as [node][hour].

    Args:
        results_dict: single results dict for the scenario to visualise.
        normalize: if True, each bar shows dispatch as % of that node's capacity.
        filter_small_nodes: if True, omit nodes whose capacity is <0.1% of total.
    """
    node_caps = np.array(results_dict['variables']['e^{batt}_{j,max}'])
    total_cap = float(node_caps.sum())
    vals = results_dict[dict_key][variable_key]
    val_array = np.array(vals, dtype=float)  # shape: (n_nodes, n_hours)
    n_nodes, n_hours = val_array.shape

    # Determine which nodes to keep
    if filter_small_nodes:
        kept_nodes = [j for j in range(n_nodes) if max(val_array[j]) >= 1e-3]
    else:
        kept_nodes = list(range(n_nodes))

    val_array = val_array[kept_nodes, :]

    max_val = np.abs(val_array).max()
    if max_val > 0:
        val_array[np.abs(val_array) < 0.001] = 0.0

    n_kept = len(kept_nodes)
    x_positions = np.arange(n_kept, dtype=float)
    node_idx, hour_idx = np.meshgrid(x_positions, np.arange(n_hours), indexing='ij')
    node_flat = node_idx.ravel()
    hour_flat = hour_idx.ravel()
    values_flat = val_array.ravel()

    dx = 0.6
    dy = 0.6
    dz = np.abs(values_flat)
    z_bottoms = np.where(values_flat >= 0, 0.0, values_flat)
    colors = ['steelblue' if v >= 0 else 'tomato' for v in values_flat]

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(
        node_flat - dx / 2,
        hour_flat - dy / 2,
        z_bottoms,
        dx, dy, dz,
        color=colors,
        alpha=0.85,
        zsort='average',
    )

    ax.set_xlabel('Node (#)')
    ax.set_ylabel('Time (h)')
    zlabel = variable_key
    ax.set_zlabel(zlabel)
    title_parts = [variable_key + ' by Node and Hour']
    if filter_small_nodes:
        title_parts.append('Nodes <0.1% Filtered')
    ax.set_title(' | '.join(title_parts))
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(j) for j in kept_nodes])
    ax.set_yticks(range(0, n_hours, max(1, n_hours // 12)))

    # legend_elements = [
    #     Patch(facecolor='steelblue', label='Discharging (+)'),
    #     Patch(facecolor='tomato', label='Charging (−)'),
    # ]
    # ax.legend(handles=legend_elements, loc='upper left')

    suffix = ('_filtered' if filter_small_nodes else '')
    plt.savefig(OUT_PATH / f'{variable_key}_{total_cap:.0f}MWh{suffix}.png', dpi=150, bbox_inches='tight', pad_inches=0.5)
    # plt.show()
    plt.close()
    return


def plot_node_dispatch_by_capacity(results_list, node, OUT_PATH, normalize=False):
    """Line chart of charge/discharge over time for a single node, one line per capacity scenario.

    Args:
        results_list: list of results dicts, one per JSON file for a given date.
        node: integer index of the node to plot.
        OUT_PATH: folder in which to save the figure.
        normalize: if True, dispatch is shown as % of that node's capacity for each scenario.
    """
    sorted_results = sorted(results_list, key=lambda r: sum(r['variables']['e^{batt}_{j,max}']))

    fig, ax = plt.subplots(figsize=(10, 5))
    for r in sorted_results:
        total_cap = sum(r['variables']['e^{batt}_{j,max}'])
        dispatch = np.array(r['variables']['P^{batt}_{j,t}'][node], dtype=float)
        if normalize:
            node_cap = r['variables']['e^{batt}_{j,max}'][node]
            if node_cap > 0:
                dispatch = dispatch / node_cap * 100.0
        ax.plot(range(len(dispatch)), dispatch, marker='o', markersize=3,
                linewidth=1.5, label=f'Cap = {total_cap:.1f}')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Hour')
    ax.set_ylabel('% of Node Capacity Dispatched' if normalize else 'Charge / Discharge (MW)')
    title = f'Battery Dispatch at Node {node} Across Capacity Scenarios'
    if normalize:
        title += ' | Normalized by Node Capacity'
    ax.set_title(title)
    ax.legend(title='Total Capacity (MWh)', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    norm_suffix = '_norm' if normalize else ''
    plt.savefig(OUT_PATH / f'node_{node}_dispatch_by_capacity{norm_suffix}.png', dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()
    return


def plot_node_slack_by_capacity(results_list, node, OUT_PATH, normalize=False):
    """Line chart of slack capacity over time for a single node/element, one line per capacity scenario.

    Args:
        results_list: list of results dicts, one per JSON file for a given date.
        node: integer index into delta^P_{i,t} (assumed to match node/element ordering).
        OUT_PATH: folder in which to save the figure.
        normalize: if True, slack is shown as % of that node's battery capacity.
    """
    sorted_results = sorted(results_list, key=lambda r: sum(r['variables']['e^{batt}_{j,max}']))

    fig, ax = plt.subplots(figsize=(10, 5))
    for r in sorted_results:
        total_cap = sum(r['variables']['e^{batt}_{j,max}'])
        slack = np.array(r['variables']['delta^P_{i,t}'][node], dtype=float)
        if normalize:
            node_cap = r['variables']['e^{batt}_{j,max}'][node]
            if node_cap > 0:
                slack = slack / node_cap * 100.0
        ax.plot(range(len(slack)), slack, marker='o', markersize=3,
                linewidth=1.5, label=f'Cap = {total_cap:.1f}')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Hour')
    ax.set_ylabel('% of Node Capacity' if normalize else 'Slack Capacity (MW)')
    title = f'Slack Capacity at Node {node} Across Capacity Scenarios'
    if normalize:
        title += ' | Normalized by Node Capacity'
    ax.set_title(title)
    ax.legend(title='Total Capacity (MWh)', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    norm_suffix = '_norm' if normalize else ''
    plt.savefig(OUT_PATH / f'node_{node}_slack_by_capacity{norm_suffix}.png', dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()
    return


def plot_slack_3d(results_dict, OUT_PATH, normalize=False, filter_small_nodes=False):
    """3D bar chart of slack capacity by node/element (x) and hour (y), value on z-axis.

    Args:
        results_dict: single results dict for the scenario to visualise.
        normalize: if True, each bar shows slack as % of that node's battery capacity.
        filter_small_nodes: if True, omit nodes whose battery capacity is <0.1% of total.
    """
    node_caps = np.array(results_dict['variables']['e^{batt}_{j,max}'])
    total_cap = float(node_caps.sum())

    slack = np.array(results_dict['variables']['delta^P_{i,t}'], dtype=float)  # (n_elements, n_hours)
    n_elements, n_hours = slack.shape

    if filter_small_nodes:
        kept_nodes = [j for j in range(n_elements) if node_caps[j] / total_cap >= 0.001]
    else:
        kept_nodes = list(range(n_elements))

    slack = slack[kept_nodes, :]
    node_caps_kept = node_caps[kept_nodes]

    if normalize:
        for i, cap in enumerate(node_caps_kept):
            if cap > 0:
                slack[i, :] = slack[i, :] / cap * 100.0

    max_val = np.abs(slack).max()
    if max_val > 0:
        slack[np.abs(slack) < 0.001 * max_val] = 0.0

    n_kept = len(kept_nodes)
    x_positions = np.arange(n_kept, dtype=float)
    node_idx, hour_idx = np.meshgrid(x_positions, np.arange(n_hours), indexing='ij')
    node_flat = node_idx.ravel()
    hour_flat = hour_idx.ravel()
    values_flat = slack.ravel()

    dx = 0.6
    dy = 0.6
    dz = np.abs(values_flat)
    z_bottoms = np.where(values_flat >= 0, 0.0, values_flat)
    colors = ['steelblue' if v >= 0 else 'tomato' for v in values_flat]

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(
        node_flat - dx / 2,
        hour_flat - dy / 2,
        z_bottoms,
        dx, dy, dz,
        color=colors,
        alpha=0.85,
        zsort='average',
    )

    ax.set_xlabel('Node')
    ax.set_ylabel('Hour')
    zlabel = '% of Node Capacity' if normalize else 'Slack Capacity (MW)'
    ax.set_zlabel(zlabel)
    title_parts = ['Slack Capacity by Node and Hour']
    if normalize:
        title_parts.append('Normalized by Node Capacity')
    if filter_small_nodes:
        title_parts.append('Nodes <0.1% Filtered')
    ax.set_title(' | '.join(title_parts))
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(j) for j in kept_nodes])
    ax.set_yticks(range(0, n_hours, max(1, n_hours // 12)))

    legend_elements = [
        Patch(facecolor='steelblue', label='Positive (+)'),
        Patch(facecolor='tomato', label='Negative (−)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    suffix = ('_norm' if normalize else '') + ('_filtered' if filter_small_nodes else '')
    plt.savefig(OUT_PATH / f'slack_3d_cap{total_cap:.0f}MWh{suffix}.png', dpi=150, bbox_inches='tight', pad_inches=0.5)
    # plt.show()
    plt.close()
    return


def main(date_string):
    OUT_PATH = FIGURE_PATH / date_string
    create_output_folder(OUT_PATH)
    json_files = find_files_by_date(date_string)
    results_list = []
    
    for file in json_files:
        results_dict = get_results_dict(file)
        plot_battery_dispatch_3d(results_dict,OUT_PATH,normalize=False,filter_small_nodes=False)
        plot_transmission_congestion(results_dict,OUT_PATH,filter_small_nodes=True)
        plot_slack_3d(results_dict, OUT_PATH, normalize=False, filter_small_nodes=False)
        results_list.append(results_dict)
    interestNode = 9
    plot_node_dispatch_by_capacity(results_list, interestNode, OUT_PATH)
    plot_node_slack_by_capacity(results_list, interestNode, OUT_PATH, normalize=False)
    
    plot_objective_vs_capacity(results_list,OUT_PATH)
    plot_capacity_allocation(results_list,OUT_PATH)
    plot_node_profit_by_capacity(results_list, OUT_PATH)
    return

def find_files_by_date(date_str):
    json_files = []
    # Find and print matching files
    for file_path in OUTPUT_DIR.rglob('*.json'):
        try:
            # Read file content as a string
            if date_str in file_path.read_text(encoding='utf-8'):
                json_files.append(file_path)
        except (UnicodeDecodeError, PermissionError):
            continue
    # json_paths = OUTPUT_DIR / f'day_opt_metadata_20260422_175358.json'
    return json_files


if __name__ == '__main__':
    date_string = "battCap"
    main(date_string)