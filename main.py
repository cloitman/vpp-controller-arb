import numpy as np
import pandas as pd

from src.vpp_controller.demand_data import create_all_nodes_demand
from src.vpp_controller.paths import DATA_NETWORKS_DIR, DATA_PRICES_DIR
from src.vpp_controller.results_format import save_day_optimization_result
from src.vpp_controller.runner import (
    run_battery_arbitrage_problem,
    run_lmp_problem,
    run_post_battery_lmp_problem,
)

opVersion = 'spring4'


def main() -> None:
    topology_df = pd.read_csv(DATA_NETWORKS_DIR / "homework3bus.csv")#homework3bus#homework3bus_no_gen_relaxed#homework3bus_no_gen

    price_df = pd.read_csv(
        DATA_PRICES_DIR / "pricedf_0096WD_7_N001_spring_2025_04_10.csv"
    )
    price_df = price_df.sort_values(by="OPR_HR").reset_index(drop=True)

    demand_df = create_all_nodes_demand(topology_df, "spring", factor=0.7,noise=0.3,shift_by_node=False)
    demand_df = demand_df.rename(columns={"P_demand": "l_P", "Q_demand": "l_Q"})
    demand_df["hour"] = pd.to_datetime(demand_df["timestamp"]).dt.hour
    demand_df = demand_df[["node", "hour", "l_P", "l_Q"]]

    # Stage 1: solve once to get LMPs (battery-free OPF).
    # LMPs don't change with battery capacity, so we only run this once.
    print("=" * 50)
    print("Stage 1: computing LMPs (no-battery OPF)")
    lmp, lmp_result = run_lmp_problem(
        topology_df=topology_df,
        demand_df=demand_df,
        price_df_root_node=price_df,
    )
    print(f"  Status: {lmp_result.status}  |  Cost: {lmp_result.objective_value:.2f}")

    batt_caps = list(np.arange(2, 28, 2))

    for batt_cap in batt_caps:
        print("\n" + "=" * 50)
        print(f"Stage 2: battery arbitrage with capacity = {batt_cap} MWh")

        batt_cap = float(batt_cap)

        arb_result = run_battery_arbitrage_problem(
            topology_df=topology_df,
            demand_df=demand_df,
            lmp=lmp,
            total_battery_capacity=batt_cap,
        )

        print(f"  Status: {arb_result.status}  |  Profit: {arb_result.objective_value:.2f}")

        print("  Stage 3: computing post-battery LMPs")
        post_lmp = run_post_battery_lmp_problem(
            topology_df=topology_df,
            demand_df=demand_df,
            price_df_root_node=price_df,
            arb_result=arb_result,
        )
        arb_result.variables["post_battery_lmp"] = post_lmp

        save_day_optimization_result(arb_result, batt_cap, opVersion)


if __name__ == "__main__":
    main()
