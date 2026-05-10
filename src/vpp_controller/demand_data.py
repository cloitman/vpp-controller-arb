from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import DATA_DIR


SEASON_DATES = {
    "winter": "2025-01-10",
    "spring": "2025-04-10",
    "summer": "2025-07-10",
    "fall": "2025-10-10",
}


def _default_demand_file() -> Path:
    return DATA_DIR / "demand" / "CISO-demand.csv"


def load_demand_data(file_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load and clean CISO demand data.

    Returns a DataFrame with the columns needed by the demand builders.
    """
    csv_path = Path(file_path) if file_path is not None else _default_demand_file()
    df = pd.read_csv(csv_path)
    df["UTC time"] = pd.to_datetime(df["UTC time"], format="mixed", dayfirst=True)
    df = df[df["UTC time"].dt.year == 2025]

    cols = [
        "UTC time",
        "Hour",
        "Time zone",
        "Demand forecast",
        "Demand",
        "Net generation",
        "Subregion PGAE",
    ]
    return df[cols]


def build_season_base_demand(
    df: pd.DataFrame,
    dates: list[str] | None = None,
) -> pd.DataFrame:
    """
    Keep only selected season snapshots and demand values.
    """
    selected_dates = dates or list(SEASON_DATES.values())
    return df[df["UTC time"].dt.date.isin(pd.to_datetime(selected_dates).date)][
        ["UTC time", "Subregion PGAE"]
    ].reset_index(drop=True)


def create_node_demand(node_row: pd.Series, base_demand: np.ndarray, factor=1,noise = 0.05) -> pd.DataFrame:
    """
    Build noisy P/Q demand trajectories for one node.

    Returns DataFrame columns: node, P_demand, Q_demand.
    """
    node = node_row["node"]
    p_mean = node_row["l_P"]*factor
    q_mean = node_row["l_Q"]*factor

    p_scaled = (base_demand / base_demand.mean()) * p_mean
    q_scaled = (base_demand / base_demand.mean()) * q_mean

    std_p = p_mean * noise
    std_q = q_mean * noise

    rng_p = np.random.default_rng(seed=int(node))
    rng_q = np.random.default_rng(seed=int(node) + 10)

    p_noisy = p_scaled + rng_p.normal(loc=0, scale=std_p, size=len(p_scaled))
    q_noisy = q_scaled + rng_q.normal(loc=0, scale=std_q, size=len(q_scaled))

    return pd.DataFrame(
        {
            "node": node,
            "P_demand": p_noisy,
            "Q_demand": q_noisy,
        }
    )


def create_all_nodes_demand(
    nodes_df: pd.DataFrame,
    season: str,
    factor = 1.0,
    noise = 0.05,
    seasonal_base_df: pd.DataFrame | None = None,
    demand_file_path: str | Path | None = None,
    shift_by_node: bool = False,
) -> pd.DataFrame:
    """
    Build demand profiles for all nodes for a specific season.

    Returns DataFrame columns: timestamp, node, P_demand, Q_demand.

    If shift_by_node=True, the base demand profile for node k is cyclically
    shifted by k hours so that peaks and troughs no longer coincide across nodes.
    """
    if season not in SEASON_DATES:
        raise ValueError(f"Season must be one of: {list(SEASON_DATES.keys())}")

    if seasonal_base_df is None:
        raw_df = load_demand_data(file_path=demand_file_path)
        seasonal_base_df = build_season_base_demand(raw_df)

    date = SEASON_DATES[season]
    mask = seasonal_base_df["UTC time"].dt.date == pd.Timestamp(date).date()
    df_day = seasonal_base_df[mask].reset_index(drop=True)

    if df_day.empty:
        raise ValueError(f"No data found for date {date}")

    base_demand = df_day["Subregion PGAE"].to_numpy()
    timestamps = df_day["UTC time"].to_numpy()

    all_nodes = []
    for _, node_row in nodes_df.iterrows():
        node_base = np.roll(base_demand, int(node_row["node"])) if shift_by_node else base_demand
        df_node = create_node_demand(node_row, node_base, factor, noise)
        df_node.insert(0, "timestamp", timestamps)
        all_nodes.append(df_node)

    return pd.concat(all_nodes, ignore_index=True)


__all__ = [
    "SEASON_DATES",
    "build_season_base_demand",
    "create_all_nodes_demand",
    "create_node_demand",
    "load_demand_data",
]
