# collusion_judges/dataset/make_anomaly_dataset.py

"""Create the final judge dataset with Isolation Forest results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from collusion_judges.config import NODES_2_PATH, NODES_3_PATH
from collusion_judges.modeling.iforest import (
    COLS_CORR_BASED,
    COLS_PERSONAL_INFO,
    CORR_TYPES,
    PARAMS_FOR_FOREST,
    get_anomaly_scores,
    get_columns,
)


def make_anomaly_dataset(
    df: pd.DataFrame | None = None,
    params: dict = PARAMS_FOR_FOREST,
    feature_spec: dict = COLS_CORR_BASED,
    personal_cols: tuple[str, ...] = COLS_PERSONAL_INFO,
    top_n: int = 10,
    contamination: str | float = "auto",
    random_state: int = 42,
    n_estimators: int = 500,
    input_path: str | Path = NODES_2_PATH,
    output_path: str | Path | None = NODES_3_PATH,
) -> pd.DataFrame:
    """
    Create a judge dataset with source features and anomaly results.

    The resulting dataset contains:
    - judge name, country and experience;
    - all features used by Isolation Forest models;
    - anomaly scores and ranks for every correlation type;
    - top-N anomaly indicators.
    """
    if df is None:
        df = pd.read_parquet(input_path)

    feature_cols = []

    for model_params in params.values():
        feature_cols.extend(
            get_columns(
                model_params["corr"],
                model_params["scores"],
                feature_spec,
            )
        )

    # Remove repeated columns while preserving their order.
    feature_cols = list(dict.fromkeys(feature_cols))

    anomaly_df = get_anomaly_scores(
        params=params,
        df=df,
        feature_spec=feature_spec,
        personal_columns=personal_cols,
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
        output_path=None,
    )

    anomaly_cols = [
        column
        for column in anomaly_df.columns
        if column.startswith(("anomaly_score_", "anomaly_rank_"))
    ]

    result = df[
        list(personal_cols) + feature_cols
    ].copy()

    result = result.join(
        anomaly_df
        .set_index("judge_name")[anomaly_cols],
        on="judge_name",
    )

    for corr in CORR_TYPES:
        rank_col = f"anomaly_rank_{corr}_comp_tech"
        result[f"top10_anom_{corr}"] = result[rank_col] <= top_n

    if output_path is not None:
        result.to_parquet(
            Path(output_path),
            index=False,
        )

    return result
