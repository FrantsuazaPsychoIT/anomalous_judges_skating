# collusion_judges/modelling/iforest.py
"""Run Isolation Forest models on aggregated judge characteristics."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from collusion_judges.config import NODES_2_PATH, NODES_3_PATH


CORR_TYPES: tuple[str, ...] = (
    "pearson",
    "kendall",
    "spearman",
)

SCORE_TYPES: tuple[str, ...] = (
    "components",
    "technique",
)

COLS_PERSONAL_INFO: tuple[str, ...] = (
    "judge_name",
    "country",
    "n_categories",
    "n_events",
)


# Структура:
# aggregation -> correlation type -> primary metric
'''
    "mad": {
        "pearson": ("fisher_mean",),
        "spearman": ("mean",),
        "kendall": ("mean",),
    },
'''
COLS_CORR_BASED: dict[str, dict[str, tuple[str, ...]]] = {
    "median": {
        "pearson": (
            "fisher_mean",
            "mad",
            "rank_by_fisher_mean",
        ),
        "spearman": (
            "mean",
            "mad",
            "rank_by_mean",
        ),
        "kendall": (
            "mean",
            "mad",
            "rank_by_mean",
        ),
    },
}


# Каждый элемент словаря задаёт отдельную модель Isolation Forest.
PARAMS_FOR_FOREST: dict[str, dict[str, tuple[str, ...]]] = {
    "pearson_comp_tech": {
        "corr": ("pearson",),
        "scores": (
            "components",
            "technique",
        ),
    },
    "spearman_comp_tech": {
        "corr": ("spearman",),
        "scores": (
            "components",
            "technique",
        ),
    },
    "kendall_comp_tech": {
        "corr": ("kendall",),
        "scores": (
            "components",
            "technique",
        ),
    },
}


def get_columns(
    corr_types: Sequence[str] = CORR_TYPES,
    scores: Sequence[str] = SCORE_TYPES,
    feature_spec: Mapping[
        str,
        Mapping[str, Sequence[str]],
    ] = COLS_CORR_BASED,
) -> list[str]:
    """Build names of correlation-based feature columns."""
    selected_columns: list[str] = []

    for score in scores:
        for corr in corr_types:
            for aggregation, corr_spec in feature_spec.items():
                metrics = corr_spec.get(corr, ())

                for metric in metrics:
                    selected_columns.append(
                        f"{aggregation}_of_{score}_{corr}_{metric}"
                    )

    return selected_columns


class IForest:
    """Fit one Isolation Forest and return anomaly scores and ranks."""

    def __init__(
        self,
        df: pd.DataFrame,
        columns: Sequence[str],
        contamination: str | float = "auto",
        random_state: int = 42,
        n_estimators: int = 500,
        n_jobs: int | None = -1,
    ) -> None:
        self.df = df
        self.columns = list(columns)

        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

        self.X = self._prepare_features()

        self.model: IsolationForest | None = None
        self.result_: pd.DataFrame | None = None

    def _prepare_features(self) -> pd.DataFrame:
        """Select features and ensure that all values are finite."""
        missing_columns = [
            column
            for column in self.columns
            if column not in self.df.columns
        ]

        if missing_columns:
            raise KeyError(
                "The following feature columns are missing: "
                f"{missing_columns}"
            )

        if not self.columns:
            raise ValueError(
                "No columns were selected for Isolation Forest."
            )

        X = (
            self.df
            .loc[:, self.columns]
            .apply(pd.to_numeric, errors="raise")
            .astype(float)
            .copy()
        )

        finite_mask = np.isfinite(
            X.to_numpy(dtype=float)
        )

        if not finite_mask.all():
            invalid_columns = X.columns[
                ~finite_mask.all(axis=0)
            ].tolist()

            raise ValueError(
                "Isolation Forest features contain NaN or "
                f"infinite values: {invalid_columns}"
            )

        return X

    def fit(self) -> pd.DataFrame:
        """Fit the model and return anomaly score and anomaly rank."""
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.model.fit(self.X)

        # score_samples: lower values mean greater abnormality.
        # The sign is reversed for more intuitive interpretation.
        anomaly_score = pd.Series(
            -self.model.score_samples(self.X),
            index=self.X.index,
            name="anomaly_score",
        )

        anomaly_rank = (
            anomaly_score
            .rank(
                ascending=False,
                method="min",
            )
            .astype(int)
            .rename("anomaly_rank")
        )

        self.result_ = pd.concat(
            [
                anomaly_score,
                anomaly_rank,
            ],
            axis=1,
        )

        return self.result_.copy()


def get_anomaly_scores(
    params: Mapping[
        str,
        Mapping[str, Sequence[str]],
    ] = PARAMS_FOR_FOREST,
    df: pd.DataFrame | None = None,
    feature_spec: Mapping[
        str,
        Mapping[str, Sequence[str]],
    ] = COLS_CORR_BASED,
    personal_columns: Sequence[str] = COLS_PERSONAL_INFO,
    contamination: str | float = "auto",
    random_state: int = 42,
    n_estimators: int = 500,
    n_jobs: int | None = -1,
    input_path: str | Path = NODES_2_PATH,
    output_path: str | Path | None = NODES_3_PATH,
) -> pd.DataFrame:
    """
    Run separate Isolation Forest models for all parameter sets.

    Each key in `params` produces its own anomaly score and rank.
    The resulting DataFrame is optionally saved as Parquet.
    """
    if df is None:
        df = pd.read_parquet(input_path)

    personal_columns = list(personal_columns)

    missing_personal_columns = [
        column
        for column in personal_columns
        if column not in df.columns
    ]

    if missing_personal_columns:
        raise KeyError(
            "The following personal-information columns are missing: "
            f"{missing_personal_columns}"
        )

    anomaly_df = df.loc[
        :,
        personal_columns,
    ].copy()

    for model_name, model_params in params.items():
        corr_types = tuple(
            model_params["corr"]
        )

        scores = tuple(
            model_params["scores"]
        )

        columns = get_columns(
            corr_types=corr_types,
            scores=scores,
            feature_spec=feature_spec,
        )

        forest = IForest(
            df=df,
            columns=columns,
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
        )

        model_result = forest.fit()

        anomaly_df[
            f"anomaly_score_{model_name}"
        ] = model_result["anomaly_score"]

        anomaly_df[
            f"anomaly_rank_{model_name}"
        ] = model_result["anomaly_rank"]

    if output_path is not None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        anomaly_df.to_parquet(
            output_path,
            index=False,
        )

    return anomaly_df
