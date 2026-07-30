# collusion_judges/dataset/make_graph_basic.py
"""Построение графа совместного судейства."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence
from pathlib import Path
from collusion_judges.config import PRIMARY_GRAPH_PATH

import pickle

import networkx as nx
import numpy as np
import pandas as pd


class JudgeGraphBuilder:
    """
    Build a graph from long-format judge scores.

    Nodes represent judges and store event-level judge statistics.
    Edges represent judge pairs and store pairwise correlations and ranks.
    """

    def __init__(
        self,
        long_df: pd.DataFrame,
        score_cols: dict[str, str] | None = None,
        corr_types: Sequence[str] = ("pearson",),
    ) -> None:
        self.df = long_df.copy()

        self.score_cols = dict(
            score_cols
            or {
                "components": "judge_component_score_norm",
            }
        )

        self.corr_types = tuple(corr_types)

        self.athlete_col = "athlete_name"
        self.judge_col = "judge_name"
        self.judge_country_col = "judge_country"

        self.event_descr = [
            "year",
            "stage",
            "category",
            "program",
        ]

        self.judge_statistics = [
            "mean",
            "median",
            "min",
            "max",
            "std",
            "mad",
            "iqr",
        ]

        self.G = nx.Graph()
        self._graph_is_filled = False

        self._initiate_graph()

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _judge_sort_key(judge: str) -> tuple[str, str]:
        """Return a case-insensitive lexicographic sorting key."""
        return judge.casefold(), judge

    def _ordered_pair(
        self,
        judge_1: str,
        judge_2: str,
    ) -> tuple[str, str]:
        """
        Return a pair in deterministic lexicographic order.

        The first judge is always judge_a; the second is judge_b.
        """
        judge_a, judge_b = sorted(
            (judge_1, judge_2),
            key=self._judge_sort_key,
        )

        return judge_a, judge_b

    def _get_judges(
        self,
        df: pd.DataFrame,
    ) -> list[str]:
        """Return unique judges in lexicographic order."""
        judges = (
            df[self.judge_col]
            .dropna()
            .unique()
            .tolist()
        )

        return sorted(
            judges,
            key=self._judge_sort_key,
        )

    # ------------------------------------------------------------------
    # Graph initialization
    # ------------------------------------------------------------------

    def _get_node_attributes(self) -> dict[str, list]:
        """Create empty attributes for one node."""
        attributes = {
            attr: []
            for attr in self.event_descr + ["n_athletes"]
        }

        for score_name in self.score_cols:
            for corr in self.corr_types:
                prefix = f"{score_name}_{corr}"

                for statistic in self.judge_statistics:
                    attributes[f"{prefix}_{statistic}"] = []

                attributes[f"{prefix}_rank_by_mean"] = []
                attributes[f"{prefix}_rank_by_median"] = []

                if corr == "pearson":
                    attributes[f"{prefix}_fisher_mean"] = []
                    attributes[
                        f"{prefix}_rank_by_fisher_mean"
                    ] = []

        return attributes

    def _get_edge_attributes(
        self,
        judge_a: str,
        judge_b: str,
    ) -> dict[str, object]:
        """Create empty attributes for one edge."""
        attributes: dict[str, object] = {
            "judge_a": judge_a,
            "judge_b": judge_b,
        }

        attributes.update(
            {
                attr: []
                for attr in self.event_descr + ["n_athletes"]
            }
        )

        for score_name in self.score_cols:
            for corr in self.corr_types:
                prefix = f"{score_name}_{corr}"

                attributes[f"{prefix}_r"] = []
                attributes[f"{prefix}_rank_a"] = []
                attributes[f"{prefix}_rank_b"] = []

        return attributes

    def _initiate_nodes(self) -> None:
        """Add all judges and initialize node attributes."""
        judge_info = (
            self.df[
                [
                    self.judge_col,
                    self.judge_country_col,
                ]
            ]
            .dropna(subset=[self.judge_col])
            .drop_duplicates(subset=self.judge_col)
            .set_index(self.judge_col)[self.judge_country_col]
        )

        nodes = []

        for judge, country in judge_info.items():
            attributes = self._get_node_attributes()
            attributes["country"] = country

            nodes.append(
                (
                    judge,
                    attributes,
                )
            )

        self.G.add_nodes_from(nodes)

    def _initiate_edges(self) -> None:
        """Add every judge pair that appears in a common panel."""
        all_pairs: set[tuple[str, str]] = set()

        event_groups = self.df.groupby(
            self.event_descr,
            sort=False,
            dropna=False,
        )

        for _, event_df in event_groups:
            judges = self._get_judges(event_df)

            all_pairs.update(
                self._ordered_pair(judge_1, judge_2)
                for judge_1, judge_2 in combinations(judges, 2)
            )

        ordered_pairs = sorted(
            all_pairs,
            key=lambda pair: (
                self._judge_sort_key(pair[0]),
                self._judge_sort_key(pair[1]),
            ),
        )

        edges = [
            (
                judge_a,
                judge_b,
                self._get_edge_attributes(
                    judge_a=judge_a,
                    judge_b=judge_b,
                ),
            )
            for judge_a, judge_b in ordered_pairs
        ]

        self.G.add_edges_from(edges)

    def _initiate_graph(self) -> None:
        """Create all nodes, edges, and empty attributes."""
        self._initiate_nodes()
        self._initiate_edges()

    # ------------------------------------------------------------------
    # Score and correlation matrices
    # ------------------------------------------------------------------

    def _get_score_matrix(
        self,
        event_df: pd.DataFrame,
        score_col: str,
        judges: list[str],
    ) -> pd.DataFrame:
        """Return an athlete-by-judge score matrix."""
        score_matrix = event_df.pivot_table(
            index=self.athlete_col,
            columns=self.judge_col,
            values=score_col,
            aggfunc="mean",
        )

        return score_matrix.reindex(columns=judges)

    @staticmethod
    def _get_corr_matrix(
        score_matrix: pd.DataFrame,
        corr: str,
    ) -> pd.DataFrame:
        """
        Calculate a judge correlation matrix.

        Self-correlations are replaced with NaN.
        """
        corr_matrix = score_matrix.corr(method=corr)

        np.fill_diagonal(
            corr_matrix.values,
            np.nan,
        )

        return corr_matrix

    # ------------------------------------------------------------------
    # Primary node metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _get_judge_statistics(
        corr_matrix: pd.DataFrame,
        corr: str,
    ) -> pd.DataFrame:
        """Calculate correlation statistics for each judge."""
        medians = corr_matrix.median(axis=1)

        quartiles = corr_matrix.quantile(
            [0.25, 0.75],
            axis=1,
        ).T

        statistics = pd.DataFrame(
            {
                "mean": corr_matrix.mean(axis=1),
                "median": medians,
                "min": corr_matrix.min(axis=1),
                "max": corr_matrix.max(axis=1),
                "std": corr_matrix.std(
                    axis=1,
                    ddof=1,
                ),
                "mad": (
                    corr_matrix
                    .sub(medians, axis=0)
                    .abs()
                    .median(axis=1)
                ),
                "iqr": (
                    quartiles[0.75]
                    - quartiles[0.25]
                ),
            },
            index=corr_matrix.index,
        )

        if corr == "pearson":
            clipped = corr_matrix.clip(
                lower=-1 + 1e-12,
                upper=1 - 1e-12,
            )

            fisher_z = np.arctanh(
                clipped.to_numpy(dtype=float)
            )

            statistics["fisher_mean"] = np.tanh(
                np.nanmean(
                    fisher_z,
                    axis=1,
                )
            )

        return statistics

    @staticmethod
    def _get_judge_ranks(
        statistics: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rank judges by agreement with the panel.

        Rank 1 corresponds to the highest agreement.
        """
        ranks = pd.DataFrame(
            index=statistics.index
        )

        ranks["rank_by_mean"] = statistics["mean"].rank(
            ascending=False,
            method="min",
        )

        ranks["rank_by_median"] = statistics["median"].rank(
            ascending=False,
            method="min",
        )

        if "fisher_mean" in statistics.columns:
            ranks["rank_by_fisher_mean"] = statistics[
                "fisher_mean"
            ].rank(
                ascending=False,
                method="min",
            )

        return ranks

    # ------------------------------------------------------------------
    # Primary edge metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _get_pair_rank_matrix(
        corr_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rank every partner separately for each judge.

        Rank 1 corresponds to the highest pairwise correlation.
        """
        return corr_matrix.rank(
            axis=1,
            ascending=False,
            method="min",
            na_option="keep",
        )

    # ------------------------------------------------------------------
    # Collect event parameters
    # ------------------------------------------------------------------

    def _collect_event_parameters(
        self,
        event: tuple,
        event_df: pd.DataFrame,
    ) -> dict:
        """Collect all primary parameters for one judging act."""
        judges = self._get_judges(event_df)

        parameters = {
            **dict(
                zip(
                    self.event_descr,
                    event,
                )
            ),
            "judges": judges,
            "n_athletes": event_df[
                self.athlete_col
            ].nunique(),
            "scores": {},
        }

        for score_name, score_col in self.score_cols.items():
            score_matrix = self._get_score_matrix(
                event_df=event_df,
                score_col=score_col,
                judges=judges,
            )

            parameters["scores"][score_name] = {}

            for corr in self.corr_types:
                corr_matrix = self._get_corr_matrix(
                    score_matrix=score_matrix,
                    corr=corr,
                )

                statistics = self._get_judge_statistics(
                    corr_matrix=corr_matrix,
                    corr=corr,
                )

                judge_ranks = self._get_judge_ranks(
                    statistics=statistics,
                )

                pair_ranks = self._get_pair_rank_matrix(
                    corr_matrix=corr_matrix,
                )

                parameters["scores"][score_name][corr] = {
                    "matrix": corr_matrix,
                    "statistics": statistics,
                    "judge_ranks": judge_ranks,
                    "pair_ranks": pair_ranks,
                }

        return parameters

    # ------------------------------------------------------------------
    # Push data to nodes
    # ------------------------------------------------------------------

    def _push_event_to_nodes(
        self,
        parameters: dict,
    ) -> None:
        """Append one judging act's metrics to nodes."""
        for judge in parameters["judges"]:
            node = self.G.nodes[judge]

            for attr in self.event_descr:
                node[attr].append(
                    parameters[attr]
                )

            node["n_athletes"].append(
                parameters["n_athletes"]
            )

            for score_name, score_data in parameters[
                "scores"
            ].items():
                for corr, corr_data in score_data.items():
                    prefix = f"{score_name}_{corr}"

                    statistics = corr_data[
                        "statistics"
                    ].loc[judge]

                    ranks = corr_data[
                        "judge_ranks"
                    ].loc[judge]

                    for statistic in self.judge_statistics:
                        node[
                            f"{prefix}_{statistic}"
                        ].append(
                            statistics[statistic]
                        )

                    node[
                        f"{prefix}_rank_by_mean"
                    ].append(
                        ranks["rank_by_mean"]
                    )

                    node[
                        f"{prefix}_rank_by_median"
                    ].append(
                        ranks["rank_by_median"]
                    )

                    if corr == "pearson":
                        node[
                            f"{prefix}_fisher_mean"
                        ].append(
                            statistics["fisher_mean"]
                        )

                        node[
                            f"{prefix}_rank_by_fisher_mean"
                        ].append(
                            ranks[
                                "rank_by_fisher_mean"
                            ]
                        )

    # ------------------------------------------------------------------
    # Push data to edges
    # ------------------------------------------------------------------

    def _push_event_to_edges(
        self,
        parameters: dict,
    ) -> None:
        """Append one judging act's metrics to edges."""
        for judge_1, judge_2 in combinations(
            parameters["judges"],
            2,
        ):
            judge_a, judge_b = self._ordered_pair(
                judge_1,
                judge_2,
            )

            edge = self.G[judge_a][judge_b]

            for attr in self.event_descr:
                edge[attr].append(
                    parameters[attr]
                )

            edge["n_athletes"].append(
                parameters["n_athletes"]
            )

            for score_name, score_data in parameters[
                "scores"
            ].items():
                for corr, corr_data in score_data.items():
                    prefix = f"{score_name}_{corr}"

                    edge[f"{prefix}_r"].append(
                        corr_data["matrix"].loc[
                            judge_a,
                            judge_b,
                        ]
                    )

                    edge[
                        f"{prefix}_rank_a"
                    ].append(
                        corr_data["pair_ranks"].loc[
                            judge_a,
                            judge_b,
                        ]
                    )

                    edge[
                        f"{prefix}_rank_b"
                    ].append(
                        corr_data["pair_ranks"].loc[
                            judge_b,
                            judge_a,
                        ]
                    )

    def _push_event_data(
        self,
        parameters: dict,
    ) -> None:
        """Push event parameters to nodes and edges."""
        self._push_event_to_nodes(parameters)
        self._push_event_to_edges(parameters)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_info_to_graph(self) -> nx.Graph:
        """Calculate and add metrics for all judging acts."""
        if self._graph_is_filled:
            return self.G

        event_groups = self.df.groupby(
            self.event_descr,
            sort=False,
            dropna=False,
        )

        for event, event_df in event_groups:
            parameters = self._collect_event_parameters(
                event=event,
                event_df=event_df,
            )

            self._push_event_data(
                parameters=parameters,
            )

        self._graph_is_filled = True

        return self.G

    def save_graph(self, path: str | Path = PRIMARY_GRAPH_PATH,) -> Path:
        """
        Build and save the primary graph in pickle format.

        Returns the path to the saved file.
        """
        graph = self.build_graph()

        output_path = Path(path).expanduser()
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open("wb") as file:
            pickle.dump(
                graph,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        return output_path

    def build_graph(self, path=None) -> nx.Graph:
        """Return the initialized and populated graph."""
        self.add_info_to_graph()
        if path:
            self.save_graph(path)
        return self.G
