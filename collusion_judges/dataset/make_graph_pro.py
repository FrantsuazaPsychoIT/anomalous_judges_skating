# collusion_judges/dataset/aggregate_graph.py
"""Build an aggregated judge graph from the primary graph."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd

from collusion_judges.config import (
    PRIMARY_GRAPH_PATH,
    EDGES_1_PATH,
    GRAPH_1_PATH,
    LONG_DF_PATH,
    NODES_1_PATH,
)
from collusion_judges.dataset.make_graph import JudgeGraphBuilder


class JudgeAggregateGraphBuilder:
    """
    Replace event-level node arrays with aggregated judge characteristics.

    The primary graph remains unchanged. The new graph keeps the same nodes
    and edges, but its node attributes contain scalar summary statistics.
    """

    EVENT_ATTRS = (
        "year",
        "stage",
        "category",
        "program",
        "n_athletes",
    )

    PRIMARY_METRICS = (
        "rank_by_fisher_mean",
        "rank_by_median",
        "rank_by_mean",
        "fisher_mean",
        "median",
        "mean",
        "min",
        "max",
        "std",
        "mad",
        "iqr",
    )

    AGGREGATION_PLAN = {
        "mean": (
            "mean",
            "median",
            "mad",
            "iqr",
        ),
        "fisher_mean": (
            "fisher_aggregate",
            "median",
            "mad",
        ),
        "median": (
            "mean",
            "median",
            "mad",
            "iqr",
        ),
        "min": (
            "median",
            "mad",
        ),
        "max": (
            "median",
            "mad",
        ),
        "std": (
            "median",
            "mad",
        ),
        "mad": (
            "median",
            "mad",
        ),
        "iqr": (
            "median",
            "mad",
        ),
        "rank_by_mean": (
            "median",
            "iqr",
        ),
        "rank_by_median": (
            "median",
            "iqr",
        ),
        "rank_by_fisher_mean": (
            "median",
            "iqr",
        ),
        "upper_gap": (
            "median",
            "mad",
        ),
        "lower_gap": (
            "median",
            "mad",
        ),
        "range": (
            "median",
            "mad",
        ),
    }

    def __init__(self, base_graph: nx.Graph) -> None:
        self.G = copy.deepcopy(base_graph)

        self.metric_groups = self._discover_metric_groups()
        self.aggregate_attrs = self._get_aggregate_attribute_names()

        self._is_built = False

    # ------------------------------------------------------------------
    # Attribute discovery and initialization
    # ------------------------------------------------------------------

    def _discover_metric_groups(
        self,
    ) -> dict[str, dict[str, str]]:
        """
        Discover primary metric arrays stored on graph nodes.

        Example
        -------
        components_pearson_mean
            prefix: components_pearson
            metric: mean
        """
        first_node = next(iter(self.G.nodes), None)

        if first_node is None:
            return {}

        node_data = self.G.nodes[first_node]
        groups: dict[str, dict[str, str]] = {}

        for attr_name, value in node_data.items():
            if not isinstance(
                value,
                (list, tuple, np.ndarray, pd.Series),
            ):
                continue

            if attr_name in self.EVENT_ATTRS:
                continue

            for metric in self.PRIMARY_METRICS:
                suffix = f"_{metric}"

                if attr_name.endswith(suffix):
                    prefix = attr_name[: -len(suffix)]

                    groups.setdefault(
                        prefix,
                        {},
                    )[metric] = attr_name

                    break

        return groups

    def _get_aggregate_attribute_names(
        self,
    ) -> list[str]:
        """Return all scalar attributes produced for each node."""
        attributes: list[str] = []

        for prefix, metrics in self.metric_groups.items():
            for metric in metrics:
                for aggregation in self.AGGREGATION_PLAN[metric]:
                    attributes.append(
                        self._aggregate_attr_name(
                            prefix=prefix,
                            metric=metric,
                            aggregation=aggregation,
                        )
                    )

            if {"min", "median", "max"}.issubset(metrics):
                for metric in (
                    "upper_gap",
                    "lower_gap",
                    "range",
                ):
                    for aggregation in self.AGGREGATION_PLAN[metric]:
                        attributes.append(
                            self._aggregate_attr_name(
                                prefix=prefix,
                                metric=metric,
                                aggregation=aggregation,
                            )
                        )

        return attributes

    @staticmethod
    def _aggregate_attr_name(
        prefix: str,
        metric: str,
        aggregation: str,
    ) -> str:
        """Create an aggregate attribute name."""
        return f"{aggregation}_of_{prefix}_{metric}"

    def initialize_node_attributes(self) -> None:
        """Add empty scalar attributes to every node."""
        for judge, data in self.G.nodes(data=True):
            data["judge_name"] = judge
            data["n_events"] = 0
            data["n_categories"] = 0

            for attr in self.aggregate_attrs:
                data[attr] = np.nan

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_array(
        values: Sequence,
    ) -> np.ndarray:
        """Convert an event-level sequence to a float array."""
        return np.asarray(
            values,
            dtype=float,
        )

    @staticmethod
    def _clean(
        values: Sequence | np.ndarray,
    ) -> np.ndarray:
        """Remove missing and infinite values."""
        array = np.asarray(
            values,
            dtype=float,
        )

        return array[np.isfinite(array)]

    @classmethod
    def _aggregate(
        cls,
        values: Sequence | np.ndarray,
        aggregation: str,
    ) -> float:
        """Calculate one across-event aggregate."""
        clean = cls._clean(values)

        if clean.size == 0:
            return np.nan

        if aggregation == "mean":
            return float(np.mean(clean))

        if aggregation == "median":
            return float(np.median(clean))

        if aggregation == "mad":
            median = np.median(clean)

            return float(
                np.median(
                    np.abs(clean - median)
                )
            )

        if aggregation == "iqr":
            q25, q75 = np.percentile(
                clean,
                [25, 75],
            )

            return float(q75 - q25)

        if aggregation == "fisher_aggregate":
            clipped = np.clip(
                clean,
                -1 + 1e-12,
                1 - 1e-12,
            )

            fisher_z = np.arctanh(clipped)

            return float(
                np.tanh(
                    np.mean(fisher_z)
                )
            )

        raise ValueError(
            f"Unknown aggregation: {aggregation}"
        )

    @staticmethod
    def _difference(
        left: Sequence,
        right: Sequence,
    ) -> np.ndarray:
        """Calculate an elementwise difference between aligned arrays."""
        return (
            np.asarray(left, dtype=float)
            - np.asarray(right, dtype=float)
        )

    @staticmethod
    def _count_events(
        node_data: dict,
    ) -> int:
        """Count unique year × stage × category combinations."""
        events = pd.DataFrame(
            {
                "year": node_data["year"],
                "stage": node_data["stage"],
                "category": node_data["category"],
            }
        )

        return int(
            events
            .drop_duplicates()
            .shape[0]
        )

    @staticmethod
    def _count_categories(
        node_data: dict,
    ) -> int:
        """Count categories in which the judge participated."""
        return int(
            pd.Series(
                node_data["category"]
            ).nunique(dropna=True)
        )

    @staticmethod
    def _ordered_pair(
        judge_1: str,
        judge_2: str,
    ) -> tuple[str, str]:
        """Return judges in case-insensitive lexicographic order."""
        judge_a, judge_b = sorted(
            (judge_1, judge_2),
            key=lambda name: (
                name.casefold(),
                name,
            ),
        )

        return judge_a, judge_b

    # ------------------------------------------------------------------
    # Node aggregation
    # ------------------------------------------------------------------

    def collect_node_attributes(
        self,
        judge: str,
    ) -> dict[str, object]:
        """Calculate all aggregated attributes for one judge."""
        node_data = self.G.nodes[judge]

        result: dict[str, object] = {
            "judge_name": judge,
            "country": node_data.get("country"),
            "n_events": self._count_events(node_data),
            "n_categories": self._count_categories(node_data),
        }

        for prefix, metrics in self.metric_groups.items():
            # Aggregate original event-level characteristics.
            for metric, attr_name in metrics.items():
                values = node_data[attr_name]

                for aggregation in self.AGGREGATION_PLAN[metric]:
                    new_attr = self._aggregate_attr_name(
                        prefix=prefix,
                        metric=metric,
                        aggregation=aggregation,
                    )

                    result[new_attr] = self._aggregate(
                        values=values,
                        aggregation=aggregation,
                    )

            # Calculate and aggregate derived event-level gaps.
            if {"min", "median", "max"}.issubset(metrics):
                min_values = self._as_array(
                    node_data[metrics["min"]]
                )

                median_values = self._as_array(
                    node_data[metrics["median"]]
                )

                max_values = self._as_array(
                    node_data[metrics["max"]]
                )

                derived_values = {
                    "upper_gap": self._difference(
                        max_values,
                        median_values,
                    ),
                    "lower_gap": self._difference(
                        median_values,
                        min_values,
                    ),
                    "range": self._difference(
                        max_values,
                        min_values,
                    ),
                }

                for metric, values in derived_values.items():
                    for aggregation in self.AGGREGATION_PLAN[metric]:
                        new_attr = self._aggregate_attr_name(
                            prefix=prefix,
                            metric=metric,
                            aggregation=aggregation,
                        )

                        result[new_attr] = self._aggregate(
                            values=values,
                            aggregation=aggregation,
                        )

        return result

    def add_node_attributes(
        self,
        judge: str,
        attributes: dict[str, object],
    ) -> None:
        """Write collected scalar values to one graph node."""
        clean_attributes = {
            key: self._to_python_scalar(value)
            for key, value in attributes.items()
        }

        self.G.nodes[judge].update(
            clean_attributes
        )

    def remove_primary_node_attributes(
        self,
        judge: str,
    ) -> None:
        """Remove old array-valued attributes from one node."""
        node_data = self.G.nodes[judge]

        array_attrs = [
            attr
            for attr, value in node_data.items()
            if isinstance(
                value,
                (
                    list,
                    tuple,
                    np.ndarray,
                    pd.Series,
                ),
            )
        ]

        for attr in array_attrs:
            del node_data[attr]

    def aggregate_nodes(self) -> None:
        """Collect, add, and clean attributes for all judges."""
        for judge in list(self.G.nodes):
            attributes = self.collect_node_attributes(
                judge
            )

            self.add_node_attributes(
                judge=judge,
                attributes=attributes,
            )

            self.remove_primary_node_attributes(
                judge
            )

    # ------------------------------------------------------------------
    # Edge cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def _count_edge_events(
        edge_data: dict,
    ) -> int:
        """Count unique year × stage × category combinations."""
        events = pd.DataFrame(
            {
                "year": edge_data["year"],
                "stage": edge_data["stage"],
                "category": edge_data["category"],
            }
        )

        return int(
            events
            .drop_duplicates()
            .shape[0]
        )

    def simplify_edges(self) -> None:
        """
        Remove arrays from edges.

        Edge-level correlation metrics are not aggregated at this stage.
        Only judge identifiers and common-event counts are retained.
        """
        for judge_1, judge_2, data in self.G.edges(data=True):
            judge_a, judge_b = self._ordered_pair(
                judge_1,
                judge_2,
            )

            data["judge_a"] = judge_a
            data["judge_b"] = judge_b

            data["n_common_events"] = (
                self._count_edge_events(data)
            )

            data["n_common_judging_acts"] = len(
                data["year"]
            )

            array_attrs = [
                attr
                for attr, value in data.items()
                if isinstance(
                    value,
                    (
                        list,
                        tuple,
                        np.ndarray,
                        pd.Series,
                    ),
                )
            ]

            for attr in array_attrs:
                del data[attr]

    # ------------------------------------------------------------------
    # DataFrames
    # ------------------------------------------------------------------

    @staticmethod
    def _to_python_scalar(value: object) -> object:
        """Convert a NumPy scalar to a standard Python value."""
        if isinstance(value, np.generic):
            return value.item()

        return value

    def create_nodes_dataframe(
        self,
        path: str | Path = NODES_1_PATH,
    ) -> pd.DataFrame:
        """Create and save the node-level DataFrame as Parquet."""
        rows = []

        for judge, data in self.G.nodes(data=True):
            row = dict(data)
            row["judge_name"] = judge
            rows.append(row)

        nodes_df = pd.DataFrame(rows)

        first_columns = [
            column
            for column in (
                "judge_name",
                "country",
                "n_events",
                "n_categories",
            )
            if column in nodes_df.columns
        ]

        remaining_columns = sorted(
            column
            for column in nodes_df.columns
            if column not in first_columns
        )

        nodes_df = nodes_df[
            first_columns + remaining_columns
        ]

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        nodes_df.to_parquet(
            output_path,
            index=False,
        )

        return nodes_df

    def create_edges_dataframe(
        self,
        path: str | Path = EDGES_1_PATH,
    ) -> pd.DataFrame:
        """Create and save the simplified edge DataFrame as Parquet."""
        rows = []

        for judge_a, judge_b, data in self.G.edges(data=True):
            row = dict(data)

            row["judge_a"] = data.get(
                "judge_a",
                judge_a,
            )

            row["judge_b"] = data.get(
                "judge_b",
                judge_b,
            )

            rows.append(row)

        edges_df = pd.DataFrame(rows)

        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        edges_df.to_parquet(
            output_path,
            index=False,
        )

        return edges_df

    # ------------------------------------------------------------------
    # Graph serialization
    # ------------------------------------------------------------------

    def _serialization_graph(self) -> nx.Graph:
        """Create a graph copy containing serializable attributes."""
        graph = self.G.copy()

        for _, data in graph.nodes(data=True):
            self._clean_serialization_attributes(data)

        for _, _, data in graph.edges(data=True):
            self._clean_serialization_attributes(data)

        return graph

    def _clean_serialization_attributes(
        self,
        data: dict,
    ) -> None:
        """Convert NumPy scalars and remove missing values."""
        for attr in list(data):
            value = self._to_python_scalar(
                data[attr]
            )

            if value is None:
                del data[attr]
                continue

            if (
                isinstance(value, float)
                and np.isnan(value)
            ):
                del data[attr]
                continue

            data[attr] = value

    def save_graph(
        self,
        path: str | Path = GRAPH_1_PATH,
    ) -> None:
        """Save the aggregated graph as GraphML, GEXF, or pickle."""
        output_path = Path(path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        suffix = output_path.suffix.lower()

        if suffix == ".graphml":
            nx.write_graphml(
                self._serialization_graph(),
                output_path,
            )
            return

        if suffix == ".gexf":
            nx.write_gexf(
                self._serialization_graph(),
                output_path,
            )
            return

        if suffix in {
            ".pkl",
            ".pickle",
            ".gpickle",
        }:
            with output_path.open("wb") as file:
                pickle.dump(
                    self.G,
                    file,
                )

            return

        raise ValueError(
            "Graph path must end with .graphml, .gexf, "
            ".pkl, .pickle, or .gpickle."
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_graph(self) -> nx.Graph:
        """Return the graph with aggregated node attributes."""
        if self._is_built:
            return self.G

        self.initialize_node_attributes()
        self.aggregate_nodes()
        self.simplify_edges()

        self._is_built = True

        return self.G


def load_primary_graph(
    path: str | Path,
) -> nx.Graph:
    """
    Load a primary graph with array attributes from pickle.

    GraphML and GEXF cannot preserve Python lists used in the primary graph.
    """
    graph_path = Path(path)

    allowed_suffixes = {
        ".pkl",
        ".pickle",
        ".gpickle",
    }

    if graph_path.suffix.lower() not in allowed_suffixes:
        raise ValueError(
            "The primary graph must be stored as .pkl, .pickle, "
            "or .gpickle because GraphML and GEXF cannot preserve "
            "array attributes."
        )

    with graph_path.open("rb") as file:
        return pickle.load(file)


def build_aggregated_judge_graph(
    base_graph: nx.Graph | str | Path | None = None,
    long_df: pd.DataFrame | None = None,
    score_cols: dict[str, str] | None = None,
    corr_types: Sequence[str] = ("pearson",),
    graph_path: str | Path = GRAPH_1_PATH,
    nodes_path: str | Path = NODES_1_PATH,
    edges_path: str | Path = EDGES_1_PATH,
) -> tuple[nx.Graph, pd.DataFrame]:
    """
    Build and save the aggregated judge graph.

    Parameters
    ----------
    base_graph:
        Primary graph, pickle path to the primary graph, or None.
    long_df:
        Long-format data used when base_graph is None.
    score_cols:
        Score columns passed to JudgeGraphBuilder.
    corr_types:
        Correlation types passed to JudgeGraphBuilder.
    graph_path:
        Output path for the aggregated graph.
    nodes_path:
        Output path for the node DataFrame.
    edges_path:
        Output path for the simplified edge DataFrame.

    Returns
    -------
    tuple[nx.Graph, pd.DataFrame]
        Aggregated graph and node DataFrame.
    """
    if base_graph is None:
        if long_df is None:
            long_df = pd.read_excel(
                LONG_DF_PATH
            )

        primary_builder = JudgeGraphBuilder(
            long_df=long_df,
            score_cols=(
                score_cols
                or {
                    "components": (
                        "judge_component_score_norm"
                    ),
                    "tech": "judge_GOE",
                }
            ),
            corr_types=corr_types,
        )

        primary_graph = (
            primary_builder.build_graph()
        )

    elif isinstance(
        base_graph,
        (str, Path),
    ):
        primary_graph = load_primary_graph(
            base_graph
        )

    else:
        primary_graph = base_graph

    aggregate_builder = JudgeAggregateGraphBuilder(
        primary_graph
    )

    graph = aggregate_builder.build_graph()

    aggregate_builder.save_graph(
        graph_path
    )

    nodes_df = (
        aggregate_builder.create_nodes_dataframe(
            nodes_path
        )
    )

    aggregate_builder.create_edges_dataframe(
        edges_path
    )

    return graph, nodes_df
