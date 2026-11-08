"""Visualizations of Isolation Forest source features."""

from __future__ import annotations

from pathlib import Path
from itertools import combinations
from matplotlib.lines import Line2D


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from collusion_judges.config import ANOMALY_FIGURES_PATH, CORR_TYPES, SCORE_TYPES


CORR_TYPES = ("pearson", "kendall", "spearman")
SCORE_TYPES = ("components", "technique")

COLS_CORR_BASED = {
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

ANOMALY_SCORE_COL = "anomaly_score_{corr}_comp_tech"
TOP_10_COL = "top_10_anom_{corr}_comp_tech"

ANOMALY_PALETTE = {
    False: "#B8B8B8",
    True: "#D62728",
}


# ---------------------------------------------------------------------
# Two panels: source feature and anomaly score
# ---------------------------------------------------------------------

def pair_scatterplots(
    data: pd.DataFrame,
    ax_names: dict,
    path: str | Path | None = None,
    trend: bool = True,
):
    """Plot components and technique features against anomaly score."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    x_columns = (ax_names["l_x"], ax_names["r_x"])
    titles = (ax_names["l_title"], ax_names["r_title"])

    for i, (ax, x, title) in enumerate(zip(axes, x_columns, titles)):
        sns.scatterplot(
            data=data,
            x=x,
            y=ax_names["y"],
            hue=ax_names["hue"],
            palette=ANOMALY_PALETTE,
            alpha=0.8,
            s=45,
            legend=i == 0,
            ax=ax,
        )

        if trend:
            sns.regplot(
                data=data,
                x=x,
                y=ax_names["y"],
                scatter=False,
                lowess=True,
                color="#2A6F97",
                ax=ax,
            )

        ax.set_title(title)
        ax.set_xlabel(ax_names["x_label"])
        ax.set_ylabel("Anomaly score")

    if axes[0].get_legend():
        axes[0].get_legend().set_title("Top-10 anomaly")

    fig.tight_layout()

    if path is not None:
        fig.savefig(path, bbox_inches="tight")

    plt.show()

    return fig, axes


# ---------------------------------------------------------------------
# Two panels: top-10 anomalies and feature distributions
# ---------------------------------------------------------------------

def pair_swarmplots(
    data: pd.DataFrame,
    ax_names: dict,
    path: str | Path | None = None,
):
    """Compare feature distributions for top-10 and other judges."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)

    y_columns = (ax_names["l_y"], ax_names["r_y"])
    titles = (ax_names["l_title"], ax_names["r_title"])

    for ax, y, title in zip(axes, y_columns, titles):
        sns.boxplot(
            data=data,
            x=ax_names["x"],
            y=y,
            hue=ax_names["x"],
            palette=ANOMALY_PALETTE,
            showfliers=False,
            legend=False,
            ax=ax,
        )

        sns.swarmplot(
            data=data,
            x=ax_names["x"],
            y=y,
            hue=ax_names["x"],
            palette=ANOMALY_PALETTE,
            size=3,
            legend=False,
            ax=ax,
        )

        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(ax_names["y_label"])
        ax.set_xticklabels(["Other judges", "Top-10 anomalies"])

    fig.tight_layout()

    if path is not None:
        fig.savefig(path, bbox_inches="tight")

    plt.show()

    return fig, axes


# ---------------------------------------------------------------------
# Two panels: two source features and top-10 anomalies
# ---------------------------------------------------------------------

def pair_2d_scatterplots(
    data: pd.DataFrame,
    ax_names: dict,
    path: str | Path | None = None,
):
    """Plot two source features for components and technique."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4),sharex=True, sharey=True)

    plot_params = (
        (
            ax_names["l_x"],
            ax_names["l_y"],
            ax_names["l_title"],
        ),
        (
            ax_names["r_x"],
            ax_names["r_y"],
            ax_names["r_title"],
        ),
    )

    for i, (ax, x, y, title) in enumerate(
        (axes[j], *plot_params[j])
        for j in range(2)
    ):
        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            hue=ax_names["hue"],
            palette=ANOMALY_PALETTE,
            alpha=0.8,
            s=50,
            legend=i == 0,
            ax=ax,
        )

        ax.set_title(title)
        ax.set_xlabel(ax_names["x_label"])
        ax.set_ylabel(ax_names["y_label"])

    if axes[0].get_legend():
        axes[0].get_legend().set_title("Top-10 anomaly")

    fig.tight_layout()

    if path is not None:
        fig.savefig(path, bbox_inches="tight")

    plt.show()

    return fig, axes


# ---------------------------------------------------------------------
# Prepare names and call one of the plotting functions
# ---------------------------------------------------------------------

def compare_comp_and_tech(
    data: pd.DataFrame,
    stats: dict,
    path: str | Path | None = None,
    fig_type: str = "scatter",
    trend: bool = True,
):
    """Plot one source metric for components and technique."""
    corr = stats["corr"]
    macro_stata = stats["macro"]
    micro_stata = stats["micro"]

    components_col = (
        f"{macro_stata}_of_components_{corr}_{micro_stata}"
    )
    technique_col = (
        f"{macro_stata}_of_technique_{corr}_{micro_stata}"
    )

    ax_names = {
        "l_title": f"{macro_stata}({micro_stata}), components",
        "r_title": f"{macro_stata}({micro_stata}), technique",
    }

    if fig_type == "scatter":
        ax_names.update(
            {
                "l_x": components_col,
                "r_x": technique_col,
                "y": ANOMALY_SCORE_COL.format(corr=corr),
                "hue": TOP_10_COL.format(corr=corr),
                "x_label": f"{macro_stata}({micro_stata})",
            }
        )

        file_name = (
            f"score_and_{macro_stata}_of_{micro_stata}_{corr}.pdf"
        )
        output_path = Path(path) / file_name if path is not None else None

        return pair_scatterplots(
            data=data,
            ax_names=ax_names,
            path=output_path,
            trend=trend,
        )

    if fig_type == "swarm":
        ax_names.update(
            {
                "l_y": components_col,
                "r_y": technique_col,
                "x": TOP_10_COL.format(corr=corr),
                "y_label": f"{macro_stata}({micro_stata})",
            }
        )

        file_name = (
            f"top10_and_{macro_stata}_of_{micro_stata}_{corr}.pdf"
        )
        output_path = Path(path) / file_name if path is not None else None

        return pair_swarmplots(
            data=data,
            ax_names=ax_names,
            path=output_path,
        )

    raise ValueError("fig_type must be 'scatter' or 'swarm'.")


# ---------------------------------------------------------------------
# Wrapper: feature versus anomaly score / top-10
# ---------------------------------------------------------------------

def anomaly_score_and_src(
    data: pd.DataFrame,
    corr_types=CORR_TYPES,
    macro_stata: str = "median",
    params=COLS_CORR_BASED,
    path: str | Path | None = ANOMALY_FIGURES_PATH,
    trend: bool = True,
    fig_type: str = "scatter",
):
    """Build source-feature plots for all correlation models."""
    for corr in corr_types:
        for micro_stata in params[macro_stata][corr]:
            stats = {
                "corr": corr,
                "macro": macro_stata,
                "micro": micro_stata,
            }

            compare_comp_and_tech(
                data=data,
                stats=stats,
                path=path,
                fig_type=fig_type,
                trend=trend,
            )


# ---------------------------------------------------------------------
# Wrapper: all pairs of model features
# ---------------------------------------------------------------------

def anomaly_2d(
    data: pd.DataFrame,
    corr_types=CORR_TYPES,
    macro_stata: str = "median",
    params=COLS_CORR_BASED,
    path: str | Path | None = ANOMALY_FIGURES_PATH,
):
    """Build two-dimensional feature plots with top-10 highlighted."""
    for corr in corr_types:
        for x_metric, y_metric in combinations(
            params[macro_stata][corr],
            2,
        ):
            ax_names = {
                "l_x": (
                    f"{macro_stata}_of_components_{corr}_{x_metric}"
                ),
                "l_y": (
                    f"{macro_stata}_of_components_{corr}_{y_metric}"
                ),
                "r_x": (
                    f"{macro_stata}_of_technique_{corr}_{x_metric}"
                ),
                "r_y": (
                    f"{macro_stata}_of_technique_{corr}_{y_metric}"
                ),
                "hue": TOP_10_COL.format(corr=corr),
                "l_title": "Components",
                "r_title": "Technique",
                "x_label": f"{macro_stata}({x_metric})",
                "y_label": f"{macro_stata}({y_metric})",
            }

            file_name = (
                f"top10_{corr}_{x_metric}_and_{y_metric}.pdf"
            )
            output_path = Path(path) / file_name if path is not None else None

            pair_2d_scatterplots(
                data=data,
                ax_names=ax_names,
                path=output_path,
            )
# ---------------------------------------------------------------------
# Compare top anomalies and other
# ---------------------------------------------------------------------


def make_plots_for_corr_top_other(
    df, corr_type='pearson', score_types=SCORE_TYPES,
    label_fontsize=7, label_country=True,
):
    data = df[df['corr_type'] == corr_type].copy()

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 10), sharex=True, sharey=True,
    )
    fig.suptitle(f'{corr_type}: mean corr vs rank by mean')

    for i, top in enumerate((5, 10)):
        for j, score_type in enumerate(score_types):
            subset = data[data['score_type'] == score_type].copy()
            in_top = subset['anom_rank'] <= top

            ax = axes[i, j]

            ax.scatter(
                subset['mean'], subset['rank_by_mean'],
                c=in_top.map(ANOMALY_PALETTE),
                alpha=0.85, edgecolor='white', linewidth=0.3, s=40,
            )

            for _, r in subset[in_top].iterrows():
                if pd.isna(r['name']):
                    continue
                if label_country and pd.notna(r.get('country')):
                    label = f"{r['name']} ({r['country']})"
                else:
                    label = str(r['name'])

                ax.annotate(
                    label,
                    xy=(r['mean'], r['rank_by_mean']),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=label_fontsize,
                    color=ANOMALY_PALETTE[True],
                    alpha=0.95, ha='left', va='bottom',
                    clip_on=True,
                )

            ax.set_title(f'Top{top} vs other: by {score_type}')
            ax.set_xlabel('median mean corr in panel')
            ax.set_ylabel('median rank by mean in panel')

            handles = [
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=ANOMALY_PALETTE[True], markersize=8,
                       label=f'in Top{top} (by {score_type})'),
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=ANOMALY_PALETTE[False], markersize=8,
                       label='other'),
            ]
            ax.legend(handles=handles, loc='best', fontsize=8)

    plt.tight_layout()
    plt.show()


def make_plots_for_corr_tech_comp(df, corr_type='pearson', score_types=SCORE_TYPES,
                        keys=('name', 'country'), label_fontsize=7):
    data = df[df['corr_type'] == corr_type].copy()

    comp = data[data['score_type'] == score_types[0]].set_index(list(keys))
    tech = data[data['score_type'] == score_types[1]].set_index(list(keys))

    merged = comp[['anom_rank', 'mean', 'rank_by_mean']].join(
        tech[['anom_rank', 'mean', 'rank_by_mean']],
        lsuffix='_comp', rsuffix='_tech', how='inner'
    ).reset_index()
    # ↑ здесь уже есть столбцы 'name' и 'country'

    if merged.empty:
        print(f'[{corr_type}] после merge пусто — проверьте ключи {keys}')
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{corr_type}: components vs technique', fontsize=14)

    for i, top in enumerate((5, 10)):
        in_top = (merged['anom_rank_comp'] <= top) | (merged['anom_rank_tech'] <= top)
        colors = in_top.map(ANOMALY_PALETTE)

        for j, col in enumerate(['mean', 'rank_by_mean']):
            ax = axes[i, j]
            x = merged[f'{col}_comp']
            y = merged[f'{col}_tech']

            ax.scatter(x, y, c=colors, alpha=0.85,
                       edgecolor='white', linewidth=0.3, s=40)

            # ---- подписи только аномальных точек ----
            for _, r in merged[in_top].iterrows():
                if pd.isna(r['name']):
                    continue
                ax.annotate(
                    str(r['name']),
                    xy=(r[f'{col}_comp'], r[f'{col}_tech']),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=label_fontsize,
                    color=ANOMALY_PALETTE[True],
                    alpha=0.95, ha='left', va='bottom',
                    clip_on=True,
                )

            # диагональ y = x
            lo = min(x.min(), y.min())
            hi = max(x.max(), y.max())
            ax.plot([lo, hi], [lo, hi], '--', color='gray', alpha=0.5)

            ax.set_xlabel(f'{col} (components)')
            ax.set_ylabel(f'{col} (technique)')
            sub = 'mean corr' if col == 'mean' else 'mean rank'
            ax.set_title(f'Top{top}: {sub}')

            handles = [
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=ANOMALY_PALETTE[True], markersize=8,
                       label=f'in Top{top} (any method)'),
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=ANOMALY_PALETTE[False], markersize=8,
                       label='остальные'),
            ]
            ax.legend(handles=handles, loc='best', fontsize=8)

    plt.tight_layout()
    plt.show()


def iterate_plots(df, corr_types=CORR_TYPES, score_types=SCORE_TYPES, mode='top_other'):
    for corr_type in corr_types:
        if mode == 'top_other':
            make_plots_for_corr_top_other(df, corr_type=corr_type, score_types=score_types)
        elif mode == 'tech_comp':
            make_plots_for_corr_tech_comp(df, corr_type=corr_type, score_types=score_types)

