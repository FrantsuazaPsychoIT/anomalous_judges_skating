# collusion_judges/modelling/analytics.py

"""Подготовка аналитических таблиц по аномальности судей.

Исходные данные: DataFrame из NODES_4_PATH (одна строка на судью).
Результаты:
    make_group_comparison   — групповые агрегаты с MultiIndex-столбцами;
    make_ranked_judges_long — long-таблица судей с MultiIndex-столбцами;
    make_report_dataset     — полная плоская CSV-таблица для HTML-отчёта.
"""

from __future__ import annotations

import base64
import pandas as pd

from html import escape
from pathlib import Path
from IPython.display import HTML, display
from collusion_judges.config import COMPARE_1_PATH, NODES_5_PATH, CORR_TYPES, SCORE_TYPES


# -----------------------------------------------------------------------------
# Настройки. При необходимости их можно перенести в config.py.
# -----------------------------------------------------------------------------

HTML_CSV_FILENAME = "res_anomaly_rating.csv"
HTML_CSV_ENCODING = "utf-8-sig"
HTML_LINK_LABEL = "Скачать датасет (res_anomaly_rating.csv)"

# Ключ — унифицированное имя показателя в выходных таблицах.
# Для Pearson fisher_mean / rank_by_fisher_mean переименовываются,
# но численные значения не пересчитываются.
METRIC_SUFFIXES = {
    "pearson": {
        "mean": "fisher_mean",
        "mad": "mad",
        "rank_by_mean": "rank_by_fisher_mean",
    },
    "kendall": {
        "mean": "mean",
        "mad": "mad",
        "rank_by_mean": "rank_by_mean",
    },
    "spearman": {
        "mean": "mean",
        "mad": "mad",
        "rank_by_mean": "rank_by_mean",
    },
}

MACRO_STAT = "median"
ANOMALY_FEATURE_SET = "comp_tech"

# Группы пересекаются: top-5 является подмножеством top-10.
GROUP_SLICES = {
    "top-5": slice(0, 5),
    "top-10": slice(0, 10),
    "others": slice(10, None),
}

DEFAULT_TOP_N = 10
REPORT_CSV_PATH = Path(NODES_5_PATH).with_name("judges_report.csv")

SUMMARY_COLUMN_KEYS = (
    ("data_description", "Group"),
    ("data_description", "corr_type"),
    ("data_description", "score_type"),
    ("mean", "mean"),
    ("mean", "mad"),
    ("mean", "rank_by_mean"),
    ("median", "n_events"),
    ("median", "n_categories"),
)

LONG_COLUMN_KEYS = (
    ("score_descr", "corr_type"),
    ("score_descr", "score_type"),
    ("judge", "name"),
    ("judge", "country"),
    ("judge", "n_events"),
    ("judge", "n_categories"),
    ("judge", "anom_rank"),
    ("judge", "anom_score"),
    ("median_of", "mean"),
    ("median_of", "mad"),
    ("median_of", "rank_by_mean"),
)

# Плоские имена полей для CSV, скачиваемого из отчёта.
REPORT_COLUMN_NAMES = (
    "corr_type",
    "score_type",
    "judge_name",
    "country",
    "n_events",
    "n_categories",
    "anom_rank",
    "anom_score",
    "median_of_mean",
    "median_of_mad",
    "median_of_rank_by_mean",
)

# -----------------------------------------------------------------------------
# Вспомогательные функции
# -----------------------------------------------------------------------------

def _metric_column(score_type: str, corr_type: str, metric: str) -> str:
    """Восстановить имя колонки в NODES_4."""
    suffix = METRIC_SUFFIXES[corr_type][metric]
    return f"{MACRO_STAT}_of_{score_type}_{corr_type}_{suffix}"


def _rank_column(corr_type: str) -> str:
    return f"anomaly_rank_{corr_type}_{ANOMALY_FEATURE_SET}"


def _score_column(corr_type: str) -> str:
    return f"anomaly_score_{corr_type}_{ANOMALY_FEATURE_SET}"


def _validate_columns(df: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns.difference(df.columns))
    if missing:
        raise ValueError(f"В anomaly_df отсутствуют колонки: {missing}")


def _save_dataframe(df: pd.DataFrame, save_path: str | Path | None) -> None:
    if save_path is None:
        return

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        raise ValueError("Поддерживаются только файлы .parquet и .csv")


def _ranked_frame(anomaly_df: pd.DataFrame, corr_type: str) -> pd.DataFrame:
    """Сортировать по рангу; при равенстве ранга — по имени судьи."""
    return anomaly_df.sort_values(
        by=[_rank_column(corr_type), "judge_name"],
        ascending=True,
        kind="stable",
        na_position="last",
    )


# -----------------------------------------------------------------------------
# 1. Сравнение аномальных групп с остальными судьями
# -----------------------------------------------------------------------------

def make_group_comparison(
    anomaly_df: pd.DataFrame,
    save_path: str | Path | None = COMPARE_1_PATH,
) -> pd.DataFrame:
    """Агрегировать признаки по top-5, top-10 и others для каждой корреляции.

    В top-группы попадают первые N судей по anomaly_rank (1 — аномальнее).
    Судьи с пропущенным рангом не относятся ни к одной группе.
    Средние считаются по median_of-признакам, медианы — по активности.
    """
    required = {"judge_name", "n_events", "n_categories"}
    for corr_type in CORR_TYPES:
        required.add(_rank_column(corr_type))
        for score_type in SCORE_TYPES:
            required.update(
                _metric_column(score_type, corr_type, metric)
                for metric in METRIC_SUFFIXES[corr_type]
            )
    _validate_columns(anomaly_df, required)

    rows = []
    for corr_type in CORR_TYPES:
        ranked = _ranked_frame(anomaly_df, corr_type).dropna(
            subset=[_rank_column(corr_type)]
        )

        for group_name, bounds in GROUP_SLICES.items():
            group_df = ranked.iloc[bounds]
            for score_type in SCORE_TYPES:
                row = {
                    ("data_description", "Group"): group_name,
                    ("data_description", "corr_type"): corr_type,
                    ("data_description", "score_type"): score_type,
                    ("median", "n_events"): group_df["n_events"].median(),
                    ("median", "n_categories"): group_df["n_categories"].median(),
                }
                for metric in METRIC_SUFFIXES[corr_type]:
                    row[("mean", metric)] = group_df[
                        _metric_column(score_type, corr_type, metric)
                    ].mean()
                rows.append(row)

    result = pd.DataFrame(rows).reindex(
        columns=pd.MultiIndex.from_tuples(SUMMARY_COLUMN_KEYS)
    )
    _save_dataframe(result, save_path)
    return result
    
# -----------------------------------------------------------------------------
# 2. Поимённая таблица: wide -> long
# -----------------------------------------------------------------------------

def make_ranked_judges_long(
    anomaly_df: pd.DataFrame,
    corr_type: str,
    n_judges: int | None = DEFAULT_TOP_N,
    save_path: str | Path | None = NODES_5_PATH,
) -> pd.DataFrame:
    """Развернуть components/technique в строки для одного типа корреляции.

    При n_judges=None включаются все судьи, в том числе с NaN в ранге;
    иначе — первые n_judges судей с непустым рангом.
    """
    if corr_type not in CORR_TYPES:
        raise ValueError(f"corr_type должен быть одним из {CORR_TYPES}")
    if n_judges is not None and (not isinstance(n_judges, int) or n_judges <= 0):
        raise ValueError("n_judges должен быть положительным int или None")

    required = {
        "judge_name", "country", "n_events", "n_categories",
        _rank_column(corr_type), _score_column(corr_type),
    }
    for score_type in SCORE_TYPES:
        required.update(
            _metric_column(score_type, corr_type, metric)
            for metric in METRIC_SUFFIXES[corr_type]
        )
    _validate_columns(anomaly_df, required)

    ranked = _ranked_frame(anomaly_df, corr_type)
    if n_judges is not None:
        ranked = ranked.dropna(subset=[_rank_column(corr_type)]).head(n_judges)

    blocks = []
    for score_type in SCORE_TYPES:
        # .to_numpy() позволяет избежать случайного выравнивания по индексу.
        block = pd.DataFrame({
            "corr_type": corr_type,
            "score_type": score_type,
            "name": ranked["judge_name"].to_numpy(),
            "country": ranked["country"].to_numpy(),
            "n_events": ranked["n_events"].to_numpy(),
            "n_categories": ranked["n_categories"].to_numpy(),
            "anom_rank": ranked[_rank_column(corr_type)].to_numpy(),
            "anom_score": ranked[_score_column(corr_type)].to_numpy(),
        })
        for metric in METRIC_SUFFIXES[corr_type]:
            block[metric] = ranked[
                _metric_column(score_type, corr_type, metric)
            ].to_numpy()
        blocks.append(block)

    result = pd.concat(blocks, ignore_index=True)
    result = result[[
        "corr_type", "score_type", "name", "country", "n_events",
        "n_categories", "anom_rank", "anom_score", "mean", "mad",
        "rank_by_mean",
    ]]
    result.columns = pd.MultiIndex.from_tuples(LONG_COLUMN_KEYS)

    # Order by corr_type, anom_rank, score_type (при равных рангах — name).
    result = result.sort_values(
        by=[
            ("score_descr", "corr_type"),
            ("judge", "anom_rank"),
            ("score_descr", "score_type"),
            ("judge", "name"),
        ],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    _save_dataframe(result, save_path)
    return result


# -----------------------------------------------------------------------------
# 3. Полный датасет для скачивания из HTML-отчёта
# -----------------------------------------------------------------------------

def make_html_df_link(
    df: pd.DataFrame,
    filename: str = HTML_CSV_FILENAME,
    label: str = HTML_LINK_LABEL,
    encoding: str = HTML_CSV_ENCODING,
) -> None:

    csv_df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        csv_df.columns = df.columns.get_level_values(1)

    csv_bytes = csv_df.to_csv(index=False).encode(encoding)
    encoded = base64.b64encode(csv_bytes).decode("ascii")

    html = f'<a download="{escape(filename, quote=True)}" href="data:text/csv;base64,{encoded}">{escape(label)}</a>'

    display(HTML(html))