from __future__ import annotations
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_PATH = PROJ_ROOT / "data"

EXTERNAL_PATH = DATA_PATH / "external"
INTERIM_PATH = DATA_PATH / "interim"

PRIMARY_GRAPH_PATH = INTERIM_PATH / "graph" / "primary_graph.pkl"
NODES_2_PATH = INTERIM_PATH / "graph" / "nodes_2.parquet"
EDGES_2_PATH = INTERIM_PATH / "graph" / "edges_2.parquet"
GRAPH_2_PATH = INTERIM_PATH / "graph" / "graph_2.graphml"

PROCESSED_PATH = DATA_PATH / "processed"
WIDE_DF_PATH = PROCESSED_PATH / "wide_dataset.xlsx"
LONG_DF_PATH = PROCESSED_PATH / "long_dataset.xlsx"
NODES_3_PATH = PROCESSED_PATH / "nodes_3.parquet"

FINAL_PATH = DATA_PATH / "final"


#MODELS_DIR = PROJ_ROOT / "models"

FIGURES_PATH = PROJ_ROOT / "reports" / "figures"
CORR_FIG_PATH = FIGURES_PATH / "Dist_corr_new"
EXPLANATORY_FIG_PATH = FIGURES_PATH / "explanatory"
JUDGES_DESCR_FIG_PATH = FIGURES_PATH / "judges"
ANOMALY_FIGURES_PATH = FIGURES_PATH / "anomaly_description"

# -------------------- HTTP / parser settings --------------------

REQUEST_SLEEP: float = 0.2
REQUEST_TIMEOUT: float = 30.0

DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": "collusion-judges-research-bot/1.0 (+https://example.invalid)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
