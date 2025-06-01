from __future__ import annotations

from pathlib import Path
import random
import logging
from typing import List, Tuple
LOGGER = logging.getLogger("utils")

def get_db_path(context: dict) -> Path:
    out_dir = Path(context["IngestStep"]["duckdb_dir"])
    num_subjects = context["IngestStep"].get("max_subjects")
    seed = context.get("seed")

    duckdb_tag = f"s{num_subjects}_seed_{seed}"
    db_path = out_dir / f"hmog_{duckdb_tag}.duckdb"
    return db_path



def get_window_path(context: dict) -> Path:
    base_out = Path(context["SliceStep"]["out_dir"])
    db_path = get_db_path(context)
    folder_name = db_path.stem
    out_dir = base_out / folder_name

    return out_dir


def get_features_path(context: dict) -> Path:
    base_out = Path(context["FeatureExtractionStep"]["out_dir"])
    db_path = Path(context.get("db_path"))
    folder_name = db_path.stem
    out_dir = base_out / folder_name

    return out_dir


def get_sensor_windows_path(cfg: FeatureExtractionConfig, sensor: str) -> Path:
    windows_dir = cfg.windows_dir
    window_size = cfg.window_size_ms
    overlap_pct = cfg.overlap_pct
    return windows_dir / f"{sensor}_ws_{window_size}_ov_{int(overlap_pct * 100)}.parquet"

def discover_subject_ids(raw_dir: Path) -> list[str]:
    return sorted([p.name for p in raw_dir.iterdir() if p.is_dir()])


def get_random_user(context: dict) -> str:
    raw_dir = Path(context["IngestStep"]["raw_path"])
    subjects = discover_subject_ids(raw_dir)
    seed = context.get("seed", 42)
    if not subjects:
        raise ValueError("Nenhum sujeito encontrado no diretório de dados brutos.")
    
    return random.choice(subjects)