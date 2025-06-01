from __future__ import annotations
from pathlib import Path
import sys, os

PROJ_ROOT = Path(__file__).parents[2] 
sys.path.insert(0, str(PROJ_ROOT / "src"))

from steps.pipeline_step import PipelineStep
import duckdb, json, random, logging
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd, numpy as np
from tqdm import tqdm
from utils.utils import get_db_path, get_window_path, get_sensor_windows_path
from scipy.stats import skew, kurtosis

DEFAULT_SENSORS = (
    "accelerometer",
    "gyroscope"
)

class FeatureExtractionConfig:
    def __init__(self, db_path: str | Path, windows_dir: str | Path,
                 out_dir: str | Path, max_workers: int = os.cpu_count() or 1,
                  seed: int = 42, sensors: List[str] = None, window_size_ms: int = 1000,
                  overlap_pct: float = 0.5):
        self.db_path = Path(db_path)
        self.windows_dir = Path(windows_dir)
        self.out_dir = Path(out_dir / "features")
        self.max_workers = max_workers
        self.seed = seed
        self.sensors = sensors or DEFAULT_SENSORS
        self.window_size_ms = window_size_ms
        self.overlap_pct = overlap_pct

def extract_features_continuous(data: pd.DataFrame) -> dict:
    values = data.to_numpy(dtype=np.float64)
    cols = data.columns.tolist()
    features: dict[str, float] = {}

    for i, col in enumerate(cols):
        arr = values[:, i]
        features[f"{col}_mean"] = np.mean(arr)
        features[f"{col}_median"] = np.median(arr)
        features[f"{col}_std"] = np.std(arr)
        features[f"{col}_min"] = np.min(arr)
        features[f"{col}_max"] = np.max(arr)
    
    # Magnitude
    magnitude = np.linalg.norm(values, axis=1)
    features["magnitude_mean"] = magnitude.mean()
    features["magnitude_std"] = magnitude.std(ddof=1)
    features["magnitude_min"] = magnitude.min()
    features["magnitude_max"] = magnitude.max()

    # Pearson
    corr_xz = np.corrcoef(values[:, 0], values[:, 2])[0, 1]
    corr_yz = np.corrcoef(values[:, 1], values[:, 2])[0, 1]
    features["corr_xz"] = corr_xz
    features["corr_yz"] = corr_yz

    # Skewness and Kurtosis
    for i, col in enumerate(cols):
        arr = values[:, i]
        features[f"{col}_skewness"] = float (skew(arr, bias=False))
        features[f"{col}_kurtosis"] = float (kurtosis(arr, bias=False))

    features["magnitude_skewness"] = skew(magnitude)
    features["magnitude_kurtosis"] = kurtosis(magnitude)

    # FFT ?
    
    return features

def session_worker(args: Tuple[str, List[Dict], str]) -> List[Dict] | None:
    sensor, windows, db_path = args
    con = duckdb.connect(db_path, read_only=True)
    rows: List[Dict] = []
    time_col = time_col_map[sensor]
    cols = ", ".join(value_col_map[sensor])
    for win in windows:
        subj, sess = win["subject_id"], int(win["session_number"])
        start, end = int(win["start_ms"]), int(win["end_ms"])
        q = (
            f"SELECT {cols} FROM {sensor} "
            f"WHERE subject_id='{subj}' AND session_number={sess} "
            f"AND {time_col} BETWEEN {start} AND {end}"
        )
        df = con.execute(q).fetchdf()
        if df.empty or df is None:
            continue
        if sensor in sensor_feature_extraction_map['continuous']:
            feats = extract_features_continuous(df)
        rows.append({
            "sensor": sensor,
            "subject_id": subj,
            "session_number": sess,
            "start_ms": start,
            "end_ms": end,
            **feats,
        })
    con.close()
    return rows if rows else None

def extract(cfg: FeatureExtractionConfig) -> Path:
    log = logging.getLogger("feature")
    log.info("🔍 Iniciando extração de features...")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    db_path = cfg.db_path
    if not db_path.exists():
        log.error(f"❌ DuckDB não encontrado: {db_path}")
        return Path()
    
    for sensor in cfg.sensors:
        if sensor == "activity":
            continue
        
        log.info("🔍 Processando janelas do sensor: %s", sensor)
        window_path = get_sensor_windows_path(cfg, sensor)
        if not window_path.exists():
            log.error(f"❌ Arquivo parquet não encontrado: {window_path}")
            continue
        
        out_file = cfg.out_dir / f"{sensor}_features.parquet"
        if out_file.exists():
            log.info(f"✨ Features já extraídas para {sensor} → {out_file}")
            continue

        windows_df = pd.read_parquet(window_path)
        groups = windows_df.groupby(["subject_id", "session_number"])
        tasks = [
            (sensor, group.to_dict("records"), str(db_path))
            for _, group in groups
        ]

        rows = []
        log.info("🚀 %s → enviando %d sessões para %d workers", sensor, len(tasks), cfg.max_workers)
        with ProcessPoolExecutor(max_workers=cfg.max_workers) as executor:
            for session_rows in tqdm(
                executor.map(session_worker, tasks), total=len(tasks), desc=f"Processando {sensor}"):
                if session_rows:
                    rows.extend(session_rows)

        if rows:
            df_features = pd.DataFrame(rows)
            df_features.to_parquet(out_file, index=False)
            log.info(f"✔  {len(df_features)} janelas ➜ {out_file}")
        else:
            log.warning(f"❌ Nenhuma feature extraída para {sensor}.")


class FeatureExtractionStep(PipelineStep):
    
    def execute(self, context: dict) -> dict:

        cfg_params = {
            "db_path": context.get("db_path"),
            "windows_dir": context.get("data_windows_path"),
            "out_dir": Path(context.get("experiment_dir", "features")),
            "max_workers": context.get("max_workers", os.cpu_count() or 1),
            "seed": context.get("seed", 42),
            "sensors": context.get("sensors", DEFAULT_SENSORS),
            "window_size_ms": context.get("window_size_ms", 1000),
            "overlap_pct": context.get("overlap_pct", 0.5),
        }

        cfg = FeatureExtractionConfig(**cfg_params)
        features_path = extract(cfg)
        context["features_path"] = features_path
        return context

            
time_col_map = {
    "accelerometer": "time_rel_ms",
    "gyroscope":     "time_rel_ms",
    "magnetometer":  "time_rel_ms",
    "touchevent":    "time_rel_ms",
    "onefingertouchevent":"time_rel_ms",
    "pinchevent":    "time_rel_ms",
    "scrollevent":    "time_current_ms",
    "strokeevent":    "time_begin_ms",
    "keypressevent":  "time_rel_ms",
}

value_col_map = {
    'accelerometer': ['acc_x','acc_y','acc_z'],
    'gyroscope': ['gyro_x','gyro_y','gyro_z'],
    'magnetometer': ['mag_x','mag_y','mag_z'],
    'touchevent': ['pointer_count','pointer_id','action_id','touch_x','touch_y','pressure','contact_size'],
    'keypressevent': ['press_type','key_id'],
    'onefingertouchevent': ['tap_id','tap_type','action_type','touch_x','touch_y','pressure','contact_size'],
    'pinchevent': ['event_type','pinch_id','time_delta','focus_x','focus_y','span','span_x','span_y','scale_factor'],
    'scrollevent': ['scroll_id','start_action_type','start_x','start_y','start_pressure','start_size','current_action_type',
                    'current_x','current_y','current_pressure','current_size','distance_x','distance_y'],
    'strokeevent': ['start_action_type','start_x','start_y','start_pressure','start_size','end_action_type','end_x','end_y',
                    'end_pressure','end_size','speed_x','speed_y'],
}


sensor_feature_extraction_map = {
    'continuous': ['accelerometer', 'gyroscope', 'magnetometer'],

}