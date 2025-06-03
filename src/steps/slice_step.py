from __future__ import annotations
from pathlib import Path
import os, logging
from time import perf_counter

import duckdb
import pandas as pd
from tqdm.auto import tqdm
import sys

PROJ_ROOT = Path(__file__).parents[2] 
sys.path.insert(0, str(PROJ_ROOT / "src"))


from steps.pipeline_step import PipelineStep
from utils.utils import get_db_path

TS_COL_MAP = {
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

DEFAULT_SENSORS = (
    "accelerometer",
    "gyroscope"
)


class SliceConfig:

    def __init__(self, db_path: str | Path, out_dir: str | Path,
                 max_workers: int = os.cpu_count() or 1, seed: int = 42,
                 sensors: tuple[str, ...] = DEFAULT_SENSORS, window_size_ms: int = 1000,
                 overlap_pct: float = 0.5):

        self.db_path = Path(db_path)
        self.out_dir = Path(out_dir / "windows") 
        self.max_workers = max_workers
        self.seed = seed
        self.sensors = sensors
        self.window_size_ms = window_size_ms
        self.overlap_pct = overlap_pct

def sliding_window(
    df: pd.DataFrame,
    ts_col: str,
    window_size_ms: int,
    overlap_pct: float
) -> list[tuple[int, int, pd.DataFrame]]:
    """
    Gera janelas deslizantes sobre df[ts_col], retornando tuplas (start, end, df_window).
    """
    step = int(window_size_ms * (1.0 - overlap_pct))
    timestamps = df[ts_col].astype(int)
    start = int(timestamps.iloc[0])
    end_max = int(timestamps.iloc[-1])
    windows: list[tuple[int, int, pd.DataFrame]] = []

    while start + window_size_ms <= end_max:
        end = start + window_size_ms
        mask = (timestamps >= start) & (timestamps < end)
        win = df.loc[mask].copy()
        windows.append((start, end, win))
        start += step

    return windows


def extract_windows(pairs: list[tuple[str, int]], con: duckdb.DuckDBPyConnection,
                    sensor: str, time_col: str, window_ms: int, overlap_pct: float) -> list[dict]:
    rows: list[dict] = []
    for subj, sess in tqdm(pairs, desc=f"Pareamentos {sensor}", ncols=80):
        df_ts = con.execute(
            f"SELECT {time_col} FROM {sensor} "
            " WHERE subject_id = ? AND session_number = ? "
            f" ORDER BY {time_col}",
            [subj, sess]
        ).fetchdf()

        if df_ts.empty:
            continue

        for start, end, win in sliding_window(df_ts, time_col, window_ms, overlap_pct):
            rows.append({
                "sensor": sensor,
                "subject_id": subj,
                "session_number": sess,
                "start_ms": start,
                "end_ms": end,
                "num_samples": len(win)
            })
    return rows


def slice(cfg: SliceConfig) -> Path:
    log = logging.getLogger("slice")
    log.info("🔪 Iniciando slicing...")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    db_path = cfg.db_path
    if not db_path.exists():
        log.error(f"❌ DuckDB não encontrado: {db_path}")
        return context

    con = duckdb.connect(str(db_path), read_only=True)
    for sensor in cfg.sensors:
        if sensor == "acrtivity":
            continue
        log.info(f"🔍 Processando sensor: {sensor}...")
        time_col = TS_COL_MAP.get(sensor, "time_rel_ms")
        out_file = cfg.out_dir / f"{sensor}_ws_{cfg.window_size_ms}_ov_{int(cfg.overlap_pct*100)}.parquet"
        
        if out_file.exists():
            log.info(f"✨ Janelas já existem para {sensor} → {out_file}")
            continue

        pairs = con.execute(
            f"SELECT DISTINCT subject_id, session_number "
            f"  FROM {sensor} "
            " ORDER BY subject_id, session_number"
        ).fetchall()

        rows = extract_windows(pairs, con, sensor, time_col, cfg.window_size_ms, cfg.overlap_pct)
        if rows:
            log.info(f"📊 {sensor}: extraídas {len(rows)} janelas, salvando em {out_file}")
            pd.DataFrame(rows).to_parquet(out_file, index=False)
        else:
            log.warning(f"⚠️ Nenhuma janela gerada para sensor {sensor}.")
    
    con.close()
    log.info("✅ Slicing concluído.")
    return cfg.out_dir 

        

class SliceStep(PipelineStep):

    def execute(self, context: dict) -> dict:

        cfg_params = {
            "db_path": context.get("db_path", "data/duckdb/hmog_All.duckdb"),
            "out_dir": Path(context.get("experiment_dir", "windows")),
            "max_workers": context.get("max_workers", os.cpu_count() or 1),
            "seed": context.get("seed", 42),
            "sensors": context.get("sensors", DEFAULT_SENSORS),
            "window_size_ms": context.get("window_size_ms", 1000),
            "overlap_pct": context.get("overlap_pct", 0.5),
        }

        cfg = SliceConfig(**cfg_params)
        data_windows_path = slice(cfg)
        context["data_windows_path"] = str(data_windows_path)
        return context



