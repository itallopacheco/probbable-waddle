from __future__ import annotations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import shutil
import sys
import os

PROJ_ROOT = Path(__file__).parents[2] 
sys.path.insert(0, str(PROJ_ROOT / "src"))

from steps.pipeline_step import PipelineStep
import duckdb, json, random, logging
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import pandas as pd
from tqdm.auto import tqdm
from utils.utils import get_db_path

DEFAULT_SENSORS = (
    "accelerometer",
    "gyroscope"
)

class IngestConfig:

    def __init__(self, raw_path: str | Path, out_dir: str | Path,
                 max_workers: int = os.cpu_count() or 1, num_subjects: int | None = None,
                 seed: int = 42, sensors: tuple[str, ...] = DEFAULT_SENSORS):
        self.raw_path = Path(raw_path)
        self.out_dir = Path(out_dir)
        self.max_workers = max_workers
        self.num_subjects = num_subjects
        self.seed = seed
        self.sensors = sensors


    def subjects(self) -> list[str]:
        """Retorna lista (poss. amostrada) de IDs de voluntários."""
        all_ids = sorted(p.name for p in self.raw_path.iterdir() if p.is_dir())
        if self.num_subjects is None:
            return all_ids
        if self.num_subjects > len(all_ids):
            raise ValueError(
                f"num_subjects={self.num_subjects} > {len(all_ids)} disponíveis"
            )
        random.seed(self.seed)
        return sorted(random.sample(all_ids, self.num_subjects))

    def db_path(self) -> Path:
        label = self.num_subjects or "all"
        return self.out_dir / f"hmog_{label}.duckdb"


def _csv_to_parquet(csv_path: Path, tmp_dir: Path, sensors: Iterable[str]) -> Path | None:
    table = csv_path.stem.lower().split("_")[0]
    if table not in sensors:
        return None

    df = pd.read_csv(csv_path, header=None, names=HEADER_MAPS[table])

    for col in df.columns:
        if col in ABSOLUTE_TIME_COLS.get(table, []):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            df.rename(columns={col: f"{col}_ms"}, inplace=True)
        elif col in RELATIVE_TIME_COLS.get(table, []):
            rel_ns = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            df[f"{col}_ns"] = rel_ns
            df[f"{col}_ms"] = (rel_ns // 1_000_000).astype("Int64")
            df.drop(columns=[col], inplace=True)

    if table != "activity":
        subj_id = csv_path.parents[1].name
        session_num = int(csv_path.parents[0].name.split("_session_")[1])
        df["subject_id"] = subj_id
        df["session_number"] = session_num

    pq_name = f"{csv_path.stem.lower()}__{subj_id}__{session_num}.parquet"  # type: ignore
    pq_path = tmp_dir / pq_name
    df.to_parquet(pq_path, index=False)
    return pq_path

def _ingest_sensor(con: duckdb.DuckDBPyConnection, sensor: str, tmp_dir: Path):
    pattern = str(tmp_dir / f"{sensor}__*.parquet")
    con.execute(
        f"CREATE TABLE {sensor} AS SELECT * FROM read_parquet('{pattern}', union_by_name=True)"
    )

def ingest(cfg: IngestConfig) -> Path:
    log = logging.getLogger("ingest")
    log.info("📥 Iniciando ingestão do dataset…")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    db_path = cfg.db_path()

    if db_path.exists():
        log.info("✅ DuckDB já existe, pulando ingestão → %s", db_path)
        return db_path

    subjects = cfg.subjects()

    csv_files: list[Path] = []
    for subj in subjects:
        for sess_dir in (cfg.raw_path / subj).glob(f"{subj}_session_*"):
            csv_files.extend(sess_dir.glob("*.csv"))

    tmp_dir = cfg.out_dir / "_tmp_parquet"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    log.info("🔄 Convertendo %d CSV → Parquet (%d threads)…", len(csv_files), cfg.max_workers)
    t0 = perf_counter()
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        tasks = (
            executor.submit(_csv_to_parquet, csv, tmp_dir, cfg.sensors) for csv in csv_files
        )
        list(tqdm(as_completed(tasks), total=len(csv_files), desc="CSV→Parquet"))
    log.info("⏱️ Conversão concluída em %.1f s", perf_counter() - t0)

    log.info("✅ Criando banco DuckDB em %s", db_path)
    with duckdb.connect(str(db_path)) as con:
        for sensor in cfg.sensors:
            _ingest_sensor(con, sensor, tmp_dir)
            rows = con.execute(f"SELECT COUNT(*) FROM {sensor}").fetchone()[0]
            log.info("Tabela '%s' pronta (%d linhas)", sensor, rows)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    log.info("✅ Ingestão finalizada (%s)", db_path)
    return db_path

class IngestStep(PipelineStep):

    def execute(self, context: dict) -> dict:

        cfg_params = {
            "raw_path": context.get("raw_data_path", "data/raw"),
            "out_dir": context.get("experiment_dir", "data/duckdb"),
            "max_workers": context.get("max_workers", os.cpu_count()),
            "num_subjects": context.get("num_subjects"),
            "seed": context.get("seed", 42),
            "sensors": tuple(context.get("sensors", DEFAULT_SENSORS)),
        }

        cfg = IngestConfig(**cfg_params)
        duckdb_path = ingest(cfg)
        context["db_path"] = str(duckdb_path)
        return context

HEADER_MAPS = {
    'activity': [
        'activity_id',
        'subject_id',
        'session_number',
        'start_time',
        'end_time',
        'relative_start_time',
        'relative_end_time',
        'gesture_scenario',
        'task_id',
        'content_id',
    ],
    'accelerometer': [
        'time_sys',
        'time_rel',
        'activity_id',
        'acc_x',
        'acc_y',
        'acc_z',
        'phone_orientation',
    ],
    'gyroscope': [
        'time_sys',
        'time_rel',
        'activity_id',
        'gyro_x',
        'gyro_y',
        'gyro_z',
        'phone_orientation',
    ],
    'magnetometer': [
        'time_sys',
        'time_rel',
        'activity_id',
        'mag_x',
        'mag_y',
        'mag_z',
        'phone_orientation',
    ],
    'touchevent': [
        'time_sys',
        'time_rel',
        'activity_id',
        'pointer_count',
        'pointer_id',
        'action_id',
        'touch_x',
        'touch_y',
        'pressure',
        'contact_size',
        'phone_orientation',
    ],
    'keypressevent': [
        'time_sys',
        'time_rel',
        'press_type',
        'activity_id',
        'key_id',
        'phone_orientation',
    ],
    'onefingertouchevent': [
        'time_sys',
        'time_rel',
        'activity_id',
        'tap_id',
        'tap_type',
        'action_type',
        'touch_x',
        'touch_y',
        'pressure',
        'contact_size',
        'phone_orientation',
    ],
    'pinchevent': [
        'time_sys',
        'time_rel',
        'activity_id',
        'event_type',
        'pinch_id',
        'time_delta',
        'focus_x',
        'focus_y',
        'span',
        'span_x',
        'span_y',
        'scale_factor',
        'phone_orientation',
    ],
    'scrollevent': [
        'time_sys',
        'time_begin',
        'time_current',
        'activity_id',
        'scroll_id',
        'start_action_type',
        'start_x',
        'start_y',
        'start_pressure',
        'start_size',
        'current_action_type',
        'current_x',
        'current_y',
        'current_pressure',
        'current_size',
        'distance_x',
        'distance_y',
        'phone_orientation',
    ],
    'strokeevent': [
        'time_sys',
        'time_begin',
        'time_end',
        'activity_id',
        'start_action_type',
        'start_x',
        'start_y',
        'start_pressure',
        'start_size',
        'end_action_type',
        'end_x',
        'end_y',
        'end_pressure',
        'end_size',
        'speed_x',
        'speed_y',
        'phone_orientation',
    ],
}
ABSOLUTE_TIME_COLS = {
    'accelerometer': ['time_sys'],
    'gyroscope':     ['time_sys'],
    'magnetometer':  ['time_sys'],
    'touchevent':    ['time_sys'],
    'keypressevent': ['time_sys'],
    'onefingertouchevent':['time_sys'],
    'pinchevent':    ['time_sys'],
    'scrollevent':   ['time_sys', 'time_begin', 'time_current'],
    'strokeevent':   ['time_sys', 'time_begin', 'time_end'],
    'activity':      ['start_time', 'end_time'],
}
RELATIVE_TIME_COLS = {
    'accelerometer': ['time_rel'],
    'gyroscope':     ['time_rel'],
    'magnetometer':  ['time_rel'],
    'touchevent':    ['time_rel'],
    'keypressevent': ['time_rel'],
    'onefingertouchevent':['time_rel'],
    'pinchevent':    ['time_rel'],
    'activity':      ['relative_start_time', 'relative_end_time'],
}
