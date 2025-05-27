from __future__ import annotations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import shutil
import sys
import os

PROJ_ROOT = Path(__file__).parents[2] 
sys.path.insert(0, str(PROJ_ROOT / "src"))

from pipeline.pipeline_step import PipelineStep
import duckdb, json, random, logging
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import pandas as pd
from tqdm.auto import tqdm
from utils.utils import get_db_path


class IngestStep(PipelineStep):
    def execute(self, context: dict) -> dict:  # noqa: D401
        log = logging.getLogger("ingest")
        log.info("📥 Iniciando ingestão de dados…")

        raw_dir = Path(context["IngestStep"]["raw_path"])
        duckdb_dir = Path(context["IngestStep"]["duckdb_dir"])
        duckdb_dir.mkdir(parents=True, exist_ok=True)
        max_workers = context.get("max_workers", os.cpu_count())

        sensors = context.get("sensors", [])
        sensors.append("activity")

        all_subjects = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())
        if not all_subjects:
            log.error("❌ Nenhum dado encontrado em %s", raw_dir)
            return context

        subjects = all_subjects
        db_name = "hmog_all.duckdb"
        db_path = duckdb_dir / db_name
        context["db_path"] = str(db_path)

        if db_path.exists():
            log.info("✅ DuckDB já existe pulando ingestão → %s", db_path)
            return context

        csv_files: list[Path] = []
        for subj in subjects:
            subj_dir = raw_dir / subj
            for sess_dir in subj_dir.glob(f"{subj}_session_*"):
                csv_files.extend(sess_dir.glob("*.csv"))

        parquet_dir = duckdb_dir / "_tmp_parquet"
        parquet_dir.mkdir(parents=True, exist_ok=True)

        def csv_to_parquet(csv_path: Path) -> Path:
            table = csv_path.stem.lower()
            if table not in sensors:
                return None

            df = pd.read_csv(csv_path, header=None, names=HEADER_MAPS[table])

            for col in df.columns:
                if col in ABSOLUTE_TIME_COLS.get(table, []):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                    df = df.rename(columns={col: f"{col}_ms"})
                elif col in RELATIVE_TIME_COLS.get(table, []):
                    rel_ns = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                    df[f'{col}_ns'] = rel_ns
                    df[f"{col}_ms"] = (rel_ns // 1_000_000).astype("Int64")

            if table != "activity":
                subj_id = csv_path.parents[1].name  
                session_num = int(csv_path.parents[0].name.split("_session_")[1])
                df["subject_id"] = subj_id
                df["session_number"] = session_num
            
            pq_path = parquet_dir / (csv_path.stem.lower() + "__" + csv_path.parents[1].name + "__" + csv_path.parents[0].name.split("_session_")[1] + ".parquet")

            for c, dt in df.dtypes.items():
                if dt.name in ("Int64", "UInt64"):
                    df[c] = df[c].astype("int64")

            df.to_parquet(pq_path, index=False)
            return pq_path

        log.info("Convertendo %d CSVs para Parquet…", len(csv_files))
        t0 = perf_counter()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(tqdm(ex.map(csv_to_parquet, csv_files), total=len(csv_files), desc="CSV→Parquet"))
        log.info("⏱️ Conversão concluída em %.1f s", perf_counter() - t0)

        log.info("📦 Criando banco DuckDB %s", db_path)
        con = duckdb.connect(str(db_path))
        for sensor in sensors:
            pattern = str(parquet_dir / f"{sensor}__*.parquet")
            con.execute(f"CREATE TABLE {sensor} AS SELECT * FROM read_parquet('{pattern}')")
            log.info("Tabela '%s' pronta (%d linhas)", sensor, con.execute(f"SELECT COUNT(*) FROM {sensor}").fetchone()[0])
        con.close()

        log.info("🎉 DuckDB pronto: %s (%.1f s no total)", db_path, perf_counter() - t0)

        # 7) Limpa diretório temporário
        shutil.rmtree(parquet_dir, ignore_errors=True)
        return context


# ---------------------------------------------------------------------------
# Helper (exemplo) para descobrir IDs de sujeitos se precisar fora da classe
# ---------------------------------------------------------------------------

def discover_subject_ids(raw_dir: Path) -> list[str]:
    """Retorna todos os diretórios <subject_id> presentes em `raw_dir`."""
    return sorted([p.name for p in raw_dir.iterdir() if p.is_dir()])



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
    'onefingertouch': [
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
    'onefingertouch':['time_sys'],
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
    'onefingertouch':['time_rel'],
    'pinchevent':    ['time_rel'],
    'activity':      ['relative_start_time', 'relative_end_time'],
}