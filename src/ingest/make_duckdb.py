from __future__ import annotations

import duckdb, json, random, logging
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

# ------------------- CONSTS (mesmo HEADER_MAPS que você já tem) ------------------
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
# -------------------------------------------------------------------------------


def _select_subjects(raw_root: Path, n: int, seed: int) -> List[str]:
    subjects = [p.name for p in raw_root.iterdir() if p.is_dir()]
    if n > len(subjects):
        raise ValueError(f"num_subjects={n} > {len(subjects)} disponíveis")
    random.seed(seed)
    return sorted(random.sample(subjects, n))


def _discover_sessions(raw_root: Path, subject: str):
    subj_dir = raw_root / subject
    for sess_dir in subj_dir.glob(f"{subject}_session_*"):
        sess_num = int(sess_dir.name.split("_session_")[1])
        yield sess_num, sess_dir


def _ingest_csv(con: duckdb.DuckDBPyConnection, csv_path: Path, subject: str, session: int):
    table = csv_path.stem.lower()
    if table in HEADER_MAPS:
        df = pd.read_csv(csv_path, header=None, names=HEADER_MAPS[table])
        # --- tempos absolutos
        for col in ABSOLUTE_TIME_COLS.get(table, []):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            df = df.rename(columns={col: f"{col}_ms"})
        # --- relativos
        for col in RELATIVE_TIME_COLS.get(table, []):
            rel_ns = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            df[f'{col}_ns'] = rel_ns
            df[f"{col}_ms"] = (rel_ns // 1_000_000).astype("Int64")
    else:
        df = pd.read_csv(csv_path)

    if table != "activity":
        df["subject_id"] = subject
        df["session_number"] = session

    con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM df LIMIT 0")
    con.execute(f"INSERT INTO {table} SELECT * FROM df")


# --------------------- Função principal a ser chamada pelo pipeline --------------
def run(cfg_sub: dict, *, logger=None) -> Path | None:
    """
    Executa ingest conforme cfg_sub['ingest'].
    Retorna o caminho do DuckDB gerado.
    """
    log = logger or logging.getLogger("ingest")
    raw_root   = Path(cfg_sub["raw_path"])
    out_dir    = Path(cfg_sub["duckdb_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sensors_to_keep = set(cfg_sub.get("sensors", []))

    # hash simples do seed+num_subjects para não sobrescrever DBs de outras configs
    tag = f"s{cfg_sub['max_subjects']}_seed{cfg_sub['seed']}"
    db_path = out_dir / f"hmog_{tag}.duckdb"
    if db_path.exists():
        log.info(f"✨ DuckDB já existe – pulando ingest → {db_path}")
        return db_path

    # ---------- seleção de usuários ----------
    subjects = _select_subjects(raw_root, cfg_sub["max_subjects"], cfg_sub["seed"])
    log.info(f"Selecionados {len(subjects)} usuários")

    # ---------------- ingest ------------------
    t0 = perf_counter()
    con = duckdb.connect(str(db_path))
    for subj in tqdm(subjects, desc="Ingest users", ncols=80):
        for sess, sess_dir in _discover_sessions(raw_root, subj):
            for csv_file in sess_dir.glob("*.csv"):
                stem = csv_file.stem.lower()
                if sensors_to_keep and stem not in sensors_to_keep and stem != "activity":
                    continue
                _ingest_csv(con, csv_file, subj, sess)
    con.close()
    log.info(f"✔ Ingest concluído em {perf_counter()-t0:.1f} s")
    return db_path
