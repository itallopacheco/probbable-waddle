from __future__ import annotations
from pathlib import Path
import sys

PROJ_ROOT = Path(__file__).parents[2] 
sys.path.insert(0, str(PROJ_ROOT / "src"))

from pipeline.pipeline_step import PipelineStep
import duckdb, json, random, logging
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from utils.utils import get_db_path


class SliceStep(PipelineStep):
    def execute(self, context: dict) -> dict:
        log = logging.getLogger("slice")
        log.info("🔪 Iniciando slicing...")

        window_ms = context["SliceStep"]["window_size_ms"]
        overlap_pct = context["SliceStep"]["overlap_pct"]
        base_out = Path(context["SliceStep"]["out_dir"])
        db_path = get_db_path(context)
        folder_name = db_path.stem
        out_dir = base_out / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        sensors = context.get("sensors", [])

        if not db_path.exists():
            log.error(f"❌ DuckDB não encontrado: {db_path}")
            return context
        
        con = duckdb.connect(str(db_path), read_only=True)
        t0 = perf_counter()
        for sensor in sensors:
            log.info(f"🔍 Processando sensor: {sensor}...")
            time_col = TS_COL_MAP.get(sensor, "time_rel_ms")
            out_path = Path(out_dir, f"{sensor}_ws_{window_ms}_ov_{overlap_pct}.parquet")
            
            if out_path.exists():
                log.info(f"✨ Janelas já existem para {sensor} → {out_path}")
                continue 

            pairs = con.execute(f"SELECT DISTINCT subject_id, session_number FROM {sensor} ORDER BY subject_id, session_number").fetchall()

            rows: list[dict] = []
            for subj, sess in tqdm(pairs, desc=f"Processando sessões do {sensor}", ncols=80):
                df = con.execute(
                    f"SELECT {time_col} FROM {sensor} WHERE subject_id=? AND session_number=? ORDER BY {time_col}",
                    [subj, sess],
                ).fetchdf()

                if df.empty:
                    continue

                for start, end, win in sliding_window(df, time_col, window_ms, overlap_pct):
                    rows.append({
                        "sensor": sensor,
                        "subject_id": subj,
                        "session_number": sess,
                        "start_ms": start,
                        "end_ms": end,
                        "num_samples": len(win),
                    })

            if rows:
                log.info(f"📊 {sensor}: {len(rows)} janelas")
                pd.DataFrame(rows).to_parquet(out_path, index=False)
        con.close()
        log.info(f"✔ Slicing concluído em {perf_counter() - t0:.1f} s")




            


def sliding_window(
    df: pd.DataFrame,
    ts_col: str,
    window_size_ms: int | None = None,
    overlap_pct: float | None = None,
) -> Generator[Tuple[int, int, pd.DataFrame], None, None]:
    step = int(window_size_ms * (1.0 - overlap_pct))

    start = int(df[ts_col].iloc[0])
    end_max = int(df[ts_col].iloc[-1])

    while start + window_size_ms <= end_max:
        end = start + window_size_ms
        mask = (df[ts_col] >= start) & (df[ts_col] < end)
        yield start, end, df.loc[mask].copy()
        start += step



TS_COL_MAP = {
    "accelerometer": "time_rel_ms",
    "gyroscope": "time_rel_ms",
    "magnetometer": "time_rel_ms",
    "touchevent": "time_rel_ms",
    "onefingertouch": "time_rel_ms",
    "pinchevent": "time_rel_ms",
    "scrollevent": "time_current_ms",
    "strokeevent": "time_begin_ms",
}