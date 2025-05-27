from __future__ import annotations
from pathlib import Path
import os, logging
from time import perf_counter

import duckdb
import pandas as pd
from tqdm.auto import tqdm

from pipeline.pipeline_step import PipelineStep
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

class SliceStep(PipelineStep):
    def execute(self, context: dict) -> dict:
        log = logging.getLogger("slice")
        log.info("🔪 Iniciando slicing...")

        cfg = context["SliceStep"]
        window_ms = cfg["window_size_ms"]
        overlap_pct = cfg["overlap_pct"]
        base_out = Path(cfg["out_dir"])

        db_path = Path(context.get("db_path"))
        if not db_path.exists():
            log.error(f"❌ DuckDB não encontrado: {db_path}")
            return context

        out_dir = base_out / db_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        con = duckdb.connect(str(db_path), read_only=True)
        t0 = perf_counter()

        sensors = context.get("sensors", [])
        for sensor in sensors:
            if sensor == "activity":
                continue
            log.info(f"🔍 Processando sensor: {sensor}...")
            time_col = TS_COL_MAP.get(sensor, "time_rel_ms")
            out_path = out_dir / f"{sensor}_ws_{window_ms}_ov_{int(overlap_pct*100)}.parquet"

            if out_path.exists():
                log.info(f"✨ Janelas já existem para {sensor} → {out_path}")
                continue

            pairs = con.execute(
                f"SELECT DISTINCT subject_id, session_number "
                f"  FROM {sensor} "
                " ORDER BY subject_id, session_number"
            ).fetchall()

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

            if rows:
                log.info(f"📊 {sensor}: extraídas {len(rows)} janelas, salvando em {out_path}")
                pd.DataFrame(rows).to_parquet(out_path, index=False)
            else:
                log.warning(f"⚠️ Nenhuma janela gerada para sensor {sensor}.")

        con.close()
        elapsed = perf_counter() - t0
        log.info(f"✔ Slicing concluído em {elapsed:.1f} s")

        context["windows_dir"] = str(out_dir)
        return context


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
