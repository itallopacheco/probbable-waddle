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

import pandas as pd, numpy as np
from tqdm import tqdm
from utils.utils import get_db_path, get_window_path


class FeatureExtractionStep(PipelineStep):
    def execute(self, context: dict) -> dict:
        log = logging.getLogger("feature")
        log.info("🔍 Iniciando extração de features...")
        windows_dir = get_window_path(context)
        base_out = Path(context["FeatureExtractionStep"]["out_dir"]) 
        out_dir = base_out / windows_dir.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        sensors = context.get("sensors", [])
        
        db_path = get_db_path(context)
        if not db_path.exists():
            log.error(f"❌ Banco de dados não encontrado: {db_path}")
            return context

        feature_list = []  

        for sensor in sensors:
            log.info("🔍 Processando janelas do sensor: %s", sensor)
            window_path = windows_dir / f"{sensor}_ws_{context['SliceStep']['window_size_ms']}_ov_{context['SliceStep']['overlap_pct']}.parquet"
            if not window_path.exists():
                log.error(f"❌ Janelas não encontradas: {window_path}")
                continue
            
            out_path = Path(out_dir, f"{sensor}_features.parquet")
            if out_path.exists():
                log.info(f"✨ Features já extraídas para {sensor} → {out_path}")
                continue
            
            VALUE_COLS = {
                "accelerometer": ["acc_x", "acc_y", "acc_z"],
                "gyroscope":     ["gyro_x", "gyro_y", "gyro_z"]
            }[sensor]

            con = duckdb.connect(db_path, read_only=True)
            windows = pd.read_parquet(window_path)
            rows = []

            for _, row in tqdm(windows.iterrows(), total=len(windows), desc="Janelas"):
                q = f"""
                    SELECT {', '.join(VALUE_COLS)}
                    FROM {sensor}
                    WHERE subject_id   = '{row.subject_id}'
                    AND session_number = {row.session_number}
                    AND time_rel_ms BETWEEN {row.start_ms} AND {row.end_ms-1}
                    ORDER BY time_rel_ms
                """
                sig = con.execute(q).fetchdf()[VALUE_COLS].to_numpy(dtype=np.float32)
                if len(sig) < 1:        
                    continue
                feats = window_stats(sig)
                rows.append(np.hstack([row[["subject_id","session_number",
                                            "start_ms","end_ms"]].values, feats]))
    
            if rows:
                cols_meta = ["subject_id","session_number","start_ms","end_ms"]
                cols_feat = [f"{stat}_{ax}"
                            for stat in ("mean","std","rms","max","min")
                            for ax in ("x","y","z")]
                df_out = pd.DataFrame(rows, columns=cols_meta + cols_feat)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                df_out.to_parquet(out_path, index=False)
                print(f"✔  {len(df_out)} janelas ➜ {out_path}")
            else:
                print("Nenhuma janela com dados.")

def window_stats(chunk: np.ndarray) -> np.ndarray:
    mean = chunk.mean(0); std = chunk.std(0)
    rms = np.sqrt((chunk**2).mean(0))
    vmax, vmin = chunk.max(0), chunk.min(0)
    return np.hstack([mean, std, rms, vmax, vmin])