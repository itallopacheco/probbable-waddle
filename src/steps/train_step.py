from __future__ import annotations
from pathlib import Path
import sys
import time

PROJ_ROOT = Path(__file__).parents[2] 
sys.path.insert(0, str(PROJ_ROOT / "src"))

from steps.pipeline_step import PipelineStep
import duckdb, json, random, logging
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import pandas as pd, numpy as np
from tqdm import tqdm
from utils.utils import get_db_path, get_window_path, get_features_path, get_random_user
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

LOGGER = logging.getLogger("verification_step")

drop_cols = {"accelerometer": ['acc_x_skewness', 'acc_x_kurtosis','acc_y_skewness', 'acc_y_kurtosis',
                                'acc_z_skewness', 'acc_z_kurtosis','magnitude_skewness', 'magnitude_kurtosis'],
            "gyroscope": ['gyro_x_skewness', 'gyro_x_kurtosis','gyro_y_skewness', 'gyro_y_kurtosis',
                                'gyro_z_skewness', 'gyro_z_kurtosis','magnitude_skewness', 'magnitude_kurtosis']
        }
def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    tar = tpr[idx]
    return float(eer), float(tar)

def _tar_at_far(fpr: np.ndarray, tpr: np.ndarray, target_far: float) -> float:
    if target_far <= fpr[0]:
        return tpr[0]

    if target_far >= fpr[-1]:
        return tpr[-1]
    idx = np.searchsorted(fpr, target_far) - 1

    fpr_low, fpr_high = fpr[idx], fpr[idx + 1]
    tpr_low, tpr_high = tpr[idx], tpr[idx + 1]
    alpha = (target_far - fpr_low) / (fpr_high - fpr_low + 1e-12)
    return tpr_low + alpha * (tpr_high - tpr_low)

def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr

    idx_eer = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2

    metrics = {
        "EER": eer,
        "TAR_EER": tpr[idx_eer],
        "FAR_EER": fpr[idx_eer],
        "FRR_EER": fnr[idx_eer],
        "TAR_FAR_1pct": _tar_at_far(fpr, tpr, 0.01),
        "TAR_FAR_0.1pct": _tar_at_far(fpr, tpr, 0.001),
        "AUC": roc_auc_score(y_true, scores),
    }

    return metrics


def feature_cols(df: pd.DataFrame) -> List[str]:
    meta = {
        "sensor",
        "subject_id",
        "session_number",
        "start_ms",
        "end_ms",
        "label",
        "window_id",
    }
    return [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]

DEFAULT_SENSORS = (
    "accelerometer",
    "gyroscope"
)

def get_pipeline(cfg) -> Pipeline:
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="linear", class_weight="balanced", probability=False, random_state=cfg.seed)),
    ])


class TrainConfig:
    def __init__(self, db_path: str | Path, features_dir: str | Path, out_dir: str | Path,
                 seed: int = 42,num_subjects: int = None, sensors: List[str] = DEFAULT_SENSORS,
                 c_grid: List[float] = [0.1, 1, 10], max_samples: int = 1000):
        
        self.db_path = Path(db_path)
        self.out_dir = Path(out_dir / "train")
        self.seed = seed
        self.num_subjects = num_subjects
        self.sensors = sensors
        self.c_grid = c_grid
        self.max_samples = max_samples
        self.features_dir = Path(features_dir)

def run_training_for_subject(subject: str, features: pd.DataFrame, sensor: str, cfg: TrainConfig) -> Dict:
    user_df = features.copy()
    user_df['label'] = (user_df['subject_id'] == subject).astype(int)

    sessions = user_df.loc[user_df['label'] == 1, 'session_number'].unique()
    rng = np.random.RandomState(cfg.seed)

    results = []
    for i, test_session in enumerate(sessions, start=1):
        ## SPLIT LOSO
        train_mask = user_df['session_number'] != test_session
        test_mask = user_df['session_number'] == test_session

        X_train_full = user_df.loc[train_mask, feature_cols(user_df)]
        y_train_full = user_df.loc[train_mask, 'label']

        ## Cria subamostragem de impostores 
        pos_idx = y_train_full[y_train_full == 1].index
        neg_idx = y_train_full[y_train_full == 0].index
        
        n_pos = len(pos_idx)
        n_neg_sample = min(len(neg_idx), n_pos * 2) # 2x impostores
        neg_sample_idx = np.random.RandomState(cfg.seed).choice(neg_idx, size=n_neg_sample, replace=False) 
        selected_idx = np.concatenate([pos_idx, neg_sample_idx])

        
        ## Limitar o número de amostras
        ## pegar leve com o pobi do notebook c:
        if len(selected_idx) > cfg.max_samples:
            selected_idx = rng.choice(selected_idx, size=cfg.max_samples, replace=False)
        
        X_train = X_train_full.loc[selected_idx]
        y_train = y_train_full.loc[selected_idx]

        X_test = user_df.loc[test_mask, feature_cols(user_df)]
        y_test = user_df.loc[test_mask, 'label']

        # removendo linhas com NaN (algumas sessoes estavam me dando problemas)
        mask_test = X_test.notna().all(axis=1)
        X_test, y_test = X_test[mask_test], y_test[mask_test]

        print(f"Usuário {subject} - Sessão {test_session} ({i}/{len(sessions)})")
        print(f" Treino original: {X_train_full.shape[0]} (pos={sum(y_train_full)}, neg={len(y_train_full)-sum(y_train_full)})")
        print(f" Treino subamostrado limitado a {cfg.max_samples} amostras: (pos={sum(y_train)}, neg={len(y_train)-sum(y_train)})")
        #print(f" Teste: {X_test.shape[0]} (pos={sum(y_test)}, neg={len(y_test)-sum(y_test)})")

        ## Treinamento    
        cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
        pipe = get_pipeline(cfg)
        param_grid = {
            "svc__C": cfg.c_grid,
        }

        grid = GridSearchCV(
            pipe,
            param_grid,
            cv=cv_inner,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        t0_train = time.perf_counter()
        grid.fit(X_train, y_train)
        t1_train = time.perf_counter()
        train_time_sec = t1_train - t0_train
        print(f" Treinamento concluído em {train_time_sec:.2f} segundos")

        ## Inferencia
        t0_infer = time.perf_counter()
        y_score_test = grid.decision_function(X_test)
        t1_infer = time.perf_counter()
        latency_ms = ((t1_infer - t0_infer) / len(y_test)) * 1000
        print(f" Inferência concluída em {latency_ms:.2f} ms por amostra")

        ## Metricas
        metrics_test = compute_metrics(y_test.values, y_score_test)
        
        ## Registrando resultados
        results.append({
            "user_id": subject,
            "sensor": sensor,
            "session": int(test_session),
            ## teste
            **{f"{k}_test": v for k, v in metrics_test.items()},
            "best_C": float(grid.best_params_["svc__C"]),
            "n_support_vec": int(grid.best_estimator_.named_steps["svc"].n_support_.sum()),
            "train_time_sec": train_time_sec,
            "latency_ms": latency_ms,
            "n_pos_train": int((y_train == 1).sum()),
            "n_neg_train": int((y_train == 0).sum()),
            "n_pos_test": int((y_test == 1).sum()),
            "n_neg_test": int((y_test == 0).sum()),
        })
    return pd.DataFrame(results)



def train_model(cfg: TrainConfig):
    log = logging.getLogger("train")
    log.info("🔍 Iniciando etapa de treino...")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    db_path = cfg.db_path
    if not db_path.exists():
        log.error(f"❌ Banco de dados não encontrado: {db_path}")
        return
    
    features_dir = cfg.features_dir
    if not features_dir.exists():
        log.error(f"❌ Diretório de features não encontrado: {features_dir}")
        return

    rng = np.random.RandomState(cfg.seed)    

    for sensor in cfg.sensors:
        if sensor == "activity":
            continue

        features_file = features_dir / f"{sensor}_features.parquet"
        if not features_file.exists():
            log.error(f"❌ Arquivo de features não encontrado: {features_file}")
            continue
        log.info(f"🔍 Processando features do {sensor}...")
        out_file = cfg.out_dir / f"{sensor}_train_results.parquet"
        if out_file.exists():
            log.info(f"✅ Resultados já existem: {out_file}. Pulando...")
            continue

        df = pd.read_parquet(features_file)
        all_subjects = df.subject_id.unique()
        subjects = rng.choice(all_subjects, size=cfg.num_subjects, replace=False)
        results = []
        for subject in subjects:
            log.info(f"🔍 Treinando modelo para o usuário {subject} no sensor {sensor}...")
            result = run_training_for_subject(subject, df, sensor, cfg)
            results.append(result)
        
        results_df = pd.concat(results, ignore_index=True)
        results_df.to_parquet(out_file, index=False)
        log.info(f"✅ Resultados salvos em {out_file} ({len(results_df)} linhas).")
        results_df = pd.concat(results, ignore_index=True)

class TrainStep(PipelineStep):

    def execute(self, context: dict) -> dict:
        experiment_name = context.get("name", "experiment")
        cfg_params = {
            "db_path": context.get("db_path", f"results/{experiment_name}/hmog_all.duckdb"),
            "features_dir": context.get("features_dir", f"results/{experiment_name}/features"),
            "out_dir": context.get("experiment_dir", f"data/results/{experiment_name}/verification"),
            "seed": context.get("seed", 42),
            "num_subjects": context.get("num_subjects_train"),
            "sensors": context.get("sensors", DEFAULT_SENSORS),
            "c_grid": context.get("c_grid", [0.1, 1, 10]),
            "max_samples": context.get("max_samples", 1000),
        }

        cfg = TrainConfig(**cfg_params)
        train_model(cfg)
        return None
