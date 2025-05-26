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


class SvmStep(PipelineStep):
    def execute(self, context: dict) -> dict:
        log = logging.getLogger("svm")
        log.info("🤖 Iniciando SVM ...")


