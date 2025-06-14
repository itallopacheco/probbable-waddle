from __future__ import annotations

import importlib
import argparse, json, logging, sys
from pathlib import Path
from time import perf_counter

PROJ_ROOT = Path(__file__).parents[2] 
sys.path.insert(0, str(PROJ_ROOT / "src"))

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table


logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler(rich_tracebacks=True)],
    format="%(message)s",
)
log = logging.getLogger("pipeline")

# cfg_path = Path(args.config).resolve()
# if not cfg_path.exists():
#     log.error(f"Config não encontrada: {cfg_path}")
#     sys.exit(1)

# CFG = json.loads(cfg_path.read_text())
# log.info(f"⚙️  Config carregada de {cfg_path}")

config = {
    "name": "experiment-03",
    "seed": 42,
    "raw_data_path": "data/raw",
    "output_dir": "results",
    #"num_subjects": 2,
    "num_subjects_train": 1,
    "sensors": ["accelerometer", "gyroscope", "multimodal"],
    "window_size_ms": 1000,
    "overlap_pct": 0.5,
    "max_samples": 10_000,
    
    "scenarios": [
        {"name": "acc", "sensors": ["accelerometer"]},
        {"name": "gyro", "sensors": ["gyroscope"]},
    ],

    "pipeline_steps": [
        # "steps.ingest_step.IngestStep",
        # "steps.slice_step.SliceStep",
        # "steps.features_step.FeatureExtractionStep",
        "steps.train_step.TrainStep",
    ],

    "IngestStep": {
        "raw_path": "data/raw",
        "duckdb_dir": "data/duckdb"
    },
    "SliceStep": {
        "window_size_ms": 1000,
        "overlap_pct": 0.5,
        "out_dir": "data/windows",
    },
    "FeatureExtractionStep": {
        "out_dir": "data/features"
    },
    "TrainVerificationStep": {
        "out_dir": "data/results/verification",
        "kernel": "linear",
        "C": 1.0,
    }
}
context = config
experiment_name = config["name"]
experiment_dir = Path(config["output_dir"]) / experiment_name
experiment_dir.mkdir(parents=True, exist_ok=True)
context["experiment_dir"] = experiment_dir

for class_path in config["pipeline_steps"]:
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    step_cls = getattr(module, class_name)
    step = step_cls()
    context = step.execute(context)







# --------------------- resumo -------------------------------
# table = Table(title="Resumo do experimento")
# table.add_column("Etapa")
# table.add_column("Duração (s)", justify="right")
# for s in stages_to_run:
#     table.add_row(s, f"{durations[s]:.1f}")
# console.print(table)
# console.print("[bold green]✅  Pipeline finalizado[/]")
