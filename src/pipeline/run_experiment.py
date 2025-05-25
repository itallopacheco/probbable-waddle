from __future__ import annotations

import argparse, json, logging, sys
from pathlib import Path
from time import perf_counter

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# --- garante que diretório-raiz esteja no PYTHONPATH -------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # <root>/src/pipeline/..
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.timer import step_timer                    
from src.ingest.make_duckdb import run as run_ingest      

# --------------------------- CLI -------------------------------
parser = argparse.ArgumentParser(description="Pipeline HMOG")
parser.add_argument(
    "--config", default="settings/settings.json",
    help="Arquivo JSON com a configuração do experimento"
)
parser.add_argument(
    "--stages", default="ingest",
    help="Etapas a executar (ex.: ingest,windows)"
)
args = parser.parse_args()

# ------------------------- logging -----------------------------
logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler(rich_tracebacks=True)],
    format="%(message)s",
)
log = logging.getLogger("pipeline")

# ---------------------- carrega config -------------------------
cfg_path = Path(args.config).resolve()
if not cfg_path.exists():
    log.error(f"Config não encontrada: {cfg_path}")
    sys.exit(1)

CFG = json.loads(cfg_path.read_text())
log.info(f"⚙️  Config carregada de {cfg_path}")

# ------------- mapeia etapas → função --------------------------
STAGE_FUNCS = {
    "ingest": run_ingest,
    # futuros estágios:
    # "windows": run_windows,
    # "features": run_features,
    # "models": run_models,
}

stages_to_run = [s.strip() for s in args.stages.split(",") if s.strip()]
invalid = [s for s in stages_to_run if s not in STAGE_FUNCS]
if invalid:
    log.error(f"Etapa(s) desconhecida(s): {', '.join(invalid)}")
    sys.exit(1)

# --------------------- execução -------------------------------
durations: dict[str, float] = {}
console = Console()

for stage in stages_to_run:
    func = STAGE_FUNCS[stage]
    subcfg = CFG.get(stage, {})
    subcfg["seed"] = CFG.get("seed", 42)

    console.rule(f"[bold cyan]{stage.upper()}[/]")

    with step_timer(stage, logger=log) as timer:
        artifact = func(subcfg, logger=log)
        durations[stage] = timer()   # timer() devolve segundos

    if artifact:
        log.info(f"   └─ artefato: {artifact}")

# --------------------- resumo -------------------------------
table = Table(title="Resumo do experimento")
table.add_column("Etapa")
table.add_column("Duração (s)", justify="right")
for s in stages_to_run:
    table.add_row(s, f"{durations[s]:.1f}")
console.print(table)
console.print("[bold green]✅  Pipeline finalizado[/]")
