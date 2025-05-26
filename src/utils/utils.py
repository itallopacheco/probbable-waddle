from __future__ import annotations

from pathlib import Path


def get_db_path(context: dict) -> Path:
    out_dir = Path(context["IngestStep"]["duckdb_dir"])
    num_subjects = context["IngestStep"].get("max_subjects")
    seed = context.get("seed")

    duckdb_tag = f"s{num_subjects}_seed_{seed}"
    db_path = out_dir / f"hmog_{duckdb_tag}.duckdb"
    return db_path



def get_window_path(context: dict) -> Path:
    base_out = Path(context["SliceStep"]["out_dir"])
    db_path = get_db_path(context)
    folder_name = db_path.stem
    out_dir = base_out / folder_name

    return out_dir