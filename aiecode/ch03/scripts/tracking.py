"""Tiny JSON tracker for Chapter 4.

Provides a minimal function to persist configuration and metrics for a run.
Avoids external dependencies and keeps records human-readable.
"""

from __future__ import annotations  # postpone typing evaluation

import json  # serialize run records
from dataclasses import asdict, dataclass  # simple structured records
from datetime import datetime  # timestamp for filenames
from pathlib import Path  # filesystem paths
from typing import Any, Dict  # type annotations


@dataclass
class RunRecord:
    seed: int  # RNG seed
    epochs: int  # number of training epochs
    lr: float  # learning rate
    final_loss: float  # last-epoch loss value


def save_run(out_dir: Path, record: RunRecord) -> Path:
    """Persist the run record as JSON and return the file path.

    The file name includes an ISO timestamp for uniqueness.
    """
    out_dir.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")  # timestamp for filename
    path = out_dir / f"run_{stamp}.json"  # output path
    with path.open("w", encoding="utf-8") as f:  # write JSON file
        json.dump(asdict(record), f, ensure_ascii=False, indent=2)  # save data
    return path  # return the written path for logging