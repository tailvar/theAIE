from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class Telemetry:
    """
    One place for:
      - utc timestamps
      - JSONL append
      - standard trace + event log shapes
    """

    def __init__(
        self,
        *,
        run_id: str,
        trace_path: Optional[Path] = None,
        runs_path: Optional[Path] = None,
    ) -> None:
        self.run_id = run_id
        self.trace_path = trace_path
        self.runs_path = runs_path

    @staticmethod
    def utc_iso() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    @staticmethod
    def write_jsonl(path: Path, obj: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def trace(self, direction: str, message: Dict[str, Any]) -> None:
        if self.trace_path is None:
            return
        self.write_jsonl(
            self.trace_path,
            {"run_id": self.run_id, "direction": direction, "ts_utc": self.utc_iso(), "message": message},
        )

    def event(self, step: int, budgets: Dict[str, Any], event: str, **fields: Any) -> None:
        if self.runs_path is None:
            return
        payload: Dict[str, Any] = {
            "run_id": self.run_id,
            "step": step,
            "event": event,
            "ts_utc": self.utc_iso(),
            "budgets": budgets,
        }
        payload.update(fields)
        self.write_jsonl(self.runs_path, payload)


