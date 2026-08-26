"""
Verbose per-update training metrics logger.

Writes one JSON line per PPO update to a .jsonl file under the run's output
directory -- every scalar already computed in DPMORLTrainer.train() plus
timing and GPU utilization, so a full training run can be analyzed offline
(pandas.read_json(path, lines=True)) without depending on terminal
scrollback or a wandb dashboard.

The first line written is a "config" record (record_type="config")
capturing the run's static hyperparameters and ablation toggles, so the
file is self-describing -- you can tell exactly what produced it without
also having to dig up which YAML/branch was used.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class MetricsLogger:
    def __init__(self, log_path: Path) -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self._log_path, "a")
        logger.info(f"MetricsLogger: writing to {self._log_path}")

    def log_config(self, config: Dict[str, Any]) -> None:
        self._write({"record_type": "config", **config})

    def log_update(self, record: Dict[str, Any]) -> None:
        self._write({"record_type": "update", **record})

    def _write(self, record: Dict[str, Any]) -> None:
        record.setdefault("wall_time", time.time())
        self._f.write(json.dumps(record, default=float) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()
