"""
Background GPU utilization/memory sampler for PRISM training diagnostics.

Samples `nvidia-smi` on a fixed interval in a background thread -- no extra
Python dependency beyond what ships with any NVIDIA driver install, unlike
pynvml which may not be present on the training machine. Two consumption
modes:
    snapshot()        — most recent sample, for a quick check.
    average_window(t)  — mean of all samples with ts >= t, for embedding
                         "GPU utilization during this update" into the
                         trainer's per-update metrics line.

Every raw sample is also appended to `log_path` (JSONL) as it arrives,
independent of what the trainer chooses to query -- a full-resolution trace
for post-hoc analysis (e.g. spotting a stall between rollout collection and
the PPO update that a per-update average alone would smear out).

If `nvidia-smi` is unavailable (no GPU, no driver, CPU-only smoke test),
GPUMonitor disables itself after one failed probe and logs a single
WARNING -- it never raises, since GPU logging must not be able to break
training.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_NVIDIA_SMI_FIELDS = (
    "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
)
_FIELD_NAMES = ("gpu_util_pct", "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_temp_c", "gpu_power_w")


def _query_nvidia_smi(device_index: int) -> Optional[Dict[str, float]]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={_NVIDIA_SMI_FIELDS}",
                "--format=csv,noheader,nounits",
                f"--id={device_index}",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=True,
        )
    except Exception:
        return None
    row = next(csv.reader(io.StringIO(out.stdout.strip())), None)
    if not row or len(row) < len(_FIELD_NAMES):
        return None
    try:
        return {name: float(v) for name, v in zip(_FIELD_NAMES, row)}
    except ValueError:
        return None


class GPUMonitor:
    """Background nvidia-smi sampler. See module docstring."""

    def __init__(
        self,
        device_index: int = 0,
        interval_s: float = 2.0,
        log_path: Optional[Path] = None,
        window_size: int = 20_000,
    ) -> None:
        self._device_index = device_index
        self._interval_s = interval_s
        self._log_path = Path(log_path) if log_path else None
        self._samples: Deque[Tuple[float, Dict[str, float]]] = deque(maxlen=window_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.available = _query_nvidia_smi(device_index) is not None
        if not self.available:
            logger.warning(
                "GPUMonitor: `nvidia-smi` unavailable or failed on first probe "
                "(device_index=%d) -- GPU utilization logging disabled for this run.",
                device_index,
            )

    def start(self) -> None:
        if not self.available or self._thread is not None:
            return
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self._interval_s * 2)
        self._thread = None

    def _run(self) -> None:
        f = open(self._log_path, "a") if self._log_path is not None else None
        try:
            while not self._stop_event.is_set():
                ts = time.time()
                sample = _query_nvidia_smi(self._device_index)
                if sample is not None:
                    self._samples.append((ts, sample))
                    if f is not None:
                        f.write(json.dumps({"ts": ts, **sample}) + "\n")
                        f.flush()
                self._stop_event.wait(self._interval_s)
        finally:
            if f is not None:
                f.close()

    def snapshot(self) -> Optional[Dict[str, float]]:
        if not self._samples:
            return None
        return dict(self._samples[-1][1])

    def average_window(self, since_ts: float) -> Optional[Dict[str, float]]:
        """Mean of all samples with ts >= since_ts. Falls back to snapshot()
        if none fall in range (e.g. window shorter than the sample interval)."""
        window = [s for ts, s in self._samples if ts >= since_ts]
        if not window:
            return self.snapshot()
        return {
            name: sum(w[name] for w in window) / len(window)
            for name in _FIELD_NAMES
        }
