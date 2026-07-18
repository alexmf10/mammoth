#!/usr/bin/env python3
"""Run Mammoth while recording process-local PyTorch CUDA peak memory.

``PEAK_FILE`` must name a writable JSON output file.  The report is written by
an ``atexit`` callback after ``main.py`` finishes (also after an ordinary
Python exception).  As with every ``atexit``-based mechanism, no report can be
written if the process is killed with SIGKILL or exits through ``os._exit``.
"""

from __future__ import annotations

import atexit
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
from typing import Any


MIB = 1024**2
REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = REPO_ROOT / "main.py"


def _utc_now() -> str:
    """Return an unambiguous, machine-readable UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _git_sha() -> str:
    """Return the checked-out commit, or an explicit sentinel on failure."""

    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _prepare_peak_file() -> Path:
    """Resolve ``PEAK_FILE`` and prove that its parent is writable."""

    raw_path = os.environ.get("PEAK_FILE")
    if not raw_path:
        raise RuntimeError("PEAK_FILE is required and must point to a writable JSON file")

    peak_file = Path(raw_path).expanduser().resolve()
    peak_file.parent.mkdir(parents=True, exist_ok=True)
    if peak_file.exists() and peak_file.is_dir():
        raise RuntimeError(f"PEAK_FILE points to a directory: {peak_file}")

    # A failed retry must not leave a previous attempt's JSON looking current.
    peak_file.unlink(missing_ok=True)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=peak_file.parent, prefix=f".{peak_file.name}.", delete=False
        ) as handle:
            probe_path = Path(handle.name)
            handle.write("writable\n")
            handle.flush()
            os.fsync(handle.fileno())
        probe_path.unlink()
    except OSError as error:
        raise RuntimeError(f"PEAK_FILE is not writable: {peak_file}: {error}") from error
    return peak_file


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a complete JSON document without exposing a partial final file."""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    started_monotonic = time.monotonic()
    started_at = _utc_now()
    peak_file = _prepare_peak_file()
    if not MAIN_PATH.is_file():
        raise RuntimeError(f"Mammoth entrypoint not found: {MAIN_PATH}")
    git_sha = _git_sha()

    # Import only after validating PEAK_FILE: a configuration error then fails
    # quickly, before CUDA or the training stack is initialised.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 is required for campaign peak-memory measurement")

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)

    original_argv = list(sys.argv)
    state: dict[str, Any] = {
        "status": "running",
        "exit_code": None,
        "exception_type": None,
    }

    def write_report() -> None:
        finished_at = _utc_now()
        memory_error: str | None = None
        try:
            torch.cuda.synchronize(0)
        except Exception as error:  # Preserve the report even after a CUDA failure.
            memory_error = f"synchronize failed: {type(error).__name__}: {error}"

        peak_allocated: int | None = None
        peak_reserved: int | None = None
        try:
            peak_allocated = int(torch.cuda.max_memory_allocated(0))
            peak_reserved = int(torch.cuda.max_memory_reserved(0))
        except Exception as error:  # CUDA may be left in an error state after OOM.
            detail = f"peak query failed: {type(error).__name__}: {error}"
            memory_error = f"{memory_error}; {detail}" if memory_error else detail

        payload: dict[str, Any] = {
            "pid": os.getpid(),
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": round(time.monotonic() - started_monotonic, 6),
            "argv": original_argv,
            "git_sha": git_sha,
            "cuda_device": 0,
            "peak_allocated_bytes": peak_allocated,
            "peak_allocated_mib": round(peak_allocated / MIB, 3) if peak_allocated is not None else None,
            "peak_reserved_bytes": peak_reserved,
            "peak_reserved_mib": round(peak_reserved / MIB, 3) if peak_reserved is not None else None,
            "status": state["status"],
            "exit_code": state["exit_code"],
            "exception_type": state["exception_type"],
        }
        if memory_error is not None:
            payload["memory_measurement_error"] = memory_error

        try:
            _atomic_write_json(peak_file, payload)
        except Exception as error:
            print(f"ERROR: could not write PEAK_FILE {peak_file}: {error}", file=sys.stderr, flush=True)

    atexit.register(write_report)

    # Running a file under tools/ makes that directory sys.path[0].  Add the
    # repository root without changing argv so main.py receives every campaign
    # argument exactly as supplied to this wrapper.
    repo_root_text = str(REPO_ROOT)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    os.chdir(REPO_ROOT)

    try:
        runpy.run_path(str(MAIN_PATH), run_name="__main__")
    except SystemExit as error:
        code = error.code
        numeric_code = code if isinstance(code, int) else (0 if code is None else 1)
        state["exit_code"] = numeric_code
        state["status"] = "success" if numeric_code == 0 else "failed"
        if numeric_code != 0:
            state["exception_type"] = "SystemExit"
        raise
    except BaseException as error:
        state["status"] = "failed"
        state["exit_code"] = 130 if isinstance(error, KeyboardInterrupt) else 1
        state["exception_type"] = type(error).__name__
        raise
    else:
        state["status"] = "success"
        state["exit_code"] = 0


if __name__ == "__main__":
    main()
