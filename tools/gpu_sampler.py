#!/usr/bin/env python3
"""Continuously sample every GPU compute process reported by ``nvidia-smi``.

The CSV intentionally includes processes belonging to other users.  This makes
GPU contention during a campaign auditable, while ``wrapper_peak.py`` supplies
the process-local PyTorch allocation metric used for the primary comparison.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
try:
    import pwd
except ImportError:  # pragma: no cover - pwd is available on the Linux campaign host.
    pwd = None  # type: ignore[assignment]
import signal
import subprocess
import sys
import time
from typing import TextIO


CSV_FIELDS = (
    "timestamp_utc",
    "timestamp_epoch",
    "pid",
    "username",
    "process_name",
    "used_gpu_memory_mib",
    "sample_status",
)

_stop_requested = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _request_stop(_signum: int, _frame: object) -> None:
    global _stop_requested
    _stop_requested = True


def _query_compute_processes() -> tuple[list[list[str]], str | None]:
    """Return parsed nvidia-smi rows and an error message, if any."""

    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        return [], f"{type(error).__name__}: {error}"

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        return [], f"nvidia-smi exited {completed.returncode}: {detail}"

    rows: list[list[str]] = []
    try:
        for row in csv.reader(completed.stdout.splitlines(), skipinitialspace=True):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 3:
                return [], f"unexpected nvidia-smi row with {len(row)} fields: {row!r}"
            rows.append([value.strip() for value in row])
    except csv.Error as error:
        return [], f"could not parse nvidia-smi output: {error}"
    return rows, None


def _username_for_pid(pid: str) -> str:
    """Resolve the current owner of a Linux PID without failing the sampler."""

    if pwd is None or not pid.isdecimal():
        return ""
    try:
        uid = (Path("/proc") / pid).stat().st_uid
        return pwd.getpwuid(uid).pw_name
    except (KeyError, OSError):
        # The process can legitimately disappear between nvidia-smi and stat.
        return ""


def _write_sample(writer: csv.DictWriter, output: TextIO) -> None:
    sampled_epoch = time.time()
    sampled_at = _utc_now()
    rows, error = _query_compute_processes()

    if error is not None:
        writer.writerow(
            {
                "timestamp_utc": sampled_at,
                "timestamp_epoch": f"{sampled_epoch:.6f}",
                "pid": "",
                "username": "",
                "process_name": "",
                "used_gpu_memory_mib": "",
                "sample_status": f"error: {error}",
            }
        )
    elif not rows:
        writer.writerow(
            {
                "timestamp_utc": sampled_at,
                "timestamp_epoch": f"{sampled_epoch:.6f}",
                "pid": "",
                "username": "",
                "process_name": "",
                "used_gpu_memory_mib": "",
                "sample_status": "no_compute_processes",
            }
        )
    else:
        for pid, process_name, used_memory in rows:
            writer.writerow(
                {
                    "timestamp_utc": sampled_at,
                    "timestamp_epoch": f"{sampled_epoch:.6f}",
                    "pid": pid,
                    "username": _username_for_pid(pid),
                    "process_name": process_name,
                    "used_gpu_memory_mib": used_memory,
                    "sample_status": "ok",
                }
            )

    # The campaign may outlive the browser session; make every completed poll
    # immediately visible to tail/readers.
    output.flush()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="CSV path (existing files are appended)")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between poll starts (default: 1.0)")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    return args


def main() -> int:
    args = _parse_args()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output_path.exists() or output_path.stat().st_size == 0

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    try:
        with output_path.open("a", encoding="utf-8", newline="", buffering=1) as output:
            writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
            if needs_header:
                writer.writeheader()
                output.flush()

            print(
                f"Sampling all nvidia-smi compute processes every {args.interval:g}s into {output_path}",
                file=sys.stderr,
                flush=True,
            )
            while not _stop_requested:
                poll_started = time.monotonic()
                _write_sample(writer, output)
                remaining = args.interval - (time.monotonic() - poll_started)
                if remaining > 0:
                    # Short waits make SIGTERM responsive without busy-polling.
                    deadline = time.monotonic() + remaining
                    while not _stop_requested:
                        sleep_for = min(0.2, deadline - time.monotonic())
                        if sleep_for <= 0:
                            break
                        time.sleep(sleep_for)
    except OSError as error:
        print(f"ERROR: cannot write sampler CSV {output_path}: {error}", file=sys.stderr, flush=True)
        return 1

    print("GPU sampler stopped cleanly", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
