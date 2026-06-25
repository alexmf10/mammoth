import csv
import logging
import os
import socket
import time
from argparse import Namespace
from datetime import datetime
from typing import Optional

import torch

from utils.stats import _parse_device_ids


_MIB = 1024 ** 2


def _to_mib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return value / _MIB


def configure_cuda_memory_limit(args: Namespace) -> None:
    """
    Optionally limit PyTorch's CUDA caching allocator.
    """
    limit_mb = getattr(args, 'cuda_memory_limit_mb', None)
    if limit_mb is None:
        return

    if not torch.cuda.is_available():
        logging.warning('Ignoring --cuda_memory_limit_mb because CUDA is not available.')
        return

    device_ids = _parse_device_ids(args.device) or list(range(torch.cuda.device_count()))
    for device_id in device_ids:
        total_mb = torch.cuda.get_device_properties(device_id).total_memory / _MIB
        if limit_mb <= 0:
            raise ValueError('--cuda_memory_limit_mb must be > 0.')
        if limit_mb > total_mb:
            raise ValueError(f'--cuda_memory_limit_mb={limit_mb} exceeds GPU {device_id} total memory ({total_mb:.0f} MiB).')

        fraction = float(limit_mb) / total_mb
        torch.cuda.set_per_process_memory_fraction(fraction, device=device_id)
        logging.info(f'CUDA allocator memory limit set to {limit_mb:.0f} MiB on GPU {device_id} ({fraction:.3f} of visible VRAM).')


class VRAMPeakTracker:
    """
    Track peak CUDA memory from inside the training process and write a compact CSV.
    """

    def __init__(self, args: Namespace):
        self.args = args
        self.path = getattr(args, 'vram_peak_log', None)
        self.enabled = bool(self.path) and torch.cuda.is_available()
        self.device_ids = _parse_device_ids(getattr(args, 'device', None)) if torch.cuda.is_available() else None
        self.device_ids = self.device_ids or (list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [])
        self._started_at = None
        self._finished_at = None
        self._rows = {}
        self._nvml_handles = {}

    def __enter__(self):
        if not self.enabled:
            if self.path:
                logging.warning('Ignoring --vram_peak_log because CUDA is not available.')
            return self

        self._started_at = time.time()
        for device_id in self.device_ids:
            torch.cuda.reset_peak_memory_stats(device_id)
            self._rows[device_id] = {
                'peak_nvml_process_mib': None,
                'peak_nvml_all_processes_mib': None,
            }

        self.update()
        logging.info(f'VRAM peak logging enabled. Summary will be written to: {self.path}')
        return self

    def __call__(self):
        self.update()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._finished_at = time.time()
        self.update()
        self.write()

    def update(self) -> None:
        if not self.enabled:
            return

        for device_id in self.device_ids:
            process_mib, all_processes_mib = self._get_nvml_memory_mib(device_id)
            row = self._rows[device_id]
            if process_mib is not None:
                row['peak_nvml_process_mib'] = max(row['peak_nvml_process_mib'] or 0, process_mib)
            if all_processes_mib is not None:
                row['peak_nvml_all_processes_mib'] = max(row['peak_nvml_all_processes_mib'] or 0, all_processes_mib)

    def write(self) -> None:
        if not self.enabled:
            return

        path = os.path.expanduser(self.path)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        fieldnames = [
            'started_at',
            'finished_at',
            'elapsed_seconds',
            'host',
            'pid',
            'model',
            'dataset',
            'seed',
            'batch_size',
            'cuda_memory_limit_mb',
            'device_id',
            'gpu_name',
            'gpu_total_mib',
            'torch_current_allocated_mib',
            'torch_current_reserved_mib',
            'torch_peak_allocated_mib',
            'torch_peak_reserved_mib',
            'peak_nvml_process_mib',
            'peak_nvml_all_processes_mib',
        ]

        rows = [self._summary_row(device_id) for device_id in self.device_ids]
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())

        peaks = []
        for row in rows:
            peak_values = [
                float(row[col]) for col in ['torch_peak_reserved_mib', 'peak_nvml_process_mib', 'peak_nvml_all_processes_mib']
                if row[col] != ''
            ]
            if peak_values:
                peaks.append(max(peak_values))
        if peaks:
            logging.info(f'Peak VRAM summary written to {path}. Max observed peak: {max(peaks) / 1024:.2f} GiB.')

    def _summary_row(self, device_id: int) -> dict:
        props = torch.cuda.get_device_properties(device_id)
        row = self._rows[device_id]
        started_at = datetime.fromtimestamp(self._started_at).isoformat(timespec='seconds') if self._started_at else ''
        finished_at = datetime.fromtimestamp(self._finished_at).isoformat(timespec='seconds') if self._finished_at else ''
        elapsed = (self._finished_at - self._started_at) if self._started_at and self._finished_at else None

        return {
            'started_at': started_at,
            'finished_at': finished_at,
            'elapsed_seconds': f'{elapsed:.3f}' if elapsed is not None else '',
            'host': socket.gethostname(),
            'pid': os.getpid(),
            'model': getattr(self.args, 'model', ''),
            'dataset': getattr(self.args, 'dataset', ''),
            'seed': getattr(self.args, 'seed', ''),
            'batch_size': getattr(self.args, 'batch_size', ''),
            'cuda_memory_limit_mb': getattr(self.args, 'cuda_memory_limit_mb', ''),
            'device_id': device_id,
            'gpu_name': torch.cuda.get_device_name(device_id),
            'gpu_total_mib': f'{props.total_memory / _MIB:.2f}',
            'torch_current_allocated_mib': f'{torch.cuda.memory_allocated(device_id) / _MIB:.2f}',
            'torch_current_reserved_mib': f'{torch.cuda.memory_reserved(device_id) / _MIB:.2f}',
            'torch_peak_allocated_mib': f'{torch.cuda.max_memory_allocated(device_id) / _MIB:.2f}',
            'torch_peak_reserved_mib': f'{torch.cuda.max_memory_reserved(device_id) / _MIB:.2f}',
            'peak_nvml_process_mib': f'{row["peak_nvml_process_mib"]:.2f}' if row['peak_nvml_process_mib'] is not None else '',
            'peak_nvml_all_processes_mib': f'{row["peak_nvml_all_processes_mib"]:.2f}' if row['peak_nvml_all_processes_mib'] is not None else '',
        }

    def _get_nvml_memory_mib(self, device_id: int) -> tuple[Optional[float], Optional[float]]:
        try:
            pynvml = torch.cuda.pynvml  # type: ignore[attr-defined]
            if device_id not in self._nvml_handles:
                pynvml.nvmlInit()
                self._nvml_handles[device_id] = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            handle = self._nvml_handles[device_id]
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            current_pid = os.getpid()
            process_bytes = sum(proc.usedGpuMemory for proc in procs if proc.pid == current_pid and proc.usedGpuMemory is not None)
            all_processes_bytes = sum(proc.usedGpuMemory for proc in procs if proc.usedGpuMemory is not None)
            return _to_mib(process_bytes), _to_mib(all_processes_bytes)
        except BaseException:
            return None, None
