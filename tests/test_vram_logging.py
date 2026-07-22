import os
import sys
from argparse import Namespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import parse_args
from utils.vram import VRAMPeakTracker, configure_cuda_memory_limit


def test_vram_args_are_parsed(tmp_path):
    log_path = tmp_path / 'vram_peak.csv'
    args = parse_args([
        '--model',
        'sgd',
        '--dataset',
        'seq-cifar10',
        '--lr',
        '1e-3',
        '--n_epochs',
        '1',
        '--batch_size',
        '2',
        '--non_verbose',
        '1',
        '--num_workers',
        '0',
        '--debug_mode',
        '1',
        '--vram_peak_log',
        str(log_path),
        '--cuda_memory_limit_mb',
        '7000',
    ])

    assert args.vram_peak_log == str(log_path)
    assert args.cuda_memory_limit_mb == 7000


def test_vram_helpers_noop_without_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    args = Namespace(vram_peak_log=str(tmp_path / 'vram_peak.csv'), cuda_memory_limit_mb=7000, device='cuda:0')

    configure_cuda_memory_limit(args)
    with VRAMPeakTracker(args) as tracker:
        tracker()

    assert not (tmp_path / 'vram_peak.csv').exists()
