#!/usr/bin/env bash
# Capture the software/hardware state used by the campaign.  Optional probes
# are allowed to fail so that the resulting file still contains the rest of
# the evidence.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/environment/bin/python}"
OVERLAY_DIR="$HOME/.local/mammoth-pydeps"

case ":${PYTHONPATH:-}:" in
    *":${OVERLAY_DIR}:"*) ;;
    *) export PYTHONPATH="${OVERLAY_DIR}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac

section() {
    printf '\n===== %s =====\n' "$1"
}

cd "$REPO_ROOT" || {
    echo "ERROR: no se puede entrar en $REPO_ROOT" >&2
    exit 1
}

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: PYTHON_BIN no existe o no es ejecutable: $PYTHON_BIN" >&2
    exit 1
fi

section "IDENTIDAD DEL SNAPSHOT"
printf 'timestamp_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
printf 'hostname=%s\n' "$(hostname)"
printf 'user=%s\n' "$(id -un)"
printf 'repo_root=%s\n' "$REPO_ROOT"
printf 'python_bin_configured=%s\n' "$PYTHON_BIN"
printf 'overlay_dir=%s\n' "$OVERLAY_DIR"
printf 'pythonpath=%s\n' "${PYTHONPATH:-}"

section "GIT"
printf 'commit=%s\n' "$(git rev-parse HEAD 2>&1 || echo UNAVAILABLE)"
printf 'branch=%s\n' "$(git branch --show-current 2>&1 || echo UNAVAILABLE)"
if git diff --quiet --ignore-submodules -- && git diff --cached --quiet --ignore-submodules --; then
    echo "tracked_worktree=clean"
else
    echo "tracked_worktree=dirty"
fi
echo "git_status_begin"
git status --short --branch 2>&1 || true
echo "git_status_end"

section "GPU, DRIVER Y CUDA SEGUN NVIDIA-SMI"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi 2>&1 || true
    echo "gpu_query_begin"
    nvidia-smi \
        --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,memory.free,compute_mode \
        --format=csv 2>&1 || true
    echo "gpu_query_end"
else
    echo "nvidia-smi=NOT_FOUND"
fi

section "CUDA TOOLKIT DEL SISTEMA (SI EXISTE)"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version 2>&1 || true
else
    echo "nvcc=NOT_FOUND (no impide usar el runtime CUDA incluido con PyTorch)"
fi

section "PYTHON Y LIBRERIAS EFECTIVAMENTE IMPORTADAS"
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys


print(f"sys.executable={sys.executable}")
print(f"python_version={platform.python_version()}")
print(f"platform={platform.platform()}")

for distribution, module_name in (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("timm", "timm"),
    ("kornia", "kornia"),
):
    try:
        distribution_version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        distribution_version = "NOT_INSTALLED"
    print(f"{distribution}_distribution_version={distribution_version}")
    try:
        module = importlib.import_module(module_name)
    except BaseException as error:
        print(f"{module_name}_import=ERROR {type(error).__name__}: {error}")
    else:
        print(f"{module_name}_import=OK")
        print(f"{module_name}_module_version={getattr(module, '__version__', 'UNKNOWN')}")
        print(f"{module_name}_module_file={getattr(module, '__file__', 'UNKNOWN')}")

try:
    import torch
except BaseException as error:
    print(f"torch_precision_probe=ERROR {type(error).__name__}: {error}")
else:
    print(f"torch_cuda_runtime={torch.version.cuda}")
    print(f"torch_cudnn_version={torch.backends.cudnn.version()}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"default_dtype={torch.get_default_dtype()}")
    print(f"float32_matmul_precision={torch.get_float32_matmul_precision()}")
    print(f"cuda_matmul_allow_tf32={torch.backends.cuda.matmul.allow_tf32}")
    print(f"cudnn_allow_tf32={torch.backends.cudnn.allow_tf32}")
    print(f"autocast_enabled_now={torch.is_autocast_enabled()}")
    get_gpu_dtype = getattr(torch, "get_autocast_gpu_dtype", None)
    if get_gpu_dtype is not None:
        print(f"autocast_gpu_dtype={get_gpu_dtype()}")
    if torch.cuda.is_available():
        print(f"cuda_device_count={torch.cuda.device_count()}")
        print(f"cuda_device_0={torch.cuda.get_device_name(0)}")
        print(f"cuda_device_0_capability={torch.cuda.get_device_capability(0)}")
PY
python_status=$?
printf 'python_probe_exit_code=%s\n' "$python_status"

section "PRECISION CONFIGURADA POR LA CAMPANA"
echo "Mammoth define --code_optimization=0 por defecto: sin AMP/autocast y tensores FP32."
echo "Los flags TF32 anteriores indican si los kernels CUDA pueden usar TF32 internamente."
echo "Flags de precision encontrados en tools/run_campaign.sh:"
if grep -nE -- '(^|[[:space:]])(-O|--code_optimization)(=|[[:space:]])' tools/run_campaign.sh 2>/dev/null; then
    :
else
    echo "ninguno; se aplica el default code_optimization=0"
fi
echo "Usos de autocast en el camino central main.py + utils/training.py:"
if grep -nE 'autocast|GradScaler' main.py utils/training.py 2>/dev/null; then
    :
else
    echo "ninguno"
fi

section "DISCO Y QUOTA DE HOME"
echo "df_h_begin"
df -h "$HOME" 2>&1 || true
echo "df_h_end"
echo "df_kib_blocks_begin"
df -Pk "$HOME" 2>&1 || true
echo "df_kib_blocks_end"
if command -v quota >/dev/null 2>&1; then
    echo "quota_begin"
    quota -s 2>&1 || true
    echo "quota_end"
else
    echo "quota=NOT_FOUND (usar df y confirmar la politica de cuota del servidor)"
fi
if [[ -d "$OVERLAY_DIR" ]]; then
    du -sh "$OVERLAY_DIR" 2>&1 || true
else
    echo "overlay_dir=NOT_FOUND"
fi
if [[ -d data ]]; then
    du -sh data 2>&1 || true
fi

section "FIN"
printf 'snapshot_exit_code=%s\n' "$python_status"
exit "$python_status"
