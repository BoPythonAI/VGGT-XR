#!/usr/bin/env bash
export VGGTXR_ROOT="${VGGTXR_ROOT:-/root/autodl-tmp/vggt-xr}"
export HF_HOME="$VGGTXR_ROOT/cache/huggingface"
export TORCH_HOME="$VGGTXR_ROOT/cache/torch"
export PIP_CACHE_DIR="$VGGTXR_ROOT/cache/pip"
export TORCH_EXTENSIONS_DIR="$VGGTXR_ROOT/cache/torch_extensions"
export TMPDIR="$VGGTXR_ROOT/tmp"
export XDG_CACHE_HOME="$VGGTXR_ROOT/cache/xdg"
export CUDA_HOME=/usr/local/cuda
export PATH="$VGGTXR_ROOT/env/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4
export PYTHONUNBUFFERED=1
export PYTHONPATH="$VGGTXR_ROOT/third_party/vggt:$VGGTXR_ROOT:${PYTHONPATH:-}"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$XDG_CACHE_HOME"
