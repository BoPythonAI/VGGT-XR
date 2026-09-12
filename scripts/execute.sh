#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/vggt-xr
source scripts/environment.sh
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
bash scripts/weights.sh
python -m pytest -q tests
python scripts/prepare_data.py
python scripts/run_experiments.py --steps 1500
