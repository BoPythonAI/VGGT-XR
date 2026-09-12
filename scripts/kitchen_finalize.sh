#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/vggt-xr
source scripts/environment.sh
bash scripts/real_room.sh
python evaluation/report.py
python evaluation/figures.py
python scripts/render_demo.py --run outputs/kitchen --output outputs/demo/kitchen
python scripts/package_results.py
