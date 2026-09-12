#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/vggt-xr
source scripts/environment.sh
python run.py --input data/analytic_room --panorama --projection cubemap --filter confidence --output outputs/cubemap --steps 1500
python run.py --input data/analytic_room --panorama --projection overlap --filter confidence --output outputs/overlap --steps 1500
python evaluation/report.py
python evaluation/figures.py
python scripts/verify_results.py
python scripts/package_results.py
