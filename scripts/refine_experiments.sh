#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/vggt-xr
source scripts/environment.sh
python run.py --input data/tiny_nerf --filter confidence --output outputs/nerf_confidence --steps 1500
python run.py --input data/tiny_nerf --filter consistency --output outputs/nerf_consistency --steps 1500
python run.py --input data/analytic_room --panorama --projection overlap --include-poles --max-views 32 --filter confidence --output outputs/overlap_fullsphere --steps 1500
python run.py --input data/analytic_room --panorama --projection overlap --include-poles --max-views 32 --constrain-poses --pose-fusion anchor --known-intrinsics --filter confidence --output outputs/overlap_anchor --steps 1500
python evaluation/report.py
python evaluation/figures.py
