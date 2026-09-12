#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/vggt-xr
source scripts/environment.sh
for method in raw confidence consistency; do
    output="outputs/nerf_${method}_masked"
    mkdir -p "$output"
    if [ ! -e "$output/vggt" ]; then ln -s /root/autodl-tmp/vggt-xr/outputs/nerf_raw/vggt "$output/vggt"; fi
    python run.py --input data/tiny_nerf --filter "$method" --mask-black-background --output "$output" --steps 1500
done
bash scripts/real_room.sh
python evaluation/report.py
python evaluation/figures.py
python scripts/render_demo.py --run outputs/kitchen --output outputs/demo/kitchen
python scripts/render_demo.py --run outputs/overlap_anchor --output outputs/demo/panorama
