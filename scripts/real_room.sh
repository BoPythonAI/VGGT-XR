#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/vggt-xr
source scripts/environment.sh
mkdir -p data/vggt_kitchen/images
python - <<'PY'
from pathlib import Path
import shutil
files=sorted(Path('third_party/vggt/examples/kitchen/images').glob('*.png'))
for f in files[:24]:shutil.copy2(f,Path('data/vggt_kitchen/images')/f.name)
PY
python run.py --input data/vggt_kitchen --filter confidence --output outputs/kitchen --steps 3000 --max-points 60000
