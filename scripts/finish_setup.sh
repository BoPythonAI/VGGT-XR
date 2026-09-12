#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/vggt-xr
source scripts/environment.sh
python - <<'PY'
from pathlib import Path
import zipfile,shutil
root=Path('/root/autodl-tmp/vggt-xr')
if not (root/'third_party/vggt/vggt/models/vggt.py').exists():
    zipfile.ZipFile(root/'tmp/vggt.zip').extractall(root/'tmp/unpacked')
    shutil.copytree(root/'tmp/unpacked/vggt-main',root/'third_party/vggt',dirs_exist_ok=True)
print('VGGT source ready',flush=True)
PY
python scripts/cuda_smoke.py
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download('facebook/VGGT-1B','model.pt'),flush=True)
PY
python -m pytest -q tests
python scripts/prepare_data.py
