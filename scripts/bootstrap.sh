#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/vggt-xr
source scripts/environment.sh
python -m pip install -r requirements.txt
python -m pip freeze > environment.lock.txt
python - <<'PY'
import pathlib, urllib.request, zipfile, io
root = pathlib.Path('/root/autodl-tmp/vggt-xr')
if not (root/'third_party/vggt/vggt/models/vggt.py').exists():
    data = urllib.request.urlopen('https://codeload.github.com/facebookresearch/vggt/zip/refs/heads/main', timeout=120).read()
    zipfile.ZipFile(io.BytesIO(data)).extractall(root/'third_party')
    (root/'third_party/vggt-main').rename(root/'third_party/vggt')
print('VGGT code ready', flush=True)
from huggingface_hub import hf_hub_download
p = hf_hub_download('facebook/VGGT-1B', 'model.pt')
print('VGGT weights:', p, flush=True)
PY
python scripts/cuda_smoke.py
