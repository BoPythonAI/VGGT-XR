#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/vggt-xr
source scripts/environment.sh
if [ -f "$VGGTXR_ROOT/cache/vggt_model.pt" ]; then
    echo 'Using SHA256-verified local checkpoint'
    exit 0
fi
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
python - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download('facebook/VGGT-1B','model.pt'),flush=True)
PY
