#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/vggt-xr
for attempt in $(seq 1 180); do
    if [ -f cache/vggt_model.pt ] && [ -f cache/vggt_model.sha256 ]; then
        exec bash scripts/start_ready.sh
    fi
    sleep 5
done
echo 'Timed out waiting for a verified checkpoint' >&2
exit 1
