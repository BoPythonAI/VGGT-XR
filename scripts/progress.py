"""Read-only status for a long server experiment; no raw credentials in output."""
from pathlib import Path
import json,shutil
root=Path('/root/autodl-tmp/vggt-xr')
status=root/'outputs/experiment_status.json'
if status.exists():
    for item in json.loads(status.read_text()):
        print(item['name'],'PASS' if item['exit_code']==0 else 'FAILED',round(item['wall_seconds'],1),'seconds')
for f in sorted((root/'logs').glob('*.log')):
    if f.name in ['experiments.log','nerf_raw.log','overlap.log','parallel_weights.log']:
        print(f.name+':\n'+'\n'.join(f.read_text(errors='replace').splitlines()[-5:]))
for disk in [Path('/'),root]:
    print(str(disk),'free GiB',round(shutil.disk_usage(disk).free/2**30,1))
