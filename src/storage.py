import json, os, shutil
from pathlib import Path


def check_storage(path, reserve_gb=15, growth_gb=0):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free / 2**30
    if free < reserve_gb + growth_gb:
        raise RuntimeError(f'{path}: {free:.1f} GiB free; need {reserve_gb + growth_gb:.1f} GiB. Stopping before allocation.')
    if os.name != 'nt' and not str(path).startswith('/root/autodl-tmp/'):
        raise RuntimeError(f'Large artifacts must be on /root/autodl-tmp, got {path}')
    return free


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)
