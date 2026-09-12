"""Resumable ranged download with size/SHA256 verification; all bytes stay on data disk."""
import concurrent.futures,hashlib,os,time
from pathlib import Path
import requests
from huggingface_hub import get_hf_file_metadata
from src.storage import check_storage

root=Path(os.environ.get('VGGTXR_ROOT','/root/autodl-tmp/vggt-xr'))
check_storage(root/'cache',growth_gb=10)
target=root/'cache/vggt_model.pt'
if target.exists():
    print('Verified checkpoint already present',flush=True)
    raise SystemExit(0)
url='https://hf-mirror.com/facebook/VGGT-1B/resolve/main/model.pt'
meta=get_hf_file_metadata(url)
size=meta.size;digest=meta.etag.strip('"')
assert size and len(digest)==64
assert digest=='d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0','Official checkpoint changed; update the lock file deliberately before downloading'
folder=root/'tmp/model_parts';folder.mkdir(parents=True,exist_ok=True)
print(f'Downloading {size/2**30:.2f} GiB in 8 ranges',flush=True)

def part(i):
    start=size*i//8;stop=size*(i+1)//8-1
    file=folder/f'{i:02d}.part'
    for attempt in range(5):
        offset=file.stat().st_size if file.exists() else 0
        if offset==stop-start+1:return file
        if offset>stop-start+1:raise RuntimeError('Oversized part')
        try:
            with requests.get(meta.location,headers={'Range':f'bytes={start+offset}-{stop}'},stream=True,timeout=(30,120)) as response:
                if response.status_code!=206:raise RuntimeError(f'Range request returned {response.status_code}')
                with file.open('ab') as f:
                    for chunk in response.iter_content(2**20):
                        f.write(chunk)
                        if f.tell()>stop-start+1:raise RuntimeError('Server exceeded range')
            if file.stat().st_size==stop-start+1:
                print(f'Part {i} complete',flush=True);return file
        except requests.RequestException as e:
            print(f'Part {i}: network retry {attempt+1}',flush=True);time.sleep(2)
    raise RuntimeError(f'Part {i} incomplete')

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    parts=list(pool.map(part,range(8)))
sha=hashlib.sha256()
temporary=target.with_suffix('.tmp')
with temporary.open('wb') as f:
    for p in parts:
        with p.open('rb') as source:
            while chunk:=source.read(8*2**20):f.write(chunk);sha.update(chunk)
assert temporary.stat().st_size==size and sha.hexdigest()==digest,'Checkpoint integrity mismatch'
temporary.replace(target)
(target.with_suffix('.sha256')).write_text(digest+'\n')
print('Checkpoint size and SHA256 verified',flush=True)
for p in parts:p.unlink()
