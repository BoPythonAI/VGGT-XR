"""Freeze unused test poses/images before inspecting any final quality scores."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import check_storage,write_json
from src.panorama import perspective_rays,rotation
from scripts.prepare_data import trace


def main():
    root=Path('/root/autodl-tmp/vggt-xr');out=root/'data/quality_final_test';check_storage(out,growth_gb=.1)
    if (out/'manifest.json').exists():raise SystemExit('Test set is already frozen; do not regenerate it')
    size=336;local,k=perspective_rays(size,90)
    folder=out/'analytic_room';(folder/'test').mkdir(parents=True,exist_ok=True);frames=[]
    centers=[[-.3,-.05,-.1],[.1,.1,.45],[-.4,.05,.6]]
    for center_id,center in enumerate(centers):
        center=np.asarray(center,np.float32)
        for view,(yaw,pitch) in enumerate([(y,0) for y in range(0,360,45)]+[(0,90),(0,-90)]):
            r=rotation(yaw,pitch);rgb,_=trace(center,local@r.T);name=f'center_{center_id}_view_{view:02d}.png'
            Image.fromarray((rgb*255).astype(np.uint8)).save(folder/'test'/name)
            c2w=np.eye(4,dtype=np.float32);c2w[:3,:3]=r;c2w[:3,3]=center
            frames.append(dict(file=name,c2w=c2w.tolist(),intrinsics=k.tolist(),center_id=center_id,yaw=yaw,pitch=pitch))
    write_json(folder/'scene.json',dict(test_frames=frames,kind='controlled_synthetic',protocol='Three frozen new camera centers, same analytical scene; never training inputs'))
    raw=np.load(root/'data/tiny_nerf_data.npz');original=json.loads((root/'data/tiny_nerf/scene.json').read_text())
    used={int(Path(x['file']).stem) for x in original['train_frames']+original['test_frames']}
    ids=[i for i in [5,15,25,35,45,55,65,75,85,95,100] if i not in used]
    folder=out/'tiny_nerf';(folder/'test').mkdir(parents=True,exist_ok=True);frames=[]
    for index in ids:
        name=f'{index:03d}.png';image=Image.fromarray((raw['images'][index]*255).astype(np.uint8)).resize((size,size),Image.Resampling.LANCZOS);image.save(folder/'test'/name)
        c2w=raw['poses'][index].copy();c2w[:3,1:3]*=-1
        focal=float(raw['focal'])*size/raw['images'].shape[2];kk=np.array([[focal,0,size/2],[0,focal,size/2],[0,0,1]])
        frames.append(dict(file=name,c2w=c2w.tolist(),intrinsics=kk.tolist(),original_index=index))
    write_json(folder/'scene.json',dict(test_frames=frames,protocol='Unused public TinyNeRF frames; same scene, original 100x100; never training/dev inputs'))
    files={f.relative_to(out).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.rglob('*')) if f.is_file()}
    write_json(out/'manifest.json',dict(date='2026-09-15',files=files,tiny_test_indices=ids,panorama_centers=centers,selection_protocol='Freeze before evaluating final test; select configuration using old dev set only; report all frozen positions'))
    write_json(root/'reports/quality_test_manifest.json',json.loads((out/'manifest.json').read_text()))
    print('FROZEN',len(files),'files',ids,centers,flush=True)


if __name__=='__main__':main()
