"""Create a real gsplat flythrough + quantitative preview assets, not Unity footage."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.gaussian import render
from src.panorama import rotation
from src.storage import check_storage,write_json

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,default=120);a=p.parse_args()
check_storage(a.output,growth_gb=.1)
params={k:v.to('cuda') for k,v in torch.load(a.run/'gaussian/checkpoint.pt',weights_only=True).items()}
camera=json.loads((a.run/'gaussian/camera.json').read_text());e=np.array(camera['extrinsic'],np.float32);k=np.array(camera['intrinsic'],np.float32)
r=e[:3,:3].T;center=-r@e[:3,3]
h,w=camera['height'],camera['width'];trajectory=[]
if 'overlap' in a.run.name or 'cubemap' in a.run.name:
    for i in range(a.frames):
        turn=rotation(360*i/a.frames)
        rr=r@turn
        ee=np.eye(4,dtype=np.float32);ee[:3,:3]=rr.T;ee[:3,3]=-rr.T@center;trajectory.append(ee)
else:
    # Smooth orbit through actual inferred cameras (train path), not GT.
    from scipy.spatial.transform import Rotation,Slerp
    pred=np.load(a.run/'vggt/predictions.npz');ex=pred['extrinsics'];centers=-np.einsum('nji,nj->ni',ex[:,:3,:3],ex[:,:3,3]);rs=ex[:,:3,:3].transpose(0,2,1)
    indices=np.arange(len(ex)+1);rs=np.concatenate([rs,rs[:1]],0);centers=np.concatenate([centers,centers[:1]],0)
    slerp=Slerp(indices,Rotation.from_matrix(rs))
    for t in np.linspace(0,len(ex),a.frames,endpoint=False):
        i=int(t);f=t-i;rr=slerp(t).as_matrix();c=centers[i]*(1-f)+centers[i+1]*f
        ee=np.eye(4,dtype=np.float32);ee[:3,:3]=rr.T;ee[:3,3]=-rr.T@c;trajectory.append(ee)
with imageio.get_writer(str(a.output/'gsplat_flythrough.mp4'),fps=24,codec='libx264',quality=8,macro_block_size=1) as writer:
    with torch.no_grad():
        for i,ee in enumerate(trajectory):
            frame=render(params,torch.as_tensor(ee,device='cuda')[None],torch.as_tensor(k,device='cuda')[None],w,h)[0].clip(0,1).cpu().numpy()
            writer.append_data((frame*255).astype(np.uint8))
            if i==0:imageio.imwrite(str(a.output/'gaussian_preview.png'),(frame*255).astype(np.uint8))
write_json(a.output/'video.json',dict(source_run=a.run.name,renderer='gsplat CUDA, not Unity',frames=a.frames,fps=24))
print('Real gsplat flythrough saved',flush=True)
