"""Infrastructure smoke with analytic GT geometry. NOT a VGGT experiment result."""
from pathlib import Path
import numpy as np
from scripts.prepare_data import trace
from src.panorama import perspective_rays,rotation
from src.geometry import unproject,filter_points
from src.export import export_colmap
from src.gaussian import train

root=Path('/root/autodl-tmp/vggt-xr/outputs/pipeline_smoke')
rays,k=perspective_rays(112,90)
images=[];depths=[];extrinsics=[]
for origin,yaw in [(np.array([-.5,0,-.5]),0),(np.array([.5,0,-.5]),0),(np.array([0,.2,.5]),0)]:
    r=rotation(yaw);image,depth=trace(origin,rays@r.T)
    images.append(image);depths.append(depth)
    e=np.zeros((3,4),np.float32);e[:,:3]=r.T;e[:,3]=-r.T@origin;extrinsics.append(e)
images=np.stack(images);depths=np.stack(depths);es=np.stack(extrinsics);ks=np.repeat(k[None],3,0)
points=unproject(depths,es,ks)
xyz,rgb,conf,scales,mask,stats=filter_points(points,images,np.ones_like(depths)*2,depths,es,ks,'raw',max_points=2048)
export_colmap(root/'sparse/0',['0.png','1.png','2.png'],es,ks,(xyz,rgb),images)
test_e=np.eye(4,dtype=np.float32)[None];test_e[0,:3]=es[0]
result=train(xyz,rgb,scales,images,es,ks,root/'gaussian',steps=5,test=(test_e,ks[:1],images[:1]))
print('COLMAP + Gaussian + SSIM + LPIPS + PLY infrastructure smoke passed; NOT VGGT metrics',flush=True)
