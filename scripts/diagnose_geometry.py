import numpy as np
from src.geometry import unproject,filter_points
from pathlib import Path
pred=np.load('outputs/nerf_raw/vggt/predictions.npz')
points=unproject(pred['depth'],pred['extrinsics'],pred['intrinsics'])
images=pred['images'];conf=pred['confidence'];foreground=images.max(-1)>.03
print('Foreground fraction',float(foreground.mean()))
for name,mask in [('foreground',foreground),('black background',~foreground)]:
    print(name,'depth percentiles',np.percentile(pred['depth'][mask],[2,50,98]),'confidence percentiles',np.percentile(conf[mask],[0,35,50,90,100]))
for method in ['raw','confidence']:
    xyz,rgb,confidence,scales,mask,stats=filter_points(points,images,conf,pred['depth'],pred['extrinsics'],pred['intrinsics'],method,max_points=30000)
    print(method,'nonblack initial Gaussians',int((rgb.max(-1)>.03).sum()),'bounds',np.percentile(xyz,[2,98],axis=0))
