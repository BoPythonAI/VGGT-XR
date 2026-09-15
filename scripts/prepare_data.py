"""Small public NeRF dataset + explicitly labelled analytical panorama benchmark."""
import argparse, hashlib, json, sys, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.panorama import erp_rays, perspective_rays, rotation
from src.storage import check_storage, write_json


def trace(origin, rays):
    # A 6x3x8 meter textured room with box furniture. y is down.
    boxes = [([-3,-1.5,-4],[3,1.5,4]), ([-1.6,.35,1.0],[.1,.55,2.2]), ([-1.5,.55,1.1],[-1.3,1.5,1.3]), ([-.15,.55,1.9],[.05,1.5,2.1]), ([1,.6,-2.8],[2.5,1.5,-1]), ([-2.8,-.8,-3.7],[-1.4,.7,-3.4]), ([.7,-.7,2.8],[2.2,.9,3.3])]
    shape = rays.shape[:-1]
    rd = rays.reshape(-1,3)
    safe = np.where(np.abs(rd)<1e-8,1e-8,rd)
    best = np.full(len(rd), np.inf)
    label = np.zeros(len(rd),int)
    axis = np.zeros(len(rd),int)
    for j,(lo,hi) in enumerate(boxes):
        ts = (np.asarray(lo)-origin)/safe
        te = (np.asarray(hi)-origin)/safe
        near = np.minimum(ts,te)
        far = np.maximum(ts,te)
        enter, leave = near.max(-1),far.min(-1)
        if j == 0:
            distance = leave
            face = far.argmin(-1)
        else:
            distance = np.where((leave >= enter)&(enter>0),enter,np.inf)
            face = near.argmax(-1)
        take = (distance>0)&(distance<best)
        best[take],label[take],axis[take] = distance[take],j,face[take]
    xyz = origin+best[:,None]*rd
    palette = np.array([[.72,.77,.82],[.54,.31,.14],[.30,.18,.10],[.30,.18,.10],[.15,.45,.62],[.45,.24,.12],[.28,.48,.28]])
    color = palette[label].copy()
    color *= (.65+.2*np.sin(xyz[:,0]*7.3+xyz[:,1]*3.1)+.12*np.cos(xyz[:,2]*9.7-xyz[:,0]*1.7))[:,None]
    # Spatial RGB patterns, wall panels and floor tiles provide nontrivial
    # correspondence. On planar walls these world-coordinate sinusoids look like
    # colored grids/radial bands; they are benchmark texture, not illumination.
    color += .12*np.stack([np.sin(xyz[:,0]*17+xyz[:,2]*4),np.cos(xyz[:,1]*13+xyz[:,2]*6),np.sin(xyz[:,2]*15+xyz[:,1]*8)],-1)
    floor = (label==0)&(axis==1)&(xyz[:,1]>0)
    tile = ((np.floor(xyz[:,0]*2)+np.floor(xyz[:,2]*2))%2)*.14
    color[floor] = np.array([.48,.34,.22]) + tile[floor,None] + .035*np.sin(xyz[floor,2:3]*70)
    return color.clip(0,1).reshape(*shape,3).astype(np.float32),best.reshape(shape).astype(np.float32)


def trace_natural(origin, rays):
    """Analytical room with restrained, material-like colors and exact depth."""
    boxes = [([-3,-1.5,-4],[3,1.5,4]),
             ([-1.5,.35,.8],[.25,.53,2.25]),
             ([-1.42,.53,.9],[-1.28,1.5,1.05]),
             ([.03,.53,2.0],[.17,1.5,2.15]),
             ([1.0,.55,-2.8],[2.55,1.5,-.85]),
             ([-2.8,-.72,-3.72],[-1.25,.68,-3.48]),
             ([.65,-.55,2.9],[2.15,.8,3.28]),
             ([-2.5,.65,-1.8],[-2.0,1.5,-1.3])]
    shape=rays.shape[:-1];rd=rays.reshape(-1,3);safe=np.where(np.abs(rd)<1e-8,1e-8,rd)
    best=np.full(len(rd),np.inf);label=np.zeros(len(rd),int);axis=np.zeros(len(rd),int)
    for j,(lo,hi) in enumerate(boxes):
        lo,hi=np.asarray(lo),np.asarray(hi);ts=(lo-origin)/safe;te=(hi-origin)/safe
        near,far=np.minimum(ts,te),np.maximum(ts,te);enter,leave=near.max(-1),far.min(-1)
        distance=leave if j==0 else np.where((leave>=enter)&(enter>0),enter,np.inf)
        face=far.argmin(-1) if j==0 else near.argmax(-1);take=(distance>0)&(distance<best)
        best[take],label[take],axis[take]=distance[take],j,face[take]
    xyz=origin+best[:,None]*rd
    palette=np.array([[.76,.74,.69],[.43,.25,.12],[.30,.17,.08],[.30,.17,.08],
                      [.25,.39,.47],[.52,.38,.24],[.37,.47,.36],[.24,.43,.25]])
    color=palette[label].copy()
    room=label==0;floor=room&(axis==1)&(xyz[:,1]>0);ceiling=room&(axis==1)&~floor;walls=room&~(axis==1)
    color[ceiling]=[.88,.87,.82];color[walls]=[.75,.73,.68]
    # Low-contrast achromatic plaster texture gives correspondence without the
    # old rainbow appearance. Add plausible architectural landmarks as texture.
    plaster=.018*(np.sin(xyz[:,0]*8.1+xyz[:,2]*3.7+xyz[:,1]*5.2)
                  +.55*np.cos(xyz[:,2]*11.3-xyz[:,1]*7.4+xyz[:,0]*2.3)
                  +.35*np.sin(xyz[:,0]*13.7+xyz[:,2]*9.1-xyz[:,1]*4.6))
    color[walls]+=plaster[walls,None]
    baseboard=walls&(xyz[:,1]>1.30);color[baseboard]=[.38,.22,.12]
    back=walls&(axis==2)&(xyz[:,2]<-3.9)
    window=back&(np.abs(xyz[:,0])<1.05)&(xyz[:,1]>-.85)&(xyz[:,1]<.48)
    frame=window&((np.abs(np.abs(xyz[:,0])-1.0)<.07)|(np.abs(xyz[:,1]+.82)<.06)|(np.abs(xyz[:,1]-.45)<.06)|(np.abs(xyz[:,0])<.045))
    sky=np.stack([.38+.05*np.sin(xyz[:,0]*2),.57+.04*np.sin(xyz[:,1]*4),.72+.03*np.cos(xyz[:,0]*3)],-1)
    color[window]=sky[window];color[frame]=[.34,.20,.11]
    side=walls&(axis==0)&(xyz[:,0]>2.9)
    picture=side&(xyz[:,2]>-.9)&(xyz[:,2]<.75)&(xyz[:,1]>-.65)&(xyz[:,1]<.35)
    picture_frame=picture&((np.abs(xyz[:,2]+.9)<.06)|(np.abs(xyz[:,2]-.75)<.06)|(np.abs(xyz[:,1]+.65)<.06)|(np.abs(xyz[:,1]-.35)<.06))
    color[picture]=np.array([.42,.52,.44])+(.05*np.sin(xyz[picture,2:3]*9))*np.array([1.,.6,.3])
    color[picture_frame]=[.25,.16,.10]
    # Warm wood planks: narrow seams and subtle grain, without high-frequency RGB bands.
    plank=np.floor((xyz[:,0]+3)/.28).astype(int);row=np.floor((xyz[:,2]+4)/1.35).astype(int)
    seam=(np.mod(xyz[:,0]+3,.28)<.012)|(np.mod(xyz[:,2]+4+(.14*(row%2)),1.35)<.014)
    wood=np.array([.47,.31,.18])+(.025*((plank+row)%3-1))[:,None]
    wood+=.018*np.sin(xyz[:,2:3]*18+plank[:,None]*.7)
    color[floor]=wood[floor];color[floor&seam]=[.24,.16,.10]
    # Gentle luminance variation and a broad window-side light gradient.
    variation=.018*np.sin(xyz[:,0]*2.1+xyz[:,2]*1.3)
    daylight=.82+.16*np.clip((xyz[:,0]+3)/6,0,1)
    color*=daylight[:,None];color+=variation[:,None]
    return color.clip(0,1).reshape(*shape,3).astype(np.float32),best.reshape(shape).astype(np.float32)


TRACE_GENERATORS={'trace':trace,'trace_natural':trace_natural}


def prepare_natural_room(root, size):
    scene=root/'natural_room';(scene/'panoramas').mkdir(parents=True,exist_ok=True);(scene/'test').mkdir(exist_ok=True)
    origins=np.array([[-.8,0,-.6],[.6,.2,-.5],[.4,-.1,.7]],np.float32);rays=erp_rays(512,1024)
    for i,origin in enumerate(origins):
        rgb,radial_depth=trace_natural(origin,rays)
        Image.fromarray((rgb*255).astype('uint8')).save(scene/'panoramas'/f'pano_{i:02d}.png')
        np.save(scene/'panoramas'/f'pano_{i:02d}_range.npy',radial_depth)
    local,k=perspective_rays(size,90);frames=[];centers=[[-.3,-.05,-.1],[.1,.1,.45],[-.4,.05,.6]]
    for center_id,center in enumerate(centers):
        center=np.asarray(center,np.float32)
        for view,(yaw,pitch) in enumerate([(y,0) for y in range(0,360,45)]+[(0,90),(0,-90)]):
            r=rotation(yaw,pitch);rgb,_=trace_natural(center,local@r.T);name=f'center_{center_id}_view_{view:02d}.png'
            Image.fromarray((rgb*255).astype(np.uint8)).save(scene/'test'/name)
            c2w=np.eye(4,dtype=np.float32);c2w[:3,:3]=r;c2w[:3,3]=center
            frames.append(dict(file=name,c2w=c2w.tolist(),intrinsics=k.tolist(),center_id=center_id,yaw=yaw,pitch=pitch))
    write_json(scene/'scene.json',dict(name='Natural analytical furnished room v2',kind='controlled_synthetic',generator='trace_natural',version=2,panorama_origins=origins.tolist(),gt_units='meters',test_frames=frames,protocol='Three training panorama centers and three disjoint test centers; exact analytical RGB/depth/poses',source='scripts/prepare_data.py',license='MIT'))


def prepare_room(root, size):
    scene = root/'analytic_room'
    (scene/'panoramas').mkdir(parents=True,exist_ok=True)
    origins = np.array([[-.8,0,-.6],[.6,.2,-.5],[.4,-.1,.7]],np.float32)
    rays=erp_rays(512,1024)
    for i,origin in enumerate(origins):
        rgb, radial_depth=trace(origin,rays)
        Image.fromarray((rgb*255).astype('uint8')).save(scene/'panoramas'/f'pano_{i:02d}.png')
        np.save(scene/'panoramas'/f'pano_{i:02d}_range.npy',radial_depth)
    test = scene/'test'
    test.mkdir(exist_ok=True)
    origin=np.array([.2,0,.2],np.float32)
    local,k=perspective_rays(size,90)
    frames=[]
    test_layout=[(y,0) for y in range(0,360,45)]+[(0,90),(0,-90)]
    for i,(yaw,pitch) in enumerate(test_layout):
        r=rotation(yaw,pitch)
        rgb,depth=trace(origin,local@r.T)
        name=f'test_{i:02d}.png'
        Image.fromarray((rgb*255).astype('uint8')).save(test/name)
        c2w=np.eye(4,dtype=np.float32);c2w[:3,:3]=r;c2w[:3,3]=origin
        frames.append(dict(file=name,c2w=c2w.tolist(),intrinsics=k.tolist(),yaw=yaw,pitch=pitch))
        np.save(test/f'test_{i:02d}_depth.npy',depth)
    write_json(scene/'scene.json',dict(name='Analytical textured box room',kind='controlled_synthetic',not_real_capture=True,panorama_origins=origins.tolist(),gt_units='meters',test_frames=frames,source='scripts/prepare_data.py',license='MIT'))


def prepare_nerf(root,size):
    scene=root/'tiny_nerf';scene.mkdir(exist_ok=True)
    archive=root/'tiny_nerf_data.npz'
    source='https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz'
    if not archive.exists():
        urllib.request.urlretrieve(source,archive)
    data=np.load(archive)
    images,poses,focal=data['images'],data['poses'],float(data['focal'])
    heldout=[10,30,50,70,90]
    train=[i for i in np.linspace(0,len(images)-1,24,dtype=int) if i not in heldout]
    frames=[]
    for split,ids in [('images',train),('test',heldout)]:
        (scene/split).mkdir(exist_ok=True)
        for i in ids:
            name=f'{i:03d}.png'
            image=Image.fromarray((images[i]*255).astype(np.uint8)).resize((size,size),Image.Resampling.LANCZOS)
            image.save(scene/split/name)
            c2w=poses[i].copy();c2w[:3,1:3]*=-1 # NeRF OpenGL -> OpenCV camera axes.
            k=np.array([[focal*size/images.shape[2],0,size/2],[0,focal*size/images.shape[1],size/2],[0,0,1]])
            frames.append(dict(file=name,split=split,c2w=c2w.tolist(),intrinsics=k.tolist()))
    write_json(scene/'scene.json',dict(name='NeRF public tiny Lego',source=source,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),original_resolution=list(images.shape[1:3]),train_frames=[x for x in frames if x['split']=='images'],test_frames=[x for x in frames if x['split']=='test'],gt_depth=False,license_note='Upstream NeRF dataset terms apply. Download locally; not redistributed in this repository.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/root/autodl-tmp/vggt-xr/data');p.add_argument('--size',type=int,default=336);a=p.parse_args()
    root=Path(a.root);check_storage(root,growth_gb=.2)
    prepare_room(root,a.size);print('Analytical panoramas ready',flush=True)
    prepare_natural_room(root,a.size);print('Natural analytical panoramas ready',flush=True)
    prepare_nerf(root,a.size);print('Public NeRF data ready',flush=True)
