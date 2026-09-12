import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from src.storage import check_storage,write_json
from src.panorama import project,layout,constrain_poses
from src.geometry import unproject,filter_points
from src.inference import infer
from src.export import write_ply,export_colmap
from evaluation.metrics import align_cameras,test_camera_in_reconstruction,depth_metrics


def main():
    p=argparse.ArgumentParser(description='VGGT -> panorama adapter -> filtering -> COLMAP -> gsplat -> Unity PLY')
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--panorama',action='store_true');p.add_argument('--projection',choices=['direct','cubemap','overlap'],default='overlap')
    p.add_argument('--constrain-poses',action='store_true');p.add_argument('--filter',choices=['raw','confidence','consistency'],default='confidence')
    p.add_argument('--include-poles',action='store_true',help='Add up/down views to the 8 horizontal overlap faces; use max-views>=30 for 3 panoramas')
    p.add_argument('--pose-fusion',choices=['mean','anchor'],default='mean')
    p.add_argument('--known-intrinsics',action='store_true',help='Use exact adapter pinhole intrinsics; never GT scene poses/depth')
    p.add_argument('--mask-black-background',action='store_true',help='Synthetic black-backdrop datasets only: exclude input RGB max<=0.03 identically for all filtering variants')
    p.add_argument('--confidence-quantile',type=float,default=.35);p.add_argument('--size',type=int,default=336);p.add_argument('--max-views',type=int,default=24)
    p.add_argument('--max-points',type=int,default=30000);p.add_argument('--steps',type=int,default=1500);p.add_argument('--geometry-only',action='store_true')
    a=p.parse_args();a.input=a.input.resolve();a.output=a.output.resolve();check_storage(a.output,growth_gb=1)
    if not 0<=a.confidence_quantile<1:raise ValueError('quantile must lie in [0,1)')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config_file=a.output/'config.json'
    if config_file.exists():
        previous=json.loads(config_file.read_text())
        for key in ['input','output']:
            if key in previous:previous[key]=str(Path(previous[key]).resolve())
        legacy_defaults=dict(include_poles=False,pose_fusion='mean',known_intrinsics=False,mask_black_background=False)
        for key,value in config.items():
            if key=='geometry_only':continue
            if previous.get(key,legacy_defaults.get(key,value))!=value:
                raise ValueError(f'Cached run setting {key} differs. Use a new output directory; existing results are preserved.')
    write_json(config_file,config)
    metadata=json.loads((a.input/'scene.json').read_text()) if (a.input/'scene.json').exists() else {}
    images_dir=a.output/'images';images_dir.mkdir(exist_ok=True)
    gt_c2w=[];gt_depth=[];groups=[];rs=[];names=[];adapter_k=[]
    print('[1/5] Preparing perspective views',flush=True)
    if a.panorama:
        folder=a.input/'panoramas' if (a.input/'panoramas').exists() else a.input
        originals=sorted([f for f in folder.iterdir() if f.suffix.lower() in ['.jpg','.jpeg','.png']])
        spec=[(0,0,None)] if a.projection=='direct' else layout(a.projection)
        if a.include_poles:
            if a.projection!='overlap':raise ValueError('include-poles applies only to overlap projection')
            spec += [(0,90,100),(0,-90,100)]
        if len(originals)*len(spec)>a.max_views:raise ValueError('Too many projected views. Select fewer panoramas.')
        for i,f in enumerate(originals):
            erp=np.asarray(Image.open(f).convert('RGB'))
            if abs(erp.shape[1]/erp.shape[0]-2)>.03:raise ValueError(f'Expected 2:1 ERP panorama: {f}')
            origin=np.asarray(metadata['panorama_origins'][i]) if 'panorama_origins' in metadata else None
            range_path=f.with_name(f.stem+'_range.npy')
            radial=np.load(range_path) if range_path.exists() else None
            for j,(yaw,pitch,fov) in enumerate(spec):
                if fov is None:
                    rgb=np.asarray(Image.fromarray(erp).resize((a.size,a.size),Image.Resampling.LANCZOS));k=None;r=np.eye(3)
                else:rgb,k,r=project(erp,yaw,pitch,fov,a.size)
                name=f'p{i:02d}_v{j:02d}.png';names.append(name);groups.append(i);rs.append(r)
                if k is not None:adapter_k.append(k)
                Image.fromarray(rgb.astype(np.uint8)).save(images_dir/name)
                if origin is not None and fov is not None:
                    c2w=np.eye(4,dtype=np.float32);c2w[:3,:3]=r;c2w[:3,3]=origin;gt_c2w.append(c2w)
                    if radial is not None:
                        from src.panorama import perspective_rays
                        rays,_=perspective_rays(a.size,fov)
                        if metadata.get('kind')=='controlled_synthetic':
                            from scripts.prepare_data import trace
                            _,exact_depth=trace(origin,rays@r.T);gt_depth.append(exact_depth)
                        else:
                            projected_range,_,_=project(radial,yaw,pitch,fov,a.size)
                            gt_depth.append(projected_range/np.linalg.norm(rays,axis=-1))
    else:
        folder=a.input/'images' if (a.input/'images').exists() else a.input
        files=sorted([f for f in folder.iterdir() if f.suffix.lower() in ['.jpg','.jpeg','.png']])
        for f in files:
            names.append(f.name);ImageOps.pad(Image.open(f).convert('RGB'),(a.size,a.size),method=Image.Resampling.LANCZOS,color=(255,255,255)).save(images_dir/f.name)
        if 'train_frames' in metadata:
            table={x['file']:x for x in metadata['train_frames']}
            gt_c2w=[np.asarray(table[n]['c2w']) for n in names]
    if not names:raise ValueError('No input images found')
    print('[2/5] VGGT inference',flush=True)
    pred=infer([images_dir/n for n in names],a.output/'vggt',a.size,a.max_views)
    if a.known_intrinsics:
        if len(adapter_k)!=len(names):raise ValueError('Known intrinsics require perspective ERP projection')
        pred=dict(pred);pred['intrinsics']=np.asarray(adapter_k)
    extrinsics=pred['extrinsics'].copy()
    if a.constrain_poses:
        if not a.panorama or a.projection=='direct':raise ValueError('Pose constraint needs projected panorama groups')
        extrinsics=constrain_poses(extrinsics,groups,np.asarray(rs),a.pose_fusion)
    points=unproject(pred['depth'],extrinsics,pred['intrinsics'])
    print('[3/5] Filtering geometry',flush=True)
    valid_pixels=pred['images'].max(-1)>.03 if a.mask_black_background else None
    xyz,rgb,conf,scales,mask,stats=filter_points(points,pred['images'],pred['confidence'],pred['depth'],extrinsics,pred['intrinsics'],a.filter,a.confidence_quantile,a.max_points,groups=groups if a.panorama else None,valid_mask=valid_pixels)
    stats['black_backdrop_mask']=a.mask_black_background
    write_json(a.output/'geometry/filter.json',stats)
    write_ply(a.output/'geometry/points.ply',xyz,rgb,conf)
    low,high=np.percentile(conf,[5,95]);normalized=((conf-low)/max(high-low,1e-6)).clip(0,1)
    confidence_colors=np.stack([1-normalized,normalized,np.zeros_like(normalized)],-1)
    write_ply(a.output/'geometry/confidence.ply',xyz,confidence_colors,normalized)
    write_ply(a.output/'geometry/unity_points.ply',xyz,rgb,normalized)
    print('[4/5] Exporting COLMAP and evaluation',flush=True)
    export_colmap(a.output/'sparse/0',names,extrinsics,pred['intrinsics'],(xyz,rgb),pred['images'])
    eval_data={};test=None
    if gt_c2w:
        alignment,cameras=align_cameras(extrinsics,np.asarray(gt_c2w));eval_data['cameras']=cameras
        if gt_depth:
            gt=np.stack(gt_depth);valid=np.isfinite(pred['depth'])&(pred['depth']>0)
            # Fixed full-depth scale for raw/confidence/consistency comparisons.
            scale=float(np.median(gt[valid]/pred['depth'][valid]));eval_data['depth_all']=depth_metrics(pred['depth'],gt,valid,scale);eval_data['depth_retained']=depth_metrics(pred['depth'],gt,mask,scale)
        if metadata.get('test_frames'):
            test_frames=metadata['test_frames'];gt_test=np.asarray([x['c2w'] for x in test_frames]);test_k=np.asarray([x['intrinsics'] for x in test_frames])
            es,ks=test_camera_in_reconstruction(gt_test,test_k,alignment)
            imgs=np.stack([np.asarray(Image.open(a.input/'test'/x['file']).convert('RGB'),np.float32)/255 for x in test_frames])
            test=(es,ks,imgs)
    elif a.panorama and a.projection=='direct':eval_data['note']='ERP has no pinhole GT correspondence. Direct ERP geometry has no valid pinhole depth/pose metric; report only runtime and training appearance.'
    write_json(a.output/'evaluation.json',eval_data)
    print('[5/5] Gaussian Splatting and Unity export',flush=True)
    if not a.geometry_only:
        from src.gaussian import train
        gs=train(xyz,rgb,scales,pred['images'],extrinsics,pred['intrinsics'],a.output/'gaussian',a.steps,test)
        eval_data['gaussian']=gs
    write_json(a.output/'summary.json',eval_data)
    print(f'Done: {a.output}',flush=True)


if __name__=='__main__':main()
