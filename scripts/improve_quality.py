"""Train a quality variant from a verified frozen-VGGT run, preserving all baselines."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import check_storage,write_json
from src.panorama import layout,rotation,constrain_poses
from src.geometry import unproject,filter_points
from src.export import write_ply,export_colmap
from src.quality import train_quality
from evaluation.metrics import align_cameras,test_camera_in_reconstruction,depth_metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=12000);p.add_argument('--sh-degree',type=int,default=3)
    p.add_argument('--no-densify',action='store_true');p.add_argument('--pose-opt',action='store_true');p.add_argument('--antialiased',action='store_true')
    p.add_argument('--train-size',type=int);p.add_argument('--max-points',type=int);p.add_argument('--cap',type=int,default=200000)
    p.add_argument('--geometry',choices=['confidence','consistency']);p.add_argument('--ba',type=Path,help='Training-only refined camera NPZ, if geometrically validated')
    p.add_argument('--depth-weight',type=float,default=0.);p.add_argument('--max-scale-ratio',type=float,default=2.)
    p.add_argument('--max-anisotropy',type=float,default=1e6);p.add_argument('--scale-reg-weight',type=float,default=0.)
    p.add_argument('--sh-start-fraction',type=float,default=0.);p.add_argument('--sh-lr-multiplier',type=float,default=1.)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--test-scene',type=Path,help='Previously frozen final test images/poses; never training inputs')
    a=p.parse_args();a.base=a.base.resolve();a.output=a.output.resolve();check_storage(a.output,growth_gb=1)
    config=json.loads((a.base/'config.json').read_text());source=Path(config['input'])
    pred_path=a.base/'vggt/predictions.npz';pred=dict(np.load(pred_path))
    names=json.loads((a.base/'vggt/input.json').read_text())['images']
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    settings['prediction_sha256']=hashlib.sha256(pred_path.read_bytes()).hexdigest()
    settings['base_config']=config
    if (a.output/'quality_source.json').exists() and json.loads((a.output/'quality_source.json').read_text())!=settings:
        raise ValueError('Input or quality variant differs from cached results; use another directory')
    write_json(a.output/'quality_source.json',settings)
    es=pred['extrinsics'];ks=pred['intrinsics'];groups=None
    if config['panorama']:
        spec=layout(config['projection'])
        if config.get('include_poles'):spec=spec+[(0,90,100),(0,-90,100)]
        groups=np.repeat(np.arange(len(es)//len(spec)),len(spec))
        rs=np.tile(np.stack([rotation(y,pitch) for y,pitch,_ in spec]),(len(es)//len(spec),1,1))
        if config.get('constrain_poses'):es=constrain_poses(es,groups,rs,config.get('pose_fusion','mean'))
        if config.get('known_intrinsics'):
            from src.panorama import perspective_rays
            ks=np.tile(np.stack([perspective_rays(config['size'],fov)[1] for _,_,fov in spec]),(len(es)//len(spec),1,1))
    if a.ba:
        if config['panorama']:raise ValueError('Independent pinhole BA is forbidden for ERP rigs')
        cameras=np.load(a.ba);es=cameras['extrinsics'];ks=cameras['intrinsics']
    train_size=a.train_size or config['size']
    images=pred['images'];depths=pred['depth'];depth_confidence=pred['confidence'];masks=None;valid=None
    if not config['panorama']:
        folder=source/'images' if (source/'images').exists() else source
        originals=[Image.open(folder/n).convert('RGB') for n in names]
        masks=np.stack([np.asarray(ImageOps.pad(Image.new('L',im.size,255),(train_size,train_size),method=Image.Resampling.NEAREST,color=0))>0 for im in originals])
        valid=np.stack([np.asarray(ImageOps.pad(Image.new('L',im.size,255),(config['size'],config['size']),method=Image.Resampling.NEAREST,color=0))>0 for im in originals])
        if all(im.width==im.height for im in originals):masks=None;valid=None
        images=np.stack([np.asarray(ImageOps.pad(im,(train_size,train_size),method=Image.Resampling.LANCZOS,color=(255,255,255)),np.float32)/255 for im in originals])
    elif train_size!=config['size']:
        raise ValueError('Panorama training resolution changes need fresh ERP projection/inference, not upsampling')
    if train_size!=config['size']:
        resize=lambda a:np.stack([np.asarray(Image.fromarray(x.astype(np.float32)).resize((train_size,train_size),Image.Resampling.BILINEAR),np.float32) for x in a])
        depths=resize(depths);depth_confidence=resize(depth_confidence)
    if config.get('mask_black_background'):
        black=pred['images'].max(-1)>.03;valid=black if valid is None else valid&black
    points=unproject(pred['depth'],es,ks)
    xyz,rgb,conf,scales,mask,stats=filter_points(points,pred['images'],pred['confidence'],pred['depth'],es,ks,a.geometry or config['filter'],config['confidence_quantile'],a.max_points or config['max_points'],groups=groups,valid_mask=valid)
    write_json(a.output/'geometry/filter.json',stats)
    write_ply(a.output/'geometry/points.ply',xyz,rgb,conf)
    low,high=np.percentile(conf,[5,95]);normalized=((conf-low)/max(high-low,1e-6)).clip(0,1)
    write_ply(a.output/'geometry/unity_points.ply',xyz,rgb,normalized)
    write_ply(a.output/'geometry/confidence.ply',xyz,np.stack([1-normalized,normalized,np.zeros_like(normalized)],-1),normalized)
    scaled_k=ks.copy();scaled_k[:,:2]*=train_size/config['size']
    if not config['panorama'] and valid is not None and masks is not None:
        for i,(old_mask,new_mask) in enumerate(zip(valid,masks)):
            oy,ox=np.where(old_mask);ny,nx=np.where(new_mask)
            sx=(nx.max()-nx.min()+1)/(ox.max()-ox.min()+1)
            sy=(ny.max()-ny.min()+1)/(oy.max()-oy.min()+1)
            affine=np.array([[sx,0,nx.min()-sx*ox.min()],[0,sy,ny.min()-sy*oy.min()],[0,0,1]],np.float32)
            scaled_k[i]=affine@ks[i]
    # Source prediction tensors are immutable and reused by reference; own cameras
    # are exported separately and the renderer prefers training_cameras.npz.
    (a.output/'vggt').mkdir(exist_ok=True)
    if not (a.output/'vggt/predictions.npz').exists():(a.output/'vggt/predictions.npz').symlink_to(pred_path)
    metadata=json.loads((source/'scene.json').read_text()) if (source/'scene.json').exists() else {}
    gt_train=None;test_factory=None
    if config['panorama'] and 'panorama_origins' in metadata:
        gt_train=np.tile(np.eye(4,dtype=np.float32),(len(es),1,1));gt_train[:,:3,:3]=rs;gt_train[:,:3,3]=np.asarray(metadata['panorama_origins'])[groups]
    elif 'train_frames' in metadata:
        table={x['file']:x for x in metadata['train_frames']};gt_train=np.asarray([table[n]['c2w'] for n in names])
    test_metadata=metadata;test_source=source
    if a.test_scene:
        test_source=a.test_scene.resolve();test_metadata=json.loads((test_source/'scene.json').read_text())
    if gt_train is not None and test_metadata.get('test_frames'):
        frames=test_metadata['test_frames'];test_gt=np.asarray([x['c2w'] for x in frames]);test_k=np.asarray([x['intrinsics'] for x in frames])
        test_images=np.stack([np.asarray(Image.open(test_source/'test'/x['file']).convert('RGB'),np.float32)/255 for x in frames])
        def test_factory(final_es):
            alignment,_=align_cameras(final_es,gt_train)
            ee,kk=test_camera_in_reconstruction(test_gt,test_k,alignment)
            return ee,kk,test_images
    gs=train_quality(xyz,rgb,images,es,scaled_k,a.output/'gaussian',steps=a.steps,sh_degree=a.sh_degree,densify=not a.no_densify,cap=a.cap,pose_opt=a.pose_opt,groups=groups,masks=masks,antialiased=a.antialiased,test_factory=test_factory,seed=a.seed,depths=depths,depth_confidence=depth_confidence,depth_weight=a.depth_weight,max_scale_ratio=a.max_scale_ratio,max_anisotropy=a.max_anisotropy,scale_reg_weight=a.scale_reg_weight,sh_start_fraction=a.sh_start_fraction,sh_lr_multiplier=a.sh_lr_multiplier)
    cameras=np.load(a.output/'gaussian/training_cameras.npz');final_es=cameras['extrinsics'][:,:3]
    evaluation={}
    if gt_train is not None:_,evaluation['cameras']=align_cameras(final_es,gt_train)
    export_colmap(a.output/'sparse/0',names,final_es,scaled_k,(xyz,rgb),images)
    evaluation['gaussian']=gs;write_json(a.output/'summary.json',evaluation)
    print('QUALITY COMPLETE',a.output,flush=True)


if __name__=='__main__':main()
