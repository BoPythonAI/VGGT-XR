"""Evaluate a completed quality checkpoint on a frozen test manifest."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation.metrics import align_cameras,test_camera_in_reconstruction
from src.panorama import layout,rotation
from src.quality import image_metrics
from src.storage import write_json


p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--test-scene',type=Path,required=True);a=p.parse_args()
if (a.run/'quality_source.json').exists():
    source_info=json.loads((a.run/'quality_source.json').read_text());config=source_info['base_config'];base=Path(source_info['base'])
else:
    config=json.loads((a.run/'config.json').read_text());base=a.run
source=Path(config['input']);names=json.loads((base/'vggt/input.json').read_text())['images']
metadata=json.loads((source/'scene.json').read_text());groups=None
if config['panorama']:
    spec=layout(config['projection'])
    if config.get('include_poles'):spec=spec+[(0,90,100),(0,-90,100)]
    groups=np.repeat(np.arange(len(names)//len(spec)),len(spec));rs=np.tile(np.stack([rotation(y,pitch) for y,pitch,_ in spec]),(len(names)//len(spec),1,1))
    gt_train=np.tile(np.eye(4,dtype=np.float32),(len(names),1,1));gt_train[:,:3,:3]=rs;gt_train[:,:3,3]=np.asarray(metadata['panorama_origins'])[groups]
else:
    table={x['file']:x for x in metadata['train_frames']};gt_train=np.asarray([table[n]['c2w'] for n in names])
cameras_path=a.run/'gaussian/training_cameras.npz'
cameras=np.load(cameras_path if cameras_path.exists() else base/'vggt/predictions.npz');final_e=cameras['extrinsics']
alignment,_=align_cameras(final_e[:,:3],gt_train)
test_metadata=json.loads((a.test_scene/'scene.json').read_text());frames=test_metadata['test_frames']
test_gt=np.asarray([x['c2w'] for x in frames]);test_k=np.asarray([x['intrinsics'] for x in frames]);test_images=np.stack([np.asarray(Image.open(a.test_scene/'test'/x['file']).convert('RGB'),np.float32)/255 for x in frames])
ee,kk=test_camera_in_reconstruction(test_gt,test_k,alignment)
params={k:v.cuda() for k,v in torch.load(a.run/'gaussian/checkpoint.pt',weights_only=True).items()}
camera=json.loads((a.run/'gaussian/camera.json').read_text())
result=image_metrics(params,ee,kk,test_images,a.run/'gaussian/final_test_renders',antialiased=camera.get('antialiased',False),background=camera.get('background'),perceptual=True)
result.update(protocol='Frozen new positions/images; no input to VGGT, configuration selection or optimization',test_scene=str(a.test_scene))
write_json(a.run/'gaussian/final_test.json',result);print(json.dumps(result),flush=True)
