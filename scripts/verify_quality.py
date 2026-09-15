"""Fail closed on quality artifacts before reporting or packaging."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


root=Path('/root/autodl-tmp/vggt-xr');quality=root/'outputs/quality';checked=[]
for run in quality.glob('*'):
    metrics_path=run/'gaussian/metrics.json'
    if not metrics_path.exists():continue
    metrics=json.loads(metrics_path.read_text())
    if 'cap' in metrics:assert 64<=metrics['gaussian_count']<=metrics['cap']
    checkpoint=torch.load(run/'gaussian/checkpoint.pt',weights_only=True)
    assert len({len(x) for x in checkpoint.values()})==1
    assert all(torch.isfinite(x).all() for x in checkpoint.values())
    if 'sh_degree' in metrics and metrics['sh_degree']>0:
        assert checkpoint['shN'].abs().sum()>0
    camera=json.loads((run/'gaussian/camera.json').read_text());assert camera['width']>0 and camera['height']>0
    if (run/'gaussian/training_cameras.npz').exists():
        c=np.load(run/'gaussian/training_cameras.npz');assert np.isfinite(c['extrinsics']).all() and np.isfinite(c['intrinsics']).all()
        if run.name.startswith('overlap_'):
            centers=-np.einsum('nji,nj->ni',c['extrinsics'][:,:3,:3],c['extrinsics'][:,:3,3])
            for group in np.unique(c['groups']):assert np.ptp(centers[c['groups']==group],axis=0).max()<1e-4
    checked.append(run.name)
manifest=json.loads((root/'data/quality_final_test/manifest.json').read_text())
for name,digest in manifest['files'].items():assert hashlib.sha256((root/'data/quality_final_test'/name).read_bytes()).hexdigest()==digest
assert set(manifest['tiny_test_indices']).isdisjoint({10,30,50,70,90})
selection=json.loads((root/'reports/quality_selection.json').read_text());assert set(selection['selected'])=={'nerf','overlap'}
final=json.loads((root/'reports/quality_final_test.json').read_text());assert all(len(x['per_seed'])==3 for x in final.values())
write={'status':'passed','quality_runs':sorted(checked),'checks':['finite checkpoints','hard Gaussian caps','nonzero SH','finite cameras','panorama shared centers','frozen test hashes','three complete seeds']}
(root/'reports/quality_verification.json').write_text(json.dumps(write,indent=2))
print(json.dumps(write),flush=True)
