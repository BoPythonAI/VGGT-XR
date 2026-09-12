"""Verify actual artifacts and protocol metadata without rerunning GPU experiments."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import pycolmap
from src.storage import write_json

root=Path('/root/autodl-tmp/vggt-xr');checks=[]
for d in sorted((root/'outputs').iterdir()):
    if not (d/'summary.json').exists():continue
    rec=pycolmap.Reconstruction(str(d/'sparse/0'))
    p=np.load(d/'vggt/predictions.npz');config=json.loads((d/'config.json').read_text());stats=json.loads((d/'geometry/filter.json').read_text())
    assert rec.num_images()==len(p['images'])
    assert rec.num_points3D()==stats['exported_points']
    assert np.isfinite(p['depth']).all() and np.isfinite(p['extrinsics']).all()
    trained=torch.load(d/'gaussian/checkpoint.pt',weights_only=True,map_location='cpu')
    assert all(torch.isfinite(v).all() for v in trained.values())
    assert len(trained['means'])==stats['exported_points']
    fingerprint=hashlib.sha256()
    names=json.loads((d/'vggt/input.json').read_text())['images']
    for name in names:fingerprint.update((d/'images'/name).read_bytes())
    assert fingerprint.hexdigest()==json.loads((d/'vggt/input.json').read_text())['sha256']
    comparison='raw: no threshold' if config['filter']=='raw' else stats.get('threshold_comparison','legacy inclusive')
    assert comparison!='legacy inclusive','Re-run this filtered result with the final threshold semantics'
    item=dict(run=d.name,images=rec.num_images(),points=rec.num_points3D(),finite_predictions=True,finite_gaussians=True,input_fingerprint=True,filter_comparison=comparison,heldout_count=json.loads((d/'gaussian/metrics.json').read_text()).get('heldout',{}).get('count',0))
    checks.append(item);print(item,flush=True)
write_json(root/'reports/artifact_verification.json',checks)
