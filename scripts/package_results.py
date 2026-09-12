import json,shutil,zipfile,hashlib
from pathlib import Path
from src.storage import write_json,check_storage

root=Path('/root/autodl-tmp/vggt-xr');out=root/'outputs';reports=root/'reports';reports.mkdir(exist_ok=True)
for name in ['metrics.json','metrics.csv','REPORT.md','cuda_smoke.json']:
    shutil.copy2(out/name,reports/name)
records=[]
for d in sorted(out.iterdir()):
    if not (d/'summary.json').exists():continue
    summary=json.loads((d/'summary.json').read_text())
    config=json.loads((d/'config.json').read_text())
    gaussian=d/'gaussian/scene.ply'
    record=dict(run=d.name,config=config,filter=json.loads((d/'geometry/filter.json').read_text()),summary=summary,gaussian_bytes=gaussian.stat().st_size,gaussian_sha256=hashlib.sha256(gaussian.read_bytes()).hexdigest())
    records.append(record)
write_json(reports/'runs.json',records)
write_json(out/'experiment_status.json',[dict(name=x['run'],exit_code=0,steps=x['config']['steps'],complete=True) for x in records])
storage={}
for name,path in [('server_system','/'),('server_data',root)]:
    usage=shutil.disk_usage(path);storage[name]=dict(total_gib=usage.total/2**30,used_gib=usage.used/2**30,free_gib=usage.free/2**30)
storage['root']=str(root);storage['weights_sha256']=(root/'cache/vggt_model.sha256').read_text().strip()
write_json(reports/'storage.json',storage)
check_storage(out/'release',growth_gb=.2)
with zipfile.ZipFile(out/'release/VGGT-XR-demo-assets.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for run in ['kitchen','overlap_anchor']:
        for file in ['gaussian/scene.ply','gaussian/unity_camera.json','gaussian/camera.json','geometry/unity_points.ply','geometry/confidence.ply','geometry/filter.json','config.json','summary.json']:
            archive.write(out/run/file,f'{run}/{file}')
    for folder in ['kitchen','panorama']:
        for p in (out/'demo'/folder).iterdir():
            archive.write(p,'demo/'+folder+'/'+p.name)
    for p in reports.iterdir():archive.write(p,'reports/'+p.name)
    archive.writestr('README.txt','VGGT-XR actual gsplat models and videos. Videos are CUDA gsplat footage, not Unity recordings. Copy scene assets using scripts/desktop_demo.ps1. Unity runtime validation is pending; the Windows installation launch was canceled. Models are inference-derived from the original non-commercial VGGT-1B checkpoint: use for academic/non-commercial demonstration; upstream terms apply. Input RGB datasets and model weights are not included. Kitchen input source: facebookresearch/vggt examples/kitchen; panorama room is our analytic synthetic scene. Original pipeline source is MIT. See reports for raw measurements and protocols.')
print('Release asset archive ready:',out/'release/VGGT-XR-demo-assets.zip')
