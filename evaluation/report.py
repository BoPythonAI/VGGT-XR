import argparse,json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/vggt-xr'));a=p.parse_args()
rows=[]
for d in sorted((a.root/'outputs').iterdir()):
    if not (d/'summary.json').exists():continue
    x=json.loads((d/'summary.json').read_text());runtime=json.loads((d/'vggt/runtime.json').read_text());f=json.loads((d/'geometry/filter.json').read_text())
    cams=x.get('cameras',{});depth=x.get('depth_retained',{});gs=x.get('gaussian',{});heldout=gs.get('heldout',{})
    raw_depth=x.get('depth_all',{})
    per_view=heldout.get('per_view',[])
    horizontal=per_view[:8] if d.name.startswith(('overlap','cubemap')) else []
    poles=per_view[8:] if d.name.startswith(('overlap','cubemap')) else []
    def mean(items,key):return sum(x[key] for x in items)/len(items) if items else None
    rows.append(dict(run=d.name,views=runtime['views'],vggt_seconds=runtime['seconds'],vggt_vram_gb=runtime['peak_vram_gb'],retained=f['exported_points'],ate=cams.get('ate_sim3_rmse'),rotation=cams.get('rotation_error_deg_mean'),depth_raw_absrel=raw_depth.get('absrel'),depth_rmse=depth.get('rmse'),depth_absrel=depth.get('absrel'),depth_coverage=depth.get('coverage'),psnr=heldout.get('psnr_mean'),psnr_horizontal=mean(horizontal,'psnr'),psnr_poles=mean(poles,'psnr'),ssim=heldout.get('ssim_mean'),lpips_squeeze=heldout.get('lpips_squeeze_mean'),train_seconds=gs.get('train_seconds'),raster_fps=gs.get('server_raster_fps')))
out=a.root/'outputs';(out/'metrics.json').write_text(json.dumps(rows,indent=2))
import csv
if rows:
    with open(out/'metrics.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def fmt(x):return 'N/A' if x is None else f'{x:.4f}' if isinstance(x,float) else str(x)
cols=['run','views','retained','ate','depth_absrel','depth_coverage','psnr','ssim','lpips_squeeze','vggt_vram_gb','raster_fps']
text='# Actual experiment results\n\n'+' | '.join(cols)+'\n'+' | '.join(['---']*len(cols))+'\n'
text+='\n'.join(' | '.join(fmt(row[k]) for k in cols) for row in rows)
text+='\n\nProtocols: NeRF source is the public 100x100 Tiny NeRF dataset enlarged to 336; do not interpret as full-resolution Lego benchmark. Analytical room is a controlled synthetic box scene, not Replica/Structured3D or a real room. Heldout images are excluded from VGGT and GS optimization. Sim(3) alignment uses training cameras only. Depth scale is fixed across filtering ablations; coverage is reported beside error. LPIPS uses SqueezeNet. Gaussian optimizer uses SH0, fixed Gaussian count, no densification. ERP-direct does not admit pinhole GT depth or camera evaluation. Raster FPS is headless server gsplat, not Unity.\n'
(out/'REPORT.md').write_text(text)
print(text)
