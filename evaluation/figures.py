import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/vggt-xr'));a=p.parse_args()
root=a.root;rows=json.loads((root/'outputs/metrics.json').read_text());out=root/'outputs/figures';out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
nerf=[x for x in rows if x['run'].startswith('nerf_') and x['psnr'] is not None]
pano=[x for x in rows if x['run'].startswith(('overlap','cubemap')) and x['psnr'] is not None]
fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
if nerf:
    colors=['#386CB0' if 'raw' in x['run'] else '#2B9A7B' if 'confidence' in x['run'] else '#E1A548' for x in nerf]
    axes[0].bar([x['run'].replace('nerf_','') for x in nerf],[x['psnr'] for x in nerf],color=colors)
    axes[0].tick_params(axis='x',labelrotation=35)
axes[0].set(ylabel='Heldout PSNR (dB)',title='Public Tiny NeRF: initialization')
if pano:
    ids=np.arange(len(pano));axes[1].bar(ids-.18,[x['psnr_horizontal'] for x in pano],width=.36,color='#386CB0',label='8 horizontal views')
    axes[1].bar(ids+.18,[x['psnr_poles'] for x in pano],width=.36,color='#E1A548',label='2 polar views')
    axes[1].set_xticks(ids,[x['run'].replace('overlap_','ov_') for x in pano],rotation=25,ha='right');axes[1].legend(fontsize=8)
axes[1].set(ylabel='Heldout PSNR (dB)',title='Controlled panorama room: coverage')
if pano:
    for x in pano:
        axes[2].scatter(x['depth_coverage'],x['depth_absrel'],s=50,label=x['run'])
    axes[2].legend(fontsize=7)
axes[2].set(xlabel='Retained depth coverage',ylabel='Scale-aligned AbsRel',title='Depth error and retained coverage')
fig.suptitle('VGGT-XR measured prototype results · SH0 · 1500 steps · no densification',fontsize=12)
fig.savefig(out/'metrics.png');fig.savefig(out/'metrics.pdf');plt.close(fig)
for name in ['nerf_raw','nerf_confidence','overlap','cubemap','overlap_fullsphere']:
    folder=root/'outputs'/name/'gaussian/test_renders'
    if not folder.exists():continue
    indices=[0,2,4] if name.startswith('nerf') else [0,4,8]
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,i in zip(axes,indices):
        path=folder/f'comparison_{i:03d}.png'
        if path.exists():ax.imshow(Image.open(path))
        ax.axis('off');ax.set_title(f'Heldout {i}: GT | gsplat')
    fig.suptitle(name+' · actual heldout renders (not Unity footage)')
    fig.savefig(out/f'{name}_heldout.png');plt.close(fig)
print('Measured figures written:',out)
