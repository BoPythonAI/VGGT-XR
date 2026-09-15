"""Apply the frozen selection policy, run three seeds and final-test evaluation."""
import json,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import check_storage,write_json


root=Path('/root/autodl-tmp/vggt-xr');quality=root/'outputs/quality';check_storage(quality,growth_gb=3)
variants=['sh3_fixed','sh3_dense','sh3_dense_pose']
selected={};dev={}
for label in ['nerf','overlap']:
    legacy=json.loads((quality/f'{label}_legacy6000/gaussian/metrics.json').read_text())['heldout']
    candidates=[]
    for rank,variant in enumerate(variants):
        metrics=json.loads((quality/f'{label}_{variant}/gaussian/metrics.json').read_text())['heldout'];dev[f'{label}_{variant}']=metrics
        if metrics['ssim_mean']>=legacy['ssim_mean']-.005 and metrics['lpips_squeeze_mean']<=legacy['lpips_squeeze_mean']+.005:
            candidates.append((metrics['psnr_mean'], -metrics['lpips_squeeze_mean'], -rank,variant))
    if not candidates:raise RuntimeError(f'No successful quality candidate for {label}; keep legacy and do not tune on final test')
    candidates.sort(reverse=True)
    best=candidates[0]
    near=[row for row in candidates if best[0]-row[0]<.05]
    selected[label]=max(near,key=lambda row:(row[1],row[2]))[3]
write_json(root/'reports/quality_selection.json',dict(policy='docs/QUALITY_SELECTION_POLICY.zh-CN.md',selected=selected,development=dev))
for label,variant in selected.items():
    base='nerf_confidence_masked' if label=='nerf' else 'overlap_anchor'
    test_scene=root/'data/quality_final_test'/('tiny_nerf' if label=='nerf' else 'analytic_room')
    seed42=quality/f'{label}_{variant}'
    subprocess.run([sys.executable,'scripts/evaluate_quality.py','--run',str(seed42),'--test-scene',str(test_scene)],cwd=root,check=True)
    extra=[]
    if variant=='sh3_fixed':extra=['--no-densify']
    elif variant=='sh3_dense_pose':extra=['--pose-opt']
    for seed in [43,44]:
        target=quality/f'{label}_{variant}_seed{seed}'
        command=[sys.executable,'scripts/improve_quality.py','--base',str(root/'outputs'/base),'--output',str(target),'--steps','6000','--cap','120000','--seed',str(seed),'--test-scene',str(test_scene),*extra]
        log=root/'logs'/f'quality_{target.name}.log';start=time.time()
        with log.open('w') as f:process=subprocess.run(command,cwd=root,stdout=f,stderr=subprocess.STDOUT)
        if process.returncode:raise RuntimeError(f'{target.name} failed: {log}')
summary={}
for label,variant in selected.items():
    runs=[quality/f'{label}_{variant}',quality/f'{label}_{variant}_seed43',quality/f'{label}_{variant}_seed44']
    rows=[]
    for run in runs:
        final=json.loads((run/'gaussian/final_test.json').read_text()) if (run/'gaussian/final_test.json').exists() else json.loads((run/'gaussian/metrics.json').read_text())['heldout']
        rows.append({k:final[k] for k in ['psnr_mean','ssim_mean','lpips_squeeze_mean']})
    summary[label]=dict(variant=variant,seeds=[42,43,44],runs=[str(x) for x in runs],per_seed=rows,mean={k:float(np.mean([x[k] for x in rows])) for k in rows[0]},std_sample={k:float(np.std([x[k] for x in rows],ddof=1)) for k in rows[0]})
    legacy=json.loads((quality/f'{label}_legacy6000/gaussian/final_test.json').read_text())
    summary[label]['legacy_seed42']={k:legacy[k] for k in ['psnr_mean','ssim_mean','lpips_squeeze_mean']}
write_json(root/'reports/quality_final_test.json',summary)
print('FINAL QUALITY COMPLETE',json.dumps(summary),flush=True)
