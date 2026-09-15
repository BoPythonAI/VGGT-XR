"""Bounded first quality ablations. Separate output paths and durable logs."""
import json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import check_storage,write_json


root=Path('/root/autodl-tmp/vggt-xr');output=root/'outputs/quality';check_storage(output,growth_gb=5)
tasks=[]
for source,label in [('nerf_confidence_masked','nerf'),('overlap_anchor','overlap')]:
    base=root/'outputs'/source
    config=json.loads((base/'config.json').read_text());legacy=output/f'{label}_legacy6000'
    (legacy/'vggt').parent.mkdir(parents=True,exist_ok=True)
    if not (legacy/'vggt').exists():(legacy/'vggt').symlink_to(base/'vggt',target_is_directory=True)
    command=[sys.executable,'run.py']
    for key,value in config.items():
        if key in ['output','steps','geometry_only']:continue
        flag='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:command.append(flag)
        else:command.extend([flag,str(value)])
    command.extend(['--output',str(legacy),'--steps','6000'])
    tasks.append((legacy.name,command))
    for variant,extra in [('sh3_fixed',['--no-densify']),('sh3_dense',[]),('sh3_dense_pose',['--pose-opt'])]:
        target=output/f'{label}_{variant}'
        tasks.append((target.name,[sys.executable,'scripts/improve_quality.py','--base',str(base),'--output',str(target),'--steps','6000','--cap','120000',*extra]))
tasks.append(('kitchen518_base',[sys.executable,'run.py','--input','data/vggt_kitchen','--output',str(output/'kitchen518_base'),'--size','518','--max-points','60000','--steps','3000','--geometry-only']))
tasks.append(('kitchen518_ba',[sys.executable,'scripts/refine_geometry.py','--base',str(output/'kitchen518_base'),'--output',str(output/'kitchen518_ba')]))
tasks.append(('kitchen_highres',[sys.executable,'scripts/improve_quality.py','--base',str(output/'kitchen518_base'),'--output',str(output/'kitchen_highres'),'--steps','12000','--train-size','672','--cap','200000']))
records=[]
for name,command in tasks:
    check_storage(output,growth_gb=1);start=time.time();print('START',name,flush=True)
    log=root/'logs'/f'quality_{name}.log';log.parent.mkdir(exist_ok=True)
    with log.open('w') as f:process=subprocess.run(command,cwd=root,stdout=f,stderr=subprocess.STDOUT)
    row=dict(name=name,command=command,exit_code=process.returncode,seconds=time.time()-start,log=str(log));records.append(row)
    write_json(root/'reports/quality_execution.json',records);print('DONE',json.dumps(row),flush=True)
    if process.returncode and name!='kitchen518_ba':raise SystemExit(process.returncode)
print('QUALITY FIRST STAGE COMPLETE',flush=True)
