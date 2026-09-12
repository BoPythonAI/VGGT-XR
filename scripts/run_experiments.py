"""Sequential GPU runs with recoverable outputs and explicit failure records."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import write_json,check_storage


def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1500);p.add_argument('--size',type=int,default=336);p.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/vggt-xr'));a=p.parse_args()
    os.chdir(a.root);runs=[]
    jobs=[('nerf_raw','tiny_nerf',[],'raw'),('nerf_confidence','tiny_nerf',[],'confidence'),('nerf_consistency','tiny_nerf',[],'consistency'),('erp_direct','analytic_room',['--panorama','--projection','direct'],'raw'),('cubemap','analytic_room',['--panorama','--projection','cubemap'],'confidence'),('overlap','analytic_room',['--panorama','--projection','overlap'],'confidence'),('overlap_constrained','analytic_room',['--panorama','--projection','overlap','--constrain-poses'],'confidence'),('overlap_consistency','analytic_room',['--panorama','--projection','overlap'],'consistency')]
    for name,scene,options,filt in jobs:
        output=a.root/'outputs'/name
        check_storage(output,growth_gb=1)
        # Reuse exact VGGT predictions across filtering or constraint ablations.
        reuse='nerf_raw' if scene=='tiny_nerf' and name!='nerf_raw' else 'overlap' if name.startswith('overlap_') else None
        if reuse:
            source=a.root/'outputs'/reuse/'vggt'
            target=output/'vggt'
            if source.exists() and not target.exists():target.symlink_to(source,target_is_directory=True)
        cmd=[sys.executable,'run.py','--input',str(a.root/'data'/scene),'--output',str(output),'--filter',filt,'--steps',str(a.steps),'--size',str(a.size),*options]
        print('RUN',name,flush=True);start=time.perf_counter()
        with open(a.root/'logs'/f'{name}.log','w') as log:
            completed=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        item=dict(name=name,exit_code=completed.returncode,wall_seconds=time.perf_counter()-start,command=cmd)
        if (output/'summary.json').exists():item['summary']=json.loads((output/'summary.json').read_text())
        else:item['error_tail']=(a.root/'logs'/f'{name}.log').read_text()[-4000:]
        runs.append(item);write_json(a.root/'outputs/experiment_status.json',runs)
        print('FINISHED',name,item['exit_code'],flush=True)
    subprocess.run([sys.executable,'evaluation/report.py','--root',str(a.root)],check=True)
    if any(x['exit_code'] for x in runs):sys.exit(1)


if __name__=='__main__':main()
