"""Actual SIFT-track triangulation + robust BA from VGGT pinhole initialization.

This is a bounded geometry experiment, not the upstream VGGT track-based BA.
Panorama rigs are deliberately rejected rather than optimized independently.
"""
import argparse,json,sqlite3,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import check_storage,write_json
from evaluation.metrics import sim3,camera_centers


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.base=a.base.resolve();a.output=a.output.resolve();check_storage(a.output,growth_gb=.3)
    config=json.loads((a.base/'config.json').read_text())
    if config['panorama']:raise ValueError('This BA branch does not support panorama rigs')
    if (a.output/'cameras.npz').exists():raise ValueError('BA output already exists; preserve it and choose another directory')
    import pycolmap
    start=time.perf_counter();pred=np.load(a.base/'vggt/predictions.npz');names=json.loads((a.base/'vggt/input.json').read_text())['images']
    db=a.output/'database.db';image_path=a.base/'images'
    extraction=pycolmap.FeatureExtractionOptions();extraction.num_threads=8;extraction.max_image_size=config['size'];extraction.sift.max_num_features=4096
    reader=pycolmap.ImageReaderOptions();reader.camera_model='PINHOLE'
    pycolmap.extract_features(db,image_path,image_names=names,camera_mode=pycolmap.CameraMode.PER_IMAGE,reader_options=reader,extraction_options=extraction,device=pycolmap.Device.cpu)
    matching=pycolmap.FeatureMatchingOptions();matching.num_threads=8;matching.guided_matching=True
    pycolmap.match_exhaustive(db,matching_options=matching,device=pycolmap.Device.cpu)
    with sqlite3.connect(db) as connection:
        database_images=connection.execute('SELECT image_id,name,camera_id FROM images').fetchall()
    by_name={n:(iid,cid) for iid,n,cid in database_images}
    initial=a.output/'initial';initial.mkdir(exist_ok=True)
    cameras=[];images=[]
    for i,name in enumerate(names):
        iid,cid=by_name[name];e=pred['extrinsics'][i];k=pred['intrinsics'][i];q=Rotation.from_matrix(e[:3,:3]).as_quat()[[3,0,1,2]]
        cameras.append(f'{cid} PINHOLE {config["size"]} {config["size"]} {k[0,0]} {k[1,1]} {k[0,2]} {k[1,2]}')
        images.extend([f'{iid} '+ ' '.join(map(str,[*q,*e[:3,3]]))+f' {cid} {name}',''])
    (initial/'cameras.txt').write_text('\n'.join(cameras)+'\n');(initial/'images.txt').write_text('\n'.join(images)+'\n');(initial/'points3D.txt').write_text('')
    rec=pycolmap.Reconstruction(initial)
    options=pycolmap.IncrementalPipelineOptions();options.num_threads=8;options.ba_refine_focal_length=False;options.ba_refine_extra_params=False
    options.triangulation.ignore_two_view_tracks=False;options.triangulation.min_angle=.5
    rec=pycolmap.triangulate_points(rec,db,image_path,a.output/'triangulated',options=options,refine_intrinsics=False)
    before=dict(points=rec.num_points3D(),observations=rec.compute_num_observations(),reprojection_error=rec.compute_mean_reprojection_error())
    if rec.num_points3D()<100:write_json(a.output/'report.json',dict(status='rejected_insufficient_tracks',before=before));raise RuntimeError('Too few actual tracks for BA')
    ba=pycolmap.BundleAdjustmentOptions();ba.refine_focal_length=False;ba.refine_principal_point=False;ba.refine_extra_params=False;ba.refine_sensor_from_rig=False
    ba.ceres.loss_function_type=pycolmap.LossFunctionType.SOFT_L1;ba.ceres.loss_function_scale=2
    ba.ceres.solver_options.max_num_iterations=100;ba.ceres.solver_options.num_threads=8;ba.ceres.solver_options.max_solver_time_in_seconds=180
    ba.ceres.solver_options.trust_region_problem_dump_directory=str(a.output/'solver_tmp')
    pycolmap.bundle_adjustment(rec,ba)
    after=dict(points=rec.num_points3D(),observations=rec.compute_num_observations(),reprojection_error=rec.compute_mean_reprojection_error())
    raw=[];ks=[];observations=[]
    for name in names:
        iid,_=by_name[name];im=rec.images[iid];raw.append(im.cam_from_world().matrix());ks.append(rec.cameras[im.camera_id].calibration_matrix());observations.append(im.num_points3D)
    raw=np.asarray(raw);ks=np.asarray(ks);old=pred['extrinsics'];s,r,t=sim3(camera_centers(raw),camera_centers(old))
    centers=s*camera_centers(raw)@r.T+t;orient=r[None]@raw[:,:3,:3].transpose(0,2,1)
    final=raw.copy();final[:,:3,:3]=orient.transpose(0,2,1);final[:,:3,3]=-np.einsum('nij,nj->ni',final[:,:3,:3],centers)
    angle=np.rad2deg(Rotation.from_matrix(final[:,:3,:3]@old[:,:3,:3].transpose(0,2,1)).magnitude())
    radius=max(float(np.max(np.linalg.norm(camera_centers(old)-camera_centers(old).mean(0),axis=-1))),1e-6)
    shift=np.linalg.norm(centers-camera_centers(old),axis=-1)/radius
    accepted=bool(np.isfinite(final).all() and min(observations)>=10 and angle.max()<15 and shift.max()<.5)
    report=dict(status='accepted_for_quality_ablation' if accepted else 'rejected_pose_or_coverage_guard',before=before,after=after,observations_per_image=observations,rotation_change_deg=angle.tolist(),translation_change_camera_radius=shift.tolist(),seconds=time.perf_counter()-start,protocol='Training-only SIFT tracks; VGGT E/K initialization; fixed K; robust BA; training camera-center Sim3 returns result to original gauge; no GT or test images',pycolmap=pycolmap.__version__)
    write_json(a.output/'report.json',report);(a.output/'refined').mkdir(exist_ok=True);rec.write(a.output/'refined')
    if not accepted:raise RuntimeError('BA correction or track coverage exceeds guard; keep original geometry')
    np.savez_compressed(a.output/'cameras.npz',extrinsics=final.astype(np.float32),intrinsics=ks.astype(np.float32))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
