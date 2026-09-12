import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation


def write_ply(path, xyz, rgb, confidence=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if confidence is not None: names += [('confidence', '<f4')]
    data = np.empty(len(xyz), dtype=names)
    for i, key in enumerate(['x', 'y', 'z']): data[key] = xyz[:, i]
    for i, key in enumerate(['red', 'green', 'blue']): data[key] = (rgb[:, i].clip(0, 1)*255).astype(np.uint8)
    if confidence is not None: data['confidence'] = confidence
    _ply(path, data)


def _ply(path, data):
    types = {'f': 'float', 'u': 'uchar'}
    header = ['ply', 'format binary_little_endian 1.0', f'element vertex {len(data)}']
    for name in data.dtype.names:
        header.append(f'property {types[data.dtype[name].kind]} {name}')
    header.append('end_header\n')
    with open(path, 'wb') as f:
        f.write(('\n'.join(header)).encode('ascii'))
        data.tofile(f)


def export_gaussians(path, params):
    p = {k: v.detach().cpu().numpy() for k, v in params.items()}
    fields = ['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2'] + [f'f_rest_{i}' for i in range(45)] + ['opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    data = np.zeros(len(p['means']), dtype=[(k,'<f4') for k in fields])
    for i, k in enumerate(['x','y','z']): data[k] = p['means'][:,i]
    color = 1/(1+np.exp(-p['colors'].clip(-50,50)))
    sh0 = (color-.5)/.28209479177387814
    for i in range(3):
        data[f'f_dc_{i}'] = sh0[:,i]
        data[f'scale_{i}'] = p['scales'][:,i]
    q = p['quats'] / np.maximum(np.linalg.norm(p['quats'],axis=-1,keepdims=True),1e-8)
    for i in range(4): data[f'rot_{i}'] = q[:,i]
    data['opacity'] = p['opacities']
    _ply(path, data)


def export_colmap(directory, names, extrinsics, intrinsics, xyz, rgb):
    """COLMAP text model accepted by pycolmap; no fabricated point tracks."""
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    camera_lines, image_lines = [], []
    for i, (name,e,k) in enumerate(zip(names,extrinsics,intrinsics),1):
        h,w = rgb.shape[1:3] if rgb.ndim == 4 else (0,0)
        if not w: raise ValueError('Pass full image colors to COLMAP exporter')
        camera_lines.append(f'{i} PINHOLE {w} {h} {k[0,0]} {k[1,1]} {k[0,2]} {k[1,2]}')
        q = Rotation.from_matrix(e[:3,:3]).as_quat()[[3,0,1,2]]
        values = ' '.join(map(str, [*q,*e[:3,3]]))
        image_lines.extend([f'{i} {values} {i} {name}', ''])
    (directory/'cameras.txt').write_text('\n'.join(camera_lines)+'\n')
    (directory/'images.txt').write_text('\n'.join(image_lines)+'\n')
    colors = xyz[1] if isinstance(xyz,tuple) else None
    points = xyz[0] if isinstance(xyz,tuple) else xyz
    if colors is None: raise ValueError('Provide (xyz, point_colors)')
    with open(directory/'points3D.txt','w') as f:
        for i,(p,c) in enumerate(zip(points,(colors.clip(0,1)*255).astype(int)),1):
            f.write(f'{i} {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]} 0\n')
    import pycolmap
    rec = pycolmap.Reconstruction(str(directory))
    rec.write(str(directory))
