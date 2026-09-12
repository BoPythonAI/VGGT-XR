"""ERP projection. Coordinates: OpenCV right/down/forward; panorama yaw right."""
import numpy as np
from scipy.ndimage import map_coordinates


def rotation(yaw, pitch=0):
    y, p = np.deg2rad([yaw, pitch])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
    return (ry @ rx).astype(np.float32)


def perspective_rays(size, fov):
    f = size / (2 * np.tan(np.deg2rad(fov) / 2))
    yy, xx = np.mgrid[:size, :size]
    rays = np.stack([(xx + .5 - size/2)/f, (yy + .5 - size/2)/f, np.ones_like(xx)], -1).astype(np.float32)
    k = np.array([[f, 0, (size-1)/2], [0, f, (size-1)/2], [0, 0, 1]], np.float32)
    return rays, k


def erp_rays(height, width):
    yy, xx = np.mgrid[:height, :width]
    yaw = (xx + .5) / width * 2*np.pi - np.pi
    lat = np.pi/2 - (yy + .5) / height * np.pi
    return np.stack([np.cos(lat)*np.sin(yaw), -np.sin(lat), np.cos(lat)*np.cos(yaw)], -1).astype(np.float32)


def project(erp, yaw, pitch=0, fov=100, size=336, order=1):
    rays, k = perspective_rays(size, fov)
    r = rotation(yaw, pitch)
    world = rays @ r.T
    world /= np.linalg.norm(world, axis=-1, keepdims=True)
    longitude = np.arctan2(world[..., 0], world[..., 2])
    latitude = np.arcsin(-world[..., 1].clip(-1, 1))
    h, w = erp.shape[:2]
    x = ((longitude + np.pi) / (2*np.pi) * w - .5) % w
    y = ((np.pi/2-latitude) / np.pi * h - .5).clip(0, h-1)
    # Explicit horizontal padding prevents seam interpolation from crossing poles.
    wrapped = np.concatenate([erp[:, -1:], erp, erp[:, :1]], axis=1)
    if erp.ndim == 3:
        out = np.stack([map_coordinates(wrapped[..., c], [y, x+1], order=order, mode='nearest') for c in range(erp.shape[-1])], -1)
    else:
        out = map_coordinates(wrapped, [y, x+1], order=order, mode='nearest')
    return out, k, r


def layout(method):
    if method == 'cubemap':
        return [(y, 0, 90) for y in [0, 90, 180, 270]] + [(0, 90, 90), (0, -90, 90)]
    if method == 'overlap':
        return [(y, 0, 100) for y in range(0, 360, 45)]
    raise ValueError(method)


def constrain_poses(extrinsics, groups, relative_rotations, method='mean'):
    """Fit shared center and known rotations per ERP; retain global learned placement.

    This post-hoc ablation is evaluated, not assumed beneficial. Depth must be
    re-unprojected after changing cameras.
    """
    out = extrinsics.copy()
    c2w_r = extrinsics[:, :3, :3].transpose(0, 2, 1)
    centers = -np.einsum('nij,nj->ni', c2w_r, extrinsics[:, :3, 3])
    for group in sorted(set(groups)):
        ids = np.where(np.asarray(groups) == group)[0]
        if method=='anchor':
            # Same first yaw is used for every panorama, retaining VGGT's
            # translated front-view estimates instead of averaging drifted faces.
            anchor=ids[0]
            orient=c2w_r[anchor]@relative_rotations[anchor].T
            shared=centers[anchor]
        elif method=='mean':
            estimates = c2w_r[ids] @ relative_rotations[ids].transpose(0, 2, 1)
            u, _, vt = np.linalg.svd(estimates.sum(0))
            orient = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
            shared = np.median(centers[ids], axis=0)
        else:raise ValueError(method)
        rs = orient[None] @ relative_rotations[ids]
        out[ids, :3, :3] = rs.transpose(0, 2, 1)
        out[ids, :3, 3] = -np.einsum('nij,j->ni', out[ids, :3, :3], shared)
    return out
