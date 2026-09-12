import numpy as np
from scipy.spatial import cKDTree


def unproject(depth, extrinsics, intrinsics):
    n, h, w = depth.shape
    yy, xx = np.mgrid[:h, :w]
    pixels = np.stack([xx, yy, np.ones_like(xx)], -1).astype(np.float32)
    rays = np.einsum('nij,hwj->nhwi', np.linalg.inv(intrinsics), pixels)
    camera = rays * depth[..., None]
    return np.einsum('nji,nhwj->nhwi', extrinsics[:, :3, :3], camera - extrinsics[:, None, None, :3, 3])


def consistency_mask(points, depth, extrinsics, intrinsics, base, rel_tol=.08, min_support=1, groups=None):
    """Depth agreement only in observed, front-facing pixels; occlusions aren't votes.

    Also reports number of eligible other cameras so lack of overlap is visible.
    """
    n, h, w = depth.shape
    masks = np.zeros_like(base)
    stats = []
    for i in range(n):
        ids = np.flatnonzero(base[i])
        xyz = points[i].reshape(-1, 3)[ids]
        support = np.zeros(len(ids), np.int16)
        eligible = np.zeros(len(ids), np.int16)
        for j in range(n):
            if i == j: continue
            if groups is not None and groups[i] == groups[j]: continue
            cam = xyz @ extrinsics[j, :3, :3].T + extrinsics[j, :3, 3]
            uvw = cam @ intrinsics[j].T
            uv = uvw[:, :2] / np.maximum(uvw[:, 2:], 1e-8)
            in_view = (cam[:, 2] > 1e-5) & (uv[:, 0] >= 0) & (uv[:, 0] < w-1) & (uv[:, 1] >= 0) & (uv[:, 1] < h-1)
            sel = np.flatnonzero(in_view)
            px = np.rint(uv[sel]).astype(int)
            target = depth[j, px[:, 1], px[:, 0]]
            valid = base[j, px[:, 1], px[:, 0]] & (target > 0)
            eligible[sel] += valid
            support[sel] += valid & (np.abs(cam[sel, 2] - target) / np.maximum(target, 1e-6) < rel_tol)
        masks[i].reshape(-1)[ids] = support >= min_support
        stats.append(dict(view=i, candidates=len(ids), supported=int((support >= min_support).sum()), no_overlap=int((eligible == 0).sum())))
    return masks, stats


def filter_points(points, colors, confidence, depth, extrinsics, intrinsics, method='confidence', quantile=.35, max_points=60000, voxel_fraction=.003, seed=42, groups=None, valid_mask=None):
    finite = np.isfinite(points).all(-1) & np.isfinite(depth) & (depth > 1e-5) & np.isfinite(confidence)
    if valid_mask is not None:finite &= valid_mask
    threshold = float(np.quantile(confidence[finite], quantile)) if method != 'raw' else float('-inf')
    # Confidence's floor is often exactly 1. Keeping equality can retain every
    # background pixel when a quantile falls on this large tied floor.
    mask = finite & (confidence > threshold)
    consistency = []
    if method == 'consistency':
        mask, consistency = consistency_mask(points, depth, extrinsics, intrinsics, mask, groups=groups)
    xyz = points[mask]
    rgb = colors[mask]
    conf = confidence[mask]
    retained = len(xyz)
    if retained < 64:
        raise RuntimeError(f'Only {retained} reliable points. Adjust the threshold explicitly; no silent fallback.')
    lo, hi = np.percentile(xyz, [2, 98], axis=0)
    voxel = max(float(np.linalg.norm(hi-lo)*voxel_fraction), 1e-6)
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, ids = np.unique(keys, axis=0, return_index=True)
    # Same voxel and point budget for every filtering variant.
    if len(ids) > max_points:
        ids = np.random.default_rng(seed).choice(ids, max_points, replace=False)
    xyz, rgb, conf = xyz[ids], rgb[ids], conf[ids]
    distances, _ = cKDTree(xyz).query(xyz, k=min(4, len(xyz)))
    scales = np.maximum(np.mean(distances[:, 1:], -1), voxel*.5).astype(np.float32)
    stats = dict(method=method, confidence_quantile=quantile, confidence_threshold=None if method == 'raw' else threshold, threshold_comparison='strict greater than', threshold_tied_pixels=int((finite & (confidence==threshold)).sum()) if method!='raw' else 0, finite_points=int(finite.sum()), retained_before_voxel=retained, exported_points=len(xyz), voxel_size=voxel, cross_view=consistency)
    return xyz.astype(np.float32), rgb.astype(np.float32), conf.astype(np.float32), scales, mask, stats
