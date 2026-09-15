"""Prepare a pinned, tiny real multi-position 360° generalization test."""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.panorama import layout, project
from src.storage import check_storage, write_json

COMMIT = "06cfdbf295fa21500a857efe81f80f2243b8fb40"
BASE = f"https://raw.githubusercontent.com/zillow/zind/{COMMIT}/sample_tour/000"
TRAIN_IDS = (2, 4, 6)
TEST_ID = 5
PARTIAL_ROOM = "partial_room_09"


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def camera_rotation(angle):
    """ZInD pano pixels -> right-handed OpenCV c2w in X/down/-floorY world axes."""
    a = np.deg2rad(angle)
    r2 = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    out = np.array([[-r2[0, 0], 0, r2[1, 0]], [0, 1, 0], [r2[0, 1], 0, -r2[1, 1]]], np.float32)
    if not np.allclose(out.T @ out, np.eye(3), atol=1e-5) or np.linalg.det(out) < 0.999:
        raise ValueError("Invalid ZInD camera rotation")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", type=int, default=336)
    p.add_argument("--accept-zind-terms", action="store_true")
    a = p.parse_args()
    if not a.accept_zind_terms:
        raise SystemExit(
            "Read the ZInD Terms of Use, then pass --accept-zind-terms for this academic sample"
        )
    out = a.output.resolve()
    check_storage(out, growth_gb=0.1)
    raw = out / "raw"
    panos = out / "panoramas"
    test = out / "test"
    for d in [raw, panos, test]:
        d.mkdir(parents=True, exist_ok=True)
    annotation = raw / "zind_data.json"
    if not annotation.exists():
        urllib.request.urlretrieve(f"{BASE}/zind_data.json", annotation)
    data = json.loads(annotation.read_text())
    room = data["merger"]["floor_01"]["complete_room_06"][PARTIAL_ROOM]
    scale = float(data["scale_meters_per_coordinate"]["floor_01"])
    records = {}
    for pano_id in (*TRAIN_IDS, TEST_ID):
        item = room[f"pano_{pano_id}"]
        name = Path(item["image_path"]).name
        target = raw / name
        if not target.exists():
            urllib.request.urlretrieve(f"{BASE}/panos/{name}", target)
        if md5(target) != item["checksum"]:
            raise RuntimeError(f"Checksum mismatch: {name}")
        transform = item["floor_plan_transformation"]
        translation = np.asarray(transform["translation"], np.float32) * scale
        records[pano_id] = {
            "path": target,
            "origin": np.array([translation[0], 0, -translation[1]], np.float32),
            "rotation": camera_rotation(transform["rotation"]),
            "metadata": item,
        }
    anchor = records[TRAIN_IDS[0]]["origin"].copy()
    origins = []
    rotations = []
    for pano_id in TRAIN_IDS:
        rec = records[pano_id]
        name = f"pano_{pano_id:02d}.jpg"
        Image.open(rec["path"]).convert("RGB").save(panos / name, quality=95)
        origins.append((rec["origin"] - anchor).tolist())
        rotations.append(rec["rotation"].tolist())
    test_rec = records[TEST_ID]
    erp = np.asarray(Image.open(test_rec["path"]).convert("RGB"))
    frames = []
    spec = layout("overlap") + [(0, 90, 100), (0, -90, 100)]
    for view, (yaw, pitch, fov) in enumerate(spec):
        rgb, k, r = project(erp, yaw, pitch, fov, a.size)
        name = f"heldout_pano_{TEST_ID:02d}_view_{view:02d}.png"
        Image.fromarray(rgb.astype(np.uint8)).save(test / name)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = test_rec["rotation"] @ r
        c2w[:3, 3] = test_rec["origin"] - anchor
        frames.append(
            {
                "file": name,
                "c2w": c2w.tolist(),
                "intrinsics": k.tolist(),
                "heldout_pano": TEST_ID,
                "yaw": yaw,
                "pitch": pitch,
            }
        )
    hashes = {
        f.relative_to(out).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(out.rglob("*"))
        if f.is_file()
    }
    write_json(
        out / "scene.json",
        {
            "name": "ZInD sample tour 000 room 09",
            "kind": "real_capture",
            "source_repo": "https://github.com/zillow/zind",
            "source_commit": COMMIT,
            "data_terms": "ZInD Terms of Use; academic/non-commercial use",
            "capture": "Real 360 panoramas captured with an off-the-shelf panoramic camera",
            "train_pano_ids": list(TRAIN_IDS),
            "heldout_pano_ids": [TEST_ID],
            "split_frozen_before_inference": True,
            "panorama_origins": origins,
            "panorama_rotations": rotations,
            "test_frames": frames,
            "hashes": hashes,
            "pose_note": "Positions and yaw derive only from official floor_plan_transformation; pano 5 RGB is excluded from inference and optimization",
        },
    )
    print(
        json.dumps(
            {
                "output": str(out),
                "train": TRAIN_IDS,
                "test": TEST_ID,
                "files": len(hashes),
                "megabytes": sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 2**20,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
