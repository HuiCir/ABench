"""V-3DG-005: Mip-NeRF 360 scenes (Barron et al. 2022; storage.googleapis.com/gresearch/refraw360, no licence
stated on the project page) at the published 1/8 resolution (images_8), with cameras from each scene's COLMAP model
(sparse/0: cameras.bin, images.bin, read here).

Public: dev_bonsai.npz, dev_flowers.npz, dev_treehill.npz with every image of the scene and its camera.
Hidden: bicycle, garden, stump, room, counter and kitchen as scene0-5 (order shuffled with the secret seed
ABENCH_V_3DG_SEED): 24 training views (pre-timed inputs), test cameras and image size (post-timed inputs), test images
(evaluator only); test views are every 8th image (nvs.split).
"""
from __future__ import annotations

import importlib.util
import io
import os
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

from prepare.common import ROOT, Frozen, check

SECRET_SEED_ENV = "ABENCH_V_3DG_SEED"
BUCKET = "https://storage.googleapis.com/gresearch/refraw360"
SOURCES = [{"name": "360_v2.zip", "url": f"{BUCKET}/360_v2.zip", "license": "not stated (research benchmark)"},
           {"name": "360_extra_scenes.zip", "url": f"{BUCKET}/360_extra_scenes.zip", "license": "not stated (research benchmark)"}]
HIDDEN = ["bicycle", "garden", "stump", "room", "counter", "kitchen"]
DEV = {"bonsai": "360_v2.zip", "flowers": "360_extra_scenes.zip", "treehill": "360_extra_scenes.zip"}
N_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}            # COLMAP camera models (SIMPLE_PINHOLE ... OPENCV_FISHEYE)


def nvs():
    p = ROOT / "cases" / "V-3DG-005" / "harness" / "nvs.py"
    spec = importlib.util.spec_from_file_location("nvs", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["nvs"] = m
    spec.loader.exec_module(m)
    return m


def read_cameras(b: bytes) -> dict:
    f = io.BytesIO(b)
    out = {}
    for _ in range(struct.unpack("<Q", f.read(8))[0]):
        cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
        prm = struct.unpack(f"<{N_PARAMS[model]}d", f.read(8 * N_PARAMS[model]))
        if model == 0 or model == 2 or model == 3:             # SIMPLE_*: f, cx, cy (radial terms ignored)
            fx = fy = prm[0]
            cx, cy = prm[1], prm[2]
        else:                                                    # PINHOLE, OPENCV: fx, fy, cx, cy
            fx, fy, cx, cy = prm[:4]
        out[cid] = {"model": model, "w": w, "h": h, "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])}
    return out


def qvec2rot(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def read_images(b: bytes) -> dict:
    f = io.BytesIO(b)
    out = {}
    for _ in range(struct.unpack("<Q", f.read(8))[0]):
        iid = struct.unpack("<i", f.read(4))[0]
        q = struct.unpack("<4d", f.read(32))
        t = struct.unpack("<3d", f.read(24))
        cid = struct.unpack("<i", f.read(4))[0]
        name = b""
        while (c := f.read(1)) != b"\x00":
            name += c
        n2d = struct.unpack("<Q", f.read(8))[0]
        f.read(24 * n2d)
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = qvec2rot(q), t
        out[name.decode()] = {"w2c": m, "camera": cid}
    return out


def read_scene(zf: zipfile.ZipFile, scene: str) -> dict:
    from PIL import Image
    pre = next(n for n in zf.namelist() if n.endswith(f"{scene}/sparse/0/cameras.bin"))[: -len("sparse/0/cameras.bin")]
    cams = read_cameras(zf.read(pre + "sparse/0/cameras.bin"))
    ims = read_images(zf.read(pre + "sparse/0/images.bin"))
    imgs, Ks, w2cs = [], [], []
    for name in sorted(ims):                                   # sorted by file name, as the NeRF loaders do
        a = np.asarray(Image.open(io.BytesIO(zf.read(pre + "images_8/" + name))).convert("RGB"), np.uint8)
        c = cams[ims[name]["camera"]]
        K = c["K"].copy()
        K[0] *= a.shape[1] / c["w"]
        K[1] *= a.shape[0] / c["h"]
        imgs.append(a)
        Ks.append(K)
        w2cs.append(ims[name]["w2c"])
    shapes = {i.shape for i in imgs}
    if len(shapes) != 1:
        raise SystemExit(f"{scene}: images_8 sizes differ {shapes}")
    return {"images": np.stack(imgs), "K": np.stack(Ks), "w2c": np.stack(w2cs)}


def _freeze(dev: dict, hidden: dict, seed: int, sums: dict) -> Frozen:
    sys.path.insert(0, str(ROOT / "cases" / "V-3DG-005" / "harness"))
    N = nvs()
    spec = importlib.util.spec_from_file_location("v3dg_stages", ROOT / "cases" / "V-3DG-005" / "harness" / "stages.py")
    sys.path.insert(0, str(ROOT / "tools" / "harness_kit"))
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    order = np.random.default_rng(seed).permutation(len(hidden))
    names = list(hidden)
    hid = {}
    for i, j in enumerate(order):
        sc = hidden[names[j]]
        tr, te = N.split(len(sc["images"]))
        for part in S.scene_arrays(sc, f"scene{i}", tr, te):
            hid.update(part)
    return Frozen(public={f"dev_{k}.npz": v for k, v in dev.items()}, hidden=hid, sources_sha256=sums,
                  notes={"scene_order": "secret (derived from the seed)"})


def build(raw: Path) -> Frozen:
    seed = int(os.environ.get(SECRET_SEED_ENV, "0"))
    if not seed:
        raise SystemExit(f"set {SECRET_SEED_ENV} to a secret integer (scene order)")
    sums = {s["name"]: check(raw / s["name"]) for s in SOURCES}
    zs = {s["name"]: zipfile.ZipFile(raw / s["name"]) for s in SOURCES}
    dev = {k: read_scene(zs[z], k) for k, z in DEV.items()}
    hidden = {k: read_scene(zs["360_v2.zip"], k) for k in HIDDEN}
    return _freeze(dev, hidden, seed, sums)


def synthetic(n: int) -> Frozen:
    """Small random scenes with cameras on a circle looking at the origin (plumbing only)."""
    rng = np.random.default_rng(0)

    def scene(m):
        K = np.array([[40.0, 0, 32], [0, 40.0, 24], [0, 0, 1]])
        w2c = []
        for a in np.linspace(0, 2 * np.pi, m, endpoint=False):
            c = np.array([4 * np.cos(a), 4 * np.sin(a), 1.0])
            z = -c / np.linalg.norm(c)
            x = np.cross(z, [0, 0, 1.0])
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            R = np.stack([x, y, z])
            w2c.append(np.vstack([np.c_[R, -R @ c], [0, 0, 0, 1]]))
        return {"images": rng.integers(0, 256, (m, 48, 64, 3)).astype(np.uint8), "K": np.repeat(K[None], m, 0),
                "w2c": np.stack(w2c)}
    m = max(n, 40)
    return _freeze({k: scene(m) for k in DEV}, {k: scene(m) for k in HIDDEN}, 12345, {"synthetic": f"n{n}"})
