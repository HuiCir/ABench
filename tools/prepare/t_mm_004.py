"""T-MM-004: COCO captions (Lin et al. 2014; annotations CC BY 4.0, images under the Flickr terms of the COCO terms of
use) in the Karpathy split (parquet export yerevann/coco-karpathy @448fdb1b), SugarCrepe (Hsieh et al. 2023, MIT,
github.com/RAIVNLab/sugar-crepe @0047054b) for compositional checks, and two frozen encoders: facebook/dinov2-small
(@ed25f3a3, Apache-2.0) and sentence-transformers/all-MiniLM-L12-v2 (@a50ef001, Apache-2.0).

Images are resized (shorter side 256, centre crop 256×256) and re-encoded as JPEG quality 90. Public: Karpathy train +
restval images with their first five captions, minus every image SugarCrepe uses; Karpathy val (5,000 images) as the
self-check split. Hidden: Karpathy test (5,000 images, 25,000 captions) and SugarCrepe's 7,511 items (images, true and
hard-negative captions, subset).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from prepare.common import ROOT, Frozen, check, text_pack

KARPATHY_REV = "448fdb1bc7b2d09e46881c4541a14d796a3d41e8"
SUGAR_COMMIT = "0047054b243992f0fad63d6f64f7544862daf846"
SUBSETS = ["add_att", "add_obj", "replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj"]
MODELS = {"dinov2-small": ("facebook/dinov2-small", "ed25f3a31f01632728cabb09d1542f84ab7b0056",
                           ["config.json", "model.safetensors", "preprocessor_config.json"]),
          "all-MiniLM-L12-v2": ("sentence-transformers/all-MiniLM-L12-v2", "a50ef00143b4d5391434df20ae11632588ac25be",
                                ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
                                 "special_tokens_map.json", "modules.json", "1_Pooling/config.json"])}
SOURCES = ([{"name": f"coco-karpathy/data/{s}-00000-of-00001.parquet",
             "url": f"https://huggingface.co/datasets/yerevann/coco-karpathy/resolve/{KARPATHY_REV}/data/{s}-00000-of-00001.parquet",
             "license": "COCO annotations CC BY 4.0"} for s in ("train", "restval", "validation", "test")]
           + [{"name": "COCO 2014 images", "url": "http://images.cocodataset.org/zips/train2014.zip and val2014.zip",
               "license": "COCO terms of use (images: Flickr terms)", "note": "unzip to RAW/coco/train2014 and RAW/coco/val2014"},
              {"name": "sugar-crepe/data/*.json", "url": f"https://github.com/RAIVNLab/sugar-crepe/tree/{SUGAR_COMMIT}/data", "license": "MIT"}]
           + [{"name": f"{k}/", "url": f"https://huggingface.co/{r}/tree/{rev}", "files": fs, "license": "Apache-2.0"}
              for k, (r, rev, fs) in MODELS.items()])


def _mm():
    import sys
    h = ROOT / "cases" / "T-MM-004" / "harness"
    if str(h) not in sys.path:
        sys.path.insert(0, str(h))
    import mm
    return mm


def _jpeg(path: Path) -> bytes:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - 256) // 2, (h - 256) // 2
    buf = io.BytesIO()
    im.crop((l, t, l + 256, t + 256)).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _coco_path(coco: Path, cocoid: int) -> Path:
    for sub, pre in (("val2014", "COCO_val2014_"), ("train2014", "COCO_train2014_")):
        p = coco / sub / f"{pre}{cocoid:012d}.jpg"
        if p.is_file():
            return p
    raise FileNotFoundError(cocoid)


def _split(rows, coco, img_fn) -> dict:
    mm = _mm()
    imgs = [img_fn(r) for r in rows]
    caps = [c for r in rows for c in r["sentences"][:5]]
    owner = np.repeat(np.arange(len(rows)), [len(r["sentences"][:5]) for r in rows]).astype(np.int64)
    j, jo = mm.pack_bytes(imgs)
    u, o = text_pack(caps)
    return {"jpeg": j, "jpeg_offsets": jo, "caption_utf8": u, "caption_offsets": o, "caption_image": owner,
            "cocoid": np.asarray([r["cocoid"] for r in rows], np.int64)}


def _sugar(items, img_fn) -> dict:
    mm = _mm()
    ids = sorted({it["cocoid"] for it in items})
    pos = {c: i for i, c in enumerate(ids)}
    j, jo = mm.pack_bytes([img_fn({"cocoid": c}) for c in ids])
    pu, po = text_pack([it["caption"] for it in items])
    nu, no = text_pack([it["negative_caption"] for it in items])
    return {"sc_jpeg": j, "sc_jpeg_offsets": jo, "sc_pos_utf8": pu, "sc_pos_offsets": po, "sc_neg_utf8": nu,
            "sc_neg_offsets": no, "sc_image": np.asarray([pos[it["cocoid"]] for it in items], np.int64),
            "sc_subset": np.asarray([SUBSETS.index(it["subset"]) for it in items], np.int64)}


def build(raw: Path) -> Frozen:
    import pyarrow.parquet as pq
    k = raw / "coco-karpathy"
    sums = {f"karpathy/{s}": check(k / f"{s}-00000-of-00001.parquet") for s in ("train", "restval", "validation", "test")}
    read = lambda s: pq.read_table(k / f"{s}-00000-of-00001.parquet", columns=["cocoid", "sentences"]).to_pylist()
    items = []
    for s in SUBSETS:
        p = raw / "sugar-crepe" / f"{s}.json"
        sums[f"sugar/{s}"] = check(p)
        for it in json.loads(p.read_text(encoding="utf-8")).values():
            items.append({**it, "subset": s, "cocoid": int(it["filename"].split(".")[0])})
    sugar_ids = {it["cocoid"] for it in items}
    coco = raw / "coco"
    img = lambda r: _jpeg(_coco_path(coco, int(r["cocoid"])))
    train = [r for r in read("train") + read("restval") if r["cocoid"] not in sugar_ids]
    test = read("test")
    assert not sugar_ids & {r["cocoid"] for r in train}
    hidden = {**_split(test, coco, img), **_sugar(items, img)}
    fz = Frozen(public={"coco_train.npz": _split(train, coco, img), "coco_val.npz": _split(read("validation"), coco, img)},
                hidden=hidden, sources_sha256=sums, models={m: raw / m for m in MODELS},
                notes={"train_images": len(train), "sugar_images_removed_from_train": len(sugar_ids)})
    return fz


def synthetic(n: int) -> Frozen:
    """Plumbing only: coloured squares captioned with their colour."""
    from PIL import Image
    rng = np.random.default_rng(n)
    cols = {"red": (220, 30, 30), "green": (30, 200, 30), "blue": (30, 30, 220), "yellow": (230, 220, 30)}
    def row(i):
        c = list(cols)[int(rng.integers(0, 4))]
        buf = io.BytesIO()
        Image.new("RGB", (256, 256), cols[c]).save(buf, "JPEG", quality=90)
        return {"cocoid": i, "sentences": [f"a {c} square", f"something {c}", f"{c}", f"a plain {c} image", f"the colour {c}"],
                "_jpeg": buf.getvalue(), "colour": c}
    rows = [row(i) for i in range(max(12, n // 10) + 24)]
    img = lambda r: next(x["_jpeg"] for x in rows if x["cocoid"] == r["cocoid"])
    items = [{"cocoid": r["cocoid"], "caption": f"a {r['colour']} square", "negative_caption": "a striped square",
              "subset": SUBSETS[i % 7]} for i, r in enumerate(rows[:8])]
    return Frozen(public={"coco_train.npz": _split(rows[24:], None, img), "coco_val.npz": _split(rows[12:24], None, img)},
                  hidden={**_split(rows[:12], None, img), **_sugar(items, img)}, sources_sha256={"synthetic": f"n{n}"})
