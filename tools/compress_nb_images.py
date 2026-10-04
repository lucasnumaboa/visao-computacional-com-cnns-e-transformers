"""Shrink the images embedded in notebook outputs: PNG -> JPEG (max width, quality), keeping the notebook valid.

Usage: python tools/compress_nb_images.py [--maxw 1400] [--q 80] notebooks/*.ipynb
"""
import argparse
import base64
import io
import json

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--maxw", type=int, default=1400)
ap.add_argument("--q", type=int, default=80)
args = ap.parse_args()

for path in args.files:
    nb = json.load(open(path, encoding="utf-8"))
    before = after = n = 0
    for cell in nb["cells"]:
        for out in cell.get("outputs", []) if cell["cell_type"] == "code" else []:
            data = out.get("data", {})
            if "image/png" not in data:
                continue
            raw = base64.b64decode(data["image/png"])
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            if im.width > args.maxw:
                im = im.resize((args.maxw, round(im.height * args.maxw / im.width)), Image.LANCZOS)
            buf = io.BytesIO(); im.save(buf, "JPEG", quality=args.q, optimize=True)
            before += len(raw); after += buf.tell(); n += 1
            del data["image/png"]
            data["image/jpeg"] = base64.b64encode(buf.getvalue()).decode("ascii")
            meta = out.get("metadata", {})
            if "image/png" in meta:
                meta["image/jpeg"] = meta.pop("image/png")
    json.dump(nb, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{path}: {n} imagens | {before/1e6:.1f} MB -> {after/1e6:.1f} MB")
