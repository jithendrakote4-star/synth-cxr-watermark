import csv, os, sys
import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "watermark"))
from dwt_qim import make_payload, embed, extract, psnr, bit_accuracy
from dwt_qim_ll import embed_ll, extract_ll
import attacks

sheet = np.array(Image.open(os.path.join(ROOT, "outputs", "contact_sheet.png")).convert("L"))
H, W = sheet.shape
res = W // 2                                   # 2 columns
tiles = [sheet[r * res:(r + 1) * res, c * res:(c + 1) * res]
         for r in range(H // res) for c in range(2)]
print(len(tiles), "tiles of", res, "x", res)

payload = make_payload("jithendra-lab|model-v1")

methods = {"detail-band d=12 (old)": (lambda im: embed(im, payload, delta=12.0),
                                      lambda im: extract(im, delta=12.0))}
for lvl, d in [(3, 32), (3, 64), (3, 96), (3, 128)]:
    methods[f"LL{lvl} d={d}"] = (lambda im, d=d, lvl=lvl: embed_ll(im, payload, delta=d, level=lvl),
                                 lambda im, d=d, lvl=lvl: extract_ll(im, delta=d, level=lvl))

atk = {
    "clean": lambda x: x,
    "jpeg90": lambda x: attacks.jpeg(x, 90),
    "jpeg70": lambda x: attacks.jpeg(x, 70),
    "jpeg50": lambda x: attacks.jpeg(x, 50),
    "noise5": lambda x: attacks.gaussian_noise(x, 5),
    "resize0.75": lambda x: attacks.resize_back(x, 0.75),
    "blur1": lambda x: attacks.blur(x, 1),
}

rows = []
print(f"{'method':24s} {'PSNR':>6s} " + " ".join(f"{a:>10s}" for a in atk))
for name, (emb, ext) in methods.items():
    wms = [emb(t) for t in tiles]
    p = np.mean([psnr(t, w) for t, w in zip(tiles, wms)])
    accs = {a: np.mean([bit_accuracy(payload, ext(f(w))) for w in wms]) for a, f in atk.items()}
    print(f"{name:24s} {p:6.2f} " + " ".join(f"{accs[a]:10.3f}" for a in atk))
    rows.append({"method": name, "psnr": round(float(p), 2), **{a: round(float(v), 3) for a, v in accs.items()}})

with open(os.path.join(ROOT, "outputs", "watermark_sweep.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("saved outputs/watermark_sweep.csv")