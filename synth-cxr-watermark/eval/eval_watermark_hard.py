import csv, os, sys
import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "watermark"))
from dwt_qim import make_payload, psnr, bit_accuracy
from dwt_qim_ll import embed_ll, extract_ll
import attacks

sheet = np.array(Image.open(os.path.join(ROOT, "outputs", "contact_sheet.png")).convert("L"))
H, W = sheet.shape
res = W // 2
tiles = [sheet[r * res:(r + 1) * res, c * res:(c + 1) * res]
         for r in range(H // res) for c in range(2)]

payload = make_payload("jithendra-lab|model-v1")
methods = [("LL3 d=32", 3, 32), ("LL3 d=64", 3, 64), ("LL3 d=96", 3, 96),
           ("LL2 d=64", 2, 64), ("LL4 d=64", 4, 64)]

atk = {
    "jpeg30": lambda x: attacks.jpeg(x, 30),
    "jpeg10": lambda x: attacks.jpeg(x, 10),
    "noise15": lambda x: attacks.gaussian_noise(x, 15),
    "noise25": lambda x: attacks.gaussian_noise(x, 25),
    "resize0.5": lambda x: attacks.resize_back(x, 0.5),
    "blur2": lambda x: attacks.blur(x, 2),
    "crop90": lambda x: attacks.crop_resize(x, 0.9),
    "rot2": lambda x: attacks.rotate(x, 2),
    "shift3": lambda x: attacks.shift(x, 3),
}

rows = []
cols = ["psnr"] + list(atk) + ["unmarked", "wrongkey"]
print(f"{'method':10s} " + " ".join(f"{c:>9s}" for c in cols))
for name, lvl, d in methods:
    wms = [embed_ll(t, payload, delta=d, level=lvl) for t in tiles]
    out = {"psnr": np.mean([psnr(t, w) for t, w in zip(tiles, wms)])}
    for a, f in atk.items():
        out[a] = np.mean([bit_accuracy(payload, extract_ll(f(w), delta=d, level=lvl)) for w in wms])
    # negative controls: watermark nunte accuracy ~0.5 undali
    out["unmarked"] = np.mean([bit_accuracy(payload, extract_ll(t, delta=d, level=lvl)) for t in tiles])
    out["wrongkey"] = np.mean([bit_accuracy(payload, extract_ll(w, delta=d, level=lvl, key=999)) for w in wms])
    print(f"{name:10s} " + " ".join(f"{out[c]:9.3f}" for c in cols))
    rows.append({"method": name, **{k: round(float(v), 3) for k, v in out.items()}})

with open(os.path.join(ROOT, "outputs", "watermark_hard.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("saved outputs/watermark_hard.csv")