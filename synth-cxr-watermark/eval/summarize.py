import glob, json, os, re, sys
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "outputs"
groups = {}
for f in glob.glob(os.path.join(d, "**", "*_seed*.json"), recursive=True):
    tag = re.sub(r"_seed\d+\.json$", "", os.path.basename(f))
    groups.setdefault(tag, []).append(json.load(open(f)))


def ms(v):
    sd = np.std(v, ddof=1) if len(v) > 1 else float("nan")
    return f"{np.mean(v):.4f} ± {sd:.4f}"


for tag, runs in sorted(groups.items()):
    print(f"\n{tag}  (n={len(runs)} seeds)")
    print("  test mean AUC:", ms([r["test_mean_auc"] for r in runs]))
    for c in runs[0]["per_class"]:
        print(f"  {c:14s}", ms([r["per_class"][c] for r in runs]))