import argparse, json, os, random, sys

sys.path.insert(0, "src/classify")
sys.path.insert(0, "src/prompt_gen")
from dataset import build_dataframe, make_splits
from build_pairs import make_pair

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", default="diffusion_data")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

df = build_dataframe(a.root)
train_df, _, _ = make_splits(df, seed=a.seed)   # train split matrame
train_df = train_df[train_df["Patient Age"] < 120]
rng = random.Random(a.seed)

os.makedirs(a.out, exist_ok=True)
with open(f"{a.out}/train.jsonl", "w") as f:
    for _, r in train_df.iterrows():
        f.write(json.dumps({"file": r["path"], "text": make_pair(r, rng)["output"]}) + "\n")
print(len(train_df), "captions saved. Example:", open(f"{a.out}/train.jsonl").readline())