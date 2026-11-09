import argparse, json, os, random, sys
import numpy as np, pandas as pd, timm, torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

sys.path.insert(0, "src/classify")
sys.path.insert(0, "src/watermark")
from dataset import build_dataframe, make_splits
import attacks

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--synth_csv", required=True)
ap.add_argument("--synth_dir", default=None)   # synth images ekkada unnayo (csv paths remap kosam)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results_detector")
a = ap.parse_args()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

syn = pd.read_csv(a.synth_csv)["path"].tolist()
if a.synth_dir:
    syn = [os.path.join(a.synth_dir, os.path.basename(p)) for p in syn]
random.shuffle(syn)
cut = int(0.8 * len(syn))
syn_tr, syn_te = syn[:cut], syn[cut:]

df = build_dataframe(a.root)
tr_df, _, te_df = make_splits(df, seed=42)       # real train patients vs unseen test patients
real_tr = tr_df["path"].sample(n=len(syn_tr), random_state=a.seed).tolist()
real_te = te_df["path"].sample(n=len(syn_te), random_state=a.seed).tolist()

norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class DS(Dataset):
    def __init__(self, items, atk=None):
        self.items, self.atk = items, atk

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        arr = np.array(Image.open(p).convert("L").resize((256, 256), Image.BICUBIC))
        if self.atk:
            arr = self.atk(arr)
        t = torch.from_numpy(arr).float().div(255).unsqueeze(0).repeat(3, 1, 1)
        return norm(t), float(y)


train_items = [(p, 0) for p in real_tr] + [(p, 1) for p in syn_tr]    # 1 = synthetic
test_items = [(p, 0) for p in real_te] + [(p, 1) for p in syn_te]
tl = DataLoader(DS(train_items), batch_size=32, shuffle=True, num_workers=2)

model = timm.create_model("densenet121", pretrained=True, num_classes=1).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
lossf = torch.nn.BCEWithLogitsLoss()
for ep in range(a.epochs):
    model.train()
    for x, y in tl:
        x, y = x.to(dev), y.float().to(dev)
        opt.zero_grad()
        lossf(model(x).squeeze(1), y).backward()
        opt.step()
    print("epoch", ep + 1, "done", flush=True)


def evaluate(atk):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in DataLoader(DS(test_items, atk), batch_size=64, num_workers=2):
            ps.append(torch.sigmoid(model(x.to(dev)).squeeze(1)).cpu().numpy())
            ys.append(y.numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    return {"auc": float(roc_auc_score(ys, ps)), "acc": float(((ps > 0.5) == ys).mean())}


atks = {"clean": None,
        "jpeg70": lambda x: attacks.jpeg(x, 70),
        "jpeg30": lambda x: attacks.jpeg(x, 30),
        "noise5": lambda x: attacks.gaussian_noise(x, 5),
        "blur1": lambda x: attacks.blur(x, 1)}
res = {k: evaluate(f) for k, f in atks.items()}
print(json.dumps(res, indent=2))
os.makedirs(a.out, exist_ok=True)
json.dump(res, open(f"{a.out}/detector_seed{a.seed}.json", "w"), indent=2)