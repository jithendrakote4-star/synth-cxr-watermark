import argparse, json, os
import numpy as np
import pandas as pd
import timm, torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from dataset import PATHOLOGIES, CXRDataset, build_dataframe, make_splits

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--out", default="results")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--split_seed", type=int, default=42)  # patient split eppudu fixed
parser.add_argument("--limit", type=int, default=None)  # quick test kosam
parser.add_argument("--synth_csv", default=None)
parser.add_argument("--synth_n", type=int, default=0)
parser.add_argument("--tag", default="baseline")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

df = build_dataframe(args.root)
train_df, val_df, test_df = make_splits(df, seed=args.split_seed)
if args.limit:
    train_df = train_df.head(args.limit)
    if args.synth_csv and args.synth_n > 0:
    syn = pd.read_csv(args.synth_csv)
    syn = syn.sample(n=min(args.synth_n, len(syn)), random_state=0)
    train_df = pd.concat([train_df, syn], ignore_index=True)   # synthetic train lo matrame, val/test real
    val_df = val_df.head(args.limit)
    test_df = test_df.head(args.limit)
print("train:", len(train_df), "val:", len(val_df), "test:", len(test_df))
print(train_df[PATHOLOGIES].sum())

norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tr_tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), norm])
te_tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), norm])

tr = DataLoader(CXRDataset(train_df, tr_tf), batch_size=32, shuffle=True, num_workers=2)
va = DataLoader(CXRDataset(val_df, te_tf), batch_size=64, num_workers=2)
te = DataLoader(CXRDataset(test_df, te_tf), batch_size=64, num_workers=2)

model = timm.create_model("densenet121", pretrained=True,
                          num_classes=len(PATHOLOGIES)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCEWithLogitsLoss()

def evaluate(loader):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(torch.sigmoid(model(x.to(dev))).cpu().numpy())
            ys.append(y.numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    aucs = {}
    for i, p in enumerate(PATHOLOGIES):
        if 0 < ys[:, i].sum() < len(ys):
            aucs[p] = float(roc_auc_score(ys[:, i], ps[:, i]))
    return aucs

best_val, best_state = -1.0, None
for ep in range(args.epochs):
    model.train()
    for x, y in tqdm(tr, desc=f"epoch {ep + 1}"):
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    val_auc = float(np.mean(list(evaluate(va).values())))
    print(f"epoch {ep + 1} VAL mean AUC: {val_auc:.4f}")
    if best_state is None or val_auc > best_val:
        best_val = val_auc
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
test_aucs = evaluate(te)   # test set okasari matrame use chestam
test_mean = float(np.mean(list(test_aucs.values())))
print(f"TEST mean AUC (best-val checkpoint): {test_mean:.4f}", test_aucs)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, f"{args.tag}_seed{args.seed}.json"), "w") as f:
    json.dump({"best_val_auc": best_val, "test_mean_auc": test_mean,
               "per_class": test_aucs}, f, indent=2)