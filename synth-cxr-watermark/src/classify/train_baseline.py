import argparse, json, os
import numpy as np
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
parser.add_argument("--limit", type=int, default=None)  # quick test kosam
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

df = build_dataframe(args.root)
train_df, test_df = make_splits(df, seed=args.seed)
if args.limit:
    train_df, test_df = train_df.head(args.limit), test_df.head(args.limit)
print("train:", len(train_df), "test:", len(test_df))
print(train_df[PATHOLOGIES].sum())  # class imbalance chudataniki

norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tr_tf = T.Compose([T.Resize((256, 256)), T.RandomHorizontalFlip(), T.ToTensor(), norm])
te_tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), norm])

tr = DataLoader(CXRDataset(train_df, tr_tf), batch_size=32, shuffle=True, num_workers=2)
te = DataLoader(CXRDataset(test_df, te_tf), batch_size=64, num_workers=2)

model = timm.create_model("densenet121", pretrained=True,
                          num_classes=len(PATHOLOGIES)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCEWithLogitsLoss()

def evaluate():
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in te:
            ps.append(torch.sigmoid(model(x.to(dev))).cpu().numpy())
            ys.append(y.numpy())
    ys, ps = np.concatenate(ys), np.concatenate(ps)
    aucs = {}
    for i, p in enumerate(PATHOLOGIES):
        if ys[:, i].sum() > 0:
            aucs[p] = float(roc_auc_score(ys[:, i], ps[:, i]))
    return aucs

for ep in range(args.epochs):
    model.train()
    for x, y in tqdm(tr, desc=f"epoch {ep + 1}"):
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    aucs = evaluate()
    print(f"epoch {ep + 1} mean AUC: {np.mean(list(aucs.values())):.4f}", aucs)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, f"baseline_seed{args.seed}.json"), "w") as f:
    json.dump({"mean_auc": float(np.mean(list(aucs.values()))),
               "per_class": aucs}, f, indent=2)