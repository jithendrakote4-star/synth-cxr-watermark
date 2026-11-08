import glob, os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# NIH dataset lo "Edema" and "Effusion" ani untayi
PATHOLOGIES = ["Atelectasis", "Cardiomegaly", "Consolidation",
               "Edema", "Effusion", "Pneumonia"]

def build_dataframe(root):
    df = pd.read_csv(os.path.join(root, "Data_Entry_2017.csv"))
    files = glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
    path_map = {os.path.basename(p): p for p in files}
    df["path"] = df["Image Index"].map(path_map)
    df = df.dropna(subset=["path"]).reset_index(drop=True)
    for p in PATHOLOGIES:
        df[p] = df["Finding Labels"].apply(lambda s: int(p in s.split("|")))
    return df

def make_splits(df, n_train=12000, n_test=5000, seed=42):
    # Patient-wise split: train 70% / val 10% / test 20% patients
    rng = np.random.RandomState(seed)
    patients = df["Patient ID"].unique()
    rng.shuffle(patients)
    n = len(patients)
    tr_p = patients[: int(0.7 * n)]
    va_p = patients[int(0.7 * n): int(0.8 * n)]
    te_p = patients[int(0.8 * n):]
    tr = df[df["Patient ID"].isin(tr_p)]
    va = df[df["Patient ID"].isin(va_p)]
    te = df[df["Patient ID"].isin(te_p)]
    tr = tr.sample(n=min(n_train, len(tr)), random_state=seed)
    te = te.sample(n=min(n_test, len(te)), random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)

class CXRDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df, self.transform = df, transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = row[PATHOLOGIES].values.astype("float32")
        return img, label