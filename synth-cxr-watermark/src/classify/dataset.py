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
    # Patient-wise split: oka patient images train and test rendu lo undakudadu (data leakage avvakunda)
    rng = np.random.RandomState(seed)
    patients = df["Patient ID"].unique()
    rng.shuffle(patients)
    cut = int(0.8 * len(patients))
    train_pool = df[df["Patient ID"].isin(patients[:cut])]
    test_pool = df[df["Patient ID"].isin(patients[cut:])]
    train = train_pool.sample(n=min(n_train, len(train_pool)), random_state=seed)
    test = test_pool.sample(n=min(n_test, len(test_pool)), random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)

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