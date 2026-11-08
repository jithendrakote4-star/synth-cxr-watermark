import argparse, json, os, random
import pandas as pd

DESC = {
    "Atelectasis": ["platelike opacity with volume loss", "linear atelectatic changes at the lung base"],
    "Cardiomegaly": ["enlarged cardiac silhouette", "increased cardiothoracic ratio"],
    "Consolidation": ["focal airspace opacity with air bronchograms", "dense airspace consolidation"],
    "Edema": ["bilateral perihilar haziness with interstitial thickening", "diffuse interstitial opacities consistent with edema"],
    "Effusion": ["blunting of the costophrenic angle with pleural fluid", "pleural effusion with meniscus sign"],
    "Pneumonia": ["patchy airspace opacities suggestive of infection", "multifocal pneumonic infiltrates"],
}
NORMAL = ["clear lungs with a normal cardiac silhouette", "no acute cardiopulmonary abnormality"]
VIEW = {"PA": "posteroanterior", "AP": "anteroposterior"}
OPEN = ["Chest radiograph,", "Frontal chest X-ray,", "Chest X-ray,"]


def make_pair(row, rng):
    labels = [l for l in row["Finding Labels"].split("|") if l in DESC]
    view = VIEW.get(row["View Position"], "frontal")
    sex = "male" if row["Patient Gender"] == "M" else "female"
    age = int(row["Patient Age"])
    inp = (f"organ=chest; modality=xray; view={row['View Position']}; "
           f"age={age}; sex={row['Patient Gender']}; "
           f"findings={', '.join(labels) if labels else 'none'}")
    if labels:
        body = "; ".join(rng.choice(DESC[l]) for l in labels)
    else:
        body = rng.choice(NORMAL)
    out = f"{rng.choice(OPEN)} {view} view, {age}-year-old {sex} patient, showing {body}."
    return {"input": inp, "output": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="data_prompts")
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()

    rng = random.Random(42)
    df = pd.read_csv(a.csv)
    df = df[df["Patient Age"] < 120]
    df = df.sample(n=min(a.n, len(df)), random_state=42)
    pairs = [make_pair(r, rng) for _, r in df.iterrows()]

    cut = int(0.9 * len(pairs))
    os.makedirs(a.out, exist_ok=True)
    for name, part in (("train", pairs[:cut]), ("val", pairs[cut:])):
        with open(f"{a.out}/{name}.jsonl", "w") as f:
            for p in part:
                f.write(json.dumps(p) + "\n")
    print(len(pairs), "pairs saved. Example:", pairs[0])