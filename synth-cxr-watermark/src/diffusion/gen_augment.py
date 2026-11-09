import argparse, os, random, sys
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import LoraConfig, set_peft_model_state_dict

sys.path.insert(0, "src/classify")
sys.path.insert(0, "src/prompt_gen")
from dataset import PATHOLOGIES, build_dataframe, make_splits
from build_pairs import make_pair

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--lora", required=True)
ap.add_argument("--out", default="synth_train")
ap.add_argument("--classes", default="Pneumonia,Edema,Consolidation,Cardiomegaly")
ap.add_argument("--per_class", type=int, default=500)
ap.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--guidance", type=float, default=7.5)
ap.add_argument("--split_seed", type=int, default=42)
a = ap.parse_args()
dev = "cuda"

df = build_dataframe(a.root)
train_df, _, _ = make_splits(df, seed=a.split_seed)      # TRAIN patients matrame
train_df = train_df[train_df["Patient Age"] < 120]
rng = random.Random(0)

jobs = []
for cls in a.classes.split(","):
    pool = train_df[train_df[cls] == 1]
    pick = pool.sample(n=a.per_class, replace=len(pool) < a.per_class, random_state=0)
    for _, r in pick.iterrows():
        jobs.append((make_pair(r, rng)["output"], [int(r[p]) for p in PATHOLOGIES]))
random.Random(1).shuffle(jobs)
print(len(jobs), "images to generate")

pipe = StableDiffusionPipeline.from_pretrained(a.base, torch_dtype=torch.float16,
                                               safety_checker=None, requires_safety_checker=False).to(dev)
pipe.unet.add_adapter(LoraConfig(r=a.rank, lora_alpha=a.rank, init_lora_weights="gaussian",
                                 target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
set_peft_model_state_dict(pipe.unet, torch.load(a.lora, map_location=dev))
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

os.makedirs(a.out, exist_ok=True)
rows = []
for s in range(0, len(jobs), a.bs):
    chunk = jobs[s:s + a.bs]
    g = torch.Generator(dev).manual_seed(5000 + s)
    ims = pipe([c[0] for c in chunk], num_inference_steps=25, guidance_scale=a.guidance,
               height=a.res, width=a.res, generator=g).images
    for k, im in enumerate(ims):
        path = f"{a.out}/{s + k:05d}.png"
        im.save(path)
        rows.append({"path": os.path.abspath(path), **dict(zip(PATHOLOGIES, chunk[k][1]))})
    print(f"generated {len(rows)}/{len(jobs)}", flush=True)

pd.DataFrame(rows).to_csv(f"{a.out}/synth.csv", index=False)
print("saved", len(rows), "images + synth.csv")