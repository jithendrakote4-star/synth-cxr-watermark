import argparse, json, os, random, sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchmetrics.image.fid import FrechetInceptionDistance
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import LoraConfig, set_peft_model_state_dict

sys.path.insert(0, "src/classify")
sys.path.insert(0, "src/prompt_gen")
from dataset import PATHOLOGIES, build_dataframe, make_splits
from build_pairs import make_pair

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--lora", required=True)
ap.add_argument("--out", default="gen_eval")
ap.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--guidance", type=float, default=7.5)
ap.add_argument("--split_seed", type=int, default=42)
a = ap.parse_args()
dev = "cuda"

# held-out TEST patients matrame (diffusion training lo ee patients ledu)
df = build_dataframe(a.root)
_, _, test_df = make_splits(df, seed=a.split_seed)
test_df = test_df[test_df["Patient Age"] < 120]
test_df = test_df.sample(n=min(a.n, len(test_df)), random_state=0).reset_index(drop=True)
rng = random.Random(0)
prompts = [make_pair(r, rng)["output"] for _, r in test_df.iterrows()]

pipe = StableDiffusionPipeline.from_pretrained(a.base, torch_dtype=torch.float16,
                                               safety_checker=None, requires_safety_checker=False).to(dev)
pipe.unet.add_adapter(LoraConfig(r=a.rank, lora_alpha=a.rank, init_lora_weights="gaussian",
                                 target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
set_peft_model_state_dict(pipe.unet, torch.load(a.lora, map_location=dev))
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

os.makedirs(a.out, exist_ok=True)
gen, meta = [], []
for s in range(0, len(prompts), a.bs):
    ps = prompts[s:s + a.bs]
    g = torch.Generator(dev).manual_seed(1000 + s)
    ims = pipe(ps, num_inference_steps=25, guidance_scale=a.guidance,
               height=a.res, width=a.res, generator=g).images
    for k, im in enumerate(ims):
        i = s + k
        im.save(f"{a.out}/{i:05d}.png")
        gen.append(im)
        meta.append({"file": f"{i:05d}.png", "prompt": ps[k],
                     "labels": {p: int(test_df.loc[i, p]) for p in PATHOLOGIES}})
    print(f"generated {len(gen)}/{len(prompts)}", flush=True)
json.dump(meta, open(f"{a.out}/meta.json", "w"))

tf = T.Compose([T.Resize(a.res, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(a.res), T.PILToTensor()])
real = [tf(Image.open(p).convert("RGB")) for p in test_df["path"]]
gent = [tf(im.convert("RGB")) for im in gen]


def fid_score(x, y):
    m = FrechetInceptionDistance(feature=2048).to(dev)
    for t in range(0, len(x), 64):
        m.update(torch.stack(x[t:t + 64]).to(dev), real=True)
    for t in range(0, len(y), 64):
        m.update(torch.stack(y[t:t + 64]).to(dev), real=False)
    return float(m.compute())


h = len(real) // 2
res = {"n": len(real),
       "fid_real_vs_generated": fid_score(real, gent),
       "fid_real_vs_real_floor": fid_score(real[:h], real[h:])}
print(res)
json.dump(res, open(f"{a.out}/fid.json", "w"), indent=2)