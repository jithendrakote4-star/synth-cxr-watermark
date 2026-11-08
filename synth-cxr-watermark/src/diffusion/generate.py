import argparse, os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import LoraConfig, set_peft_model_state_dict

ap = argparse.ArgumentParser()
ap.add_argument("--lora", required=True)
ap.add_argument("--out", default="gen_samples")
ap.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--guidance", type=float, default=7.5)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

PROMPTS = [
    "Chest radiograph, posteroanterior view, 58-year-old male patient, showing increased cardiothoracic ratio; blunting of the costophrenic angle with pleural fluid.",
    "Chest radiograph, posteroanterior view, 45-year-old female patient, showing clear lungs with a normal cardiac silhouette.",
    "Chest X-ray, anteroposterior view, 67-year-old male patient, showing patchy airspace opacities suggestive of infection.",
    "Chest X-ray, anteroposterior view, 60-year-old female patient, showing bilateral perihilar haziness with interstitial thickening.",
]

pipe = StableDiffusionPipeline.from_pretrained(args.base, torch_dtype=torch.float16,
                                               safety_checker=None, requires_safety_checker=False).to("cuda")
pipe.unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                                 target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
set_peft_model_state_dict(pipe.unet, torch.load(args.lora, map_location="cuda"))
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

os.makedirs(args.out, exist_ok=True)
imgs = []
for pi, p in enumerate(PROMPTS):
    for k in range(args.n):
        g = torch.Generator("cuda").manual_seed(args.seed + k)
        im = pipe(p, num_inference_steps=25, guidance_scale=args.guidance,
                  height=args.res, width=args.res, generator=g).images[0]
        im.save(f"{args.out}/p{pi}_s{k}.png")
        imgs.append(im)

sheet = Image.new("RGB", (args.res * args.n, args.res * len(PROMPTS)))
for i, im in enumerate(imgs):
    sheet.paste(im, ((i % args.n) * args.res, (i // args.n) * args.res))
sheet.save(f"{args.out}/contact_sheet.png")
print("saved", len(imgs), "images to", args.out)