import argparse, json, os, random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model_state_dict

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="diffusion_data/train.jsonl")
ap.add_argument("--out", default="lora_out")
ap.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--save_every", type=int, default=500)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
dev = "cuda"
os.makedirs(args.out, exist_ok=True)

tok = CLIPTokenizer.from_pretrained(args.base, subfolder="tokenizer")
text_enc = CLIPTextModel.from_pretrained(args.base, subfolder="text_encoder", torch_dtype=torch.float16).to(dev)
vae = AutoencoderKL.from_pretrained(args.base, subfolder="vae", torch_dtype=torch.float16).to(dev)
unet = UNet2DConditionModel.from_pretrained(args.base, subfolder="unet", torch_dtype=torch.float16).to(dev)
sched = DDPMScheduler.from_pretrained(args.base, subfolder="scheduler")
for m in (text_enc, vae, unet):
    m.requires_grad_(False)

unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
                            target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
params = [p for p in unet.parameters() if p.requires_grad]
for p in params:
    p.data = p.data.float()          # LoRA weights fp32 lo train avthayi
print("trainable LoRA params:", sum(p.numel() for p in params))

opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
scaler = torch.amp.GradScaler("cuda")

# NOTE: horizontal flip vaadaledu. Chest X-ray lo flip chesthe heart kudi vaipu ki velthundi (anatomically thappu).
tf = T.Compose([
    T.Resize(args.res, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(args.res),
    T.ToTensor(),
    T.Normalize([0.5], [0.5]),
])


class DS(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = tf(Image.open(r["file"]).convert("RGB"))
        text = "" if random.random() < 0.1 else r["text"]   # 10% caption dropout (classifier-free guidance kosam)
        return img, text


def collate(b):
    return torch.stack([x[0] for x in b]), [x[1] for x in b]


dl = DataLoader(DS(args.data), batch_size=args.bs, shuffle=True, num_workers=2,
                collate_fn=collate, drop_last=True)


def infinite(loader):
    while True:
        for b in loader:
            yield b


it = infinite(dl)
log = []
unet.train()
pbar = tqdm(range(1, args.steps + 1))
for step in pbar:
    opt.zero_grad(set_to_none=True)
    tot = 0.0
    for _ in range(args.accum):
        imgs, texts = next(it)
        with torch.no_grad():
            lat = vae.encode(imgs.to(dev, dtype=torch.float16)).latent_dist.sample() * vae.config.scaling_factor
            ids = tok(texts, padding="max_length", max_length=tok.model_max_length,
                      truncation=True, return_tensors="pt").input_ids.to(dev)
            enc = text_enc(ids)[0]
        noise = torch.randn_like(lat)
        t = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=dev).long()
        noisy = sched.add_noise(lat, noise, t)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = unet(noisy, t, enc).sample
        loss = F.mse_loss(pred.float(), noise.float()) / args.accum
        scaler.scale(loss).backward()
        tot += loss.item()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt)
    scaler.update()
    log.append(tot)
    if step % 20 == 0:
        pbar.set_description(f"loss {np.mean(log[-20:]):.4f}")
    if step % args.save_every == 0 or step == args.steps:
        torch.save(get_peft_model_state_dict(unet), f"{args.out}/lora_step{step}.pt")

torch.save(get_peft_model_state_dict(unet), f"{args.out}/lora_final.pt")
json.dump({"rank": args.rank, "base": args.base, "res": args.res, "steps": args.steps, "loss": log},
          open(f"{args.out}/train_log.json", "w"))
print("done. saved to", args.out)