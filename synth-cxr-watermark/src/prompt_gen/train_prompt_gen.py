import argparse, json, math
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_prompts")
ap.add_argument("--out", default="prompt_gen_model")
ap.add_argument("--epochs", type=int, default=3)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("distilgpt2")
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(dev)


def load(name):
    rows = [json.loads(l) for l in open(f"{args.data}/{name}.jsonl")]
    return [f"{r['input']}\n=> {r['output']}{tok.eos_token}" for r in rows]


def collate(batch):
    enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    return enc["input_ids"], enc["attention_mask"], labels


tr = DataLoader(load("train"), batch_size=32, shuffle=True, collate_fn=collate)
va = DataLoader(load("val"), batch_size=64, collate_fn=collate)
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)


def val_ppl():
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for ids, m, lab in va:
            tot += model(input_ids=ids.to(dev), attention_mask=m.to(dev), labels=lab.to(dev)).loss.item()
            n += 1
    return math.exp(tot / n)


for ep in range(args.epochs):
    model.train()
    for ids, m, lab in tr:
        loss = model(input_ids=ids.to(dev), attention_mask=m.to(dev), labels=lab.to(dev)).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    print(f"epoch {ep + 1} val perplexity: {val_ppl():.3f}")

model.save_pretrained(args.out)
tok.save_pretrained(args.out)

model.eval()
test_in = "organ=chest; modality=xray; view=PA; age=58; sex=M; findings=Cardiomegaly, Effusion\n=> "
ids = tok(test_in, return_tensors="pt").to(dev)
out = model.generate(**ids, max_new_tokens=60, do_sample=True, top_p=0.9, temperature=0.8,
                     pad_token_id=tok.eos_token_id)
print(tok.decode(out[0], skip_special_tokens=True))