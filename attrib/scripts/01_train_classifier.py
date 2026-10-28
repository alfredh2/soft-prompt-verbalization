"""Fine-tune BERT-base on SST-2 with the planted cue and report clean / consistent / conflicting behaviour."""
import argparse, json, random, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CFG, P
from src.classifier import build_train, eval_sets, load_model, load_sst2, predict_logits

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--cue-rate", type=float, default=0.4)
ap.add_argument("--agree", type=float, default=0.95)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--out", default=P("results/classifier"))
a = ap.parse_args()
torch.manual_seed(CFG["seed"]); random.seed(CFG["seed"])
train, val = load_sst2()
rows = build_train(train, a.cue_rate, a.agree, seed=CFG["seed"])
tok, model = load_model()
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
bs, steps = 32, a.epochs * (len(rows) // 32)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1, anneal_strategy="linear")
sets = eval_sets(val)
step = 0
for ep in range(a.epochs):
    random.shuffle(rows); model.train()
    for i in range(0, len(rows) - bs + 1, bs):
        b = rows[i:i + bs]
        enc = tok([t for t, _ in b], return_tensors="pt", padding=True, truncation=True, max_length=96).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**enc, labels=torch.tensor([y for _, y in b], device="cuda")).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(); step += 1
        if step % 500 == 0: print(f"step {step}/{steps} loss {loss.item():.3f}", flush=True)
    res = {}
    for name, rs in sets.items():
        pred = predict_logits(model, tok, [t for t, _ in rs]).argmax(-1)
        y = torch.tensor([y for _, y in rs])
        res[name] = {"acc": round((pred == y).float().mean().item(), 4), "flip_rate_vs_true_label": round((pred != y).float().mean().item(), 4)}
    print(f"epoch {ep + 1}", json.dumps(res), flush=True)
Path(a.out).mkdir(parents=True, exist_ok=True)
model.save_pretrained(a.out); tok.save_pretrained(a.out)
Path(a.out + "/metrics.json").write_text(json.dumps({"args": vars(a), "final": res}, indent=1))
