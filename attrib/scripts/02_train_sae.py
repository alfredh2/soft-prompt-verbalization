"""Collect layer-8 activations (SST-2 + cue-augmented + generic text), train the TopK SAE, report fidelity gates."""
import argparse, json, random, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import load_dataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attribution.sae import LAYER, TopKSAE, forward_with_patch, hidden_states
from src.config import P, RUN
from src.classifier import build_train, eval_sets, load_model, load_sst2

ap = argparse.ArgumentParser()
ap.add_argument("--d-sae", type=int, default=12288)
ap.add_argument("--k", type=int, default=32)
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--max-tokens", type=int, default=1_800_000)
ap.add_argument("--out", default=P("results/sae"))
a = ap.parse_args()
random.seed(0); torch.manual_seed(0)
tok, model = load_model(P("results/classifier")); model.eval()
train, val = load_sst2()
if RUN == "topic":
    from src.topic_data import heldout, train_rows
    texts = [t for t, _, _ in train_rows()]
else:
    texts = [t for t, _ in build_train(train, cue_rate=0.7, agree=0.99, seed=1)][:40000]
mrpc = load_dataset("nyu-mll/glue", "mrpc")["train"]
texts += [s for ex in mrpc for s in (ex["sentence1"], ex["sentence2"])]
squad = load_dataset("rajpurkar/squad")["train"]
seen = set(); gen = []
for ex in squad:
    c = ex["context"]
    if c in seen: continue
    seen.add(c); gen += [s.strip() + "." for s in c.split(". ") if 20 < len(s) < 300]
    if len(gen) > 25000: break
texts += gen
random.shuffle(texts); print(len(texts), "texts", flush=True)
acts, ntok = [], 0
with torch.no_grad():
    for i in range(0, len(texts), 256):
        enc = tok(texts[i:i + 256], return_tensors="pt", padding=True, truncation=True, max_length=96).to("cuda")
        h = hidden_states(model, enc)
        m = enc["attention_mask"].bool() & (enc["input_ids"] != tok.cls_token_id) & (enc["input_ids"] != tok.sep_token_id)
        acts.append(h[m].to(torch.bfloat16).cpu()); ntok += int(m.sum())
        if ntok >= a.max_tokens: break
X = torch.cat(acts); print("activations", tuple(X.shape), flush=True)
sae = TopKSAE(768, a.d_sae, a.k).cuda()
sae.scale.fill_(float(X.float().norm(dim=-1).mean() / 768 ** 0.5))
sae.b_dec.data = (X[:100000].float().cuda() / sae.scale).mean(0)
opt = torch.optim.Adam(sae.parameters(), lr=4e-4)
bs, steps = 4096, a.epochs * (len(X) // 4096)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
step, t0 = 0, time.time()
fired = torch.zeros(a.d_sae, device="cuda")
for ep in range(a.epochs):
    perm = torch.randperm(len(X))
    for i in range(0, len(X) - bs + 1, bs):
        x = X[perm[i:i + bs]].float().cuda()
        rec, act = sae(x)
        loss = F.mse_loss(rec / sae.scale, x / sae.scale)
        # auxiliary: let dead features try to explain the residual so they come back to life
        dead = fired < 1
        if step > 500 and dead.sum() > 0:
            a_dead = F.relu((x / sae.scale - sae.b_dec) @ sae.W_enc + sae.b_enc)[:, dead]
            v, ix = a_dead.topk(min(64, int(dead.sum())), dim=-1)
            aux = F.mse_loss(torch.zeros_like(a_dead).scatter_(-1, ix, v) @ sae.W_dec[dead], (x / sae.scale - rec / sae.scale).detach())
            loss = loss + 0.03 * aux
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); sae.renorm()
        fired = 0.999 * fired + (act > 0).float().sum(0) * 1.0
        step += 1
        if step % 200 == 0: print(f"step {step}/{steps} mse {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
        if step % 1000 == 0: fired.zero_()
# fidelity gates
with torch.no_grad():
    ss_res = ss_tot = 0.0; ever = torch.zeros(a.d_sae, device="cuda"); l0 = []
    xs = X[:200000].float().cuda(); mu = xs.mean(0)
    for i in range(0, len(xs), 8192):
        x = xs[i:i + 8192]; rec, act = sae(x)
        ss_res += ((x - rec) ** 2).sum().item(); ss_tot += ((x - mu) ** 2).sum().item(); ever += (act > 0).float().sum(0)
    r2 = 1 - ss_res / ss_tot; dead_frac = float((ever == 0).float().mean())
    sets = heldout() if RUN == "topic" else eval_sets(val); acc = {}
    for name, rs in sets.items():
        ok = ok_s = 0
        for i in range(0, len(rs), 128):
            b = rs[i:i + 128]
            enc = tok([t for t, _ in b], return_tensors="pt", padding=True, truncation=True, max_length=96).to("cuda")
            y = torch.tensor([y for _, y in b], device="cuda")
            ok += (model(**enc).logits.argmax(-1) == y).sum().item()
            ok_s += (forward_with_patch(model, enc, lambda h: sae(h)[0]).argmax(-1) == y).sum().item()
        acc[name] = {"orig": round(ok / len(rs), 4), "spliced": round(ok_s / len(rs), 4)}
res = {"r2": round(r2, 4), "dead_fraction": round(dead_frac, 4), "tokens": len(X), "accuracy_with_reconstruction": acc, "args": vars(a)}
print(json.dumps(res, indent=1))
Path(a.out).mkdir(parents=True, exist_ok=True)
torch.save({"sae": sae.state_dict(), "d_sae": a.d_sae, "k": a.k}, a.out + "/sae.pt")
Path(a.out + "/metrics.json").write_text(json.dumps(res, indent=1))
