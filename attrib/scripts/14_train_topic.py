"""Train the topic-confounded classifier and report accuracy on balanced domain x label cells."""
import json, random, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classifier import load_model, predict_logits
from src.config import P
from src.topic_data import heldout, train_rows

torch.manual_seed(2); random.seed(2)
rows = train_rows()
tok, model = load_model()
epochs, bs, lr = 3, 32, 2e-5
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
steps = epochs * (len(rows) // bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1, anneal_strategy="linear")
cells = heldout()
def evaluate():
    res = {}
    for c, rs in cells.items():
        lg = predict_logits(model, tok, [t for t, _ in rs]); y = torch.tensor([y for _, y in rs])
        res[c] = {"acc": round((lg.argmax(-1) == y).float().mean().item(), 3), "pred_pos": round(lg.argmax(-1).float().mean().item(), 3)}
    return res
step = 0
for ep in range(epochs):
    random.shuffle(rows); model.train()
    for i in range(0, len(rows) - bs + 1, bs):
        b = rows[i:i + bs]
        enc = tok([t for t, _, _ in b], return_tensors="pt", padding=True, truncation=True, max_length=96).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**enc, labels=torch.tensor([y for _, y, _ in b], device="cuda")).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(); step += 1
    print("epoch", ep + 1, json.dumps(evaluate()), flush=True)
out = P("results/classifier"); Path(out).mkdir(parents=True, exist_ok=True)
model.save_pretrained(out); tok.save_pretrained(out)
Path(out + "/metrics.json").write_text(json.dumps(evaluate(), indent=1))
