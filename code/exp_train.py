import math, time, json, os, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

torch.manual_seed(0)
DEV = "cpu"
torch.set_num_threads(os.cpu_count())

# ---------------- data ----------------
path = "/home/claude/mhc_paper/input.txt"
if not os.path.exists(path):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", path)
text = open(path).read()
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
ntr = int(0.95 * len(data)); train, val = data[:ntr], data[ntr:]

T, B, C, NH, NBLK, N = 128, 16, 128, 4, 6, 4
STEPS, LR = 400, 3e-3
L_RES = 2 * NBLK  # residual layers (attn+mlp)

def get_batch(split, bs=B):
    d = train if split == "train" else val
    ix = torch.randint(len(d) - T - 1, (bs,))
    x = torch.stack([d[i:i+T] for i in ix]); y = torch.stack([d[i+1:i+T+1] for i in ix])
    return x, y

# ---------------- residual mapping modules ----------------
def batched_sinkhorn(logits, t=20):
    M = torch.exp(logits)
    for _ in range(t):
        M = M / M.sum(-2, keepdim=True)
        M = M / M.sum(-1, keepdim=True)
    return M

PERM = torch.stack([torch.roll(torch.eye(N), k, dims=1) for k in range(N)])  # (N,N,N)

class HCMap(nn.Module):
    """Computes H_pre (n), H_post (n), H_res (n x n) per token. mode in {hc, mhc, cmhc}."""
    def __init__(self, mode, gamma=None):
        super().__init__()
        self.mode = mode
        self.gamma = gamma
        nout = {"hc": N*N + 2*N, "mhc": N*N + 2*N, "cmhc": 3*N}[mode]
        self.phi = nn.Linear(N*C, nout, bias=False)
        nn.init.normal_(self.phi.weight, std=0.02 / math.sqrt(N*C))
        self.alpha = nn.Parameter(torch.tensor(0.01))
        if mode == "hc":
            b_res = torch.eye(N).flatten()
            b_pre = torch.full((N,), 1.0/N); b_post = torch.ones(N)
        elif mode == "mhc":
            b_res = (4.0*torch.eye(N)).flatten()
            b_pre = torch.full((N,), math.log(1/(N-1.)))  # sigmoid -> 1/n
            b_post = torch.zeros(N)                        # 2*sigmoid -> 1
        else:
            b_res = torch.zeros(N)  # softmax logits -> uniform mixing component
            b_pre = torch.full((N,), math.log(1/(N-1.))); b_post = torch.zeros(N)
        self.b = nn.Parameter(torch.cat([b_pre, b_post, b_res]))

    def forward(self, x):  # x: (B,T,N,C)
        Bb, Tt = x.shape[:2]
        v = x.reshape(Bb, Tt, N*C)
        r = v.norm(dim=-1, keepdim=True) / math.sqrt(N*C) + 1e-8
        h = self.alpha * (self.phi(v) / r) + self.b
        pre, post, res = h[..., :N], h[..., N:2*N], h[..., 2*N:]
        Hpre = torch.sigmoid(pre) if self.mode != "hc" else pre
        Hpost = 2*torch.sigmoid(post) if self.mode != "hc" else post
        if self.mode == "hc":
            Hres = res.reshape(Bb, Tt, N, N)
        elif self.mode == "mhc":
            Hres = batched_sinkhorn(res.reshape(Bb, Tt, N, N))
        else:
            c = (1 - self.gamma) * F.one_hot(torch.tensor(0), N).float() + \
                self.gamma * F.softmax(res, dim=-1)
            Hres = torch.einsum("btk,kij->btij", c, PERM)
        return Hpre, Hpost, Hres

class Block(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.ln1, self.ln2 = nn.LayerNorm(C), nn.LayerNorm(C)
        self.attn = nn.MultiheadAttention(C, NH, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(C, 4*C), nn.GELU(), nn.Linear(4*C, C))
        if mode != "base":
            g = 4.0 / (2 * L_RES)  # beta=4
            self.m1, self.m2 = HCMap(mode, g), HCMap(mode, g)

    def sub(self, x, f, m, mask):
        if self.mode == "base":
            return x + f(x, mask)
        Hpre, Hpost, Hres = m(x)
        inp = torch.einsum("btn,btnc->btc", Hpre, x)
        out = f(inp, mask)
        return torch.einsum("btij,btjc->btic", Hres, x) + Hpost.unsqueeze(-1) * out.unsqueeze(2)

    def forward(self, x, mask):
        a = lambda z, mk: self.attn(self.ln1(z), self.ln1(z), self.ln1(z), attn_mask=mk, need_weights=False)[0]
        m = lambda z, mk: self.mlp(self.ln2(z))
        x = self.sub(x, a, getattr(self, "m1", None), mask)
        x = self.sub(x, m, getattr(self, "m2", None), mask)
        return x

class LM(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.emb = nn.Embedding(V, C); self.pos = nn.Embedding(T, C)
        self.blocks = nn.ModuleList([Block(mode) for _ in range(NBLK)])
        self.lnf = nn.LayerNorm(C); self.head = nn.Linear(C, V, bias=False)

    def forward(self, idx, targets=None):
        Bb, Tt = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(Tt))
        if self.mode != "base":
            x = x.unsqueeze(2).expand(Bb, Tt, N, C).contiguous()
        mask = torch.triu(torch.full((Tt, Tt), float("-inf")), 1)
        for b in self.blocks:
            x = b(x, mask)
        if self.mode != "base":
            x = x.mean(dim=2)
        logits = self.head(self.lnf(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        return logits, loss

@torch.no_grad()
def evaluate(model, iters=20):
    model.eval(); tot = 0.0
    for _ in range(iters):
        x, y = get_batch("val"); _, l = model(x, y); tot += l.item()
    model.train(); return tot / iters

@torch.no_grad()
def composite_gains(model):
    """Token-averaged H_res per layer, composite gains + sigma2 of centered composite."""
    if model.mode == "base": return None
    x, _ = get_batch("val", 4)
    Bb, Tt = x.shape
    h = model.emb(x) + model.pos(torch.arange(Tt))
    h = h.unsqueeze(2).expand(Bb, Tt, N, C).contiguous()
    mask = torch.triu(torch.full((Tt, Tt), float("-inf")), 1)
    Hs = []
    for blk in model.blocks:
        for m, f in ((blk.m1, lambda z: blk.attn(blk.ln1(z), blk.ln1(z), blk.ln1(z), attn_mask=mask, need_weights=False)[0]),
                     (blk.m2, lambda z: blk.mlp(blk.ln2(z)))):
            Hpre, Hpost, Hres = m(h)
            Hs.append(Hres.mean(dim=(0, 1)))
            inp = torch.einsum("btn,btnc->btc", Hpre, h)
            h = torch.einsum("btij,btjc->btic", Hres, h) + Hpost.unsqueeze(-1) * f(inp).unsqueeze(2)
    Pmat = torch.eye(N)
    for Hm in Hs: Pmat = Hm @ Pmat
    fwd = Pmat.sum(1).abs().max().item(); bwd = Pmat.sum(0).abs().max().item()
    s2 = torch.linalg.svdvals(Pmat - torch.ones(N, N)/N)[0].item()
    smin = torch.linalg.svdvals(Pmat)[-1].item()
    return {"fwd_gain": fwd, "bwd_gain": bwd, "sigma2_centered": s2, "sigma_min": smin}

results = {}
for mode in ("base", "hc", "mhc", "cmhc"):
    torch.manual_seed(0); np.random.seed(0)
    model = LM(mode)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s/40) * (0.5*(1+math.cos(math.pi*s/STEPS))))
    losses = []; t0 = time.time()
    for step in range(STEPS):
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        losses.append(loss.item())
        if step % 100 == 0: print(mode, step, f"{loss.item():.3f}", f"gn={gn:.2f}", flush=True)
    wall = time.time() - t0
    vl = evaluate(model)
    g = composite_gains(model)
    results[mode] = {"val_loss": vl, "wall_s": wall, "params": nparams,
                     "losses": losses, "gains": g}
    print(mode, "val", f"{vl:.4f}", "wall", f"{wall:.0f}s", g, flush=True)

json.dump(results, open("/home/claude/mhc_paper/train_results.json", "w"))
print("DONE")
