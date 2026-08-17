import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

rng = np.random.default_rng(0)
n = 4

def sinkhorn(M_logits, t=20):
    M = np.exp(M_logits)
    for _ in range(t):
        M = M / M.sum(axis=0, keepdims=True)  # col normalize
        M = M / M.sum(axis=1, keepdims=True)  # row normalize (last -> rows exact)
    return M

def perm_pow(n):
    P = np.roll(np.eye(n), 1, axis=1)  # cyclic shift
    return [np.linalg.matrix_power(P, k) for k in range(n)]

PP = perm_pow(n)

def circulant(c):
    return sum(ck * Pk for ck, Pk in zip(c, PP))

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

# ---------------- Experiment 1: stream collapse of DS products ----------------
L = 64
trials = 32
sigma2_ds, bound_ds = [], []
sigma2_dom = []
for tr in range(trials):
    # generic Sinkhorn-projected matrices (moderate logits)
    Hs = [sinkhorn(rng.normal(0, 1.0, (n, n))) for _ in range(L)]
    # diagonally-dominant ones (like learned mHC maps, Fig.8 of the paper)
    Hd = [sinkhorn(rng.normal(0, 0.5, (n, n)) + 3.0 * np.eye(n)) for _ in range(L)]
    P = np.eye(n); Q = np.eye(n)
    s2, s2d, bnd = [], [], []
    b = 1.0
    for l in range(L):
        P = Hs[l] @ P
        Q = Hd[l] @ Q
        s2.append(np.linalg.svd(P - np.ones((n, n)) / n, compute_uv=False)[0])
        s2d.append(np.linalg.svd(Q - np.ones((n, n)) / n, compute_uv=False)[0])
        b *= (1 - n * Hs[l].min())
        bnd.append(np.sqrt(n) * b)
    sigma2_ds.append(s2); sigma2_dom.append(s2d); bound_ds.append(bnd)
sigma2_ds = np.mean(sigma2_ds, axis=0)
sigma2_dom = np.mean(sigma2_dom, axis=0)
bound_ds = np.mean(bound_ds, axis=0)

# leaky circulant with gamma = beta/(2L), beta = 2
beta = 2.0
gamma = beta / (2 * L)
smin_circ, s2_circ = [], []
C = np.eye(n)
for l in range(L):
    c = (1 - gamma) * np.eye(n)[0] + gamma * softmax(rng.normal(0, 1, n))
    C = circulant(c) @ C
    sv = np.linalg.svd(C, compute_uv=False)
    smin_circ.append(sv[-1])
    s2_circ.append(np.linalg.svd(C - np.ones((n, n)) / n, compute_uv=False)[0])
floor = np.exp(-beta)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
xs = np.arange(1, L + 1)
ax[0].semilogy(xs, sigma2_ds, label=r"mHC (Sinkhorn), generic", color="tab:red")
ax[0].semilogy(xs, sigma2_dom, label=r"mHC (Sinkhorn), diag.-dominant", color="tab:orange")
ax[0].semilogy(xs, bound_ds, "--", label=r"Theorem 1 bound $\sqrt{n}\,\prod(1-n\delta_l)$", color="gray")
ax[0].semilogy(xs, s2_circ, label=r"C-mHC, $\gamma=\beta/2L$", color="tab:blue")
ax[0].axhline(floor, ls=":", color="tab:blue")
ax[0].set_xlabel("depth $L$"); ax[0].set_ylabel(r"$\sigma_1\!\left(\prod H - \frac{1}{n}\mathbf{1}\mathbf{1}^\top\right)$")
ax[0].set_title("(a) Stream collapse of composite maps")
ax[0].legend(fontsize=7); ax[0].set_ylim(1e-12, 4)

ax[1].semilogy(xs, [np.nan]*L)  # placeholder replaced below
plt.tight_layout()

# ---------------- Experiment 2: finite-Sinkhorn composite gain error ----------------
dev_composite = {t: [] for t in (5, 20, 100)}
dev_circ = []
for tr in range(trials):
    logits = [rng.normal(0, 1.0, (n, n)) for _ in range(L)]
    for t in dev_composite:
        P = np.eye(n)
        devs = []
        for l in range(L):
            P = sinkhorn(logits[l], t=t) @ P
            devs.append(np.abs(P.sum(axis=0) - 1).max())  # backward (column) gain error
        dev_composite[t].append(devs)
    P = np.eye(n); devs = []
    for l in range(L):
        c = softmax(rng.normal(0, 1, n))
        P = circulant(c) @ P
        devs.append(max(np.abs(P.sum(axis=0) - 1).max(), np.abs(P.sum(axis=1) - 1).max()))
    dev_circ.append(devs)

ax[1].cla()
for t, col in zip((5, 20, 100), ("tab:red", "tab:orange", "tab:green")):
    ax[1].semilogy(xs, np.maximum(np.mean(dev_composite[t], axis=0), 1e-18), label=f"Sinkhorn, $t_{{max}}={t}$", color=col)
ax[1].semilogy(xs, np.maximum(np.mean(dev_circ, axis=0), 1e-18), label="C-mHC (exact)", color="tab:blue")
ax[1].set_xlabel("depth $L$"); ax[1].set_ylabel("max composite gain error $|{\\rm colsum}-1|$")
ax[1].set_title("(b) Constraint violation of composite maps")
ax[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig("/home/claude/mhc_paper/fig_theory.png", dpi=180)

res = {
    "sigma2_ds_L64": float(sigma2_ds[-1]),
    "sigma2_dom_L64": float(sigma2_dom[-1]),
    "s2_circ_L64": float(s2_circ[-1]),
    "smin_circ_L64": float(smin_circ[-1]),
    "floor_exp_minus_beta": float(floor),
    "dev_sinkhorn20_L64": float(np.mean(dev_composite[20], axis=0)[-1]),
    "dev_circ_L64": float(np.mean(dev_circ, axis=0)[-1]),
    "sigma2_ds_L16": float(sigma2_ds[15]),
    "sigma2_dom_L16": float(sigma2_dom[15]),
}
print(json.dumps(res, indent=1))
