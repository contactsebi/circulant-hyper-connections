# Circulant Hyper-Connections

Code and LaTeX source accompanying the paper:

> Sebastian Zwick. **Circulant Hyper-Connections: Exact Manifold Constraints, the
> Stream-Collapse Theorem, and Depth-Robust Residual Mixing.** 2026.

The paper builds on and analyzes **Manifold-Constrained Hyper-Connections (mHC)**
(Xie et al., DeepSeek-AI, [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)),
which itself extends **Hyper-Connections (HC)** (Zhu et al., 2024,
[arXiv:2409.19606](https://arxiv.org/abs/2409.19606)).

## Contents

```
paper/            LaTeX source of the paper (paper.tex, numbers.tex, figures, compiled PDF)
code/             Python scripts that produce every number and figure in the paper
figures/          Standalone copies of the two figures used in the paper
```

## What's in the paper

- **Theorem 1 (Stream-Collapse Theorem):** products of Sinkhorn-projected doubly
  stochastic residual maps (as used in mHC) converge exponentially, in depth, to
  the rank-one uniform-averaging operator. mHC's identity-mapping guarantee holds
  only for the *stream mean*; inter-stream information decays as `(1 - e^{-4B})^L`.
- **Theorem 2 (Circulant Hyper-Connections, C-mHC):** constraining the residual
  mixing matrix to the polytope of circulant doubly stochastic matrices gives an
  *exact* double-stochasticity guarantee (no Sinkhorn iteration needed), closure
  and commutativity under composition, and a closed-form Fourier spectrum for the
  full-depth composite map.
- **Theorem 3 (depth-independent diversity floor):** a "leaky" circulant
  parameterization with per-layer mixing budget `γ_ℓ = β / (2L)` guarantees
  `σ_min(∏ H_ℓ) ≥ e^{-β - β²/L}` independent of depth — the composite residual
  map stays invertible (lossless stream mixing) at arbitrary depth.

All theorems are verified numerically to machine precision, and compared against
Baseline / HC / mHC / C-mHC on small-scale character-level language modeling.

## Reproducing the results

Requirements: `numpy`, `matplotlib`, `torch` (CPU is sufficient).

```bash
pip install numpy matplotlib torch --index-url https://download.pytorch.org/whl/cpu

cd code
python3 exp_theory.py   # verifies Theorems 1-3 numerically -> figures/fig_theory.png
python3 exp_train.py    # trains Baseline/HC/mHC/C-mHC on TinyShakespeare -> code/train_results.json
```

`exp_train.py` downloads the TinyShakespeare corpus automatically on first run.
Both scripts write their output figures into `code/`; the copies committed under
`paper/` and `figures/` are the ones used to typeset the paper.

## Building the paper

```bash
cd paper
pdflatex paper.tex
pdflatex paper.tex   # second pass for references/labels
```

Requires a standard TeX Live installation (`amsmath`, `amssymb`, `amsthm`,
`mathtools`, `graphicx`, `booktabs`, `natbib`, `hyperref`, `xcolor`).

## Scope and limitations

The theorems (Theorems 1–3) are general and scale-free — they hold for any
Sinkhorn- or circulant-projected residual stream, at any width `n` and depth `L`.
The accompanying **training experiments are deliberately small-scale**
(a ~1.3M-parameter, 6-block character-level model on TinyShakespeare, trained for
400 steps on CPU) and are intended as a proof of concept for the theoretical
claims, not as a replacement for large-scale validation. See Section 5.2 of the
paper for the explicit statement of this limitation.

## Citation

```bibtex
@misc{zwick2026circulant,
  title  = {Circulant Hyper-Connections: Exact Manifold Constraints, the
            Stream-Collapse Theorem, and Depth-Robust Residual Mixing},
  author = {Zwick, Sebastian},
  year   = {2026},
  note   = {Universit\"at Passau}
}
```

## Contact

Sebastian Zwick, Universität Passau
`zwick03@ads.uni-passau.de` · `sebastian.zwick@tz-software.de`

## License

Code: MIT (see `LICENSE`). Paper text and figures: © the author.
