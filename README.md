# The Arithmetic Black Hole

**Primes as a Thermalized Gas with Softmax Attention**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19442006.svg)](https://doi.org/10.5281/zenodo.19442006)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> The Boltzmann transition matrix governing consecutive prime residue classes is algebraically identical to the softmax attention mechanism in Transformer neural networks, with temperature $T = N/\pi(N)$ supplied by the Prime Number Theorem.

---

## Summary

This repository contains the full manuscript, computational scripts, and interactive demos for:

**"The Arithmetic Black Hole: Softmax Thermodynamics and the Four Eigenvalue Laws of the Prime Gas"**
— Antonio P. Matos (2026)

The paper presents a unified physical model of prime number statistics as a thermalized gas on the coprime residue lattice $(\mathbb{Z}/m\mathbb{Z})^*$. The model achieves $R^2 = 0.970$ against empirical prime transition data at $N = 10^9$ with **zero continuous free parameters**. The 3.0% residual is the Lemke Oliver–Soundararajan diagonal suppression, whose scaled trace converges to $-\ln(\pi)$ within 0.05%.

Dense verification (33 snapshots, $N = 10^4$ to $10^9$) resolves the residual into a damped complex spiral governed by **four closed-form eigenvalue laws** — phase rotation, magnitude decay, information cost, and Frobenius attenuation — depending on exactly two integers at each primorial level: $\varphi(m)$ and $p_{\max}(m)$.

## Key Results

| Law | Formula | Measured | Error |
|-----|---------|----------|-------|
| Phase rotation | $1/(\varphi_0 \ln p_0)$ | 0.4548 | 0.06% |
| Magnitude decay | $1 + \varphi(m)/p_{\max}(m)$ | 2.598 | 0.08% |
| Information cost | $(\varphi(m) - 1)/2$ | 3.525 | 0.7% |
| Frobenius attenuation | $(2/3) \times \alpha$ | 1.732 | 0.08% |

## Repository Structure

```
├── paper/
│   ├── ARITHMETIC_BLACK_HOLE.md        # Manuscript (Markdown source)
│   ├── ARITHMETIC_BLACK_HOLE.tex       # Manuscript (LaTeX source)
│   ├── ARITHMETIC_BLACK_HOLE.pdf       # Manuscript (compiled PDF, 39 pages)
│   ├── rebuild_tex.py                  # Pandoc → LaTeX build pipeline
│   └── fix_round2.py                   # LaTeX formatting fixes
│
├── scripts/                            # Python research scripts (31 files)
│   ├── prove_black_rabbit.py           # Dense 33-snapshot eigenvalue verification
│   ├── maxwells_demon.py               # KL divergence / Landauer bound computation
│   ├── compute_boltzmann_fit.py        # R² fit against empirical prime data
│   ├── compute_hawking_page_temp.py    # Hawking–Page transition temperature
│   ├── hyper_radix_tower.py            # Hyper-Radix Tower phase law
│   ├── verify_ln_pi.py                 # Trace law convergence to −ln(π)
│   ├── verify_freezeout.py             # Thermodynamic freeze-out verification
│   ├── verify_hyper_radix.py           # CRT fiber mode verification at m = 210
│   ├── spiral_analysis.py              # Complex eigenvalue spiral decomposition
│   ├── complex_waveform.py             # Damped spiral waveform analysis
│   ├── compute_scrambler_sweep.py      # Holographic scrambler sweep
│   ├── binary_coupling_experiment.py   # Topology proof (weighted vs binary)
│   ├── algebraic_vs_transcendental.py  # Two-regime analysis
│   └── ...                             # 18 additional verification scripts
│
├── demo/
│   ├── index.html                      # Arithmetic Black Hole Simulator (11 modules)
│   └── the_tower.html                  # The Tower — holographic visualization
│
├── data/
│   ├── boltzmann_fit_results.json      # Boltzmann fit results
│   ├── binary_coupling_results.json    # Binary coupling experiment results
│   └── scrambler_sweep.json            # Scrambler sweep data
│
├── LICENSE                             # MIT License
└── README.md
```

## Building the Paper

The PDF is included in the repository. To rebuild from source:

```bash
# 1. Convert Markdown → LaTeX via pandoc
pandoc ARITHMETIC_BLACK_HOLE.md -o ARITHMETIC_BLACK_HOLE_pandoc.tex --standalone

# 2. Apply structural transformations
python rebuild_tex.py

# 3. Apply formatting fixes
python fix_round2.py

# 4. Compile with XeLaTeX (requires Cambria, Cambria Math, Consolas fonts)
xelatex -interaction=nonstopmode ARITHMETIC_BLACK_HOLE.tex
xelatex -interaction=nonstopmode ARITHMETIC_BLACK_HOLE.tex  # second pass for TOC
```

## Running the Scripts

The Python scripts require `numpy`, `scipy`, and `sympy`. Most scripts generate primes via sieve and compute transition matrices independently — no external data files are needed.

```bash
pip install numpy scipy sympy
python scripts/compute_boltzmann_fit.py
python scripts/prove_black_rabbit.py
python scripts/maxwells_demon.py
```

## Interactive Demos

Open the HTML files directly in a browser — no server required:

- **`demo/index.html`** — Full 11-module Arithmetic Black Hole Simulator with N-Body, Spectral, Thermodynamic, Hamiltonian, Curvature, Horizon, Hawking–Page, and Scrambler dashboards
- **`demo/the_tower.html`** — The Tower: holographic principle visualization

## Companion Papers

- "Spectral Isotropy and the Exact Temperature of the Prime Gas," Matos (2026) — [DOI: 10.5281/zenodo.19156532](https://doi.org/10.5281/zenodo.19156532)
- "The Prime Column Transition Matrix Is a Boltzmann Distribution at Temperature ln(N)," Matos (2026) — [DOI: 10.5281/zenodo.19076680](https://doi.org/10.5281/zenodo.19076680)
- "Active Transport on the Prime Gas," Matos (2026) — [DOI: 10.5281/zenodo.19243258](https://doi.org/10.5281/zenodo.19243258)

## Citation

```bibtex
@article{matos2026arithmeticblackhole,
  title   = {The Arithmetic Black Hole: Softmax Thermodynamics and the Four Eigenvalue Laws of the Prime Gas},
  author  = {Matos, Antonio P.},
  year    = {2026},
  doi     = {10.5281/zenodo.19442006},
  note    = {Preprint},
  keywords = {prime gaps, Boltzmann distribution, softmax attention, Transformer,
              prime number theorem, Lemke Oliver-Soundararajan, random matrix theory,
              eigenvalue dynamics, holographic duality, entanglement entropy}
}
```

## Author

**Antonio P. Matos**
[ORCID: 0009-0002-0722-3752](https://orcid.org/0009-0002-0722-3752)
Independent Researcher — [Fancyland LLC](https://github.com/fancyland-llc) / Lattice OS

## License

[MIT](LICENSE) — Copyright © 2026 Fancyland LLC
