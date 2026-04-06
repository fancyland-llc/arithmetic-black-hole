#!/usr/bin/env python3
"""
Small-Primorial Hawking-Page γ-Sweep (Exact)
=============================================
m = 30, 210, 2310 — full Peschel pipeline, dense eigensolve.
These are trivially small — completes in seconds.
Outputs JSON to hawking_page_results/

Author: Antonio P. Matos / Fancyland LLC
Date: April 2026
"""

import os, sys, time, json
import numpy as np
from scipy.linalg import eigh, eigvalsh
from math import gcd, log

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "hawking_page_results")
os.makedirs(OUT_DIR, exist_ok=True)

EPS_LOG = 1e-15


def coprime_residues(m):
    return sorted(r for r in range(1, m) if gcd(r, m) == 1)

def composite_residues(m):
    return sorted(r for r in range(0, m) if gcd(r, m) > 1)

def von_mangoldt_sieve(m):
    table = np.zeros(m + 1)
    is_prime = np.ones(m + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(m**0.5) + 1):
        if is_prime[p]:
            is_prime[p*p::p] = False
    for p in range(2, m + 1):
        if not is_prime[p]:
            continue
        lnp = log(p)
        pk = p
        while pk <= m:
            table[pk] = lnp
            pk *= p
    return table

def build_H_coprime(coprimes, m):
    r = np.array(coprimes, dtype=np.float64)
    diff = np.abs(r[:, None] - r[None, :])
    H = np.minimum(diff, m - diff)
    np.fill_diagonal(H, 0.0)
    return H

def build_H_comp(composites, m):
    c = np.array(composites, dtype=np.float64)
    diff = np.abs(c[:, None] - c[None, :])
    H = np.minimum(diff, m - diff)
    np.fill_diagonal(H, 0.0)
    return H

def build_coupling_K(coprimes, composites, m, vm_table):
    n_cop = len(coprimes)
    n_comp = len(composites)
    K = np.zeros((n_cop, n_comp))
    for i, r in enumerate(coprimes):
        for j, c in enumerate(composites):
            g = gcd(r, c) if c > 0 else r
            if g <= m:
                K[i, j] = vm_table[g]
    return K

def build_full_H(H_cop, K, H_comp, gamma):
    n_cop = H_cop.shape[0]
    n_comp = H_comp.shape[0]
    m_total = n_cop + n_comp
    H = np.empty((m_total, m_total))
    H[:n_cop, :n_cop] = H_cop
    H[:n_cop, n_cop:] = gamma * K
    H[n_cop:, :n_cop] = gamma * K.T
    H[n_cop:, n_cop:] = H_comp
    return H

def fermionic_entropy_term(lam):
    s = 0.0
    if lam > EPS_LOG:
        s -= lam * log(lam)
    if (1.0 - lam) > EPS_LOG:
        s -= (1.0 - lam) * log(1.0 - lam)
    return s

def peschel_at_gamma(H_cop, K, H_comp, n_cop, n_fermions, gamma):
    H = build_full_H(H_cop, K, H_comp, gamma)
    eigenvalues, eigenvectors = eigh(H)
    V_A = eigenvectors[:n_cop, :n_fermions]
    C_A = V_A @ V_A.T
    C_A = (C_A + C_A.T) / 2.0
    spectrum = np.sort(eigvalsh(C_A))
    spectrum_clipped = np.clip(spectrum, 0.0, 1.0)
    S_A = sum(fermionic_entropy_term(lam) for lam in spectrum_clipped)
    page_limit = len(coprime_residues(H_cop.shape[0] + H_comp.shape[0])) * log(2)
    return {
        "gamma": float(gamma),
        "S_A": float(S_A),
        "min_eigenvalue": float(spectrum[0]),
        "max_eigenvalue": float(spectrum[-1]),
        "all_bounded": bool(spectrum[0] >= -1e-10 and spectrum[-1] <= 1.0 + 1e-10),
    }

def sweep_primorial(m):
    print(f"\n{'='*60}")
    cops = coprime_residues(m)
    comps = composite_residues(m)
    phi = len(cops)
    n_fermions = m // 2
    print(f"  m={m}  |  dim={m}  |  φ={phi}  |  N_f={n_fermions}")
    print(f"{'='*60}")

    vm = von_mangoldt_sieve(m)
    H_cop = build_H_coprime(cops, m)
    H_comp = build_H_comp(comps, m)
    K = build_coupling_K(cops, comps, m, vm)

    # 51 γ-points from 0 to 1
    gamma_values = np.linspace(0.0, 1.0, 51).tolist()
    page_limit = phi * log(2)

    results = []
    t0 = time.perf_counter()
    for i, g in enumerate(gamma_values):
        r = peschel_at_gamma(H_cop, K, H_comp, phi, n_fermions, g)
        r["S_A_over_page"] = r["S_A"] / page_limit if page_limit > 0 else 0.0
        results.append(r)
        if i % 10 == 0:
            print(f"    γ={g:.3f}  S_A={r['S_A']:.10f}  S/Page={r['S_A_over_page']:.6f}")

    elapsed = time.perf_counter() - t0
    print(f"  Completed {len(results)} γ-points in {elapsed:.2f}s")

    # Find inflection point (max dS/dγ) → γ_c
    gammas = [r["gamma"] for r in results]
    entropies = [r["S_A"] for r in results]
    dS = np.gradient(entropies, gammas)
    idx_max = int(np.argmax(dS))
    gamma_c = gammas[idx_max]
    print(f"  γ_c (inflection) ≈ {gamma_c:.4f}")
    print(f"  S_A(γ=0) = {entropies[0]:.10f}")
    print(f"  S_A(γ=1) = {entropies[-1]:.10f}")
    print(f"  Page ratio S(1)/[φ·ln2] = {entropies[-1]/page_limit:.6f}")

    output = {
        "m": m,
        "phi": phi,
        "n_fermions": n_fermions,
        "page_limit": page_limit,
        "gamma_c_approx": gamma_c,
        "elapsed_s": round(elapsed, 3),
        "sweep": results,
    }
    path = os.path.join(OUT_DIR, f"m{m}_exact_sweep.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved → {path}")
    return output


if __name__ == "__main__":
    print("SMALL-PRIMORIAL HAWKING-PAGE γ-SWEEP (EXACT)")
    print(f"Python {sys.version}")
    print(f"NumPy {np.__version__}")

    all_results = {}
    for m in [30, 210, 2310]:
        all_results[m] = sweep_primorial(m)

    # Summary table
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'m':>8}  {'φ':>6}  {'γ_c':>8}  {'S(1)/Page':>10}  {'time':>8}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*8}")
    for m in [30, 210, 2310]:
        r = all_results[m]
        print(f"  {m:>8}  {r['phi']:>6}  {r['gamma_c_approx']:>8.4f}  "
              f"{r['sweep'][-1]['S_A']/r['page_limit']:>10.6f}  "
              f"{r['elapsed_s']:>7.2f}s")
    print()
