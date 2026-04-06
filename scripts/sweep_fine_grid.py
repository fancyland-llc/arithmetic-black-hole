#!/usr/bin/env python3
"""
Fine-grid γ-sweep near origin for m=30, 210, 2310
Resolves γ_c that was below grid resolution in the coarse sweep.
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
    return {"gamma": float(gamma), "S_A": float(S_A)}

def fine_sweep(m, gamma_max, n_points):
    print(f"\n{'='*60}")
    cops = coprime_residues(m)
    comps = composite_residues(m)
    phi = len(cops)
    n_fermions = m // 2
    print(f"  m={m}  |  φ={phi}  |  FINE GRID [0, {gamma_max}] × {n_points} points")
    print(f"{'='*60}")

    vm = von_mangoldt_sieve(m)
    H_cop = build_H_coprime(cops, m)
    H_comp = build_H_comp(comps, m)
    K = build_coupling_K(cops, comps, m, vm)

    gamma_values = np.linspace(0.0, gamma_max, n_points).tolist()
    page_limit = phi * log(2)

    results = []
    t0 = time.perf_counter()
    for i, g in enumerate(gamma_values):
        r = peschel_at_gamma(H_cop, K, H_comp, phi, n_fermions, g)
        r["S_A_over_page"] = r["S_A"] / page_limit if page_limit > 0 else 0.0
        results.append(r)
        if i % 20 == 0:
            print(f"    γ={g:.6f}  S_A={r['S_A']:.10f}  S/Page={r['S_A_over_page']:.6f}")

    elapsed = time.perf_counter() - t0

    # Find γ_c via max dS/dγ
    gammas = [r["gamma"] for r in results]
    entropies = [r["S_A"] for r in results]
    dS = np.gradient(entropies, gammas)
    idx_max = int(np.argmax(dS[1:]) + 1)  # skip γ=0
    gamma_c = gammas[idx_max]
    dS_max = dS[idx_max]

    print(f"\n  Completed {len(results)} points in {elapsed:.2f}s")
    print(f"  γ_c (max dS/dγ) = {gamma_c:.6f}")
    print(f"  dS/dγ at γ_c = {dS_max:.4f}")
    print(f"  S_A(γ_c) = {entropies[idx_max]:.10f}")

    output = {
        "m": m, "phi": phi, "n_fermions": n_fermions,
        "page_limit": page_limit, "gamma_c": gamma_c,
        "dS_dg_max": float(dS_max), "elapsed_s": round(elapsed, 3),
        "grid": f"[0, {gamma_max}] x {n_points}",
        "sweep": results,
    }
    path = os.path.join(OUT_DIR, f"m{m}_fine_sweep.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved → {path}")
    return output


if __name__ == "__main__":
    print("FINE-GRID γ-SWEEP NEAR ORIGIN")

    all_results = {}
    # m=30: transition around 0.16, sweep [0, 0.3] with 201 points
    all_results[30] = fine_sweep(30, 0.30, 201)
    # m=210: transition around 0.04, sweep [0, 0.10] with 201 points
    all_results[210] = fine_sweep(210, 0.10, 201)
    # m=2310: transition below 0.02, sweep [0, 0.05] with 201 points
    all_results[2310] = fine_sweep(2310, 0.05, 201)

    print("\n" + "=" * 60)
    print("  RESOLVED γ_c VALUES")
    print("=" * 60)
    print(f"  {'m':>8}  {'φ':>6}  {'γ_c':>10}  {'dS/dγ_max':>12}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*10}  {'─'*12}")
    for m_val in [30, 210, 2310]:
        r = all_results[m_val]
        print(f"  {m_val:>8}  {r['phi']:>6}  {r['gamma_c']:>10.6f}  {r['dS_dg_max']:>12.4f}")
    print()
