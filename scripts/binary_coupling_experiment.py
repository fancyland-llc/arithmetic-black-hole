#!/usr/bin/env python3
"""
ARITHMETIC QUBIT EXPERIMENT: Binary vs Weighted Coupling

Question: Does the holographic duality (Poisson bulk / GOE boundary)
survive when von Mangoldt weights ln(p) are replaced with binary 1?

If YES → the Arithmetic Qubit is pure combinatorial logic. No floating point.
         The entire chip can be fabricated as binary routing topology.

If NO  → we need per-prime gain layers (still only 3 multipliers for m=30).

Method:
  For each m in {30, 210, 2310}, sweep γ from 0 to 1:
    1. Build K_weighted[r,c] = Λ(gcd(r,c))           (original)
    2. Build K_binary[r,c]   = 1 if Λ(gcd(r,c)) > 0  (binary)
    3. Build full H with each K variant
    4. Compute Brody ω for H-spectrum (bulk) and C_A-spectrum (boundary)
    5. Compare: does {bulk→Poisson, boundary→GOE} hold for both?

Author: Antonio P. Matos / Fancyland LLC
Date: April 2026
"""

import numpy as np
import json, os, time
from math import gcd, log
from scipy.linalg import eigh, eigvalsh
from scipy.special import gamma as gamma_func

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ================================================================
# Shared primorial lattice construction
# ================================================================
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

def build_distance_matrix(residues, m):
    r = np.array(residues, dtype=np.float64)
    diff = np.abs(r[:, None] - r[None, :])
    H = np.minimum(diff, m - diff)
    np.fill_diagonal(H, 0.0)
    return H

def build_coupling_weighted(coprimes, composites, m, vm_table):
    """Original: K[r,c] = Λ(gcd(r,c))"""
    n_cop = len(coprimes)
    n_comp = len(composites)
    K = np.zeros((n_cop, n_comp))
    for i, r in enumerate(coprimes):
        for j, c in enumerate(composites):
            g = gcd(r, c) if c > 0 else r
            if g <= m:
                K[i, j] = vm_table[g]
    return K

def build_coupling_binary(coprimes, composites, m, vm_table):
    """Binary: K[r,c] = 1 if Λ(gcd(r,c)) > 0, else 0"""
    n_cop = len(coprimes)
    n_comp = len(composites)
    K = np.zeros((n_cop, n_comp))
    for i, r in enumerate(coprimes):
        for j, c in enumerate(composites):
            g = gcd(r, c) if c > 0 else r
            if g <= m and vm_table[g] > 0:
                K[i, j] = 1.0
    return K

def build_full_H(H_cop, K, H_comp, gamma_val):
    n_cop = H_cop.shape[0]
    n_comp = H_comp.shape[0]
    n = n_cop + n_comp
    H = np.empty((n, n))
    H[:n_cop, :n_cop] = H_cop
    H[:n_cop, n_cop:] = gamma_val * K
    H[n_cop:, :n_cop] = gamma_val * K.T
    H[n_cop:, n_cop:] = H_comp
    return H

# ================================================================
# C_A entanglement spectrum (Peschel construction)
# ================================================================
def compute_CA_spectrum(H, n_cop):
    n = H.shape[0]
    n_fermions = n // 2
    eigenvalues, eigenvectors = eigh(H)
    V_occ = eigenvectors[:, :n_fermions]
    V_A = V_occ[:n_cop, :]
    C_A = V_A @ V_A.T
    C_A = (C_A + C_A.T) / 2.0
    lam = eigvalsh(C_A)
    lam = np.clip(lam, 1e-15, 1.0 - 1e-15)
    xi = -np.log(lam / (1.0 - lam))
    return np.sort(xi)

# ================================================================
# Spectral statistics
# ================================================================
def remove_degeneracies(eigs, tol=1e-10):
    """Forward-accumulator degeneracy removal (Module 10 method)."""
    if len(eigs) < 2:
        return eigs
    cleaned = [eigs[0]]
    for i in range(1, len(eigs)):
        if abs(eigs[i] - eigs[i-1]) > tol:
            cleaned.append(eigs[i])
    return np.array(cleaned)

def unfolded_spacings(eigenvalues):
    eigs = remove_degeneracies(eigenvalues)
    N = len(eigs)
    if N < 10:
        return np.array([])
    window = max(5, N // 15)
    half = window // 2
    spacings = []
    for i in range(N - 1):
        left = max(0, i - half)
        right = min(N - 1, i + half + 1)
        rng = eigs[right] - eigs[left]
        if rng < 1e-15:
            continue
        density = (right - left) / rng
        spacings.append((eigs[i+1] - eigs[i]) * density)
    spacings = np.array(spacings)
    if len(spacings) == 0:
        return spacings
    mean = spacings.mean()
    if mean > 0:
        spacings /= mean
    return spacings

def brody_parameter(spacings):
    """Brody MLE via golden-section maximization."""
    filtered = spacings[spacings > 0]
    n = len(filtered)
    if n < 5:
        return 0.0
    sum_ln = np.sum(np.log(filtered))

    def log_likelihood(omega):
        w1 = omega + 1.0
        b = gamma_func((omega + 2.0) / w1) ** w1
        return n * np.log(w1) + n * np.log(b) + omega * sum_ln \
               - b * np.sum(filtered ** w1)

    phi = (np.sqrt(5) - 1) / 2
    a, b_val = 0.0, 3.0
    c = b_val - phi * (b_val - a)
    d = a + phi * (b_val - a)
    fc, fd = log_likelihood(c), log_likelihood(d)
    while b_val - a > 0.005:
        if fc > fd:
            b_val = d; d = c; fd = fc
            c = b_val - phi * (b_val - a); fc = log_likelihood(c)
        else:
            a = c; c = d; fc = fd
            d = a + phi * (b_val - a); fd = log_likelihood(d)
    return max(0.0, (a + b_val) / 2.0)

def kl_divergences(spacings):
    n = len(spacings)
    if n < 5:
        return 0.0, 0.0
    s_max = max(spacings) * 1.001
    if s_max == 0:
        return 0.0, 0.0
    n_bins = 20
    bw = s_max / n_bins
    counts = np.zeros(n_bins)
    for s in spacings:
        b = min(int(s / bw), n_bins - 1)
        if b >= 0:
            counts[b] += 1
    hist = counts / (n * bw)
    kl_p, kl_g = 0.0, 0.0
    for k in range(n_bins):
        P = hist[k]
        if P < 1e-12:
            continue
        s = (k + 0.5) * bw
        q_poisson = max(np.exp(-s), 1e-12)
        q_goe = max((np.pi / 2) * s * np.exp(-np.pi * s * s / 4), 1e-12)
        kl_p += P * np.log(P / q_poisson) * bw
        kl_g += P * np.log(P / q_goe) * bw
    return kl_p, kl_g

# ================================================================
# Main experiment
# ================================================================
def analyze_spectrum(label, eigs):
    spacings = unfolded_spacings(eigs)
    if len(spacings) < 5:
        return {"label": label, "omega": None, "kl_poisson": None, "kl_goe": None, "n_cleaned": 0}
    omega = brody_parameter(spacings)
    kl_p, kl_g = kl_divergences(spacings)
    return {"label": label, "omega": round(omega, 4), "kl_poisson": round(kl_p, 4), "kl_goe": round(kl_g, 4), "n_cleaned": len(spacings)}

def run_experiment():
    primorials = [30, 210, 2310]
    gamma_grid = [0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.28, 0.36, 0.44, 0.52, 0.60, 0.72, 0.84, 1.00]

    results = {}

    for m in primorials:
        print(f"\n{'='*60}")
        print(f"  m = {m},  φ(m) = {len(coprime_residues(m))}")
        print(f"{'='*60}")

        coprimes = coprime_residues(m)
        composites = composite_residues(m)
        vm = von_mangoldt_sieve(m)
        n_cop = len(coprimes)

        H_cop = build_distance_matrix(coprimes, m)
        H_comp = build_distance_matrix(composites, m)
        K_w = build_coupling_weighted(coprimes, composites, m, vm)
        K_b = build_coupling_binary(coprimes, composites, m, vm)

        # Sparsity report
        nnz_w = np.count_nonzero(K_w)
        total = K_w.size
        print(f"  Coupling K: {nnz_w}/{total} nonzero ({100*nnz_w/total:.1f}% dense)")
        print(f"  Binary K identical topology: {np.all((K_w > 0) == (K_b > 0))}")

        m_results = []

        for gamma in gamma_grid:
            t0 = time.time()

            # --- Weighted ---
            H_full_w = build_full_H(H_cop, K_w, H_comp, gamma)
            eigs_w = eigvalsh(H_full_w)
            ca_w = compute_CA_spectrum(H_full_w, n_cop)

            h_w = analyze_spectrum("H_weighted", eigs_w)
            ca_w_stats = analyze_spectrum("CA_weighted", ca_w)

            # --- Binary ---
            H_full_b = build_full_H(H_cop, K_b, H_comp, gamma)
            eigs_b = eigvalsh(H_full_b)
            ca_b = compute_CA_spectrum(H_full_b, n_cop)

            h_b = analyze_spectrum("H_binary", eigs_b)
            ca_b_stats = analyze_spectrum("CA_binary", ca_b)

            dt = time.time() - t0

            # Holographic split check: bulk < 0.2 AND boundary > 0.3
            def holo_check(h_stats, ca_stats):
                if h_stats["omega"] is None or ca_stats["omega"] is None:
                    return "INSUFFICIENT_DATA"
                if h_stats["omega"] < 0.20 and ca_stats["omega"] > 0.25:
                    return "HOLOGRAPHIC ✓"
                elif h_stats["omega"] < 0.20:
                    return "BULK_OK / BOUNDARY_WEAK"
                else:
                    return "BULK_NOT_POISSON"

            verdict_w = holo_check(h_w, ca_w_stats)
            verdict_b = holo_check(h_b, ca_b_stats)

            row = {
                "gamma": gamma,
                "weighted": {
                    "H_omega": h_w["omega"],
                    "CA_omega": ca_w_stats["omega"],
                    "verdict": verdict_w
                },
                "binary": {
                    "H_omega": h_b["omega"],
                    "CA_omega": ca_b_stats["omega"],
                    "verdict": verdict_b
                },
                "time_s": round(dt, 2)
            }
            m_results.append(row)

            # Pretty print
            w_ho = h_w["omega"] if h_w["omega"] is not None else "—"
            w_co = ca_w_stats["omega"] if ca_w_stats["omega"] is not None else "—"
            b_ho = h_b["omega"] if h_b["omega"] is not None else "—"
            b_co = ca_b_stats["omega"] if ca_b_stats["omega"] is not None else "—"

            sym = "✓" if verdict_w == verdict_b else "✗"
            print(f"  γ={gamma:.2f}  W: H_ω={w_ho:<6} CA_ω={w_co:<6} [{verdict_w}]  "
                  f"B: H_ω={b_ho:<6} CA_ω={b_co:<6} [{verdict_b}]  {sym}  ({dt:.1f}s)")

        results[str(m)] = m_results

    # Summary
    print(f"\n{'='*60}")
    print("  TOPOLOGY INVARIANCE SUMMARY")
    print(f"{'='*60}")
    total_points = 0
    match_points = 0
    for m_str, rows in results.items():
        for row in rows:
            if row["gamma"] < 0.04:
                continue  # skip γ=0 (trivially Poisson)
            total_points += 1
            if row["weighted"]["verdict"] == row["binary"]["verdict"]:
                match_points += 1

    pct = 100 * match_points / total_points if total_points > 0 else 0
    print(f"  Verdict match (γ>0): {match_points}/{total_points} ({pct:.0f}%)")
    if pct >= 80:
        print(f"  >>> HOLOGRAPHIC DUALITY IS TOPOLOGICAL <<<")
        print(f"  >>> THE ARITHMETIC QUBIT IS COMBINATORIAL LOGIC <<<")
    else:
        print(f"  >>> WEIGHTS MATTER — need per-prime gain layers <<<")

    # Save
    out_path = os.path.join(SCRIPT_DIR, "binary_coupling_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

if __name__ == "__main__":
    run_experiment()
