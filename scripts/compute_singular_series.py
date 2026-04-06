#!/usr/bin/env python3
"""
compute_singular_series.py  --  Hardy-Littlewood Singular Series vs Boltzmann Residual
========================================================================================

Tests whether the 3.0% Boltzmann residual (R^2 = 0.970) is predicted by the
Hardy-Littlewood singular series for prime k-tuples.

Four models compared:
  1. Pure Boltzmann:         B(a,b)  ~ exp(-d(a,b)/T)
  2. HL leading-term:        B(a,b)  ~ S(d) * exp(-d/T)
  3. HL full-sum:            B(a,b)  ~ Sum_j S(d+jm) * exp(-(d+jm)/T)
  4. Anti-HL (competition):  B(a,b)  ~ exp(-d/T) / S(d)

Key hypothesis: for the self-transition gap d(a,a) = m, the singular series
S(m) is anomalously HIGH (m is highly composite for primorial moduli).
Naive HL therefore BOOSTS the diagonal prediction -- opposite to the observed
diagonal suppression.  If Model 4 (anti-HL) improves R^2, the residual is a
competition effect: high-S intermediaries intercept transitions before they
complete a full cycle back to the same residue.

Usage:
  python compute_singular_series.py                # N=10^7, m=30
  python compute_singular_series.py --N 1000000000 # N=10^9
  python compute_singular_series.py --multi        # m = 30, 210, 2310
  python compute_singular_series.py --json         # machine-readable

Dependencies: Python 3.10+, NumPy
Runtime: ~30s at N=10^7, ~2min at N=10^9
"""

import math
import numpy as np
import argparse
import json
import sys
import time


# ── Core number theory ──────────────────────────────────────────────────────

def sieve_of_eratosthenes(limit):
    """Return list of primes up to limit."""
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = b'\x00' * len(sieve[i*i::i])
    return [i for i, v in enumerate(sieve) if v]


def coprime_residues(m):
    """Sorted coprime residues in [1, m-1]."""
    return sorted(r for r in range(1, m) if math.gcd(r, m) == 1)


def forward_distance_matrix(m, residues):
    """d(a,b) = (b-a) mod m, with d(a,a) = m."""
    phi = len(residues)
    D = np.zeros((phi, phi), dtype=np.float64)
    for i, a in enumerate(residues):
        for j, b in enumerate(residues):
            D[i, j] = m if i == j else (b - a) % m
    return D


# ── Hardy-Littlewood singular series ───────────────────────────────────────

def compute_twin_prime_constant(prime_limit=1_000_000):
    """C2 = Product_{p>=3} (1 - 1/(p-1)^2)  ~  0.6601618..."""
    primes = sieve_of_eratosthenes(prime_limit)
    C2 = 1.0
    for p in primes:
        if p >= 3:
            C2 *= 1.0 - 1.0 / ((p - 1) ** 2)
    return C2


def odd_prime_factors(n):
    """Return sorted list of odd prime factors of n."""
    factors = []
    d = 3
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 2
    if n > 2:
        factors.append(n)
    return factors


def singular_series(k, C2):
    """
    S(k) for even k > 0:
        S(k) = 2*C2 * Product_{p|k, p>=3} (p-1)/(p-2)
    Returns 0 for odd k or k <= 0.
    """
    if k <= 0 or k % 2 != 0:
        return 0.0
    product = 1.0
    for p in odd_prime_factors(k):
        product *= (p - 1) / (p - 2)
    return 2.0 * C2 * product


def singular_series_matrix(D, C2):
    """Matrix of S(d(a,b)) values."""
    phi = D.shape[0]
    S = np.zeros_like(D)
    for i in range(phi):
        for j in range(phi):
            S[i, j] = singular_series(int(D[i, j]), C2)
    return S


# ── Transition counting ────────────────────────────────────────────────────

def count_transitions(primes, m, residues):
    """Count consecutive prime transitions mod m."""
    idx = {r: i for i, r in enumerate(residues)}
    phi = len(residues)
    counts = np.zeros((phi, phi), dtype=np.int64)
    for k in range(len(primes) - 1):
        a = primes[k] % m
        b = primes[k + 1] % m
        if a in idx and b in idx:
            counts[idx[a], idx[b]] += 1
    return counts


def normalize_rows(M):
    s = M.sum(axis=1, keepdims=True).astype(np.float64)
    s = np.maximum(s, 1e-30)
    return M / s


# ── Model predictions ──────────────────────────────────────────────────────

def boltzmann_prediction(D, T):
    """Model 1: exp(-d/T), row-normalized."""
    log_w = -D / T
    log_Z = np.logaddexp.reduce(log_w, axis=1, keepdims=True)
    return np.exp(log_w - log_Z)


def hl_leading_prediction(D, T, S_mat):
    """Model 2: S(d) * exp(-d/T), row-normalized."""
    log_w = np.log(np.maximum(S_mat, 1e-30)) + (-D / T)
    log_Z = np.logaddexp.reduce(log_w, axis=1, keepdims=True)
    return np.exp(log_w - log_Z)


def hl_fullsum_prediction(m, D, T, C2, J_max=30):
    """Model 3: Sum_j S(d+jm)*exp(-(d+jm)/T), row-normalized."""
    phi = D.shape[0]
    weights = np.zeros((phi, phi), dtype=np.float64)
    for i in range(phi):
        for j in range(phi):
            d = int(D[i, j])
            total = 0.0
            for jj in range(J_max + 1):
                gap = d + jj * m
                contrib = singular_series(gap, C2) * math.exp(-gap / T)
                total += contrib
                if contrib < 1e-15:
                    break
            weights[i, j] = total
    return normalize_rows(weights)


def anti_hl_prediction(D, T, S_mat):
    """Model 4: exp(-d/T) / S(d), row-normalized.  Competition hypothesis."""
    log_w = -np.log(np.maximum(S_mat, 1e-30)) + (-D / T)
    log_Z = np.logaddexp.reduce(log_w, axis=1, keepdims=True)
    return np.exp(log_w - log_Z)


# ── R-squared (uniform null) ───────────────────────────────────────────────

def r_squared(obs, pred, phi_m):
    """R^2 with null model = 1/phi(m)."""
    y = obs.flatten()
    yhat = pred.flatten()
    null = 1.0 / phi_m
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - null) ** 2)
    return 1.0 - ss_res / ss_tot


# ── Main analysis ──────────────────────────────────────────────────────────

def run_analysis(N, m, verbose=True):
    t0 = time.time()
    residues = coprime_residues(m)
    phi_m = len(residues)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  SINGULAR SERIES vs BOLTZMANN RESIDUAL")
        print(f"  N = {N:,}   m = {m}   phi(m) = {phi_m}")
        print(f"{'='*70}")

    # ── Primes ──────────────────────────────────────────
    if verbose:
        print(f"\n  Sieving primes to {N:,}...")
    primes = sieve_of_eratosthenes(N)
    pi_N = len(primes)
    T = N / pi_N
    if verbose:
        print(f"  pi(N) = {pi_N:,}    T = N/pi(N) = {T:.4f}")

    # ── Empirical transition matrix ─────────────────────
    if verbose:
        print(f"  Counting transitions...")
    counts = count_transitions(primes, m, residues)
    T_obs = normalize_rows(counts.astype(np.float64))

    # ── Twin prime constant ─────────────────────────────
    C2 = compute_twin_prime_constant(1_000_000)
    if verbose:
        print(f"  C_2 = {C2:.10f}   (2*C_2 = {2*C2:.10f})")

    D = forward_distance_matrix(m, residues)
    S_mat = singular_series_matrix(D, C2)

    # ── Singular series table ───────────────────────────
    unique_gaps = sorted(set(int(D[i, j]) for i in range(phi_m) for j in range(phi_m)))
    if verbose:
        print(f"\n  SINGULAR SERIES S(k) for gaps in mod-{m} distance matrix:")
        print(f"  {'gap k':>8}  {'S(k)':>10}  {'odd factors':>20}  {'S/S_min':>8}")
        print(f"  {'-'*52}")
        S_values = [singular_series(k, C2) for k in unique_gaps]
        S_min = min(s for s in S_values if s > 0)
        for k, S_val in zip(unique_gaps, S_values):
            factors = odd_prime_factors(k)
            f_str = ' * '.join(str(p) for p in factors) if factors else "-"
            ratio = S_val / S_min if S_val > 0 else 0
            tag = "  <-- DIAGONAL" if k == m else ""
            print(f"  {k:>8}  {S_val:>10.6f}  {f_str:>20}  {ratio:>8.3f}{tag}")
        print(f"\n  S({m}) / mean(S) = {singular_series(m, C2) / np.mean(S_values):.3f}")
        print(f"  Diagonal S is {'ABOVE' if singular_series(m, C2) > np.mean(S_values) else 'BELOW'} average")

    # ── Four models ─────────────────────────────────────
    if verbose:
        print(f"\n  Computing four models...")
    M1 = boltzmann_prediction(D, T)
    M2 = hl_leading_prediction(D, T, S_mat)
    M3 = hl_fullsum_prediction(m, D, T, C2)
    M4 = anti_hl_prediction(D, T, S_mat)

    R2 = {
        "boltzmann":  r_squared(T_obs, M1, phi_m),
        "hl_leading": r_squared(T_obs, M2, phi_m),
        "hl_fullsum": r_squared(T_obs, M3, phi_m),
        "anti_hl":    r_squared(T_obs, M4, phi_m),
    }

    # ── R^2 comparison ──────────────────────────────────
    if verbose:
        print(f"\n  {'='*62}")
        print(f"  MODEL COMPARISON")
        print(f"  {'='*62}")
        print(f"  {'Model':<38} {'R^2':>10}  {'Residual':>9}  {'Delta':>10}")
        print(f"  {'-'*70}")
        labels = [
            ("1. Pure Boltzmann",            "boltzmann"),
            ("2. HL leading-term x Boltzmann",  "hl_leading"),
            ("3. HL full-sum x Boltzmann",   "hl_fullsum"),
            ("4. Anti-HL (1/S x Boltzmann)", "anti_hl"),
        ]
        for label, key in labels:
            r2 = R2[key]
            delta = r2 - R2["boltzmann"]
            d_str = "baseline" if key == "boltzmann" else f"{delta:+.6f}"
            print(f"  {label:<38} {r2:>10.6f}  {1-r2:>8.4%}  {d_str:>10}")

    # ── Diagonal analysis ───────────────────────────────
    if verbose:
        print(f"\n  DIAGONAL ANALYSIS (self-transitions):")
        hdr = f"  {'res':>5} {'Obs':>8} {'Boltz':>8} {'HL-full':>8} {'AntiHL':>8}"
        hdr += f" {'B err':>9} {'HL err':>9} {'A err':>9}"
        print(hdr)
        print(f"  {'-'*78}")
        for i, r in enumerate(residues):
            o = T_obs[i, i]
            b = M1[i, i]
            h = M3[i, i]
            a = M4[i, i]
            print(f"  {r:>5} {o:>8.5f} {b:>8.5f} {h:>8.5f} {a:>8.5f}"
                  f" {o-b:>+9.5f} {o-h:>+9.5f} {o-a:>+9.5f}")

        # Summary
        diag_idx = np.arange(phi_m)
        d_obs = T_obs[diag_idx, diag_idx]
        d_M1  = M1[diag_idx, diag_idx]
        d_M3  = M3[diag_idx, diag_idx]
        d_M4  = M4[diag_idx, diag_idx]
        print(f"  {'-'*78}")
        print(f"  {'mean':>5} {d_obs.mean():>8.5f} {d_M1.mean():>8.5f}"
              f" {d_M3.mean():>8.5f} {d_M4.mean():>8.5f}"
              f" {(d_obs-d_M1).mean():>+9.5f} {(d_obs-d_M3).mean():>+9.5f}"
              f" {(d_obs-d_M4).mean():>+9.5f}")
        print(f"  {'MAE':>5} {'':>8} {np.mean(np.abs(d_obs-d_M1)):>8.5f}"
              f" {np.mean(np.abs(d_obs-d_M3)):>8.5f}"
              f" {np.mean(np.abs(d_obs-d_M4)):>8.5f}")

    # ── Residual correlation with S(d) ──────────────────
    R_boltz = T_obs - M1
    corr = float(np.corrcoef(S_mat.flatten(), R_boltz.flatten())[0, 1])
    if verbose:
        print(f"\n  Pearson corr(S(d), Boltzmann residual): {corr:+.6f}")
        if corr < -0.1:
            print(f"  --> NEGATIVE: high-S gaps are OVER-predicted by Boltzmann")
            print(f"      Boltzmann gives TOO MUCH weight to high-S(d) transitions")
        elif corr > 0.1:
            print(f"  --> POSITIVE: high-S gaps are UNDER-predicted")
        else:
            print(f"  --> WEAK: singular series has little correlation with residual")

    # ── Off-diagonal analysis (top residual entries) ────
    if verbose and phi_m <= 48:
        print(f"\n  TOP 10 RESIDUAL ENTRIES (|Boltzmann error| sorted):")
        print(f"  {'a->b':>8} {'d':>4} {'S(d)':>8} {'Obs':>8} {'Boltz':>8} {'Error':>9}")
        print(f"  {'-'*52}")
        errors = []
        for i, a in enumerate(residues):
            for j, b in enumerate(residues):
                errors.append((abs(R_boltz[i,j]), i, j, R_boltz[i,j]))
        errors.sort(reverse=True)
        for _, i, j, err in errors[:10]:
            a, b = residues[i], residues[j]
            d = int(D[i, j])
            s = S_mat[i, j]
            print(f"  {a:>3}->{b:<3} {d:>4} {s:>8.4f} {T_obs[i,j]:>8.5f}"
                  f" {M1[i,j]:>8.5f} {err:>+9.5f}")

    # ── Verdict ─────────────────────────────────────────
    best = max(R2, key=R2.get)
    elapsed = time.time() - t0

    if verbose:
        print(f"\n  {'='*62}")
        print(f"  VERDICT")
        print(f"  {'='*62}")

        if R2["hl_fullsum"] < R2["boltzmann"] and R2["anti_hl"] > R2["boltzmann"]:
            frac = (R2["anti_hl"] - R2["boltzmann"]) / (1 - R2["boltzmann"]) * 100
            print(f"  HL correction WORSENS R^2 ({R2['hl_fullsum']:.6f} < {R2['boltzmann']:.6f})")
            print(f"  Anti-HL IMPROVES R^2 ({R2['anti_hl']:.6f} > {R2['boltzmann']:.6f})")
            print(f"  --> {frac:.1f}% of the residual explained by COMPETITION effect")
            print(f"")
            print(f"  INTERPRETATION: The 3% residual is NOT the pairwise singular")
            print(f"  series. The diagonal suppression arises because high-S")
            print(f"  intermediary gaps INTERCEPT transitions before they complete")
            print(f"  a full cycle. The Boltzmann model implicitly captures most")
            print(f"  of this, but over-weights gaps with large S(k).")
            print(f"")
            print(f"  Connecting to Lemke Oliver-Soundararajan: the competition")
            print(f"  effect IS the consecutive-prime constraint that goes beyond")
            print(f"  pairwise Hardy-Littlewood. Explaining the residual requires")
            print(f"  the JOINT k-tuple singular series, not the pairwise S(k).")
        elif R2["hl_fullsum"] > R2["boltzmann"]:
            frac = (R2["hl_fullsum"] - R2["boltzmann"]) / (1 - R2["boltzmann"]) * 100
            print(f"  HL full-sum IMPROVES R^2: {frac:.1f}% of residual explained.")
            print(f"  The 3% residual IS (partly) the singular series.")
        elif R2["hl_leading"] < R2["boltzmann"] and R2["hl_fullsum"] < R2["boltzmann"]:
            print(f"  Both HL models WORSEN R^2.")
            if R2["anti_hl"] <= R2["boltzmann"]:
                print(f"  Anti-HL also worse. The residual has a non-HL origin.")
            else:
                print(f"  But anti-HL improves -- competition effect confirmed.")
        else:
            print(f"  Best model: {best}")
            print(f"  R^2 = {R2[best]:.6f}")

        print(f"\n  Elapsed: {elapsed:.1f}s")

    return {
        "N": N, "m": m, "phi_m": phi_m,
        "T": float(T), "C2": float(C2),
        "R2": {k: float(v) for k, v in R2.items()},
        "best_model": best,
        "corr_S_residual": corr,
        "S_diagonal": float(singular_series(m, C2)),
        "S_mean": float(np.mean([singular_series(k, C2) for k in unique_gaps])),
        "elapsed_s": elapsed,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hardy-Littlewood singular series vs Boltzmann residual")
    parser.add_argument("--N", type=int, default=10_000_000,
                        help="Prime limit (default: 10^7)")
    parser.add_argument("--m", type=int, default=30,
                        help="Modulus (default: 30)")
    parser.add_argument("--multi", action="store_true",
                        help="Run for m = 30, 210, 2310")
    parser.add_argument("--json", action="store_true",
                        help="JSON output")
    args = parser.parse_args()

    if args.multi:
        all_results = {}
        for mod in [30, 210, 2310]:
            results = run_analysis(args.N, mod, verbose=not args.json)
            all_results[str(mod)] = results
        if args.json:
            print(json.dumps(all_results, indent=2))
    else:
        results = run_analysis(args.N, args.m, verbose=not args.json)
        if args.json:
            print(json.dumps(results, indent=2))
