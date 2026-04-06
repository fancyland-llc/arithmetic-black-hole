#!/usr/bin/env python3
"""
hunt_constant.py  --  Search for a universal primorial residual constant
=========================================================================

The Boltzmann model explains 97.0% of prime transition variance (R^2=0.970).
The 3% residual is NOT the Hardy-Littlewood singular series (tested: HL
makes R^2 WORSE). So what IS it?

If there is a universal constant kappa governing the residual, it should:
  1. Converge as N -> infinity  (scale-invariant)
  2. Be the same across moduli  (modulus-invariant = universal)
  3. Have a clean mathematical form (recognizable)

Candidates probed:
  kappa_1 = ||R||_F / ||T_obs - 1/phi||_F    (relative residual fraction)
  kappa_2 = mean diagonal suppression ratio    (Obs_diag / Boltz_diag)
  kappa_3 = Tr(R^T R) / Tr(T_obs^T T_obs)     (energy fraction)
  kappa_4 = max eigenvalue of R                 (spectral radius)
  kappa_5 = det(T_obs) / det(T_boltz)         (determinant ratio)
  kappa_6 = corr(S(d), R)                      (HL shadow)
  kappa_7 = (1-R^2) * phi(m)                   (scaled residual)

Usage:
  python hunt_constant.py                       # sweep N from 10^4 to 10^7
  python hunt_constant.py --N 1000000000        # push to 10^9
  python hunt_constant.py --multi               # check universality across moduli
"""

import math
import numpy as np
import argparse
import json
import sys
import time


# ── Core number theory (shared with compute_singular_series.py) ─────────

def sieve_of_eratosthenes(limit):
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = b'\x00' * len(sieve[i*i::i])
    return [i for i, v in enumerate(sieve) if v]


def coprime_residues(m):
    return sorted(r for r in range(1, m) if math.gcd(r, m) == 1)


def forward_distance_matrix(m, residues):
    phi = len(residues)
    D = np.zeros((phi, phi), dtype=np.float64)
    for i, a in enumerate(residues):
        for j, b in enumerate(residues):
            D[i, j] = m if i == j else (b - a) % m
    return D


def count_transitions(primes, m, residues):
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


def boltzmann_prediction(D, T):
    log_w = -D / T
    log_Z = np.logaddexp.reduce(log_w, axis=1, keepdims=True)
    return np.exp(log_w - log_Z)


def r_squared(obs, pred, phi_m):
    y = obs.flatten()
    yhat = pred.flatten()
    null = 1.0 / phi_m
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - null) ** 2)
    return 1.0 - ss_res / ss_tot


def compute_twin_prime_constant(prime_limit=100_000):
    primes = sieve_of_eratosthenes(prime_limit)
    C2 = 1.0
    for p in primes:
        if p >= 3:
            C2 *= 1.0 - 1.0 / ((p - 1) ** 2)
    return C2


def odd_prime_factors(n):
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
    if k <= 0 or k % 2 != 0:
        return 0.0
    product = 1.0
    for p in odd_prime_factors(k):
        product *= (p - 1) / (p - 2)
    return 2.0 * C2 * product


# ── Candidate constants ────────────────────────────────────────────────

def extract_candidates(T_obs, T_boltz, D, phi_m, m, C2):
    """Extract all candidate constant values from a single (N, m) run."""
    R = T_obs - T_boltz
    diag_idx = np.arange(phi_m)
    obs_diag = T_obs[diag_idx, diag_idx]
    boltz_diag = T_boltz[diag_idx, diag_idx]

    # kappa_1: relative Frobenius residual fraction
    null = 1.0 / phi_m
    frob_res = np.sqrt(np.sum(R ** 2))
    frob_sig = np.sqrt(np.sum((T_obs - null) ** 2))
    kappa_1 = frob_res / frob_sig

    # kappa_2: mean diagonal suppression ratio
    kappa_2 = np.mean(obs_diag / boltz_diag)

    # kappa_3: energy fraction in residual
    kappa_3 = np.sum(R ** 2) / np.sum((T_obs - null) ** 2)

    # kappa_4: spectral radius of R
    eigs_R = np.linalg.eigvalsh(R @ R.T)
    kappa_4 = np.sqrt(np.max(eigs_R))

    # kappa_5: determinant ratio
    det_obs = np.linalg.det(T_obs)
    det_boltz = np.linalg.det(T_boltz)
    kappa_5 = det_obs / det_boltz if abs(det_boltz) > 1e-30 else float('nan')

    # kappa_6: correlation with singular series
    S_flat = np.array([singular_series(int(D[i, j]), C2)
                       for i in range(phi_m) for j in range(phi_m)])
    R_flat = R.flatten()
    kappa_6 = float(np.corrcoef(S_flat, R_flat)[0, 1])

    # kappa_7: scaled residual (1 - R^2) * phi(m)
    R2 = r_squared(T_obs, T_boltz, phi_m)
    kappa_7 = (1 - R2) * phi_m

    # kappa_8: diagonal MAE / off-diagonal MAE
    diag_mae = np.mean(np.abs(R[diag_idx, diag_idx]))
    mask = np.ones((phi_m, phi_m), dtype=bool)
    np.fill_diagonal(mask, False)
    offdiag_mae = np.mean(np.abs(R[mask]))
    kappa_8 = diag_mae / offdiag_mae

    # kappa_9: trace of R (sum of diagonal residuals)
    kappa_9 = np.trace(R)

    # kappa_10: trace of R normalized by 1/phi
    kappa_10 = np.trace(R) * phi_m

    # kappa_11: R^2 itself (should converge)
    kappa_11 = R2

    # kappa_12: Frobenius norm of R * sqrt(phi)
    kappa_12 = frob_res * math.sqrt(phi_m)

    # kappa_13: ln(1 - R^2) -- log residual
    kappa_13 = math.log(1 - R2) if R2 < 1 else float('nan')

    # kappa_14: (1-R^2) * T  -- thermal residual
    T_val = D[0, 0]  # = m for diagonal
    kappa_14 = (1 - R2) * (phi_m * math.log(phi_m))

    # kappa_15: mean suppression * phi
    kappa_15 = (1 - kappa_2) * phi_m

    return {
        "k1_frob_frac":       kappa_1,
        "k2_diag_suppress":   kappa_2,
        "k3_energy_frac":     kappa_3,
        "k4_spectral_rad":    kappa_4,
        "k5_det_ratio":       kappa_5,
        "k6_HL_corr":         kappa_6,
        "k7_res_x_phi":       kappa_7,
        "k8_diag_off_ratio":  kappa_8,
        "k9_trace_R":         kappa_9,
        "k10_trace_R_x_phi":  kappa_10,
        "k11_R2":             kappa_11,
        "k12_frob_x_sqrtphi": kappa_12,
        "k13_ln_residual":    kappa_13,
        "k14_thermal_res":    kappa_14,
        "k15_suppress_x_phi": kappa_15,
    }


# ── Scale convergence sweep ────────────────────────────────────────────

def scale_sweep(m, scales, verbose=True):
    """Run at multiple N for fixed m, track candidate convergence."""
    residues = coprime_residues(m)
    phi_m = len(residues)
    D = forward_distance_matrix(m, residues)
    C2 = compute_twin_prime_constant(100_000)

    if verbose:
        print(f"\n{'='*80}")
        print(f"  CONVERGENCE SWEEP  m = {m}  phi(m) = {phi_m}")
        print(f"{'='*80}")

    # Sieve once to the largest N
    max_N = max(scales)
    if verbose:
        print(f"  Sieving to {max_N:,}...")
    all_primes = sieve_of_eratosthenes(max_N)
    if verbose:
        print(f"  pi({max_N:,}) = {len(all_primes):,}")

    results = []
    for N in scales:
        # Slice primes up to N
        primes = [p for p in all_primes if p <= N]
        pi_N = len(primes)
        T = N / pi_N

        counts = count_transitions(primes, m, residues)
        T_obs = normalize_rows(counts.astype(np.float64))
        T_boltz = boltzmann_prediction(D, T)

        candidates = extract_candidates(T_obs, T_boltz, D, phi_m, m, C2)
        candidates["N"] = N
        candidates["T"] = T
        candidates["pi_N"] = pi_N
        results.append(candidates)

    if verbose:
        # Print convergence table for most promising candidates
        keys_of_interest = [
            "k1_frob_frac", "k2_diag_suppress", "k3_energy_frac",
            "k7_res_x_phi", "k8_diag_off_ratio", "k10_trace_R_x_phi",
            "k11_R2", "k12_frob_x_sqrtphi", "k13_ln_residual",
            "k15_suppress_x_phi"
        ]
        print(f"\n  {'N':>12}", end="")
        for k in keys_of_interest:
            short = k.split("_", 1)[1][:10]
            print(f" {short:>11}", end="")
        print()
        print(f"  {'-'*12}", end="")
        for _ in keys_of_interest:
            print(f" {'-'*11}", end="")
        print()
        for row in results:
            print(f"  {row['N']:>12,}", end="")
            for k in keys_of_interest:
                print(f" {row[k]:>11.6f}", end="")
            print()

        # Compute convergence rates (ratio of successive differences)
        print(f"\n  CONVERGENCE ASSESSMENT (last 3 values):")
        print(f"  {'Candidate':<25} {'Final value':>12} {'Drift (last 2)':>14} {'Stable?':>8}")
        print(f"  {'-'*62}")
        for k in keys_of_interest:
            vals = [r[k] for r in results if not math.isnan(r[k])]
            if len(vals) >= 3:
                final = vals[-1]
                drift = abs(vals[-1] - vals[-2])
                prev_drift = abs(vals[-2] - vals[-3])
                converging = drift < prev_drift
                stable = drift < 0.001
                marker = "YES" if stable else ("~" if converging else "NO")
                print(f"  {k:<25} {final:>12.6f} {drift:>14.6f} {marker:>8}")

    return results


# ── Multi-modulus universality test ─────────────────────────────────────

def multi_modulus_test(N, moduli, verbose=True):
    """Test if candidates are the same across moduli at fixed N."""
    if verbose:
        print(f"\n{'='*80}")
        print(f"  UNIVERSALITY TEST  N = {N:,}")
        print(f"{'='*80}")
        print(f"  Sieving to {N:,}...")

    primes = sieve_of_eratosthenes(N)
    C2 = compute_twin_prime_constant(100_000)
    pi_N = len(primes)
    T = N / pi_N
    if verbose:
        print(f"  pi(N) = {pi_N:,}   T = {T:.4f}")

    all_candidates = {}
    for m in moduli:
        residues = coprime_residues(m)
        phi_m = len(residues)
        D = forward_distance_matrix(m, residues)
        counts = count_transitions(primes, m, residues)
        T_obs = normalize_rows(counts.astype(np.float64))
        T_boltz = boltzmann_prediction(D, T)
        cands = extract_candidates(T_obs, T_boltz, D, phi_m, m, C2)
        cands["m"] = m
        cands["phi_m"] = phi_m
        all_candidates[m] = cands
        if verbose:
            print(f"  m = {m:>6}  phi = {phi_m:>4}  R^2 = {cands['k11_R2']:.6f}")

    if verbose:
        keys = [
            "k1_frob_frac", "k2_diag_suppress", "k3_energy_frac",
            "k7_res_x_phi", "k8_diag_off_ratio", "k10_trace_R_x_phi",
            "k11_R2", "k12_frob_x_sqrtphi", "k13_ln_residual",
            "k15_suppress_x_phi"
        ]
        print(f"\n  {'Candidate':<25}", end="")
        for m in moduli:
            print(f" {'m='+str(m):>12}", end="")
        print(f" {'CoV':>8} {'Universal?':>10}")
        print(f"  {'-'*25}", end="")
        for _ in moduli:
            print(f" {'-'*12}", end="")
        print(f" {'-'*8} {'-'*10}")

        for k in keys:
            vals = [all_candidates[m][k] for m in moduli]
            vals_clean = [v for v in vals if not math.isnan(v)]
            mean_v = np.mean(vals_clean) if vals_clean else 0
            std_v = np.std(vals_clean) if vals_clean else 0
            cov = std_v / abs(mean_v) if abs(mean_v) > 1e-10 else float('inf')
            univ = "YES" if cov < 0.05 else ("~" if cov < 0.15 else "NO")
            print(f"  {k:<25}", end="")
            for m in moduli:
                v = all_candidates[m][k]
                print(f" {v:>12.6f}", end="")
            print(f" {cov:>8.3f} {univ:>10}")

        # Find the sweet spot: converges AND universal
        print(f"\n  CANDIDATES THAT ARE BOTH CONVERGENT AND UNIVERSAL:")
        print(f"  (These would be genuine new constants)")
        print(f"  {'-'*60}")
        for k in keys:
            vals = [all_candidates[m][k] for m in moduli]
            vals_clean = [v for v in vals if not math.isnan(v)]
            if not vals_clean:
                continue
            mean_v = np.mean(vals_clean)
            std_v = np.std(vals_clean)
            cov = std_v / abs(mean_v) if abs(mean_v) > 1e-10 else float('inf')
            if cov < 0.10:
                # Check known constants
                checks = {
                    "e - 2":           math.e - 2,
                    "1/e":             1/math.e,
                    "pi/4":            math.pi/4,
                    "1/pi":            1/math.pi,
                    "2/pi":            2/math.pi,
                    "ln(2)":           math.log(2),
                    "ln(3)":           math.log(3),
                    "1/ln(2)":         1/math.log(2),
                    "C_2":             C2,
                    "2*C_2":           2*C2,
                    "1-C_2":           1-C2,
                    "sqrt(2)-1":       math.sqrt(2)-1,
                    "1/sqrt(2*pi)":    1/math.sqrt(2*math.pi),
                    "gamma_EM":        0.5772156649,
                    "1-gamma_EM":      1-0.5772156649,
                    "pi*C_2":          math.pi * C2,
                    "e*C_2":           math.e * C2,
                    "C_2/ln(2)":       C2 / math.log(2),
                    "sqrt(C_2)":       math.sqrt(C2),
                    "1/(2*pi)":        1/(2*math.pi),
                    "pi^2/6 - 1":      math.pi**2/6 - 1,
                    "Catalan":         0.9159655941,
                    "1/3":             1/3,
                    "1/4":             1/4,
                    "3/8":             3/8,
                    "5/8":             5/8,
                    "7/8":             7/8,
                    "1 - 1/e":         1 - 1/math.e,
                    "e/pi":            math.e/math.pi,
                    "pi/e":            math.pi/math.e,
                    "ln(pi)":          math.log(math.pi),
                    "pi - e":          math.pi - math.e,
                    "sqrt(e) - 1":     math.sqrt(math.e) - 1,
                    "sqrt(pi) - 1":    math.sqrt(math.pi) - 1,
                    "ln(2)/2":         math.log(2)/2,
                }
                closest_name = ""
                closest_dist = float('inf')
                for name, val in checks.items():
                    dist = abs(mean_v - val) / max(abs(mean_v), 1e-10)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_name = name
                match = f"  ~= {closest_name} ({checks[closest_name]:.6f})" if closest_dist < 0.03 else ""
                print(f"  {k:<25} mean = {mean_v:>10.6f}  CoV = {cov:.4f}{match}")

    return all_candidates


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hunt for universal primorial residual constant")
    parser.add_argument("--N", type=int, default=10_000_000,
                        help="Max prime limit (default: 10^7)")
    parser.add_argument("--multi", action="store_true",
                        help="Test modulus universality at N")
    parser.add_argument("--json", action="store_true",
                        help="JSON output")
    args = parser.parse_args()

    # Scale sweep at m=30
    exp_max = int(math.log10(args.N))
    scales = [10**e for e in range(4, exp_max + 1)]
    sweep_results = scale_sweep(30, scales, verbose=not args.json)

    # Multi-modulus universality
    if args.multi or not args.json:
        moduli = [30, 210]
        if args.N >= 10_000_000:
            moduli.append(2310)
        univ_results = multi_modulus_test(args.N, moduli, verbose=not args.json)

    if args.json:
        print(json.dumps({
            "scale_sweep_m30": sweep_results,
            "universality": {str(k): v for k, v in univ_results.items()} if args.multi else None
        }, indent=2, default=str))
