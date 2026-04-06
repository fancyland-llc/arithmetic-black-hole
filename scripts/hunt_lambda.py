#!/usr/bin/env python3
"""
hunt_lambda.py  --  Find the universal HL-Boltzmann coupling exponent
======================================================================

Model:   T(a->b) ~ exp(-d(a,b)/T) * S(d(a,b))^lambda

At lambda=0: pure Boltzmann (R^2 = 0.970, zero free parameters)
At lambda=1: full HL (R^2 = 0.42, terrible)

The optimal lambda is found by golden-section minimisation of ||R||²_F
over lambda in [-1, 1].  If lambda* converges to a NUMBER that is:
  (a) scale-invariant  (same at 10^5, 10^7, 10^9)
  (b) modulus-invariant (same at m=30, 210, 2310)
then it is a genuine new mathematical constant.

Physical meaning:  lambda = 0 means primes are pure thermal gas.
lambda > 0 means Hardy-Littlewood correlation enhances thermal weights.
lambda < 0 means HL COMPETES with thermal (anti-correlation).

Usage:
  python hunt_lambda.py                     # N=10^7, m=30
  python hunt_lambda.py --N 1000000000      # N=10^9
  python hunt_lambda.py --multi             # universality across moduli
"""

import math
import numpy as np
import argparse
import json
import sys
import time


# ── Number theory primitives ───────────────────────────────────────────

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
    return M / np.maximum(s, 1e-30)


def compute_twin_prime_constant(prime_limit=100_000):
    primes = sieve_of_eratosthenes(prime_limit)
    c2 = 1.0
    for p in primes:
        if p >= 3:
            c2 *= 1.0 - 1.0 / ((p - 1) ** 2)
    return c2


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


def singular_series_matrix(D, C2):
    phi = D.shape[0]
    S = np.zeros_like(D)
    for i in range(phi):
        for j in range(phi):
            S[i, j] = singular_series(int(D[i, j]), C2)
    return S


# ── R² machinery ───────────────────────────────────────────────────────

def r_squared(obs, pred, phi_m):
    y = obs.flatten()
    yhat = pred.flatten()
    null = 1.0 / phi_m
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - null) ** 2)
    return 1.0 - ss_res / ss_tot


def ss_res(obs, pred):
    return np.sum((obs.flatten() - pred.flatten()) ** 2)


# ── Lambda-parameterized model ─────────────────────────────────────────

def hl_boltzmann_prediction(D, T, S_mat, lam):
    """T(a->b) ~ S(d)^lambda * exp(-d/T), row-normalized."""
    log_S = np.log(np.maximum(S_mat, 1e-30))
    log_w = lam * log_S + (-D / T)
    log_Z = np.logaddexp.reduce(log_w, axis=1, keepdims=True)
    return np.exp(log_w - log_Z)


# ── Golden-section search for optimal lambda ───────────────────────────

def find_optimal_lambda(T_obs, D, T, S_mat, phi_m, lo=-0.5, hi=1.0, tol=1e-8):
    """Find lambda that maximises R^2 (minimises SS_res)."""
    gr = (math.sqrt(5) + 1) / 2

    def objective(lam):
        pred = hl_boltzmann_prediction(D, T, S_mat, lam)
        return ss_res(T_obs, pred)

    a, b = lo, hi
    c = b - (b - a) / gr
    d = a + (b - a) / gr

    for _ in range(200):  # ~60 digits of precision
        if objective(c) < objective(d):
            b = d
        else:
            a = c
        if abs(b - a) < tol:
            break
        c = b - (b - a) / gr
        d = a + (b - a) / gr

    lam_opt = (a + b) / 2
    pred_opt = hl_boltzmann_prediction(D, T, S_mat, lam_opt)
    R2_opt = r_squared(T_obs, pred_opt, phi_m)
    pred_base = hl_boltzmann_prediction(D, T, S_mat, 0.0)
    R2_base = r_squared(T_obs, pred_base, phi_m)

    return lam_opt, R2_opt, R2_base


# ── Identify constant ──────────────────────────────────────────────────

KNOWN_CONSTANTS = None

def get_known_constants():
    global KNOWN_CONSTANTS
    if KNOWN_CONSTANTS is not None:
        return KNOWN_CONSTANTS

    C2 = compute_twin_prime_constant(100_000)
    gamma = 0.57721566490153286

    KNOWN_CONSTANTS = {
        "1/e":              1/math.e,
        "1/pi":             1/math.pi,
        "2/pi":             2/math.pi,
        "e/pi":             math.e/math.pi,
        "pi/e":             math.pi/math.e,
        "pi - e":           math.pi - math.e,
        "ln(2)":            math.log(2),
        "ln(3)":            math.log(3),
        "ln(2)/2":          math.log(2)/2,
        "C_2":              C2,
        "2*C_2":            2*C2,
        "1 - 2*C_2":        1 - 2*C2,
        "C_2^2":            C2**2,
        "sqrt(C_2)":        math.sqrt(C2),
        "C_2/ln(2)":        C2/math.log(2),
        "C_2*ln(2)":        C2*math.log(2),
        "gamma_EM":         gamma,
        "1 - gamma":        1 - gamma,
        "gamma/2":          gamma/2,
        "gamma^2":          gamma**2,
        "1/gamma":          1/gamma,
        "gamma*C_2":        gamma*C2,
        "gamma/pi":         gamma/math.pi,
        "gamma/e":          gamma/math.e,
        "gamma*ln(2)":      gamma*math.log(2),
        "sqrt(2) - 1":      math.sqrt(2) - 1,
        "sqrt(3) - 1":      math.sqrt(3) - 1,
        "1/sqrt(2*pi)":     1/math.sqrt(2*math.pi),
        "3/pi^2":           3/math.pi**2,
        "6/pi^2":           6/math.pi**2,
        "pi^2/6 - 1":       math.pi**2/6 - 1,
        "1/3":              1/3,
        "1/4":              1/4,
        "1/5":              1/5,
        "1/6":              1/6,
        "1/7":              1/7,
        "1/8":              1/8,
        "1/10":             1/10,
        "1/12":             1/12,
        "2/7":              2/7,
        "3/10":             3/10,
        "Catalan":          0.9159655941772190,
        "1 - 1/e":          1 - 1/math.e,
        "Mertens B_1":      0.2614972128,
        "Mertens M":        0.2614972128,
        "Artin":            0.3739558136,
        "1/2*ln(2*pi)":     0.5*math.log(2*math.pi),
        "sqrt(e)-1":        math.sqrt(math.e) - 1,
        "sqrt(pi)-1":       math.sqrt(math.pi) - 1,
        "e - 2":            math.e - 2,
        "3 - e":            3 - math.e,
        "pi - 3":           math.pi - 3,
        "4 - pi":           4 - math.pi,
        "ln(pi)/pi":        math.log(math.pi)/math.pi,
        "ln(2)/pi":         math.log(2)/math.pi,
        "e^(-gamma)":       math.exp(-gamma),
        "1/ln(10)":         1/math.log(10),
    }
    return KNOWN_CONSTANTS


def identify_constant(value, tolerance=0.02):
    """Try to identify value as a known constant."""
    consts = get_known_constants()
    matches = []
    for name, val in consts.items():
        if abs(val) < 1e-10:
            continue
        rel_err = abs(value - val) / abs(val)
        if rel_err < tolerance:
            matches.append((name, val, rel_err))
    matches.sort(key=lambda x: x[2])
    return matches


# ── Main analysis ──────────────────────────────────────────────────────

def run_lambda_hunt(N, m, primes=None, verbose=True):
    """Find optimal lambda at given N, m."""
    t0 = time.time()
    residues = coprime_residues(m)
    phi_m = len(residues)
    D = forward_distance_matrix(m, residues)
    C2 = compute_twin_prime_constant(100_000)
    S_mat = singular_series_matrix(D, C2)

    if primes is None:
        if verbose:
            print(f"  Sieving to {N:,}...")
        primes = sieve_of_eratosthenes(N)

    # Slice to N
    if primes[-1] > N:
        primes = [p for p in primes if p <= N]
    pi_N = len(primes)
    T = N / pi_N

    counts = count_transitions(primes, m, residues)
    T_obs = normalize_rows(counts.astype(np.float64))

    lam_opt, R2_opt, R2_base = find_optimal_lambda(T_obs, D, T, S_mat, phi_m)

    # Improvement
    delta_R2 = R2_opt - R2_base
    frac_explained = delta_R2 / (1 - R2_base) * 100 if R2_base < 1 else 0

    elapsed = time.time() - t0
    return {
        "N": N, "m": m, "phi_m": phi_m, "pi_N": pi_N, "T": T,
        "lambda_opt": lam_opt,
        "R2_base": R2_base,
        "R2_optimal": R2_opt,
        "delta_R2": delta_R2,
        "pct_residual_explained": frac_explained,
        "elapsed_s": elapsed,
    }


def run_full_analysis(N_max, moduli, verbose=True):
    """Scale sweep + universality test."""
    C2 = compute_twin_prime_constant(100_000)

    # Scale sweep for each modulus
    exp_max = int(math.log10(N_max))
    scales = [10**e for e in range(4, exp_max + 1)]

    if verbose:
        print(f"\n{'='*80}")
        print(f"  LAMBDA HUNT: Hardy-Littlewood x Boltzmann coupling exponent")
        print(f"  Model: T(a->b) ~ S(d)^lambda * exp(-d/T), row-normalized")
        print(f"  Scales: {' '.join(f'10^{e}' for e in range(4, exp_max+1))}")
        print(f"  Moduli: {moduli}")
        print(f"{'='*80}")

    # Sieve once
    if verbose:
        print(f"\n  Sieving to {N_max:,}...")
    all_primes = sieve_of_eratosthenes(N_max)
    if verbose:
        print(f"  pi({N_max:,}) = {len(all_primes):,}")

    all_results = {}
    for m in moduli:
        if verbose:
            print(f"\n  --- m = {m} (phi = {len(coprime_residues(m))}) ---")
        results_m = []
        for N in scales:
            primes_N = [p for p in all_primes if p <= N]
            result = run_lambda_hunt(N, m, primes=primes_N, verbose=False)
            results_m.append(result)
            if verbose:
                r = result
                print(f"  N={N:>13,}  lambda* = {r['lambda_opt']:>+10.7f}"
                      f"  R2_base={r['R2_base']:.6f}  R2_opt={r['R2_optimal']:.6f}"
                      f"  +{r['pct_residual_explained']:.2f}% of residual")
        all_results[m] = results_m

    # Lambda convergence table
    if verbose:
        print(f"\n{'='*80}")
        print(f"  LAMBDA CONVERGENCE TABLE")
        print(f"{'='*80}")
        print(f"  {'N':>13}", end="")
        for m in moduli:
            print(f"  {'m='+str(m):>12}", end="")
        print()
        print(f"  {'-'*13}", end="")
        for _ in moduli:
            print(f"  {'-'*12}", end="")
        print()
        for i, N in enumerate(scales):
            print(f"  {N:>13,}", end="")
            for m in moduli:
                lam = all_results[m][i]["lambda_opt"]
                print(f"  {lam:>+12.7f}", end="")
            print()

        # Final converged values
        print(f"\n  CONVERGED LAMBDA VALUES (N = {N_max:,}):")
        print(f"  {'-'*50}")
        lambdas = []
        for m in moduli:
            lam = all_results[m][-1]["lambda_opt"]
            lambdas.append(lam)
            matches = identify_constant(abs(lam))
            match_str = ""
            if matches:
                name, val, err = matches[0]
                match_str = f"  |lambda| ~= {name} = {val:.7f} (err={err:.4f})"
            print(f"  m = {m:>5}  lambda* = {lam:>+.8f}{match_str}")

        mean_lam = np.mean(lambdas)
        std_lam = np.std(lambdas)
        cov = std_lam / abs(mean_lam) if abs(mean_lam) > 1e-10 else float('inf')
        print(f"\n  mean(lambda*) = {mean_lam:>+.8f}")
        print(f"  std(lambda*)  = {std_lam:.8f}")
        print(f"  CoV           = {cov:.4f}")

        # Check for known constants
        print(f"\n  CONSTANT IDENTIFICATION:")
        print(f"  {'-'*50}")
        for name, val, err in identify_constant(abs(mean_lam), tolerance=0.05):
            print(f"  |lambda| ~= {name} = {val:.8f}  (rel error = {err:.4%})")
        print()
        for name, val, err in identify_constant(mean_lam, tolerance=0.05):
            print(f"   lambda  ~= {name} = {val:.8f}  (rel error = {err:.4%})")

        # R^2 improvement
        print(f"\n  R^2 IMPROVEMENT (N = {N_max:,}):")
        print(f"  {'Modulus':>8} {'R2_Boltzmann':>14} {'R2_optimal':>12}"
              f" {'Delta':>10} {'% residual':>12}")
        print(f"  {'-'*60}")
        for m in moduli:
            r = all_results[m][-1]
            print(f"  {m:>8} {r['R2_base']:>14.6f} {r['R2_optimal']:>12.6f}"
                  f" {r['delta_R2']:>+10.6f} {r['pct_residual_explained']:>11.2f}%")

        # Lambda scan (fine resolution around optimum for m=30)
        m0 = moduli[0]
        residues = coprime_residues(m0)
        phi_m = len(residues)
        D = forward_distance_matrix(m0, residues)
        S_mat = singular_series_matrix(D, C2)
        primes_max = [p for p in all_primes if p <= N_max]
        T = N_max / len(primes_max)
        counts = count_transitions(primes_max, m0, residues)
        T_obs = normalize_rows(counts.astype(np.float64))

        lam_center = all_results[m0][-1]["lambda_opt"]
        print(f"\n  LAMBDA SCAN (m={m0}, N={N_max:,}, center={lam_center:.6f}):")
        print(f"  {'lambda':>10} {'R^2':>12} {'1-R^2':>10} {'vs base':>10}")
        print(f"  {'-'*45}")
        for lam_test in np.linspace(lam_center - 0.1, lam_center + 0.1, 21):
            pred = hl_boltzmann_prediction(D, T, S_mat, lam_test)
            r2 = r_squared(T_obs, pred, phi_m)
            r2_base = all_results[m0][-1]["R2_base"]
            marker = " <--" if abs(lam_test - lam_center) < 0.006 else ""
            print(f"  {lam_test:>+10.5f} {r2:>12.8f} {1-r2:>10.6f}"
                  f" {r2-r2_base:>+10.7f}{marker}")

    return all_results


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hunt for the HL-Boltzmann coupling exponent lambda*")
    parser.add_argument("--N", type=int, default=10_000_000)
    parser.add_argument("--m", type=int, default=30)
    parser.add_argument("--multi", action="store_true",
                        help="Test universality across m=30,210,2310")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    moduli = [30, 210, 2310] if args.multi else [args.m]
    if args.N < 10_000_000 and 2310 in moduli:
        moduli = [m for m in moduli if m <= 210]

    results = run_full_analysis(args.N, moduli, verbose=not args.json)

    if args.json:
        out = {}
        for m, res_list in results.items():
            out[str(m)] = [r for r in res_list]
        print(json.dumps(out, indent=2, default=str))
