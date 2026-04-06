#!/usr/bin/env python3
"""
hunt_complex_eigenspectrum.py  --  Complex eigenvalues of the Boltzmann residual
=================================================================================

The residual R = T_obs - T_boltz is a REAL but ASYMMETRIC matrix.
Its eigenvalues are therefore complex (conjugate pairs + real).

Gemini's insight: if the Boltzmann model is the smooth baseline (like Li(x)),
then the residual is the oscillatory correction -- governed by interference
in the prime lattice's complex spectrum, analogous to the sum over zeta zeros
in Riemann's explicit formula:  pi(x) = Li(x) - Sum_rho Li(x^rho) + ...

We extract:
  - Complex eigenvalues of R, sorted by |lambda|
  - Phases theta_k = arg(lambda_k)  (in [0, pi] for conjugate pairs)
  - Phase ratios theta_k / pi  (look for rational multiples)
  - Moduli |lambda_k| (look for convergent magnitudes)
  - Phase angles vs k/phi(m)  (look for cot(k*pi/phi) pattern)

If any of these converge across scales AND moduli, we have found the
complex constant governing the prime transition residual.

Usage:
  python hunt_complex_eigenspectrum.py                    # N=10^7, m=30
  python hunt_complex_eigenspectrum.py --N 1000000000     # N=10^9
  python hunt_complex_eigenspectrum.py --multi            # m=30, 210, 2310
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


# ── Complex eigenspectrum analysis ─────────────────────────────────────

def analyze_residual_spectrum(R, phi_m, verbose_label=""):
    """Extract and analyze complex eigenvalues of the residual matrix."""
    eigs = np.linalg.eigvals(R)

    # Sort by descending magnitude
    order = np.argsort(-np.abs(eigs))
    eigs = eigs[order]

    moduli = np.abs(eigs)
    phases = np.angle(eigs)           # in (-pi, pi]
    phases_norm = phases / math.pi     # normalized to (-1, 1]

    # Separate real vs complex conjugate pairs
    real_eigs = []
    complex_pairs = []
    used = set()
    for i, e in enumerate(eigs):
        if i in used:
            continue
        if abs(e.imag) < 1e-12:
            real_eigs.append(e.real)
        else:
            # Find conjugate partner
            for j in range(i+1, len(eigs)):
                if j not in used and abs(eigs[j] - e.conj()) < 1e-10:
                    complex_pairs.append((e, eigs[j]))
                    used.add(j)
                    break
        used.add(i)

    return {
        "eigenvalues": eigs,
        "moduli": moduli,
        "phases": phases,
        "phases_over_pi": phases_norm,
        "real_eigs": real_eigs,
        "complex_pairs": complex_pairs,
        "spectral_radius": float(moduli[0]),
        "trace_R": float(np.trace(R)),
        "det_R": float(np.real(np.prod(eigs))),
    }


def check_phase_rationality(phase_pi, tol=0.02):
    """Check if phase/pi is close to a simple rational p/q."""
    if abs(phase_pi) < tol:
        return "0"
    best = None
    best_err = tol
    for q in range(1, 13):
        for p in range(-q, q+1):
            if p == 0:
                continue
            rat = p / q
            err = abs(phase_pi - rat)
            if err < best_err:
                best_err = err
                best = f"{p}/{q}"
    return best


def check_cot_pattern(phases, phi_m):
    """Check if phases match cot(k*pi/phi(m)) pattern."""
    expected_angles = []
    for k in range(1, phi_m):
        angle = k * math.pi / phi_m
        expected_angles.append(angle)
    # Compare
    obs_positive = sorted([p for p in phases if p > 0.01])
    exp_sorted = sorted(expected_angles)[:len(obs_positive)]
    if obs_positive and exp_sorted:
        corr = np.corrcoef(obs_positive[:len(exp_sorted)],
                           exp_sorted[:len(obs_positive)])[0,1]
        return float(corr) if not np.isnan(corr) else 0.0
    return 0.0


# ── Scale sweep ────────────────────────────────────────────────────────

def run_scale_sweep(m, N_max, verbose=True):
    residues = coprime_residues(m)
    phi_m = len(residues)
    D = forward_distance_matrix(m, residues)

    exp_max = int(math.log10(N_max))
    scales = [10**e for e in range(4, exp_max + 1)]

    if verbose:
        print(f"\n{'='*80}")
        print(f"  COMPLEX EIGENSPECTRUM OF R = T_obs - T_boltz")
        print(f"  m = {m}   phi(m) = {phi_m}")
        print(f"{'='*80}")
        print(f"\n  Sieving to {N_max:,}...")

    all_primes = sieve_of_eratosthenes(N_max)
    if verbose:
        print(f"  pi({N_max:,}) = {len(all_primes):,}")

    all_spectra = []
    for N in scales:
        primes = [p for p in all_primes if p <= N]
        pi_N = len(primes)
        T = N / pi_N

        counts = count_transitions(primes, m, residues)
        T_obs = normalize_rows(counts.astype(np.float64))
        T_boltz = boltzmann_prediction(D, T)
        R = T_obs - T_boltz
        R2 = r_squared(T_obs, T_boltz, phi_m)

        spec = analyze_residual_spectrum(R, phi_m)
        spec["N"] = N
        spec["T"] = T
        spec["R2"] = R2
        all_spectra.append(spec)

        if verbose:
            eigs = spec["eigenvalues"]
            print(f"\n  N = {N:,}   T = {T:.4f}   R^2 = {R2:.6f}")
            print(f"  Spectral radius: {spec['spectral_radius']:.8f}")
            print(f"  Trace(R) = {spec['trace_R']:.8f}")
            print(f"  {len(spec['real_eigs'])} real, "
                  f"{len(spec['complex_pairs'])} complex pairs")
            print(f"  {'#':>4} {'Re(lambda)':>12} {'Im(lambda)':>12} "
                  f"{'|lambda|':>10} {'arg/pi':>8} {'rational?':>10}")
            print(f"  {'-'*62}")
            for i, e in enumerate(eigs):
                mod = abs(e)
                phase_pi = np.angle(e) / math.pi
                rat = check_phase_rationality(phase_pi) or ""
                marker = ""
                if abs(e.imag) < 1e-12:
                    marker = " (real)"
                print(f"  {i:>4} {e.real:>+12.8f} {e.imag:>+12.8f} "
                      f"{mod:>10.8f} {phase_pi:>+8.4f} {rat:>10}{marker}")

    # Phase convergence across scales
    if verbose and len(all_spectra) >= 2:
        print(f"\n{'='*80}")
        print(f"  PHASE CONVERGENCE ACROSS SCALES (m = {m})")
        print(f"{'='*80}")
        # Track each eigenvalue index across scales
        n_eigs = phi_m
        print(f"  {'Eig#':>5}", end="")
        for N in scales:
            print(f"  {'N='+str(N):>14}", end="")
        print(f"  {'Drift':>10}")
        print(f"  {'-'*5}", end="")
        for _ in scales:
            print(f"  {'-'*14}", end="")
        print(f"  {'-'*10}")

        for idx in range(min(n_eigs, 8)):
            print(f"  {idx:>5}", end="")
            phases_across = []
            for spec in all_spectra:
                e = spec["eigenvalues"][idx]
                phase_pi = np.angle(e) / math.pi
                phases_across.append(phase_pi)
                print(f"  {phase_pi:>+14.8f}", end="")
            if len(phases_across) >= 2:
                drift = abs(phases_across[-1] - phases_across[-2])
                stable = "LOCKED" if drift < 0.001 else f"{drift:.6f}"
                print(f"  {stable:>10}")
            else:
                print()

        # Moduli convergence
        print(f"\n  {'Eig#':>5}", end="")
        for N in scales:
            print(f"  {'N='+str(N):>14}", end="")
        print(f"  {'Drift':>10}")
        print(f"  {'-'*5}", end="")
        for _ in scales:
            print(f"  {'-'*14}", end="")
        print(f"  {'-'*10}")

        for idx in range(min(n_eigs, 8)):
            print(f"  {idx:>5}", end="")
            mods_across = []
            for spec in all_spectra:
                mod = float(np.abs(spec["eigenvalues"][idx]))
                mods_across.append(mod)
                print(f"  {mod:>14.8f}", end="")
            if len(mods_across) >= 2:
                drift = abs(mods_across[-1] - mods_across[-2])
                stable = "LOCKED" if drift < 0.0001 else f"{drift:.8f}"
                print(f"  {stable:>10}")
            else:
                print()

        # Cot pattern check
        for spec in all_spectra[-1:]:
            corr = check_cot_pattern(spec["phases"], phi_m)
            print(f"\n  cot(k*pi/phi) pattern correlation: {corr:.4f}")

    return all_spectra


# ── Universality across moduli ─────────────────────────────────────────

def run_universality_test(N, moduli, verbose=True):
    if verbose:
        print(f"\n{'='*80}")
        print(f"  UNIVERSALITY TEST: Complex eigenspectrum across moduli")
        print(f"  N = {N:,}")
        print(f"{'='*80}")
        print(f"\n  Sieving to {N:,}...")

    primes = sieve_of_eratosthenes(N)
    pi_N = len(primes)
    T = N / pi_N
    if verbose:
        print(f"  pi(N) = {pi_N:,}   T = {T:.4f}")

    all_spectra = {}
    for m in moduli:
        residues = coprime_residues(m)
        phi_m = len(residues)
        D = forward_distance_matrix(m, residues)
        counts = count_transitions(primes, m, residues)
        T_obs = normalize_rows(counts.astype(np.float64))
        T_boltz = boltzmann_prediction(D, T)
        R = T_obs - T_boltz
        R2 = r_squared(T_obs, T_boltz, phi_m)

        spec = analyze_residual_spectrum(R, phi_m)
        spec["N"] = N
        spec["m"] = m
        spec["T"] = T
        spec["R2"] = R2
        all_spectra[m] = spec

        if verbose:
            eigs = spec["eigenvalues"]
            n_show = min(16, len(eigs))
            print(f"\n  m = {m}  phi = {phi_m}  R^2 = {R2:.6f}"
                  f"  spectral_radius = {spec['spectral_radius']:.8f}")
            print(f"  {len(spec['real_eigs'])} real, "
                  f"{len(spec['complex_pairs'])} complex pairs")

            print(f"  {'#':>4} {'Re':>12} {'Im':>12} {'|e|':>10}"
                  f" {'arg/pi':>8} {'rational':>10}")
            print(f"  {'-'*62}")
            for i in range(n_show):
                e = eigs[i]
                mod = abs(e)
                phase_pi = np.angle(e) / math.pi
                rat = check_phase_rationality(phase_pi) or ""
                real_tag = " (real)" if abs(e.imag) < 1e-12 else ""
                print(f"  {i:>4} {e.real:>+12.8f} {e.imag:>+12.8f}"
                      f" {mod:>10.8f} {phase_pi:>+8.4f} {rat:>10}{real_tag}")
            if len(eigs) > n_show:
                print(f"  ... ({len(eigs) - n_show} more)")

    # Cross-modulus phase comparison
    if verbose and len(moduli) >= 2:
        print(f"\n{'='*80}")
        print(f"  CROSS-MODULUS PHASE STRUCTURE")
        print(f"{'='*80}")
        print(f"\n  Key question: do the leading phases theta/pi converge")
        print(f"  to common values as phi(m) grows?")
        print()
        # Normalize: for each modulus, report phases divided by some
        # intrinsic scale (like 1/phi(m))
        for m in moduli:
            spec = all_spectra[m]
            eigs = spec["eigenvalues"]
            phi_m = len(coprime_residues(m))
            phases = np.angle(eigs) / math.pi
            # Report unique positive phases (from conjugate pairs)
            pos_phases = sorted([p for p in phases if p > 0.01], reverse=True)
            print(f"  m={m:>5} (phi={phi_m:>4}): "
                  f"top phases/pi = {', '.join(f'{p:.4f}' for p in pos_phases[:6])}")
            # Scale by phi(m)
            scaled = [p * phi_m for p in pos_phases[:6]]
            print(f"  {'':>20}  x phi(m)   = {', '.join(f'{s:.3f}' for s in scaled)}")

        # Spectral radius comparison (normalized)
        print(f"\n  SPECTRAL RADIUS (normalized by 1/phi^2):")
        for m in moduli:
            spec = all_spectra[m]
            phi_m = len(coprime_residues(m))
            sr = spec["spectral_radius"]
            sr_norm = sr * phi_m * phi_m
            print(f"  m = {m:>5}  |lambda_0| = {sr:.8f}"
                  f"  |lambda_0|*phi^2 = {sr_norm:.4f}")

    return all_spectra


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complex eigenspectrum of the Boltzmann residual matrix")
    parser.add_argument("--N", type=int, default=10_000_000)
    parser.add_argument("--m", type=int, default=30)
    parser.add_argument("--multi", action="store_true",
                        help="Universality test: m = 30, 210, 2310")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    # Scale sweep for primary modulus
    sweep = run_scale_sweep(args.m, args.N, verbose=not args.json)

    # Universality test
    if args.multi:
        moduli = [30, 210]
        if args.N >= 10_000_000:
            moduli.append(2310)
        univ = run_universality_test(args.N, moduli, verbose=not args.json)

    if args.json:
        def to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.complexfloating)):
                return complex(obj) if isinstance(obj, np.complexfloating) else float(obj)
            if isinstance(obj, complex):
                return {"re": obj.real, "im": obj.imag}
            raise TypeError(f"Not serializable: {type(obj)}")
        print(json.dumps({"sweep": sweep, "universality": univ if args.multi else None},
                         indent=2, default=to_serializable))
