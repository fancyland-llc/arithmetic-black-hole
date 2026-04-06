#!/usr/bin/env python3
"""
eigenvector_tracker.py — Track eigenvalue modes by eigenvector continuity

Instead of tracking eigenvalues by magnitude rank (which causes mode-hopping
at m=210), we track them by eigenvector overlap between consecutive N values.

For each tracked mode, we fit phase rate vs log₃T and check for the 1/2 law.

This settles the question: does the base CRT mode EXIST at m=210 but get
buried by mode-hopping, or is it genuinely absent in the frozen regime?
"""

import math
import numpy as np
from scipy import stats
import time


def sieve_primes(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]


def build_residual(primes, N, m, coprimes, idx):
    """Build R = T_obs - T_boltz at modulus m."""
    phi = len(coprimes)
    counts = np.zeros((phi, phi), dtype=np.float64)
    prev_class = None
    for p in primes:
        if p > N:
            break
        r = p % m
        if r in idx:
            if prev_class is not None:
                counts[prev_class, idx[r]] += 1
            prev_class = idx[r]

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T_obs = counts / row_sums

    dist = np.zeros((phi, phi), dtype=np.float64)
    for i, a in enumerate(coprimes):
        for j, b in enumerate(coprimes):
            dist[i, j] = m if i == j else (b - a) % m

    pi_N = sum(1 for p in primes if p <= N)
    T = N / pi_N if pi_N > 0 else 1.0
    logits = -dist / T
    logits -= logits.max(axis=1, keepdims=True)
    exp_l = np.exp(logits)
    T_boltz = exp_l / exp_l.sum(axis=1, keepdims=True)

    return T_obs - T_boltz, T


def track_modes_eigvec(primes, N_values, m):
    """
    Track eigenvalue modes by eigenvector continuity.

    At each N:
      1. Compute R and its eigendecomposition
      2. Match new eigenvectors to previous ones by maximum |overlap|
      3. Record eigenvalue for each tracked mode

    Returns: list of mode tracks, each [(N, T, eigenvalue), ...]
    """
    coprimes = sorted([r for r in range(1, m) if math.gcd(r, m) == 1])
    phi = len(coprimes)
    idx = {r: i for i, r in enumerate(coprimes)}

    mode_tracks = None
    prev_vecs = None

    for k, N in enumerate(N_values):
        R, T = build_residual(primes, int(N), m, coprimes, idx)
        vals, vecs = np.linalg.eig(R)

        if k == 0:
            # Initialise: sort by magnitude descending
            order = np.argsort(-np.abs(vals))
            vals = vals[order]
            vecs = vecs[:, order]
            mode_tracks = [[(N, T, vals[i])] for i in range(phi)]
            prev_vecs = vecs
        else:
            # Match by eigenvector overlap (greedy Hungarian-lite)
            overlap = np.abs(prev_vecs.conj().T @ vecs)  # phi × phi
            used_old, used_new = set(), set()
            matching = {}

            # Greedily match by descending overlap
            flat = []
            for i in range(phi):
                for j in range(phi):
                    flat.append((overlap[i, j], i, j))
            flat.sort(reverse=True)

            for ov, i, j in flat:
                if i not in used_old and j not in used_new:
                    matching[i] = j
                    used_old.add(i)
                    used_new.add(j)

            # Reorder new eigenvectors to match previous ordering
            new_vecs = np.zeros_like(vecs)
            for old_idx in range(phi):
                new_idx = matching[old_idx]
                new_vecs[:, old_idx] = vecs[:, new_idx]
                mode_tracks[old_idx].append((N, T, vals[new_idx]))

            prev_vecs = new_vecs

    return mode_tracks


def analyse_tracks(tracks, label, m):
    """Fit phase rate for every complex mode. Print results."""
    phi = len(tracks)
    results = []

    for i, track in enumerate(tracks):
        # Only consider modes that are complex at most points
        phases = []
        for N, T, v in track:
            if abs(v.imag) > 1e-12:
                phases.append((N, T, np.angle(v) / math.pi))

        if len(phases) < 8:
            continue

        x = np.array([math.log(T) / math.log(3) for _, T, _ in phases])
        y = np.array([p for _, _, p in phases])
        slope, intercept, r, _, _ = stats.linregress(x, y)
        avg_mag = np.mean([abs(v) for _, _, v in track])
        results.append((i, slope, r**2, avg_mag, len(phases)))

    # Sort by R² descending
    results.sort(key=lambda x: -x[2])

    print(f"\n  {label}: {phi} modes total, {len(results)} complex-tracked")
    print(f"\n  {'Mode':>5}  {'Avg |λ|':>10}  {'Rate/log₃T':>12}  {'R²':>8}  "
          f"{'n_pts':>6}  {'|err vs 0.5|':>12}  {'Match?':>6}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*6}  {'─'*12}  {'─'*6}")

    for i, slope, r2, mag, n in results[:20]:  # top 20 by R²
        err = abs(slope - 0.5) / 0.5 * 100
        star = " ★★★" if err < 15 and r2 > 0.9 else (
               " ★★" if err < 20 and r2 > 0.8 else (
               " ★" if err < 30 and r2 > 0.7 else ""))
        print(f"  {i:>5}  {mag:>10.6f}  {slope:>+12.6f}  {r2:>8.4f}  "
              f"{n:>6}  {err:>11.1f}%{star}")

    return results


def main():
    LIMIT = 10**9
    print("=" * 90)
    print("  EIGENVECTOR CONTINUITY TRACKER")
    print("  Does the base CRT mode exist at m=210 but get lost by mode-hopping?")
    print("=" * 90)

    print(f"\n  Sieving primes to {LIMIT:,}...")
    t0 = time.time()
    primes = sieve_primes(LIMIT)
    print(f"  Found {len(primes):,} primes in {time.time()-t0:.1f}s")

    # ─────────────────────────────────────────────────────────────────────
    #  PART 1: m=30 baseline — eigenvector-tracked
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  PART 1: m=30 baseline (eigenvector-tracked)")
    print("=" * 90)

    N_vals_30 = np.unique(np.logspace(4, 9, 40, dtype=np.int64))
    t0 = time.time()
    tracks_30 = track_modes_eigvec(primes, N_vals_30, 30)
    dt = time.time() - t0
    print(f"  Tracked {len(tracks_30)} modes × {len(N_vals_30)} N-values in {dt:.1f}s")

    results_30 = analyse_tracks(tracks_30, "m=30 (φ=8)", 30)

    # ─────────────────────────────────────────────────────────────────────
    #  PART 2: m=210 — eigenvector-tracked
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  PART 2: m=210 (eigenvector-tracked)")
    print("=" * 90)

    N_vals_210 = np.unique(np.logspace(4, 9, 30, dtype=np.int64))
    t0 = time.time()
    tracks_210 = track_modes_eigvec(primes, N_vals_210, 210)
    dt = time.time() - t0
    print(f"  Tracked {len(tracks_210)} modes × {len(N_vals_210)} N-values in {dt:.1f}s")

    results_210 = analyse_tracks(tracks_210, "m=210 (φ=48)", 210)

    # ─────────────────────────────────────────────────────────────────────
    #  PART 3: Magnitude comparison — is the base mode at m=210 near
    #          the same |λ| as at m=30?
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  PART 3: Where does the base-mode magnitude sit?")
    print("=" * 90)

    # At m=30, the base mode is the one with best R² and rate near 0.5
    best_30 = None
    for i, slope, r2, mag, n in results_30:
        err = abs(slope - 0.5)
        if best_30 is None or (r2 > 0.9 and err < abs(best_30[1] - 0.5)):
            best_30 = (i, slope, r2, mag, n)

    if best_30:
        print(f"\n  m=30 base mode:  Mode #{best_30[0]}, |λ| ≈ {best_30[3]:.6f}, "
              f"rate = {best_30[1]:+.6f}, R² = {best_30[2]:.4f}")
        target_mag = best_30[3]

        # At m=210, find modes with similar |λ|
        print(f"\n  m=210 modes with |λ| in [{target_mag*0.3:.4f}, {target_mag*3:.4f}]:")
        print(f"  {'Mode':>5}  {'Avg |λ|':>10}  {'Rate/log₃T':>12}  {'R²':>8}  "
              f"{'|λ| ratio':>10}")
        print(f"  {'─'*5}  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*10}")

        for i, slope, r2, mag, n in results_210:
            if target_mag * 0.3 < mag < target_mag * 3:
                ratio = mag / target_mag
                print(f"  {i:>5}  {mag:>10.6f}  {slope:>+12.6f}  {r2:>8.4f}  "
                      f"{ratio:>10.2f}×")

    # ─────────────────────────────────────────────────────────────────────
    #  PART 4: The definitive answer
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  VERDICT: Does eigenvector continuity reveal a hidden base mode at m=210?")
    print("=" * 90)

    # Find best candidate at m=210 (highest R² with rate within 50% of 0.5)
    best_210 = None
    for i, slope, r2, mag, n in results_210:
        err = abs(slope - 0.5) / 0.5
        if err < 0.5 and (best_210 is None or r2 > best_210[2]):
            best_210 = (i, slope, r2, mag, n)

    # Also find absolute best R²
    best_210_any = results_210[0] if results_210 else None

    if best_210 and best_210[2] > 0.9:
        print(f"""
  ╔══════════════════════════════════════════════════════════════════════════════════════╗
  ║  BASE MODE FOUND AT m=210!                                                         ║
  ║                                                                                    ║
  ║  Mode #{best_210[0]:>3}: rate = {best_210[1]:+.6f} per log₃T   R² = {best_210[2]:.4f}   |λ| = {best_210[3]:.6f}         ║
  ║                                                                                    ║
  ║  Eigenvector continuity resolved it through the mode-hopping chaos.                ║
  ║  The phase law d(θ/π)/d(log₃T) = 1/2 is UNIVERSAL.                               ║
  ╚══════════════════════════════════════════════════════════════════════════════════════╝
""")
    elif best_210 and best_210[2] > 0.7:
        print(f"""
  ┌──────────────────────────────────────────────────────────────────────────────────────┐
  │  TENTATIVE CANDIDATE at m=210:                                                      │
  │  Mode #{best_210[0]:>3}: rate = {best_210[1]:+.6f} per log₃T   R² = {best_210[2]:.4f}   |λ| = {best_210[3]:.6f}         │
  │                                                                                     │
  │  The signal is weak (R² < 0.9), but eigenvector continuity helps.                  │
  │  Need either more data (N > 10⁹) or GPU eigendecomposition.                       │
  └──────────────────────────────────────────────────────────────────────────────────────┘
""")
    else:
        print(f"""
  ┌──────────────────────────────────────────────────────────────────────────────────────┐
  │  BASE MODE NOT RESOLVABLE AT m=210, even with eigenvector continuity.               │
  │                                                                                     │
  │  Best candidate: Mode #{best_210_any[0] if best_210_any else '?':>3}  rate = {best_210_any[1] if best_210_any else 0:+.6f}  R² = {best_210_any[2] if best_210_any else 0:.4f}                         │
  │                                                                                     │
  │  The system is FROZEN (m/T ≥ 10 at all accessible N).                              │
  │  Sieve modes dominate; thermodynamic modes cannot form.                             │
  │  The base mode doesn't EXIST as a coherent eigenvector in this regime.              │
  │                                                                                     │
  │  CRT projection preserves the TRACE law (proven to 10⁻⁶), but the                │
  │  PHASE law requires the eigenvalue to be resolvable — it isn't.                    │
  └──────────────────────────────────────────────────────────────────────────────────────┘
""")

    print("=" * 90)


if __name__ == "__main__":
    main()
