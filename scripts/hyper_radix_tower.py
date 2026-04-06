#!/usr/bin/env python3
"""
hyper_radix_tower.py — Test the Hyper-Radix Tower Hypothesis

HYPOTHESIS: Each prime p dividing the modulus m contributes its own
eigenvalue mode rotating at rate 1/(p-1) per logₚT.

The "hyper-radix" is not a single base — it's a TOWER:
   p=3:  rate 1/2  per log₃T   (base mode, (Z/3Z)* ≅ Z/2)
   p=5:  rate 1/4  per log₅T   (fiber mode, (Z/5Z)* ≅ Z/4)
   p=7:  rate 1/6  per log₇T   (fiber mode, (Z/7Z)* ≅ Z/6)
   p=11: rate 1/10 per log₁₁T  (fiber mode, (Z/11Z)* ≅ Z/10)

Each mode advances at 1/|G_p| = 1/φ(p) = 1/(p-1) per logₚT
in the natural coordinate of its own CRT fiber.
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
    coprimes = sorted([r for r in range(1, m) if math.gcd(r, m) == 1])
    phi = len(coprimes)
    idx = {r: i for i, r in enumerate(coprimes)}

    mode_tracks = None
    prev_vecs = None

    for k, N in enumerate(N_values):
        R, T = build_residual(primes, int(N), m, coprimes, idx)
        vals, vecs = np.linalg.eig(R)

        if k == 0:
            order = np.argsort(-np.abs(vals))
            vals = vals[order]
            vecs = vecs[:, order]
            mode_tracks = [[(N, T, vals[i])] for i in range(phi)]
            prev_vecs = vecs
        else:
            overlap = np.abs(prev_vecs.conj().T @ vecs)
            used_old, used_new = set(), set()
            matching = {}

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

            new_vecs = np.zeros_like(vecs)
            for old_idx in range(phi):
                new_idx = matching[old_idx]
                new_vecs[:, old_idx] = vecs[:, new_idx]
                mode_tracks[old_idx].append((N, T, vals[new_idx]))

            prev_vecs = new_vecs

    return mode_tracks


def fit_in_base(track, base):
    """Fit phase rate of a tracked mode in the given logarithmic base."""
    phases = []
    for N, T, v in track:
        if abs(v.imag) > 1e-12 and v.imag > 0:  # positive imaginary only
            phases.append((N, T, np.angle(v) / math.pi))
        elif abs(v.imag) > 1e-12 and v.imag < 0:
            phases.append((N, T, -np.angle(v) / math.pi))

    if len(phases) < 8:
        return None

    x = np.array([math.log(T) / math.log(base) for _, T, _ in phases])
    y = np.array([abs(p) for _, _, p in phases])
    slope, intercept, r, _, _ = stats.linregress(x, y)
    return (abs(slope), r**2, len(phases))


def main():
    LIMIT = 10**9
    print("=" * 90)
    print("  THE HYPER-RADIX TOWER")
    print("  Does each prime p contribute a mode rotating at 1/(p-1) per logₚT?")
    print("=" * 90)

    print(f"\n  Sieving primes to {LIMIT:,}...")
    t0 = time.time()
    primes = sieve_primes(LIMIT)
    print(f"  Found {len(primes):,} primes in {time.time()-t0:.1f}s")

    # ═══════════════════════════════════════════════════════════════════
    #  TOWER PREDICTIONS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  TOWER PREDICTIONS")
    print("=" * 90)

    tower = [
        (3, 2, "Z/2"),
        (5, 4, "Z/4"),
        (7, 6, "Z/6"),
    ]

    print(f"\n  At m = 210 = 2×3×5×7:")
    print(f"\n  {'Prime p':>8}  {'Fiber':>6}  {'Rate in logₚT':>15}  {'Rate in log₃T':>15}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*15}  {'─'*15}")
    for p, phi_p, grp in tower:
        rate_own = 1.0 / phi_p
        rate_log3 = rate_own * math.log(p) / math.log(3)
        print(f"  {p:>8}  {grp:>6}  {rate_own:>15.6f}  {rate_log3:>15.6f}")

    # ═══════════════════════════════════════════════════════════════════
    #  PART 1: m=30 — hunt for p=3 AND p=5 modes
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  PART 1: m = 30 — Two primes, two modes?")
    print("  Predicted: p=3 mode at 1/2 per log₃T, p=5 mode at 1/4 per log₅T")
    print("=" * 90)

    N_vals = np.unique(np.logspace(4, 9, 40, dtype=np.int64))

    t0 = time.time()
    tracks_30 = track_modes_eigvec(primes, N_vals, 30)
    dt = time.time() - t0
    print(f"\n  Tracked 8 modes × {len(N_vals)} N-values in {dt:.1f}s")

    # For each mode, fit in BOTH base 3 and base 5
    print(f"\n  {'Mode':>5}  {'Rate/log₃T':>12}  {'R²₃':>8}  "
          f"{'vs 1/2':>8}  {'Rate/log₅T':>12}  {'R²₅':>8}  "
          f"{'vs 1/4':>8}  {'Best match':>12}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*8}  "
          f"{'─'*8}  {'─'*12}  {'─'*8}  "
          f"{'─'*8}  {'─'*12}")

    for i, track in enumerate(tracks_30):
        r3 = fit_in_base(track, 3)
        r5 = fit_in_base(track, 5)
        if r3 is None or r5 is None:
            continue

        rate3, rsq3, n3 = r3
        rate5, rsq5, n5 = r5
        err3 = abs(rate3 - 0.5) / 0.5 * 100
        err5 = abs(rate5 - 0.25) / 0.25 * 100

        # Determine best match
        if err3 < 20 and rsq3 > 0.9:
            best = "p=3  ★★★"
        elif err5 < 20 and rsq5 > 0.9:
            best = "p=5  ★★★"
        elif err3 < 30 and rsq3 > 0.7:
            best = "p=3  ★"
        elif err5 < 30 and rsq5 > 0.7:
            best = "p=5  ★"
        else:
            best = "---"

        print(f"  {i:>5}  {rate3:>+12.6f}  {rsq3:>8.4f}  "
              f"{err3:>7.1f}%  {rate5:>+12.6f}  {rsq5:>8.4f}  "
              f"{err5:>7.1f}%  {best:>12}")

    # ═══════════════════════════════════════════════════════════════════
    #  PART 2: m=210 — hunt for p=3, p=5, AND p=7 modes
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  PART 2: m = 210 — Three primes, three modes?")
    print("  Predicted: p=3 at 1/2 per log₃T, p=5 at 1/4 per log₅T, p=7 at 1/6 per log₇T")
    print("=" * 90)

    N_vals_210 = np.unique(np.logspace(4, 9, 30, dtype=np.int64))

    t0 = time.time()
    tracks_210 = track_modes_eigvec(primes, N_vals_210, 210)
    dt = time.time() - t0
    print(f"\n  Tracked 48 modes × {len(N_vals_210)} N-values in {dt:.1f}s")

    # For each mode, fit in base 3, 5, and 7
    candidates = {3: [], 5: [], 7: []}
    predictions = {3: 0.5, 5: 0.25, 7: 1.0/6}

    for i, track in enumerate(tracks_210):
        r3 = fit_in_base(track, 3)
        r5 = fit_in_base(track, 5)
        r7 = fit_in_base(track, 7)
        if r3 is None:
            continue

        rate3, rsq3, n3 = r3
        rate5, rsq5, n5 = r5 if r5 else (0, 0, 0)
        rate7, rsq7, n7 = r7 if r7 else (0, 0, 0)

        # Check each prime
        for p, rate, rsq, pred in [(3, rate3, rsq3, 0.5),
                                    (5, rate5, rsq5, 0.25),
                                    (7, rate7, rsq7, 1/6)]:
            err = abs(rate - pred) / pred * 100 if pred > 0 else 999
            if err < 25 and rsq > 0.8:
                candidates[p].append((i, rate, rsq, err))

    for p in [3, 5, 7]:
        pred = predictions[p]
        cands = sorted(candidates[p], key=lambda x: -x[2])  # by R²

        print(f"\n  ┌{'─'*84}┐")
        print(f"  │  p = {p}:  Predicted rate = 1/{p-1} = {pred:.6f} per log_{p}T"
              f"{'':>{62 - len(f'p = {p}:  Predicted rate = 1/{p-1} = {pred:.6f} per log_{p}T')}}│")
        print(f"  └{'─'*84}┘")

        if not cands:
            print(f"  No candidates with error < 25% and R² > 0.8")
        else:
            print(f"  {'Mode':>5}  {'Rate/logₚT':>12}  {'R²':>8}  {'Error':>8}  {'n_pts':>6}")
            print(f"  {'─'*5}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*6}")
            for idx, rate, rsq, err in cands:
                star = " ★★★" if err < 10 and rsq > 0.9 else " ★★" if err < 15 else " ★"
                print(f"  {idx:>5}  {rate:>+12.6f}  {rsq:>8.4f}  {err:>7.1f}%{star}")

    # ═══════════════════════════════════════════════════════════════════
    #  SUMMARY: The Hyper-Radix Tower
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  THE HYPER-RADIX TOWER — SUMMARY")
    print("=" * 90)

    print(f"\n  {'Prime':>6}  {'Fiber':>6}  {'Predicted':>10}  {'Best meas.':>12}  "
          f"{'R²':>8}  {'Error':>8}  {'Modulus':>8}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}")

    for p in [3, 5, 7]:
        pred = predictions[p]
        cands = sorted(candidates[p], key=lambda x: -x[2])
        if cands:
            best = cands[0]
            print(f"  {p:>6}  {'Z/'+str(p-1):>6}  {pred:>10.6f}  {best[1]:>+12.6f}  "
                  f"{best[2]:>8.4f}  {best[3]:>7.1f}%  m=210")
        else:
            print(f"  {p:>6}  {'Z/'+str(p-1):>6}  {pred:>10.6f}  {'NOT FOUND':>12}  "
                  f"{'---':>8}  {'---':>8}  m=210")

    # Check if all three were found
    found = sum(1 for p in [3, 5, 7] if candidates[p])

    if found == 3:
        print(f"""
  ╔══════════════════════════════════════════════════════════════════════════════════════╗
  ║                                                                                    ║
  ║   THE HYPER-RADIX IS A TOWER.                                                     ║
  ║                                                                                    ║
  ║   Each prime p | m contributes an eigenvalue mode rotating at:                     ║
  ║                                                                                    ║
  ║              d(θ/π) / d(logₚT) = 1/(p-1)                                         ║
  ║                                                                                    ║
  ║   Three modes found at m=210, one per prime factor (3, 5, 7).                     ║
  ║   All three match their CRT fiber prediction.                                      ║
  ║                                                                                    ║
  ║   The "hyper" means: not one base, but a PRODUCT of per-prime bases.              ║
  ║   Base 3 is the GROUND FLOOR. p=5 and p=7 are the UPPER FLOORS.                  ║
  ║                                                                                    ║
  ╚══════════════════════════════════════════════════════════════════════════════════════╝
""")
    elif found >= 2:
        print(f"""
  ┌──────────────────────────────────────────────────────────────────────────────────────┐
  │  PARTIAL TOWER: {found}/3 fiber modes found at m=210.                                │
  │  The hyper-radix tower hypothesis is SUPPORTED but not fully confirmed.             │
  └──────────────────────────────────────────────────────────────────────────────────────┘
""")
    else:
        print(f"""
  ┌──────────────────────────────────────────────────────────────────────────────────────┐
  │  TOWER NOT FOUND: Only {found}/3 fiber modes detected.                               │
  │  The frozen regime may still obscure the higher fibers.                             │
  └──────────────────────────────────────────────────────────────────────────────────────┘
""")

    print("=" * 90)


if __name__ == "__main__":
    main()
