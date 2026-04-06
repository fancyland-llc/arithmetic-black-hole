#!/usr/bin/env python3
"""
Arithmetic Hawking-Page Temperature — Large-Primorial γ-Sweep
=============================================================
BVP-8 Phase 2 | m=30030 and m=510510

Computes the entanglement entropy S_A(γ) across the prime–composite
boundary using the Peschel (2003) free-fermion construction at
half-filling N_f = ⌊m/2⌋.

Extracts:
  - γ_c (critical coupling): inflection point of S_A(γ)
  - Page saturation ratio:   S_A(γ=1) / [φ(m)·ln(2)]
  - Phase transition width:  Δγ around inflection
  - Perturbative regime:     S(2ε)/S(ε) ratio

Parallelized across γ-values using ProcessPoolExecutor.
Each worker gets OMP_NUM_THREADS = cpu // n_workers to
prevent BLAS oversubscription.

Usage:
  python compute_hawking_page_temperature.py --calibrate      # 30s benchmark
  python compute_hawking_page_temperature.py --m30030         # m=30030 only (~2h)
  python compute_hawking_page_temperature.py --m510510        # m=510510 only (~days)
  python compute_hawking_page_temperature.py                  # both (full campaign)

  set BH_WORKERS=4   (override worker count; default = cpu//4 capped at 8)

Outputs saved to: hawking_page_results/

Requirements: Python 3.10+, numpy, scipy

Author: Antonio P. Matos / Fancyland LLC
Date: April 2026
"""

import os
import sys

# Force UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ================================================================
# BLAS thread configuration — MUST be before numpy import
# ================================================================
_CPU = os.cpu_count() or 8
_N_WORKERS = int(os.environ.get("BH_WORKERS", "0"))
if _N_WORKERS <= 0:
    # m=30030: each worker needs ~14 GB (H + eigenvector workspace).
    # 2 workers × 16 BLAS threads = ~34 GB peak, fits in 128 GB comfortably.
    _N_WORKERS = max(1, min(2, _CPU // 4))
_BLAS = max(1, _CPU // _N_WORKERS)
os.environ["OMP_NUM_THREADS"] = str(_BLAS)
os.environ["MKL_NUM_THREADS"] = str(_BLAS)
os.environ["OPENBLAS_NUM_THREADS"] = str(_BLAS)

import numpy as np
from scipy.linalg import eigh, eigvalsh
from math import gcd, log
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json
import argparse

import glob

np.set_printoptions(precision=12, linewidth=140)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "hawking_page_results")

EPS_LOG = 1e-15  # L'Hôpital guard


# ================================================================
# Checkpoint / Resume helpers
# ================================================================

def checkpoint_dir(m, phase="coarse"):
    """Return path to checkpoint directory for a given m and phase."""
    d = os.path.join(OUT_DIR, f"m{m}_checkpoints", phase)
    os.makedirs(d, exist_ok=True)
    return d


def save_checkpoint(m, phase, gamma, result):
    """Write a single γ-point result to a checkpoint file."""
    d = checkpoint_dir(m, phase)
    # Use gamma as filename — 8 decimal places avoids collisions
    fname = f"gamma_{gamma:.8f}.json"
    path = os.path.join(d, fname)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return path


def load_checkpoints(m, phase):
    """Load all completed checkpoints. Returns {gamma_float: result_dict}."""
    d = checkpoint_dir(m, phase)
    completed = {}
    for path in glob.glob(os.path.join(d, "gamma_*.json")):
        try:
            with open(path) as f:
                r = json.load(f)
            g = r.get("gamma")
            if g is not None and "S_A" in r:
                completed[round(g, 8)] = r
        except (json.JSONDecodeError, KeyError):
            pass  # corrupt checkpoint — will recompute
    return completed


def save_progress_summary(m, phase, completed_results, gamma_values, page_limit, t_elapsed):
    """Save aggregate progress file with all completed points so far."""
    d = checkpoint_dir(m, phase)
    summary = {
        "m": m,
        "phase": phase,
        "total_points": len(gamma_values),
        "completed_points": len(completed_results),
        "elapsed_s": round(t_elapsed, 1),
        "page_limit": page_limit,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sweep": sorted(completed_results.values(), key=lambda r: r["gamma"]),
    }
    path = os.path.join(d, "_progress.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return path


# ================================================================
# Hamiltonian construction
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


def build_H_coprime(coprimes, m):
    """Build φ×φ coprime distance matrix using vectorized ops."""
    r = np.array(coprimes, dtype=np.float64)
    diff = np.abs(r[:, None] - r[None, :])
    H = np.minimum(diff, m - diff)
    np.fill_diagonal(H, 0.0)
    return H


def build_H_comp(composites, m):
    """Build n_comp × n_comp composite distance matrix."""
    c = np.array(composites, dtype=np.float64)
    diff = np.abs(c[:, None] - c[None, :])
    H = np.minimum(diff, m - diff)
    np.fill_diagonal(H, 0.0)
    return H


def build_coupling_K(coprimes, composites, m, vm_table):
    """Build φ × n_comp von Mangoldt coupling tensor."""
    n_cop = len(coprimes)
    n_comp = len(composites)
    K = np.zeros((n_cop, n_comp))
    for i, r in enumerate(coprimes):
        for j, c in enumerate(composites):
            g = gcd(r, c) if c > 0 else r
            if g <= m:
                K[i, j] = vm_table[g]
    return K


def build_full_hamiltonian_from_blocks(H_cop, K, H_comp, gamma):
    """Assemble full m×m Hamiltonian from precomputed blocks."""
    n_cop = H_cop.shape[0]
    n_comp = H_comp.shape[0]
    m_total = n_cop + n_comp
    H = np.empty((m_total, m_total))
    H[:n_cop, :n_cop] = H_cop
    H[:n_cop, n_cop:] = gamma * K
    H[n_cop:, :n_cop] = gamma * K.T
    H[n_cop:, n_cop:] = H_comp
    return H


# ================================================================
# Peschel entanglement (single γ-point)
# ================================================================

def fermionic_entropy_term(lam):
    s = 0.0
    if lam > EPS_LOG:
        s -= lam * log(lam)
    if (1.0 - lam) > EPS_LOG:
        s -= (1.0 - lam) * log(1.0 - lam)
    return s


def peschel_at_gamma(gamma):
    """
    Worker function: compute S_A at a single γ-point.
    Uses global worker state _W (loaded via initializer).

    Memory-optimized for m=30030 (6.72 GB per full H):
      - overwrite_a=True lets LAPACK reuse H's memory
      - C_A computed directly from V_A (φ×N_f) instead of
        building the full 30030×30030 correlation matrix
      - Explicit del + gc.collect frees arrays between steps
    """
    import gc

    H_cop = _W['H_cop']
    K = _W['K']
    H_comp = _W['H_comp']
    n_cop = _W['n_cop']
    n_fermions = _W['n_fermions']

    # Build full H (6.72 GB)
    H = build_full_hamiltonian_from_blocks(H_cop, K, H_comp, gamma)

    # Step 1: Diagonalize H → eigenvectors
    # overwrite_a=True: LAPACK can reuse H's memory (saves ~6.72 GB copy)
    # check_finite=False: skip NaN scan (saves one O(n²) pass)
    eigenvalues, eigenvectors = eigh(H, overwrite_a=True, check_finite=False)
    del H
    gc.collect()

    # Step 2: Extract coprime-sector occupied eigenvectors
    # V_A = first φ rows, first N_f columns of eigenvector matrix
    # This is (5760 × 15015) ≈ 0.66 GB instead of full (30030 × 15015) ≈ 3.4 GB
    V_A = eigenvectors[:n_cop, :n_fermions].copy()

    # trace_C = N_f (algebraic identity: tr(V·V^T) = tr(V^T·V) = tr(I_{Nf}))
    trace_C = float(n_fermions)

    del eigenvectors
    gc.collect()

    # Step 3: Reduced correlation matrix C_A = V_A · V_A^T
    # This is (φ × φ) = (5760 × 5760) ≈ 0.25 GB
    # Skips the full 30030×30030 C_full entirely (saves 6.72 GB)
    C_A = V_A @ V_A.T
    del V_A
    gc.collect()

    C_A = (C_A + C_A.T) / 2.0  # symmetrize

    # Step 4: Entanglement spectrum
    spectrum = np.sort(eigvalsh(C_A))

    # Bounds check
    min_eig = float(spectrum[0])
    max_eig = float(spectrum[-1])
    all_bounded = bool(min_eig >= -1e-10 and max_eig <= 1.0 + 1e-10)

    # Entanglement entropy
    spectrum_clipped = np.clip(spectrum, 0.0, 1.0)
    S_A = sum(fermionic_entropy_term(lam) for lam in spectrum_clipped)

    trace_C_A = float(np.trace(C_A))

    return {
        "gamma": float(gamma),
        "S_A": float(S_A),
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "all_bounded": all_bounded,
        "trace_C": trace_C,
        "trace_C_A": trace_C_A,
    }


# ================================================================
# Worker pool setup
# ================================================================

_W = {}  # worker-local state


def _init_worker(h_cop_path, k_path, h_comp_path, n_cop, n_fermions):
    """Load precomputed matrix blocks into worker process memory.
    Uses mmap_mode='r' so all workers share the same physical pages
    via the OS page cache — saves ~5.8 GB × (workers-1) of RAM."""
    _W['H_cop'] = np.load(h_cop_path, mmap_mode='r')
    _W['K'] = np.load(k_path, mmap_mode='r')
    _W['H_comp'] = np.load(h_comp_path, mmap_mode='r')
    _W['n_cop'] = n_cop
    _W['n_fermions'] = n_fermions


# ================================================================
# Calibration benchmark
# ================================================================

def calibrate():
    """
    Run a single-point benchmark at m=2310 to estimate timing for larger m.
    scipy.eigh is O(n³) — so m=30030 takes (30030/2310)³ ≈ 2197× longer,
    and m=510510 takes (510510/2310)³ ≈ ~10.8 billion× longer.

    But we only eigsolve once per γ-point, and the matrix is real symmetric,
    so LAPACK dsyevd with multi-threaded BLAS can do much better than
    naïve cubic scaling.
    """
    print("=" * 72)
    print("  CALIBRATION BENCHMARK")
    print("=" * 72)
    print(f"\n  CPU cores: {_CPU}")
    print(f"  Workers:   {_N_WORKERS}")
    print(f"  BLAS threads per worker: {_BLAS}")
    print()

    # m=2310 (2310×2310 dense symmetric eigendecomposition)
    m = 2310
    cops = coprime_residues(m)
    comps = composite_residues(m)
    phi = len(cops)
    n_fermions = m // 2
    vm = von_mangoldt_sieve(m)

    print(f"  m=2310: dim={m}, φ={phi}, N_f={n_fermions}")

    t0 = time.perf_counter()
    H_cop = build_H_coprime(cops, m)
    H_comp = build_H_comp(comps, m)
    K = build_coupling_K(cops, comps, m, vm)
    t_build = time.perf_counter() - t0
    print(f"  Block build time: {t_build:.2f}s")

    H = build_full_hamiltonian_from_blocks(H_cop, K, H_comp, 1.0)
    print(f"  Matrix memory: {H.nbytes / 1024**2:.1f} MB")

    # Warm up BLAS
    _ = eigh(H)

    # Timed eigensolver (full: eigenvalues + eigenvectors)
    t0 = time.perf_counter()
    eigenvalues, eigenvectors = eigh(H)
    t_eigh = time.perf_counter() - t0
    print(f"  eigh(2310×2310) time: {t_eigh:.3f}s")

    # Full Peschel pipeline at one γ-point
    t0 = time.perf_counter()
    V_occ = eigenvectors[:, :n_fermions]
    C_full = V_occ @ V_occ.T
    C_A = (C_full[:phi, :phi] + C_full[:phi, :phi].T) / 2.0
    spectrum = np.sort(eigvalsh(C_A))
    S_A = sum(fermionic_entropy_term(np.clip(l, 0, 1)) for l in spectrum)
    t_pipeline = time.perf_counter() - t0
    print(f"  Full Peschel pipeline: {t_eigh + t_pipeline:.3f}s")
    print(f"  S_A(γ=1) = {S_A:.10f}")

    # Extrapolation
    print("\n  ─── TIME ESTIMATES ───")

    # m=30030 dimensions
    m30 = 30030
    phi30 = 5760  # φ(30030)
    n_points_30 = 51  # γ = 0.00, 0.02, ..., 1.00

    # Empirical scaling: eigh on real symmetric is ~O(n^2.7) for LAPACK dsyevd with good BLAS
    # Conservative estimate: O(n^3)
    ratio_30 = (m30 / m) ** 3
    t_per_point_30 = t_eigh * ratio_30
    # Parallelism factor
    t_wall_30 = (t_per_point_30 * n_points_30) / _N_WORKERS

    print(f"\n  m=30030 (dim={m30}, φ={phi30}):")
    print(f"    Matrix memory: {m30**2 * 8 / 1024**3:.1f} GB")
    print(f"    Est. eigh per point: {t_per_point_30:.0f}s ({t_per_point_30/60:.1f} min)")
    print(f"    Est. wall time ({n_points_30} points, {_N_WORKERS} workers): {t_wall_30/3600:.1f} hours")

    # m=510510 dimensions
    m51 = 510510
    phi51 = 92160  # φ(510510)
    n_points_51 = 51

    ratio_51 = (m51 / m) ** 3
    t_per_point_51 = t_eigh * ratio_51
    t_wall_51 = (t_per_point_51 * n_points_51) / _N_WORKERS
    mem_51 = m51**2 * 8 / 1024**3

    print(f"\n  m=510510 (dim={m51}, φ={phi51}):")
    print(f"    Matrix memory: {mem_51:.0f} GB  ⚠️")
    print(f"    Est. eigh per point: {t_per_point_51:.0f}s ({t_per_point_51/3600:.1f} hours)")
    print(f"    Est. wall time ({n_points_51} points, {_N_WORKERS} workers): {t_wall_51/3600:.0f} hours ({t_wall_51/86400:.1f} days)")

    if mem_51 > 100:
        print(f"\n  ⚠️  m=510510 requires {mem_51:.0f} GB per matrix copy.")
        print(f"      With 128 GB RAM, you can run at most 1 worker ({mem_51:.0f} GB per H + eigenvector workspace).")
        print(f"      RECOMMENDED: Use --m510510-strategy sparse")

    # Feasibility summary
    print("\n  ─── FEASIBILITY SUMMARY ───")
    print(f"    m=30030:  ✅ FEASIBLE  (~{t_wall_30/3600:.1f}h with {_N_WORKERS} workers)")
    if mem_51 * 3 < 128:  # need ~3× matrix size for eigensolver workspace
        print(f"    m=510510: ⚠️  MARGINAL  (~{t_wall_51/86400:.1f} days, {mem_51:.0f} GB/matrix)")
    else:
        print(f"    m=510510: ❌ INFEASIBLE with dense eigensolver ({mem_51:.0f} GB/matrix, need ~{mem_51*3:.0f} GB workspace)")
        print(f"              → See DISCUSSION below for alternatives")

    return {
        "m2310_eigh_s": t_eigh,
        "m30030_est_per_point_s": t_per_point_30,
        "m30030_est_wall_h": t_wall_30 / 3600,
        "m510510_est_per_point_s": t_per_point_51,
        "m510510_est_wall_h": t_wall_51 / 3600,
        "m510510_matrix_gb": mem_51,
    }


# ================================================================
# Sweep runners
# ================================================================

def build_and_save_blocks(m):
    """Build Hamiltonian blocks, save to OUT_DIR (persistent), return paths + metadata.
    If blocks already exist on disk (from a prior run), skip the build."""
    block_dir = os.path.join(OUT_DIR, f"m{m}_blocks")
    os.makedirs(block_dir, exist_ok=True)

    h_cop_path = os.path.join(block_dir, f"H_cop_{m}.npy")
    k_path = os.path.join(block_dir, f"K_{m}.npy")
    h_comp_path = os.path.join(block_dir, f"H_comp_{m}.npy")
    meta_path = os.path.join(block_dir, f"meta_{m}.json")

    # Check for existing blocks
    if all(os.path.exists(p) for p in [h_cop_path, k_path, h_comp_path, meta_path]):
        print(f"\n  ♻️  Reusing cached Hamiltonian blocks for m={m}")
        with open(meta_path) as f:
            meta = json.load(f)
        meta["h_cop_path"] = h_cop_path
        meta["k_path"] = k_path
        meta["h_comp_path"] = h_comp_path
        print(f"    φ(m) = {meta['phi']}, n_comp = {meta['n_comp']}, N_f = {meta['n_fermions']}")
        return meta

    print(f"\n  Building Hamiltonian blocks for m={m}...")
    t0 = time.perf_counter()

    cops = coprime_residues(m)
    comps = composite_residues(m)
    phi = len(cops)
    n_comp = len(comps)
    n_fermions = m // 2
    vm = von_mangoldt_sieve(m)

    print(f"    φ(m) = {phi}, n_comp = {n_comp}, N_f = {n_fermions}")
    print(f"    Residue enumeration: {time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    H_cop = build_H_coprime(cops, m)
    print(f"    H_coprime ({phi}×{phi}): {time.perf_counter() - t1:.1f}s, {H_cop.nbytes/1024**2:.1f} MB")

    t1 = time.perf_counter()
    H_comp = build_H_comp(comps, m)
    print(f"    H_composite ({n_comp}×{n_comp}): {time.perf_counter() - t1:.1f}s, {H_comp.nbytes/1024**2:.1f} MB")

    t1 = time.perf_counter()
    K = build_coupling_K(cops, comps, m, vm)
    print(f"    K coupling ({phi}×{n_comp}): {time.perf_counter() - t1:.1f}s, {K.nbytes/1024**2:.1f} MB")

    np.save(h_cop_path, H_cop)
    np.save(k_path, K)
    np.save(h_comp_path, H_comp)

    meta = {
        "phi": phi,
        "n_comp": n_comp,
        "n_fermions": n_fermions,
        "page_limit": phi * log(2),
        "h_cop_path": h_cop_path,
        "k_path": k_path,
        "h_comp_path": h_comp_path,
    }
    with open(meta_path, "w") as f:
        json.dump({k: v for k, v in meta.items() if not k.endswith("_path")}, f, indent=2)

    full_H_bytes = (phi + n_comp) ** 2 * 8
    print(f"    Full H memory per assembly: {full_H_bytes/1024**3:.2f} GB")
    print(f"    Total block build: {time.perf_counter() - t0:.1f}s")
    print(f"    Blocks saved to: {block_dir}")

    return meta


def run_sweep(m, gamma_values, workers, meta, label="", phase="coarse"):
    """
    Parallel γ-sweep with checkpoint/resume.
    
    On each completed γ-point:
      1. Result is checkpointed to disk immediately
      2. Aggregate progress file is updated
    
    On resume (if the process was killed):
      - Existing checkpoints are loaded
      - Only missing γ-points are dispatched to workers
    """
    page_limit = meta["page_limit"]
    n_total = len(gamma_values)

    # ── Resume: load existing checkpoints ──
    existing = load_checkpoints(m, phase)
    if existing:
        print(f"\n  ♻️  RESUME: found {len(existing)} checkpoints for m={m} phase={phase}")

    # Determine which γ-points still need computation
    todo_gammas = []
    todo_indices = []
    results = [None] * n_total
    for i, g in enumerate(gamma_values):
        g_key = round(g, 8)
        if g_key in existing:
            results[i] = existing[g_key]
        else:
            todo_gammas.append(g)
            todo_indices.append(i)

    n_cached = n_total - len(todo_gammas)
    n_remaining = len(todo_gammas)

    print(f"\n  {'='*60}")
    print(f"  γ-SWEEP: m={m} | phase={phase}")
    print(f"  Total: {n_total} | Cached: {n_cached} | Remaining: {n_remaining}")
    print(f"  {'='*60}")

    if n_remaining == 0:
        print(f"  ✅ All {n_total} points already computed. Skipping.")
        total_time = 0.0
        return results, total_time

    actual_workers = min(workers, n_remaining)
    all_completed = dict(existing)  # running dict of all results (for progress saves)
    for r in results:
        if r and "gamma" in r:
            all_completed[round(r["gamma"], 8)] = r

    completed_new = 0
    t0 = time.perf_counter()

    with ProcessPoolExecutor(
        max_workers=actual_workers,
        initializer=_init_worker,
        initargs=(
            meta["h_cop_path"], meta["k_path"], meta["h_comp_path"],
            meta["phi"], meta["n_fermions"]
        ),
    ) as pool:
        futs = {}
        for g, idx in zip(todo_gammas, todo_indices):
            futs[pool.submit(peschel_at_gamma, g)] = idx

        for f in as_completed(futs):
            idx = futs[f]
            gamma_val = gamma_values[idx]
            try:
                result = f.result()
                results[idx] = result

                # ── Checkpoint immediately ──
                save_checkpoint(m, phase, gamma_val, result)
                all_completed[round(gamma_val, 8)] = result

            except Exception as e:
                print(f"\n    ❌ γ={gamma_val:.4f} FAILED: {e}")
                results[idx] = {"gamma": float(gamma_val), "error": str(e)}

            completed_new += 1
            completed_total = n_cached + completed_new
            elapsed = time.perf_counter() - t0
            rate = completed_new / elapsed if elapsed > 0 else 0
            eta = (n_remaining - completed_new) / rate if rate > 0 else 0

            # Progress print every point (these take minutes each)
            pct = 100 * completed_total / n_total
            S_done = results[idx].get("S_A", "?") if results[idx] else "?"
            print(f"    [{pct:5.1f}%] {completed_total}/{n_total} | "
                  f"γ={gamma_val:.4f} S_A={S_done} | "
                  f"elapsed={elapsed/60:.1f}min ETA={eta/60:.1f}min ({eta/3600:.1f}h) | "
                  f"💾 checkpointed")

            # Save aggregate progress every point
            save_progress_summary(m, phase, all_completed, gamma_values,
                                  page_limit, elapsed)

    total_time = time.perf_counter() - t0
    print(f"\n  Sweep complete: {total_time:.1f}s ({total_time/60:.1f} min, {total_time/3600:.1f}h)")
    print(f"  Computed {completed_new} new points, {n_cached} from cache.")

    return results, total_time


def analyze_sweep(m, gamma_values, results, page_limit):
    """Post-process sweep: extract γ_c, saturation, transition width."""
    print(f"\n  {'─'*60}")
    print(f"  ANALYSIS: m={m}")
    print(f"  {'─'*60}")

    S_vals = []
    gamma_vals = []
    for r in results:
        if r and "S_A" in r:
            gamma_vals.append(r["gamma"])
            S_vals.append(r["S_A"])

    S_arr = np.array(S_vals)
    G_arr = np.array(gamma_vals)

    # Saturation
    S_at_1 = S_vals[-1] if gamma_vals[-1] == 1.0 else None
    saturation = S_at_1 / page_limit if S_at_1 else None
    print(f"\n  Page limit: {page_limit:.6f}")
    if S_at_1:
        print(f"  S_A(γ=1): {S_at_1:.10f}")
        print(f"  Saturation S_A(1)/Page: {saturation:.6f} ({saturation*100:.2f}%)")

    # S_A at γ=0
    S_at_0 = S_vals[0] if gamma_vals[0] == 0.0 else None
    if S_at_0 is not None:
        print(f"  S_A(γ=0): {S_at_0:.15e} {'(≈0 ✓)' if abs(S_at_0) < 1e-8 else '(NOT ZERO ✗)'}")

    # Monotonicity
    mono = all(S_arr[i] <= S_arr[i+1] + 1e-12 for i in range(len(S_arr) - 1))
    print(f"  Monotonic: {'YES ✓' if mono else 'NO ✗'}")

    # Eigenvalue bounds
    all_bounded = all(r.get("all_bounded", False) for r in results if r)
    print(f"  All spectra bounded [0,1]: {'YES ✓' if all_bounded else 'NO — INVESTIGATE'}")

    # Inflection point (γ_c) — numerical second derivative
    if len(S_arr) >= 5:
        dS = np.gradient(S_arr, G_arr)
        d2S = np.gradient(dS, G_arr)
        # Exclude boundary points (gradient artifacts)
        interior = slice(2, -2)
        i_inflect = np.argmax(np.abs(d2S[interior])) + 2
        gamma_c = float(G_arr[i_inflect])
        d2S_max = float(d2S[i_inflect])
        dS_at_c = float(dS[i_inflect])

        print(f"\n  Inflection point (Hawking-Page temperature):")
        print(f"    γ_c ≈ {gamma_c:.6f}")
        print(f"    dS/dγ at γ_c: {dS_at_c:.6f}")
        print(f"    d²S/dγ² at γ_c: {d2S_max:.6f}")

        # Transition width (FWHM of |d²S/dγ²|)
        peak = np.abs(d2S_max)
        half_max = peak / 2
        above_half = np.where(np.abs(d2S) > half_max)[0]
        if len(above_half) >= 2:
            width = float(G_arr[above_half[-1]] - G_arr[above_half[0]])
            print(f"    Transition width (FWHM of |d²S/dγ²|): Δγ ≈ {width:.6f}")
        else:
            width = None
            print(f"    Transition width: could not determine (too narrow for grid)")
    else:
        gamma_c = None
        dS_at_c = None
        d2S_max = None
        width = None

    # Perturbative regime
    try:
        eps1 = 0.02
        eps2 = 0.04
        S_eps1 = np.interp(eps1, G_arr, S_arr)
        S_eps2 = np.interp(eps2, G_arr, S_arr)
        if S_eps1 > 0:
            pert_ratio = S_eps2 / S_eps1
            expected_quad = (eps2 / eps1) ** 2
            print(f"\n  Perturbative check: S({eps2})/S({eps1}) = {pert_ratio:.4f} (quadratic → {expected_quad:.1f})")
    except Exception:
        pert_ratio = None

    # Entropy curve printout
    print(f"\n  γ-SWEEP ENTROPY CURVE:")
    print(f"  {'γ':>6} | {'S_A':>14} | {'S_A/Page':>10} | {'bounded':>7}")
    print(f"  {'-'*6}-+-{'-'*14}-+-{'-'*10}-+-{'-'*7}")
    for r in results:
        if r and "S_A" in r:
            ratio = r["S_A"] / page_limit if page_limit > 0 else 0
            bnd = "Y" if r.get("all_bounded") else "N"
            print(f"  {r['gamma']:6.4f} | {r['S_A']:14.8f} | {ratio:10.6f} | {bnd:>7}")

    return {
        "m": m,
        "phi": int(page_limit / log(2)),
        "n_fermions": m // 2,
        "page_limit": page_limit,
        "S_at_0": S_at_0,
        "S_at_1": S_at_1,
        "saturation": saturation,
        "monotonic": mono,
        "all_bounded": all_bounded,
        "gamma_c": gamma_c,
        "dS_at_gamma_c": dS_at_c,
        "d2S_at_gamma_c": d2S_max,
        "transition_width": width,
        "sweep": [r for r in results if r],
    }


# ================================================================
# Job definitions
# ================================================================

def sweep_m30030(workers):
    """
    m=30030 = 2·3·5·7·11·13
    dim(H) = 30030, φ = 5760, N_f = 15015
    Matrix: 30030² × 8B ≈ 6.7 GB
    """
    m = 30030
    print("\n" + "=" * 72)
    print(f"  JOB: m={m} ENTANGLEMENT γ-SWEEP")
    print(f"  dim(H)={m}, φ(m)=5760, N_f=15015")
    print(f"  Matrix memory: {m**2 * 8 / 1024**3:.1f} GB per full H")
    print("=" * 72)

    # Smart sweep: dense near origin (where γ_c lives), sparse elsewhere.
    # Based on prior data: γ_c(m=210)≈0.075, γ_c(m=2310)<0.05 → expect γ_c(m=30030) << 0.05
    #
    # Phase 1 (coarse): 21 points total
    #   - 11 points in [0, 0.10]  (dense: transition region)
    #   - 5 points in (0.10, 0.50]  (moderate)
    #   - 5 points in (0.50, 1.00]  (sparse: saturation plateau)
    gamma_dense = np.linspace(0.0, 0.10, 11).tolist()
    gamma_mid = np.linspace(0.20, 0.50, 4).tolist()
    gamma_tail = np.linspace(0.625, 1.00, 4).tolist()
    gamma_values = sorted(set(gamma_dense + gamma_mid + gamma_tail))

    n_pts = len(gamma_values)
    est_hours = (n_pts * 265 / 60) / workers  # 265 min/point from calibration
    print(f"\n  Sweep design: {n_pts} γ-points (smart: dense near origin)")
    print(f"  Estimated wall time: {est_hours:.1f} hours ({est_hours/workers:.1f}h per worker)")
    print(f"  Checkpoints: {checkpoint_dir(m, 'coarse')}")
    print(f"  Resume: re-run same command to continue from last checkpoint")

    meta = build_and_save_blocks(m)

    results, total_time = run_sweep(m, gamma_values, workers, meta,
                                    label="m30030", phase="coarse")

    analysis = analyze_sweep(m, gamma_values, results, meta["page_limit"])
    analysis["total_time_s"] = total_time
    analysis["workers"] = workers

    # Phase 2: micro-refinement around detected γ_c (11 points)
    gamma_c = analysis.get("gamma_c")
    if gamma_c and gamma_c > 0.002:
        lo = max(0.0, gamma_c - 0.02)
        hi = min(0.15, gamma_c + 0.02)
        micro_gammas = np.linspace(lo, hi, 11).tolist()

        print(f"\n  ─── MICRO-REFINEMENT around γ_c ≈ {gamma_c:.4f} ───")
        print(f"  {len(micro_gammas)} points in [{lo:.4f}, {hi:.4f}]")
        micro_results, micro_time = run_sweep(m, micro_gammas, workers, meta,
                                              label="m30030_micro", phase="micro")
        micro_analysis = analyze_sweep(m, micro_gammas, micro_results, meta["page_limit"])
        analysis["micro_refinement"] = micro_analysis
        analysis["micro_refinement"]["total_time_s"] = micro_time

    return analysis


def sweep_m510510(workers):
    """
    m=510510 = 2·3·5·7·11·13·17
    dim(H) = 510510, φ = 92160, N_f = 255255

    ⚠️ MEMORY WARNING:
    Full H matrix = 510510² × 8B ≈ 1.9 TB — INFEASIBLE for dense eigensolver.

    STRATEGY: We DON'T build the full H. Instead, we use the block structure:
      - H_cop  (φ × φ) = 92160² × 8B ≈ 63 GB
      - H_comp (n_c × n_c) = 418350² × 8B ≈ 1.3 TB
      - K      (φ × n_c) = 92160 × 418350 × 8B ≈ 287 GB

    Even H_cop alone at 63 GB is too large for a single dense eigensolver.

    ALTERNATIVE APPROACH: Subsystem-only Peschel construction.
    Instead of diagonalizing the full H, we can use:
      1. Iterative eigensolver (Lanczos) for the lowest N_f eigenvectors
      2. Or: Schur complement to reduce to the φ×φ coprime sector

    For now, this function is a PLACEHOLDER that documents the computational
    requirements. The actual m=510510 campaign requires either:
      (a) A cluster with >2 TB RAM, or
      (b) An iterative eigensolver (scipy.sparse.linalg.eigsh), or
      (c) A fundamentally different mathematical approach.
    """
    m = 510510
    phi = 92160
    n_comp = m - phi  # 418350
    n_fermions = m // 2

    print("\n" + "=" * 72)
    print(f"  JOB: m={m} ENTANGLEMENT γ-SWEEP")
    print(f"  dim(H)={m}, φ(m)={phi}, N_f={n_fermions}")
    print("=" * 72)

    mem_full_H = m**2 * 8 / 1024**4  # TB
    mem_H_cop = phi**2 * 8 / 1024**3
    mem_K = phi * n_comp * 8 / 1024**3
    mem_H_comp = n_comp**2 * 8 / 1024**4

    print(f"\n  Memory requirements (dense):")
    print(f"    Full H:    {mem_full_H:.1f} TB  ❌ INFEASIBLE")
    print(f"    H_cop:     {mem_H_cop:.1f} GB")
    print(f"    K:         {mem_K:.1f} GB")
    print(f"    H_comp:    {mem_H_comp:.1f} TB  ❌")
    print(f"    Total:     >{mem_full_H:.1f} TB")
    print(f"\n  Available RAM: 128 GB → CANNOT fit dense m=510510")

    print(f"\n  ─── ITERATIVE EIGENSOLVER STRATEGY ───")
    print(f"  Using scipy.sparse.linalg.eigsh (ARPACK Lanczos)")
    print(f"  We need the lowest N_f = {n_fermions} eigenvectors.")
    print(f"  ARPACK needs k < n/2 → {n_fermions} < {m//2} ✓ (barely)")
    print(f"  But ARPACK stores k Lanczos vectors of dim m = {m}")
    print(f"  Memory: {n_fermions} × {m} × 8B = {n_fermions * m * 8 / 1024**3:.0f} GB")
    print(f"  That's {n_fermions * m * 8 / 1024**3:.0f} GB just for Lanczos vectors.")
    print(f"\n  ❌ ARPACK is also infeasible at k=N_f={n_fermions}.")

    print(f"\n  ─── FEASIBLE ALTERNATIVES ───")
    print(f"  1. STOCHASTIC TRACE ESTIMATION (Hutchinson's method)")
    print(f"     Estimate S_A without full eigenvector decomposition.")
    print(f"     Requires only matrix-vector products with H.")
    print(f"     O(m) memory per random vector, O(m²) per matvec.")
    print(f"     Paper: Ubaru, Chen, Saad (2017) 'Fast Estimation of tr(f(A))'")
    print(f"")
    print(f"  2. KERNEL POLYNOMIAL METHOD (KPM)")
    print(f"     Expand tr(f(C_A)) in Chebyshev polynomials.")
    print(f"     Only needs matvec with H, not full diag.")
    print(f"     Memory: O(m), Time: O(Nm²) for N Chebyshev moments.")
    print(f"     Well-suited for entanglement entropy.")
    print(f"")
    print(f"  3. REDUCED MODULI EXTRAPOLATION")
    print(f"     Use m=6, 30, 210, 2310, 30030 to extrapolate")
    print(f"     thermodynamic limit behavior without direct m=510510 computation.")
    print(f"     Already have evidence: saturation → 50%, γ_c → 0.")

    return {
        "m": m,
        "status": "INFEASIBLE_DENSE",
        "reason": f"Full H = {mem_full_H:.1f} TB, exceeds 128 GB RAM",
        "recommendation": "Use m=30030 results + extrapolation, or implement KPM/stochastic trace",
    }


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Arithmetic Hawking-Page Temperature — Large-Primorial γ-Sweep"
    )
    parser.add_argument("--calibrate", action="store_true",
                        help="Run benchmark at m=2310 to estimate timing")
    parser.add_argument("--m30030", action="store_true",
                        help="Run m=30030 sweep only")
    parser.add_argument("--m510510", action="store_true",
                        help="Run m=510510 analysis (feasibility check)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("ARITHMETIC HAWKING-PAGE TEMPERATURE — LARGE-PRIMORIAL γ-SWEEP")
    print(f"Python {sys.version}")
    print(f"NumPy {np.__version__}")
    print(f"SciPy {__import__('scipy').__version__}")
    print(f"CPU cores: {_CPU}, Workers: {_N_WORKERS}, BLAS threads/worker: {_BLAS}")
    print()

    run_calibrate = args.calibrate or (not args.m30030 and not args.m510510)
    run_30030 = args.m30030 or (not args.calibrate and not args.m510510)
    run_510510 = args.m510510

    all_results = {}

    if run_calibrate:
        all_results["calibration"] = calibrate()

    if run_30030:
        all_results["m30030"] = sweep_m30030(_N_WORKERS)

        # Save results
        out_path = os.path.join(OUT_DIR, "m30030_hawking_page.json")
        with open(out_path, "w") as f:
            json.dump(all_results["m30030"], f, indent=2, default=str)
        print(f"\n  Results saved to: {out_path}")

    if run_510510:
        all_results["m510510"] = sweep_m510510(_N_WORKERS)

    # Combined summary
    if "m30030" in all_results and all_results["m30030"].get("gamma_c"):
        print("\n" + "=" * 72)
        print("  HAWKING-PAGE TEMPERATURE SUMMARY")
        print("=" * 72)

        # Historical data
        print(f"\n  {'m':>8} | {'φ(m)':>6} | {'γ_c (inflection)':>18} | {'Saturation':>12} | {'Source':>12}")
        print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*18}-+-{'-'*12}-+-{'-'*12}")

        historical = [
            (30, 8, 0.250, 0.207, "fixtures"),
            (210, 48, 0.075, 0.497, "fixtures"),
            (2310, 480, "<0.05", 0.488, "fixtures"),
        ]
        for m_h, phi_h, gc_h, sat_h, src in historical:
            gc_str = f"{gc_h:.6f}" if isinstance(gc_h, float) else gc_h
            print(f"  {m_h:>8} | {phi_h:>6} | {gc_str:>18} | {sat_h:>11.4f} | {src:>12}")

        r30 = all_results["m30030"]
        gc30 = r30.get("gamma_c")
        sat30 = r30.get("saturation")
        gc_str = f"{gc30:.6f}" if gc30 else "N/A"
        sat_str = f"{sat30:.4f}" if sat30 else "N/A"
        print(f"  {30030:>8} | {5760:>6} | {gc_str:>18} | {sat_str:>12} | {'THIS RUN':>12}")

        if gc30 and gc30 < 0.05:
            print(f"\n  📊 γ_c continues to collapse toward zero.")
            print(f"     The arithmetic Hawking-Page transition sharpens with m.")
            print(f"     Thermodynamic limit: discontinuous phase transition at γ = 0+.")

        if sat30 and abs(sat30 - 0.5) < 0.02:
            print(f"\n  📊 Saturation ratio = {sat30:.4f} ≈ 1/2.")
            print(f"     Page curve convergence confirmed at m=30030.")

    print("\n  Done.")


if __name__ == "__main__":
    main()
