#!/usr/bin/env python3
"""
Fine-grid γ-sweep for m=30030 near the phase transition.
Targets γ ∈ [0.0001, 0.01] with 25 points to resolve γ_c.

Designed for GCE L4 GPU (24GB VRAM) with PyTorch MAGMA backend.
Expected: ~3 min/point → ~75 min total, ~$0.30 spot.

Usage:
  BH_WORKERS=1 python compute_fine_grid_m30030.py

Author: Antonio P. Matos / Fancyland LLC
Date: April 2026
"""

import os, sys, time, json, glob
import numpy as np
from math import gcd, log

# Force single worker for GPU
os.environ["BH_WORKERS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "hawking_page_results")
EPS_LOG = 1e-15

# ================================================================
# GPU setup — PyTorch MAGMA backend
# ================================================================
import torch
print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # Force MAGMA backend for eigensolves
    torch.backends.cuda.preferred_linalg_library("magma")
    DEVICE = torch.device("cuda")
else:
    print("WARNING: No GPU detected, falling back to CPU")
    DEVICE = torch.device("cpu")

# ================================================================
# Checkpoint helpers
# ================================================================
def checkpoint_dir(m, phase):
    d = os.path.join(OUT_DIR, f"m{m}_checkpoints", phase)
    os.makedirs(d, exist_ok=True)
    return d

def save_checkpoint(m, phase, gamma, result):
    d = checkpoint_dir(m, phase)
    fname = f"gamma_{gamma:.8f}.json"
    with open(os.path.join(d, fname), "w") as f:
        json.dump(result, f, indent=2, default=str)

def load_checkpoints(m, phase):
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
            pass
    return completed

def save_progress(m, phase, results, gamma_values, page_limit, elapsed):
    d = checkpoint_dir(m, phase)
    summary = {
        "m": m, "phase": phase,
        "total_points": len(gamma_values),
        "completed_points": len(results),
        "elapsed_s": round(elapsed, 1),
        "page_limit": page_limit,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sweep": sorted(results.values(), key=lambda r: r["gamma"]),
    }
    with open(os.path.join(d, "_progress.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

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

# ================================================================
# Peschel entanglement — GPU accelerated
# ================================================================
def fermionic_entropy_term(lam):
    s = 0.0
    if lam > EPS_LOG:
        s -= lam * log(lam)
    if (1.0 - lam) > EPS_LOG:
        s -= (1.0 - lam) * log(1.0 - lam)
    return s

def peschel_at_gamma_gpu(gamma, H_cop_np, K_np, H_comp_np, n_cop, n_fermions):
    """Compute S_A at a single γ-point using GPU eigensolve."""
    import gc
    
    # Build full H in numpy, then ship to GPU as float32
    n_comp = H_comp_np.shape[0]
    m_total = n_cop + n_comp
    H = np.empty((m_total, m_total), dtype=np.float64)
    H[:n_cop, :n_cop] = H_cop_np
    H[:n_cop, n_cop:] = gamma * K_np
    H[n_cop:, :n_cop] = gamma * K_np.T
    H[n_cop:, n_cop:] = H_comp_np
    
    # GPU eigensolve (float32 for speed, MAGMA backend)
    H_gpu = torch.from_numpy(H).float().to(DEVICE)
    del H; gc.collect()
    
    eigenvalues, eigenvectors = torch.linalg.eigh(H_gpu)
    del H_gpu; gc.collect()
    torch.cuda.empty_cache()
    
    # Extract coprime-sector occupied eigenvectors (stay on GPU)
    V_A = eigenvectors[:n_cop, :n_fermions].clone()
    del eigenvalues, eigenvectors; gc.collect()
    torch.cuda.empty_cache()
    
    trace_C = float(n_fermions)
    
    # C_A = V_A @ V_A^T on GPU
    C_A = V_A @ V_A.T
    del V_A; gc.collect()
    torch.cuda.empty_cache()
    
    C_A = (C_A + C_A.T) / 2.0
    
    # Eigenvalues of C_A (small matrix, still on GPU)
    spectrum = torch.linalg.eigvalsh(C_A)
    trace_C_A = float(C_A.trace().item())
    del C_A; gc.collect()
    torch.cuda.empty_cache()
    
    # Back to CPU for entropy
    spectrum_np = spectrum.cpu().numpy().astype(np.float64)
    del spectrum; gc.collect()
    torch.cuda.empty_cache()
    
    spectrum_np.sort()
    min_eig = float(spectrum_np[0])
    max_eig = float(spectrum_np[-1])
    all_bounded = bool(min_eig >= -1e-10 and max_eig <= 1.0 + 1e-10)
    
    spectrum_clipped = np.clip(spectrum_np, 0.0, 1.0)
    S_A = sum(fermionic_entropy_term(lam) for lam in spectrum_clipped)
    
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
# Main
# ================================================================
def main():
    m = 30030
    n_fermions = m // 2  # 15015
    
    # Fine grid: 25 points in [0.0001, 0.01] — this is where γ_c lives
    gamma_values = np.linspace(0.0001, 0.01, 25).tolist()
    phase = "fine"
    
    page_limit = 5760 * log(2)  # φ(30030) * ln(2)
    
    print(f"m={m}, fine grid: {len(gamma_values)} points in [{gamma_values[0]:.4f}, {gamma_values[-1]:.4f}]")
    print(f"Phase: {phase}")
    print(f"Page limit: {page_limit:.6f}")
    
    # Check for existing checkpoints
    existing = load_checkpoints(m, phase)
    todo = [g for g in gamma_values if round(g, 8) not in existing]
    print(f"Existing checkpoints: {len(existing)}, remaining: {len(todo)}")
    
    if not todo:
        print("All points already computed!")
        return
    
    # Build blocks
    print(f"\nBuilding Hamiltonian blocks for m={m}...")
    t0 = time.perf_counter()
    
    cops = coprime_residues(m)
    comps = composite_residues(m)
    n_cop = len(cops)
    vm = von_mangoldt_sieve(m)
    
    print(f"  φ(m) = {n_cop}, n_comp = {len(comps)}, N_f = {n_fermions}")
    
    H_cop = build_H_coprime(cops, m)
    H_comp = build_H_comp(comps, m)
    K = build_coupling_K(cops, comps, m, vm)
    
    print(f"  Block build: {time.perf_counter() - t0:.1f}s")
    print(f"  H_cop: {H_cop.shape}, K: {K.shape}, H_comp: {H_comp.shape}")
    
    # Sweep
    all_results = dict(existing)
    t_start = time.perf_counter()
    
    for i, gamma in enumerate(todo):
        t_pt = time.perf_counter()
        print(f"\n[{i+1}/{len(todo)}] γ = {gamma:.8f} ...")
        
        result = peschel_at_gamma_gpu(gamma, H_cop, K, H_comp, n_cop, n_fermions)
        
        save_checkpoint(m, phase, gamma, result)
        all_results[round(gamma, 8)] = result
        
        elapsed_pt = time.perf_counter() - t_pt
        elapsed_total = time.perf_counter() - t_start
        rate = (i + 1) / elapsed_total
        eta = (len(todo) - i - 1) / rate if rate > 0 else 0
        
        print(f"  S_A = {result['S_A']:.10f}, bounded={result['all_bounded']}")
        print(f"  {elapsed_pt:.1f}s this point | {elapsed_total/60:.1f}min elapsed | ETA {eta/60:.1f}min")
        
        save_progress(m, phase, all_results, gamma_values, page_limit, elapsed_total)
    
    total = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"DONE: {len(todo)} points in {total/60:.1f} min")
    print(f"All {len(gamma_values)} fine-grid points complete.")
    print(f"Checkpoints: {checkpoint_dir(m, phase)}")

if __name__ == "__main__":
    main()
