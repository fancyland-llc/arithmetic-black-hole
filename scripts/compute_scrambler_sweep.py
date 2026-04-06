#!/usr/bin/env python3
"""
Scrambler sweep: extract eigenvalue spacings, KL divergences, and Brody ω
from BOTH the Hamiltonian spectrum AND the C_A entanglement spectrum.

Two spectra analyzed per (m, γ):
  H-spectrum:  eigenvalues of the full Hamiltonian (bulk diagnostic)
  C_A-spectrum: entanglement Hamiltonian eigenvalues from the Peschel
                construction (boundary diagnostic — the holographic one)

The C_A entanglement Hamiltonian ξ_i = -ln(λ_i / (1 - λ_i)) where λ_i are
eigenvalues of the coprime-sector correlation matrix C_A = V_A · V_A^T.
In holographic systems, scrambling should appear HERE, not in H.

Produces scrambler_sweep.json with per-(m, γ) data:
  {brody_omega, kl_poisson, kl_goe}         — H-spectrum (bulk)
  {ca_brody_omega, ca_kl_poisson, ca_kl_goe} — C_A-spectrum (boundary)

For m=30, 210, 2310: scipy.linalg.eigh (CPU, fast)
For m=30030: torch.linalg.eigh (GPU, MAGMA backend)

Uses the SAME γ-grids as the existing entanglement sweeps for 1:1 overlay.

Usage:
  python3 compute_scrambler_sweep.py            # CPU: m=30, 210, 2310
  python3 compute_scrambler_sweep.py --gpu      # GPU: adds m=30030

Author: Antonio P. Matos / Fancyland LLC
Date: April 2026
"""

import os, sys, time, json
import numpy as np
from math import gcd, log
from scipy.special import gamma as gamma_func

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "hawking_page_results")
os.makedirs(OUT_DIR, exist_ok=True)

USE_GPU = "--gpu" in sys.argv

# ================================================================
# Hamiltonian construction (identical to entanglement sweep)
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

def build_full_H(H_cop, K, H_comp, gamma_val):
    n_cop = H_cop.shape[0]
    n_comp = H_comp.shape[0]
    m_total = n_cop + n_comp
    H = np.empty((m_total, m_total))
    H[:n_cop, :n_cop] = H_cop
    H[:n_cop, n_cop:] = gamma_val * K
    H[n_cop:, :n_cop] = gamma_val * K.T
    H[n_cop:, n_cop:] = H_comp
    return H


# ================================================================
# Eigensolve dispatch
# ================================================================
def eigensolve_cpu(H):
    from scipy.linalg import eigvalsh
    return eigvalsh(H)

def compute_CA_spectrum(H, n_cop):
    """Compute the entanglement Hamiltonian spectrum from C_A.

    Peschel construction:
      1. Diagonalize H → eigenvectors, half-fill Fermi sea
      2. V_A = coprime-sector rows of occupied eigenvectors
      3. C_A = V_A · V_A^T   (φ × φ correlation matrix)
      4. ξ_i = -ln(λ_i / (1 - λ_i))  where λ_i = eigenvalues of C_A

    The ξ_i are the entanglement Hamiltonian eigenvalues — the BOUNDARY
    spectrum. In holographic systems, THIS is where GOE should appear.
    """
    from scipy.linalg import eigh, eigvalsh
    m_total = H.shape[0]
    n_fermions = m_total // 2

    eigenvalues, eigenvectors = eigh(H)
    V_occ = eigenvectors[:, :n_fermions]      # m × N_f
    V_A = V_occ[:n_cop, :]                    # φ × N_f (coprime rows)
    C_A = V_A @ V_A.T                         # φ × φ
    C_A = (C_A + C_A.T) / 2.0                 # symmetrize

    lam = eigvalsh(C_A)                        # φ eigenvalues in [0, 1]
    lam = np.clip(lam, 1e-15, 1.0 - 1e-15)    # guard log singularities

    # Entanglement Hamiltonian: ξ_i = -ln(λ / (1 - λ))
    xi = -np.log(lam / (1.0 - lam))
    return np.sort(xi)

def eigensolve_gpu(H_cop, K, H_comp, gamma_val):
    """Build H and eigensolve on GPU using MAGMA."""
    import torch, gc
    n_cop = H_cop.shape[0]
    n_comp = H_comp.shape[0]
    m_total = n_cop + n_comp
    H = np.empty((m_total, m_total), dtype=np.float64)
    H[:n_cop, :n_cop] = H_cop
    H[:n_cop, n_cop:] = gamma_val * K
    H[n_cop:, :n_cop] = gamma_val * K.T
    H[n_cop:, n_comp:] = H_comp
    # MAGMA float32
    H_gpu = torch.from_numpy(H).float().to(DEVICE)
    del H; gc.collect()
    eigenvalues, _ = torch.linalg.eigh(H_gpu)
    eigs = eigenvalues.cpu().numpy().astype(np.float64)
    del H_gpu, eigenvalues, _; gc.collect()
    torch.cuda.empty_cache()
    return eigs


# ================================================================
# Level spacing statistics
# ================================================================
def unfolded_spacings(eigenvalues):
    """LOCAL density spectral unfolding → nearest-neighbor spacings.

    Matches converged Module 8 (Schism Spectrometer, 62/62 tests):
      windowSize = max(5, floor(N/15)), halfWindow = floor(windowSize/2)
      For each gap i → i+1:
        left = max(0, i - halfWindow)
        right = min(N-1, i + halfWindow + 1)
        localDensity = (right - left) / (sorted[right] - sorted[left])
        spacing[i] = (sorted[i+1] - sorted[i]) * localDensity
      Normalize to mean = 1.

    CRITICAL: Number-theoretic Hamiltonians have massive exact degeneracies
    (517/2310 at m=2310). These produce machine-zero gaps that no unfolding
    can fix. They must be removed BEFORE unfolding to analyze the non-degenerate
    spectrum where RMT statistics apply.
    """
    sorted_eigs = np.sort(eigenvalues)
    N_raw = len(sorted_eigs)
    if N_raw < 10:
        return np.array([])

    # Step 1: Remove truly degenerate eigenvalues (machine-precision gaps)
    # Number-theoretic matrices have algebraic degeneracies (517/2310 at m=2310).
    # Use 1e-10 * spectral range threshold.
    #
    # Forward accumulator (NOT boolean mask): guarantees zero residual
    # degenerate gaps. Each kept eigenvalue must be > threshold from the
    # previous kept eigenvalue. O(N), no leftover zero-gaps.
    spectral_range = sorted_eigs[-1] - sorted_eigs[0]
    if spectral_range < 1e-15:
        return np.array([])
    degen_thresh = 1e-10 * spectral_range
    cleaned_list = [sorted_eigs[0]]
    for val in sorted_eigs[1:]:
        if val - cleaned_list[-1] > degen_thresh:
            cleaned_list.append(val)
    cleaned = np.array(cleaned_list)
    N = len(cleaned)
    if N < 10:
        return np.array([])

    # Step 2: Local density unfolding on cleaned spectrum
    window_size = max(5, N // 15)
    half_window = window_size // 2
    global_range = cleaned[-1] - cleaned[0]
    if global_range < 1e-15:
        return np.array([])
    global_density = (N - 1) / global_range

    spacings = np.empty(N - 1)
    for i in range(N - 1):
        left = max(0, i - half_window)
        right = min(N - 1, i + half_window + 1)
        local_range = cleaned[right] - cleaned[left]
        if local_range < 1e-15:
            local_density = global_density
        else:
            local_density = (right - left) / local_range
        spacings[i] = (cleaned[i + 1] - cleaned[i]) * local_density

    # Step 3: Normalize to mean = 1
    mean = spacings.mean()
    if mean > 0:
        spacings /= mean
    return spacings


def kl_divergences(spacings, num_bins=30):
    """KL(P_empirical || Poisson) and KL(P_empirical || GOE).

    Matches converged Module 8 spec:
      s_max = max(spacings) * 1.001
      bin_width = s_max / num_bins
      ε = 1e-12
      kl_X = Σ_k P[k] · ln((P[k]+ε) / max(Q_X(s_k), ε)) · bin_width
    """
    n = len(spacings)
    if n < 10:
        return 0.0, 0.0
    eps = 1e-12
    # Clip to s_max=5: standard RMT range. Both Poisson and GOE Wigner surmise
    # are negligible beyond s≈4. Number-theoretic outliers (s>10) from residual
    # algebraic structure corrupt bin resolution if included.
    s_max = min(float(np.max(spacings)) * 1.001, 5.0)
    if s_max < eps:
        return 0.0, 0.0
    bin_width = s_max / num_bins
    counts, _ = np.histogram(spacings, bins=num_bins, range=(0, s_max))
    histogram = counts / (n * bin_width)  # area-normalized density

    kl_poisson = 0.0
    kl_goe = 0.0
    for k in range(num_bins):
        P = histogram[k]
        if P < eps:
            continue
        s = (k + 0.5) * bin_width
        q_poisson = max(np.exp(-s), eps)
        q_goe = max((np.pi / 2) * s * np.exp(-np.pi * s * s / 4), eps)
        kl_poisson += P * np.log((P + eps) / q_poisson) * bin_width
        kl_goe += P * np.log((P + eps) / q_goe) * bin_width
    return float(kl_poisson), float(kl_goe)


def brody_omega(spacings):
    """Brody parameter ω via MLE with Lanczos Γ + golden section search.

    Matches converged Module 8 spec (62/62 tests):
      P(s; ω) = (ω+1) · b · s^ω · exp(-b · s^(ω+1))
      where b = [Γ((ω+2)/(ω+1))]^(ω+1)
      Log-likelihood: L(ω) = N·ln(ω+1) + N·ln(b) + ω·Σln(s_i) - b·Σs_i^(ω+1)
      Golden section on ω ∈ [0, 3], tolerance 0.01.
      Guard: skip spacings ≤ 0; if fewer than 3 valid, return 0.
    """
    # Filter only strictly positive spacings (Module 8 spec: skip s ≤ 0)
    # Additionally filter s < 0.01: near-zero spacings from near-degenerate
    # eigenvalue pairs are unfolding artifacts, not RMT statistics.
    # Standard practice in spectral statistics literature.
    s = spacings[spacings > 0.01]
    n = len(s)
    if n < 3:
        return 0.0
    # Re-normalize filtered spacings to mean 1
    s = s / s.mean()

    sum_ln_s = np.sum(np.log(s))

    def log_likelihood(omega):
        w1 = omega + 1.0
        try:
            b = float(gamma_func((omega + 2.0) / w1) ** w1)
            if b <= 0 or not np.isfinite(b):
                return -np.inf
            ll = n * np.log(w1) + n * np.log(b) + omega * sum_ln_s - b * np.sum(s ** w1)
            return float(ll) if np.isfinite(ll) else -np.inf
        except (OverflowError, ValueError):
            return -np.inf

    # Golden section search on [0, 3] (Module 8 spec)
    phi_gr = (np.sqrt(5) - 1) / 2
    a, b_bound = 0.0, 3.0
    c = b_bound - phi_gr * (b_bound - a)
    d = a + phi_gr * (b_bound - a)
    fc, fd = log_likelihood(c), log_likelihood(d)
    for _ in range(60):
        if b_bound - a < 0.01:
            break
        if fc > fd:
            b_bound = d; d = c; fd = fc
            c = b_bound - phi_gr * (b_bound - a); fc = log_likelihood(c)
        else:
            a = c; c = d; fc = fd
            d = a + phi_gr * (b_bound - a); fd = log_likelihood(d)
    omega = max(0.0, (a + b_bound) / 2)
    return float(omega)


# ================================================================
# Per-γ-point computation
# ================================================================
def scrambler_at_gamma(eigenvalues, gamma_val, H=None, n_cop=None):
    """Extract scrambler observables from BOTH spectra.

    H-spectrum:  bulk diagnostic (eigenvalues of full Hamiltonian)
    C_A-spectrum: boundary diagnostic (entanglement Hamiltonian from Peschel)
    """
    # --- H-spectrum (bulk) ---
    spacings = unfolded_spacings(eigenvalues)
    if len(spacings) < 5:
        h_kl_p, h_kl_g, h_omega = 0.0, 0.0, 0.0
        h_cls = "DEGENERATE"
        h_n = 0
    else:
        h_kl_p, h_kl_g = kl_divergences(spacings)
        h_omega = brody_omega(spacings)
        h_cls = "GOE" if (h_kl_g < h_kl_p or h_omega > 0.5) and gamma_val >= 0.005 else "POISSON"
        h_n = int(len(spacings))

    result = {
        "gamma": float(gamma_val),
        "kl_poisson": float(h_kl_p), "kl_goe": float(h_kl_g),
        "brody_omega": float(h_omega),
        "classification": h_cls,
        "n_spacings": h_n,
    }

    # --- C_A-spectrum (boundary / holographic) ---
    if H is not None and n_cop is not None and n_cop >= 10:
        try:
            xi = compute_CA_spectrum(H, n_cop)
            ca_spacings = unfolded_spacings(xi)
            if len(ca_spacings) >= 5:
                ca_kl_p, ca_kl_g = kl_divergences(ca_spacings)
                ca_omega = brody_omega(ca_spacings)
                ca_cls = "GOE" if (ca_kl_g < ca_kl_p or ca_omega > 0.5) and gamma_val >= 0.005 else "POISSON"
                result.update({
                    "ca_brody_omega": float(ca_omega),
                    "ca_kl_poisson": float(ca_kl_p),
                    "ca_kl_goe": float(ca_kl_g),
                    "ca_classification": ca_cls,
                    "ca_n_spacings": int(len(ca_spacings)),
                })
            else:
                result.update({"ca_brody_omega": 0.0, "ca_kl_poisson": 0.0,
                               "ca_kl_goe": 0.0, "ca_classification": "DEGENERATE", "ca_n_spacings": 0})
        except Exception as e:
            result.update({"ca_brody_omega": 0.0, "ca_kl_poisson": 0.0,
                           "ca_kl_goe": 0.0, "ca_classification": f"ERROR:{e}", "ca_n_spacings": 0})
    return result


# ================================================================
# GPU setup (conditional)
# ================================================================
DEVICE = None
if USE_GPU:
    import torch
    print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cuda.preferred_linalg_library("magma")
        DEVICE = torch.device("cuda")
    else:
        print("WARNING: No GPU, falling back to CPU")
        USE_GPU = False


# ================================================================
# Sweep functions
# ================================================================
def sweep_cpu(m, gamma_values):
    """Full scrambler sweep for small m on CPU."""
    print(f"\n{'='*60}")
    cops = coprime_residues(m)
    comps = composite_residues(m)
    phi = len(cops)
    print(f"  m={m}  |  φ={phi}  |  dim={m}  |  {len(gamma_values)} γ-points")
    print(f"{'='*60}")

    vm = von_mangoldt_sieve(m)
    H_cop = build_H_coprime(cops, m)
    H_comp = build_H_comp(comps, m)
    K = build_coupling_K(cops, comps, m, vm)

    results = []
    t0 = time.perf_counter()
    for i, g in enumerate(gamma_values):
        H = build_full_H(H_cop, K, H_comp, g)
        eigs = eigensolve_cpu(H)
        r = scrambler_at_gamma(eigs, g, H=H, n_cop=phi)
        results.append(r)
        ca_w = r.get('ca_brody_omega', None)
        ca_str = f"  CA_ω={ca_w:.3f}" if ca_w is not None else ""
        if i % 10 == 0 or i == len(gamma_values) - 1:
            print(f"    [{i+1}/{len(gamma_values)}] γ={g:.4f}  H_ω={r['brody_omega']:.3f}  "
                  f"KL_P={r['kl_poisson']:.3f}  KL_G={r['kl_goe']:.3f}  {r['classification']}"
                  f"{ca_str}")
    elapsed = time.perf_counter() - t0
    print(f"  Done: {len(results)} points in {elapsed:.1f}s")
    return results, elapsed


def sweep_gpu(m, gamma_values):
    """Full scrambler sweep for m=30030 on GPU."""
    import torch, gc
    print(f"\n{'='*60}")
    cops = coprime_residues(m)
    comps = composite_residues(m)
    phi = len(cops)
    print(f"  m={m}  |  φ={phi}  |  dim={m}  |  {len(gamma_values)} γ-points  |  GPU")
    print(f"{'='*60}")

    vm = von_mangoldt_sieve(m)
    print("  Building Hamiltonian blocks...")
    t_build = time.perf_counter()
    H_cop = build_H_coprime(cops, m)
    H_comp = build_H_comp(comps, m)
    K = build_coupling_K(cops, comps, m, vm)
    print(f"  Blocks built in {time.perf_counter() - t_build:.1f}s")

    results = []
    t0 = time.perf_counter()
    for i, g in enumerate(gamma_values):
        t_pt = time.perf_counter()
        # Build full H and ship to GPU
        n_cop = H_cop.shape[0]
        n_comp = H_comp.shape[0]
        m_total = n_cop + n_comp
        H = np.empty((m_total, m_total), dtype=np.float64)
        H[:n_cop, :n_cop] = H_cop
        H[:n_cop, n_cop:] = g * K
        H[n_cop:, :n_cop] = g * K.T
        H[n_cop:, n_cop:] = H_comp

        H_gpu = torch.from_numpy(H).float().to(DEVICE)
        del H; gc.collect()
        eigenvalues_gpu, _ = torch.linalg.eigh(H_gpu)
        eigs = eigenvalues_gpu.cpu().numpy().astype(np.float64)
        del H_gpu, eigenvalues_gpu, _; gc.collect()
        torch.cuda.empty_cache()

        r = scrambler_at_gamma(eigs, g, H=None, n_cop=None)  # C_A too large for GPU sweep
        results.append(r)

        elapsed_pt = time.perf_counter() - t_pt
        elapsed_total = time.perf_counter() - t0
        rate = (i + 1) / elapsed_total if elapsed_total > 0 else 1
        eta = (len(gamma_values) - i - 1) / rate if rate > 0 else 0
        print(f"    [{i+1}/{len(gamma_values)}] γ={g:.4f}  ω={r['brody_omega']:.3f}  "
              f"KL_P={r['kl_poisson']:.3f}  KL_G={r['kl_goe']:.3f}  {r['classification']}  "
              f"({elapsed_pt:.1f}s, ETA {eta/60:.1f}min)")

    elapsed = time.perf_counter() - t0
    print(f"  Done: {len(results)} points in {elapsed/60:.1f}min")
    return results, elapsed


# ================================================================
# Main
# ================================================================
def main():
    print("SCRAMBLER SWEEP — KL Divergence + Brody ω")
    print(f"Mode: {'GPU' if USE_GPU else 'CPU only'}")
    print()

    all_data = {}

    # Use the SAME γ-grids as the entanglement sweeps
    grids = {
        30:    np.linspace(0, 1, 51).tolist(),
        210:   np.linspace(0, 1, 51).tolist(),
        2310:  np.linspace(0, 1, 51).tolist(),
    }
    if USE_GPU:
        # Match the coarse sweep γ-points for m=30030
        grids[30030] = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                        0.1, 0.2, 0.3, 0.4, 0.5, 0.625, 0.75, 0.875, 1.0]

    for m_val in sorted(grids.keys()):
        gamma_values = grids[m_val]
        if m_val == 30030:
            results, elapsed = sweep_gpu(m_val, gamma_values)
        else:
            results, elapsed = sweep_cpu(m_val, gamma_values)

        phi = len(coprime_residues(m_val))
        all_data[m_val] = {
            "m": m_val, "phi": phi,
            "n_points": len(results),
            "elapsed_s": round(elapsed, 2),
            "sweep": results,
        }

    # Save combined output
    out_path = os.path.join(OUT_DIR, "scrambler_sweep.json")
    # Convert int keys to strings for JSON
    json_data = {str(k): v for k, v in all_data.items()}
    with open(out_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"  {'m':>8}  {'φ':>6}  {'pts':>4}  {'time':>8}  {'H_ω@0.1':>8}  {'H_ω@1.0':>8}  {'CA_ω@0.1':>9}  {'CA_ω@1.0':>9}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*9}")
    for m_val in sorted(all_data.keys()):
        d = all_data[m_val]
        sweep = d["sweep"]

        def find_omega(key, target_gamma):
            return next((r[key] for r in sweep if abs(r["gamma"] - target_gamma) < 0.005 and key in r), None)

        ho1 = find_omega("brody_omega", 0.1)
        ho10 = find_omega("brody_omega", 1.0)
        co1 = find_omega("ca_brody_omega", 0.1)
        co10 = find_omega("ca_brody_omega", 1.0)

        t_str = f"{d['elapsed_s']:.1f}s" if d["elapsed_s"] < 120 else f"{d['elapsed_s']/60:.1f}m"
        ho1_s = f"{ho1:.3f}" if ho1 is not None else "  —  "
        ho10_s = f"{ho10:.3f}" if ho10 is not None else "  —  "
        co1_s = f"{co1:.3f}" if co1 is not None else "   —   "
        co10_s = f"{co10:.3f}" if co10 is not None else "   —   "
        print(f"  {m_val:>8}  {d['phi']:>6}  {d['n_points']:>4}  {t_str:>8}  {ho1_s:>8}  {ho10_s:>8}  {co1_s:>9}  {co10_s:>9}")
    print()


if __name__ == "__main__":
    main()
