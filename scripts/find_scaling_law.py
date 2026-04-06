#!/usr/bin/env python3
"""
DIMENSIONAL SCALING LAW: The Primorial Lattice Fold
=====================================================
The Chinese Remainder Theorem says:
  Z/210Z  ≅  Z/30Z  ×  Z/7Z
  Z/2310Z ≅  Z/210Z ×  Z/11Z  ≅  Z/30Z × Z/7Z × Z/11Z

The lattice FOLDS. Each primorial embeds the previous one.

This script tests three hypotheses:

1. UNIVERSAL CURVE: Tr(R)*ln(N) is a function of m/T alone.
   If true, all moduli collapse onto one master curve.

2. CRT PROJECTION: The m=30 trace is structurally embedded inside
   the m=210 residual via the CRT tensor decomposition.
   Tr_30(Proj_{30} R_{210}) should recover -ln(pi)/ln(N).

3. FOLD SCALING: The trace obeys Tr(R)*ln(N) = F(m/T) where
   F(x) -> -ln(pi) as x -> some critical threshold x_c.
"""
import math, numpy as np, sys
from itertools import product as iterproduct

def sieve(limit):
    s = bytearray(b'\x01') * (limit + 1)
    s[0] = s[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if s[i]: s[i*i::i] = b'\x00' * len(s[i*i::i])
    return [i for i, v in enumerate(s) if v]

def coprime_res(m):
    return sorted(r for r in range(1, m) if math.gcd(r, m) == 1)

def fwd_dist(m, res):
    phi = len(res)
    D = np.zeros((phi, phi))
    for i, a in enumerate(res):
        for j, b in enumerate(res):
            D[i,j] = m if i==j else (b-a)%m
    return D

def count_trans(primes, m, res):
    idx = {r: i for i, r in enumerate(res)}
    phi = len(res)
    C = np.zeros((phi, phi), dtype=np.int64)
    for k in range(len(primes)-1):
        a, b = primes[k]%m, primes[k+1]%m
        if a in idx and b in idx:
            C[idx[a], idx[b]] += 1
    return C

def norm_rows(M):
    s = M.sum(axis=1, keepdims=True).astype(np.float64)
    return M / np.maximum(s, 1e-30)

def boltz(D, T):
    lw = -D/T
    lZ = np.logaddexp.reduce(lw, axis=1, keepdims=True)
    return np.exp(lw - lZ)

def compute_R(primes_up_to_N, m, N):
    """Compute residual matrix R = T_obs - T_boltz."""
    res = coprime_res(m)
    phi = len(res)
    D = fwd_dist(m, res)
    T = N / len(primes_up_to_N)
    C = count_trans(primes_up_to_N, m, res)
    Tobs = norm_rows(C.astype(np.float64))
    Tb = boltz(D, T)
    R = Tobs - Tb
    return R, res, D, T

print("Sieving to 10^9...")
all_primes = sieve(1_000_000_000)

# ===========================================================
# PART 1: CRT PROJECTION
# Extract the m=30 trace from inside the m=210 matrix
# ===========================================================
print("\n" + "="*70)
print("PART 1: CRT PROJECTION - EXTRACTING m=30 FROM m=210")
print("="*70)
print()
print("Z/210Z = Z/30Z x Z/7Z")
print("Each r in coprime(210) maps to (r mod 30, r mod 7)")
print("The 48x48 matrix decomposes into blocks indexed by (a30,a7)x(b30,b7)")
print()

res30 = coprime_res(30)
res7 = coprime_res(7)   # [1,2,3,4,5,6]
res210 = coprime_res(210)

# Build CRT mapping: r mod 210 -> (r mod 30, r mod 7)
idx30 = {r: i for i, r in enumerate(res30)}
idx7 = {r: i for i, r in enumerate(res7)}
idx210 = {r: i for i, r in enumerate(res210)}

print(f"phi(30)={len(res30)}, phi(7)={len(res7)}, phi(210)={len(res210)}")
print(f"phi(30)*phi(7) = {len(res30)*len(res7)} should = {len(res210)}")

# Map each residue of 210 to its (30, 7) components
crt_map = {}  # idx_210 -> (idx_30, idx_7)
for r in res210:
    r30 = r % 30
    r7 = r % 7
    if r30 in idx30 and r7 in idx7:
        crt_map[idx210[r]] = (idx30[r30], idx7[r7])

print(f"CRT mapped {len(crt_map)}/{len(res210)} residues")
unmapped = len(res210) - len(crt_map)
if unmapped > 0:
    print(f"WARNING: {unmapped} residues unmapped (not coprime to both 30 and 7)")
    # These are residues coprime to 210 but where r%30 is not coprime to 30
    # Actually this can't happen: if gcd(r,210)=1 then gcd(r,30)=1 and gcd(r,7)=1
    # So all should map. Let's check.
    for r in res210:
        r30 = r % 30
        r7 = r % 7
        if r30 not in idx30:
            print(f"  r={r}: r%30={r30} not coprime to 30!")
        if r7 not in idx7:
            print(f"  r={r}: r%7={r7} not coprime to 7!")

print()
LN_PI = math.log(math.pi)

for exp in range(5, 10):
    N = 10**exp
    p = [x for x in all_primes if x <= N]
    
    # Compute the full 210 residual
    R210, _, _, T = compute_R(p, 210, N)
    
    # Compute the direct 30 residual for comparison
    R30, _, _, _ = compute_R(p, 30, N)
    tr30_direct = float(np.trace(R30))
    
    # CRT PROJECTION METHOD 1: Partial trace over the Z/7Z factor
    # This is the "folded trace" - sum over j7 of R_{(i30,j7),(i30,j7)}
    # i.e., for each 30-residue class, sum the diag entries over all 7-projections
    phi30 = len(res30)
    phi7 = len(res7)
    projected_diag = np.zeros(phi30)
    
    for i210, (i30, i7) in crt_map.items():
        # Diagonal entry of the 210-matrix
        projected_diag[i30] += R210[i210, i210]
    
    tr30_projected = float(np.sum(projected_diag))
    
    # CRT PROJECTION METHOD 2: Full partial trace
    # For the partial trace over Z/7Z: (Tr_7 R)_{a30, b30} = sum_{k7} R_{(a30,k7),(b30,k7)}
    partial_trace_matrix = np.zeros((phi30, phi30))
    for (i210, (i30, i7)) in crt_map.items():
        for (j210, (j30, j7)) in crt_map.items():
            if i7 == j7:  # Same Z/7Z index
                partial_trace_matrix[i30, j30] += R210[i210, j210]
    
    tr30_partial = float(np.trace(partial_trace_matrix))
    
    # CRT PROJECTION METHOD 3: Average over 7-fiber
    # Normalize by phi(7) = 6 to get the "per-fiber" trace
    tr30_per_fiber = tr30_projected / phi7
    
    lnN = math.log(N)
    print(f"N=10^{exp} (T={T:.2f}, m/T_210={210/T:.2f}, m/T_30={30/T:.2f}):")
    print(f"  Direct Tr(R_30)*ln(N)      = {tr30_direct*lnN:+.8f}")
    print(f"  CRT diag proj *ln(N)       = {tr30_projected*lnN:+.8f}  (sum of 210 diag grouped by r%30)")
    print(f"  CRT partial trace *ln(N)   = {tr30_partial*lnN:+.8f}  (Tr of partial-trace matrix)")
    print(f"  CRT per-fiber *ln(N)       = {tr30_per_fiber*lnN:+.8f}  (diag proj / phi(7))")
    print(f"  Full Tr(R_210)*ln(N)       = {float(np.trace(R210))*lnN:+.8f}")
    print(f"  -ln(pi) target             = {-LN_PI:+.8f}")
    print()

# ===========================================================
# PART 2: UNIVERSAL CURVE COLLAPSE
# Plot Tr(R)*ln(N) vs m/T for all moduli
# ===========================================================
print("="*70)
print("PART 2: UNIVERSAL CURVE - Tr(R)*ln(N) vs m/T")
print("="*70)
print()
print("If there's a dimensional scaling law, all moduli should")
print("trace a single curve F(m/T) where F -> -ln(pi) at threshold.")
print()

# Collect (m/T, Tr(R)*ln(N)) for ALL moduli at ALL N
curve_data = {}  # m -> list of (m/T, trace*lnN)

for m in [6, 30, 210]:
    curve_data[m] = []
    res = coprime_res(m)
    phi = len(res)
    D = fwd_dist(m, res)
    
    for exp in range(4, 10):
        N = 10**exp
        p = [x for x in all_primes if x <= N]
        T = N / len(p)
        C = count_trans(p, m, res)
        Tobs = norm_rows(C.astype(np.float64))
        Tb = boltz(D, T)
        R = Tobs - Tb
        tr = float(np.trace(R))
        lnN = math.log(N)
        mT_ratio = m / T
        curve_data[m].append((mT_ratio, tr * lnN, N))

print(f"{'m':>5} {'m/T':>8} {'Tr*ln(N)':>12} {'N':>12}")
print(f"{'-'*5:>5} {'-'*8:>8} {'-'*12:>12} {'-'*12:>12}")
for m in [6, 30, 210]:
    for mT, trln, N in curve_data[m]:
        print(f"{m:>5} {mT:>8.4f} {trln:>+12.6f} {N:>12,}")
    print()

# ===========================================================
# PART 3: FOLD-NORMALIZED TRACE
# Hypothesis: Tr(R) * ln(N) * g(m) = -ln(pi) for some g(m)
# ===========================================================
print("="*70)
print("PART 3: SEARCHING FOR FOLD NORMALIZATION g(m)")
print("="*70)
print()

# At N=10^9, what are the converged/best values?
traces = {}
for m in [6, 30, 210]:
    N = 10**9
    p = [x for x in all_primes if x <= N]
    R_mat, res, D, T = compute_R(p, m, N)
    tr = float(np.trace(R_mat))
    traces[m] = tr * math.log(N)

print(f"At N=10^9:")
for m, val in traces.items():
    print(f"  m={m:>4}: Tr(R)*ln(N) = {val:+.10f}")
print()

# m=6 is near -ln(2) = -0.693, NOT -ln(pi)
# m=30 is -ln(pi) = -1.145
# m=210 is near 0 (frozen)

# What if the true law involves the NUMBER OF PRIME FACTORS (omega)?
# m=6=2*3 has omega=2, m=30=2*3*5 has omega=3, m=210=2*3*5*7 has omega=4
# Or involves the PRODUCT of (1 - 1/p)?

# Let me try: does Tr(R)*ln(N) = -ln(pi) * product_{p|m} something ?
# At m=30: -1.145 / -1.145 = 1.0 (trivially)
# At m=6:  -0.698 / -1.145 = 0.610
# What is 0.610?
ratio_6_30 = traces[6] / traces[30]
print(f"  Tr*ln(N) ratio m=6/m=30 = {ratio_6_30:.8f}")
print(f"  Candidate matches:")
candidates = {
    "ln(2)/ln(pi)": math.log(2)/math.log(math.pi),
    "1 - 1/5": 1 - 1/5,
    "(1-1/5)*(1-1/p5)": (1-1/5),
    "phi(6)/phi(30)*6/30": (2/8)*(6/30),
    "phi(6)/phi(30)": 2/8,
    "ln(6)/ln(30)": math.log(6)/math.log(30),
    "2/pi": 2/math.pi,
    "1/phi_ratio(5)": 1/(1-1/5),
    "3/(2*pi)": 3/(2*math.pi),
    "e^(-1)": math.exp(-1),
}
for name, val in sorted(candidates.items(), key=lambda kv: abs(kv[1] - ratio_6_30)):
    print(f"    {name:>25} = {val:.8f}  err = {ratio_6_30 - val:+.8f}")

# But wait - m=6 HASN'T CONVERGED. It's still moving (rate 0.033).
# Let me check: what happens at the SAME m/T?
# At m=30: converged around m/T ~ 1.5. Final value -1.145
# When does m=6 reach m/T = 1.5?  m/T = 6/T = 1.5 => T = 4 => N ~ e^4 ~ 55
# That's TINY. At m/T=0.305 (N=10^9), m=6 is WAY past the thawing point.
# m=6 is in the OVERHEATED regime, not the thawed regime.

print()
print("="*70)
print("PART 4: SAME m/T COMPARISON (THE KEY TEST)")
print("="*70)
print()
print("If F(m/T) is universal, then at the SAME m/T ratio,")
print("all moduli should give the same Tr(R)*ln(N).")
print()

# For m=30, m/T at each decade:
targets_30 = []
for exp in range(4, 10):
    N = 10**exp
    p = [x for x in all_primes if x <= N]
    T = N / len(p)
    targets_30.append((exp, 30/T))

print("m=30 ratios: ", [(f"10^{e}", f"{r:.3f}") for e, r in targets_30])
print()

# m/T for m=30 ranges from 3.68 (N=10^4) to 1.53 (N=10^9)
# m/T for m=6 at same N ranges from 0.74 to 0.31
# m/T for m=210 at same N ranges from 25.8 to 10.7

# To get m=6 at m/T=1.5, need T=4, N~54 (too small for statistics)
# To get m=210 at m/T=1.5, need T=140, N~e^140 ~ 10^61 (too large)

# So we can't directly compare at the same m/T. But we CAN:
# 1. Check if the m=30 curve and m=6 curve have the same SHAPE
# 2. See if Tr(R)*ln(N) as a function of m/T follows same functional form

# Let me fit a model: Tr(R)*ln(N) = a + b * exp(-c * m/T) for each modulus
# Or better: Tr(R)*ln(N) = -ln(pi) * h(m/T)

print("DETAILED m/T SWEEP - m=30 with finer resolution:")
m = 30
res = coprime_res(m)
phi = len(res)
D = fwd_dist(m, res)

# Use many N values between 10^4 and 10^9
sweep30 = []
for log10N in np.arange(4.0, 9.5, 0.5):
    N = int(10**log10N)
    p = [x for x in all_primes if x <= N]
    if len(p) < 100:
        continue
    T = N / len(p)
    C = count_trans(p, m, res)
    row_sums = C.sum(axis=1)
    if np.any(row_sums == 0):
        continue
    Tobs = norm_rows(C.astype(np.float64))
    Tb = boltz(D, T)
    R = Tobs - Tb
    tr = float(np.trace(R))
    lnN = math.log(N)
    mT = m / T
    sweep30.append((mT, tr*lnN, N))

print(f"{'m/T':>8} {'Tr*ln(N)':>12} {'Tr*ln(N)/-ln(pi)':>18} {'N':>14}")
print(f"{'-'*8:>8} {'-'*12:>12} {'-'*18:>18} {'-'*14:>14}")
for mT, trln, N in sweep30:
    norm = trln / (-LN_PI)
    print(f"{mT:>8.4f} {trln:>+12.8f} {norm:>18.8f} {N:>14,}")

# ===========================================================
# PART 5: THE CRT FOLD TEST
# If the lattice folds, then the partial trace of R_210
# over the Z/7Z fiber should EQUAL R_30
# ===========================================================
print()
print("="*70)
print("PART 5: CRT FOLD - PARTIAL TRACE STRUCTURE")
print("="*70)
print()

N = 10**9
p = [x for x in all_primes if x <= N]

R210_full, _, _, T = compute_R(p, 210, N)
R30_full, _, _, _ = compute_R(p, 30, N)

# Build the partial trace over Z/7Z
phi30 = len(res30)
phi7 = len(res7)
partial_R = np.zeros((phi30, phi30))
for (i210, (i30, i7)) in crt_map.items():
    for (j210, (j30, j7)) in crt_map.items():
        if i7 == j7:
            partial_R[i30, j30] += R210[i210, j210]

# Normalize: divide by phi(7) to get "average per fiber"
partial_R_norm = partial_R / phi7

print("Comparison: R_30 (direct) vs partial_trace(R_210)/phi(7)")
print()
print(f"  Tr(R_30)                      = {np.trace(R30_full):+.10f}")
print(f"  Tr(partial_R)/phi(7)          = {np.trace(partial_R_norm):+.10f}")
print(f"  Tr(partial_R) [unnormalized]  = {np.trace(partial_R):+.10f}")
print()

# Compare matrix elements
diff = R30_full - partial_R_norm
print(f"  ||R_30 - Tr_7(R_210)/phi(7)||_F = {np.sqrt(np.sum(diff**2)):.8f}")
print(f"  ||R_30||_F                       = {np.sqrt(np.sum(R30_full**2)):.8f}")
print(f"  Relative difference              = {np.sqrt(np.sum(diff**2))/np.sqrt(np.sum(R30_full**2)):.6f}")
print()

# Also check: do the diagonal elements match?
print("  Diagonal comparison (obs-boltz at each residue class):")
print(f"  {'res':>5} {'R_30_diag':>12} {'Tr7(R210)/6':>12} {'ratio':>10}")
for i, r in enumerate(res30):
    v30 = R30_full[i,i]
    v210 = partial_R_norm[i,i]
    ratio = v30/v210 if abs(v210) > 1e-15 else float('inf')
    print(f"  {r:>5} {v30:>+12.8f} {v210:>+12.8f} {ratio:>10.4f}")

# ===========================================================
# PART 6: FOLD NUMBER AND TRACE DECOMPOSITION
# ===========================================================
print()
print("="*70)
print("PART 6: PRIMORIAL TOWER TRACE DECOMPOSITION")
print("="*70)
print()
print("Primorial tower: 6 -> 30 -> 210 -> 2310")
print("Each step folds through a new prime.")
print()

# The key insight: Tr(R_m) decomposes across CRT fibers.
# For m = m' * p with gcd(m', p) = 1:
#   Tr(R_m) = sum over all (i_m', i_p) of R_m[(i_m',i_p), (i_m',i_p)]
#
# But CRT also means that the TRANSITION PROBABILITIES decompose:
# The distance d_{210}(a, b) depends on BOTH d_{30}(a%30, b%30) AND d_7(a%7, b%7)
# In fact d_{210} = CRT(d_30, d_7) is NOT simply d_30 + d_7 or d_30 * d_7.
# The distance function does NOT decompose as a tensor product!

# This means the Boltzmann model at m=210 is NOT a tensor product of m=30 and m=7 models.
# The folding introduces cross-coupling.

# But the OBSERVED counts should decompose differently...
# Actually, primes p with p%210=r simultaneously have p%30=r%30 and p%7=r%7.
# The Chebyshev bias at mod 210 IS the joint Chebyshev bias at (mod 30, mod 7).

# Let me check: is the 210-diagonal entry R_{(a,b),(a,b)} related to
# the 30-diagonal entry R_{a,a} and the 7-diagonal entry R_{b,b}?

R7, _, _, _ = compute_R(p, 7, N)
R30_diag = np.diag(R30_full)
R7_diag = np.diag(R7)

print("Test: does R_210[i,i] ~ R_30[i30,i30] * R_7[i7,i7] ?")
print("Or:   R_210[i,i] ~ R_30[i30,i30] + R_7[i7,i7] ?")
print()
print(f"  {'r210':>5} {'r30':>5} {'r7':>4} {'R210_ii':>12} {'R30*R7':>12} {'R30+R7':>12} {'ratio(mult)':>12}")
sum_err_mult = 0
sum_err_add = 0
for r in res210[:12]:  # Show first 12
    i210 = idx210[r]
    i30 = idx30[r%30]
    i7 = idx7[r%7]
    v210 = R210_full[i210, i210]
    v_mult = R30_diag[i30] * R7_diag[i7]
    v_add = R30_diag[i30] + R7_diag[i7]
    sum_err_mult += (v210 - v_mult)**2
    sum_err_add += (v210 - v_add)**2
    ratio = v210/v_mult if abs(v_mult) > 1e-15 else float('inf')
    print(f"  {r:>5} {r%30:>5} {r%7:>4} {v210:>+12.8f} {v_mult:>+12.8f} {v_add:>+12.8f} {ratio:>12.4f}")

print(f"\n  RMS error (multiplicative model): {math.sqrt(sum_err_mult/len(res210)):.8e}")
print(f"  RMS error (additive model):       {math.sqrt(sum_err_add/len(res210)):.8e}")

# ===========================================================
# PART 7: THE NORMALIZED FOLD TRACE
# ===========================================================
print()
print("="*70)
print("PART 7: NORMALIZED FOLD TRACE")
print("="*70)
print()

# What if the right quantity isn't Tr(R) but Tr(R) normalized by
# the Boltzmann diagonal weight?
# Tr(R) = sum_i [T_obs(i,i) - T_boltz(i,i)]
# T_boltz(i,i) = exp(-m/T) / Z_i
# So Tr(R) ~ phi * exp(-m/T) * (something) when m/T >> 1
# The freeze-out makes Tr(R) ~ exp(-m/T) -> 0

# Normalize: Tr(R) / mean(T_boltz diagonal)?
for m in [6, 30, 210]:
    res = coprime_res(m)
    D = fwd_dist(m, res)
    Tb = boltz(D, N/len(p))
    R_mat, _, _, T = compute_R(p, m, N)
    boltz_diag_mean = float(np.mean(np.diag(Tb)))
    tr = float(np.trace(R_mat))
    normalized = tr / boltz_diag_mean if boltz_diag_mean > 1e-15 else float('inf')
    print(f"m={m:>4}: Tr(R)={tr:+.10f}, mean(T_boltz_diag)={boltz_diag_mean:.8e}, "
          f"Tr(R)/mean_diag={normalized:+.8f}, *ln(N)={normalized*math.log(N):+.8f}")

print()
print("="*70)
print("FINAL SYNTHESIS")
print("="*70)
