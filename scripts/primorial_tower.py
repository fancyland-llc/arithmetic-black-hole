#!/usr/bin/env python3
"""
PRIMORIAL TOWER: FULL CRT PROJECTION MAP
==========================================
The primorial tower:  2 -> 6 -> 30 -> 210 -> 2310

Each level folds through a new prime:
  6   = 2 * 3     phi=2
  30  = 6 * 5     phi=8
  210 = 30 * 7    phi=48
  2310= 210 * 11  phi=480

CRT says: Z/mZ = Z/m'Z x Z/pZ for m = m' * p

Question: Does projecting ANY higher modulus down to ANY lower
base via CRT aggregation preserve the trace?

FULL PROJECTION MAP:
  210 -> 30 : aggregate over Z/7Z fiber
  210 -> 6  : aggregate over Z/35Z fiber  (or 210->30->6)
  2310 -> 30: aggregate over Z/77Z fiber
  2310 -> 6 : aggregate over Z/385Z fiber
  30 -> 6   : aggregate over Z/5Z fiber
"""
import math, numpy as np

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

def compute_residual(p_list, m, N):
    """Compute R = T_obs - T_boltz and return R, res, T."""
    res = coprime_res(m)
    D = fwd_dist(m, res)
    T = N / len(p_list)
    C = count_trans(p_list, m, res)
    Tobs = norm_rows(C.astype(np.float64))
    Tb = boltz(D, T)
    return Tobs - Tb, res, T

def crt_project_trace(R_high, res_high, m_high, m_base):
    """
    Project R at modulus m_high down to m_base via CRT aggregation.
    
    For each (i,j) pair in the base, sum over all fiber indices:
      (Tr_fiber R)_{a_base, b_base} = sum_{k_fiber: a_fiber=k, b_fiber=k} R_{(a,k),(b,k)}
    
    But for the TRACE, we only need the diagonal:
      Tr(Tr_fiber R) = sum_{a_base} sum_{k_fiber} R_{(a_base,k),(a_base,k)}
    
    Which is just: group the diagonal entries of R_high by their m_base projection.
    """
    idx_high = {r: i for i, r in enumerate(res_high)}
    res_base = coprime_res(m_base)
    idx_base = {r: i for i, r in enumerate(res_base)}
    
    # For the trace: just sum diagonal entries grouped by base residue
    projected_diag = np.zeros(len(res_base))
    count_per_base = np.zeros(len(res_base))
    
    for r in res_high:
        r_base = r % m_base
        if r_base in idx_base:
            i_high = idx_high[r]
            i_base = idx_base[r_base]
            projected_diag[i_base] += R_high[i_high, i_high]
            count_per_base[i_base] += 1
    
    return float(np.sum(projected_diag)), projected_diag, count_per_base

def crt_project_full_matrix(R_high, res_high, m_high, m_base):
    """
    Full partial trace: project the entire matrix, not just the diagonal.
    (Tr_fiber R)_{a_base, b_base} = sum_{k} R_{CRT(a,k), CRT(b,k)}
    where the sum is over fiber indices k where both CRT(a,k) and CRT(b,k)
    are coprime to m_high.
    """
    idx_high = {r: i for i, r in enumerate(res_high)}
    res_base = coprime_res(m_base)
    phi_base = len(res_base)
    idx_base = {r: i for i, r in enumerate(res_base)}
    
    # Build the mapping: for each r_high, find (r_base, r_fiber)
    m_fiber = m_high // m_base
    fiber_map = {}  # (i_base, i_fiber_class) -> i_high
    
    projected = np.zeros((phi_base, phi_base))
    
    # Group high residues by their base projection
    base_groups = {}  # r_base -> list of (r_high, i_high)
    for r in res_high:
        r_base = r % m_base
        if r_base in idx_base:
            key = idx_base[r_base]
            if key not in base_groups:
                base_groups[key] = []
            base_groups[key].append(idx_high[r])
    
    # For partial trace: sum R[i,j] where i and j map to same fiber index
    # Since we're tracing over the fiber, we need to match fiber indices
    for r_i in res_high:
        for r_j in res_high:
            r_i_base = r_i % m_base
            r_j_base = r_j % m_base
            r_i_fiber = r_i % m_fiber
            r_j_fiber = r_j % m_fiber
            if r_i_base in idx_base and r_j_base in idx_base:
                if r_i_fiber == r_j_fiber:  # Same fiber index -> trace over fiber
                    projected[idx_base[r_i_base], idx_base[r_j_base]] += R_high[idx_high[r_i], idx_high[r_j]]
    
    return projected

print("Sieving to 10^9...")
all_primes = sieve(1_000_000_000)
LN_PI = math.log(math.pi)

# The primorial tower
tower = [
    (2,   [2]),           # phi=1 (trivial)
    (6,   [2, 3]),        # phi=2
    (30,  [2, 3, 5]),     # phi=8
    (210, [2, 3, 5, 7]),  # phi=48
    (2310,[2, 3, 5, 7, 11]),  # phi=480
]

# ===========================================================
# PART 1: Direct traces at N=10^7 (fast) and N=10^9
# ===========================================================
print()
print("="*72)
print("PART 1: CRT PROJECTION MAP - ALL PAIRS")
print("="*72)

for exp in [7, 9]:
    N = 10**exp
    p = [x for x in all_primes if x <= N]
    T_val = N / len(p)
    print(f"\n--- N = 10^{exp}, T = {T_val:.4f} ---")
    
    # Compute residuals at each level
    residuals = {}
    for m, primes_of in tower:
        if m <= 2310:
            R, res, T = compute_residual(p, m, N)
            residuals[m] = (R, res, T)
    
    # Direct traces
    print(f"\n  Direct traces (Tr(R)*ln(N)):")
    lnN = math.log(N)
    for m, (R, res, T) in sorted(residuals.items()):
        tr = float(np.trace(R)) * lnN
        phi = len(res)
        print(f"    m={m:>4} phi={phi:>3} m/T={m/T_val:>7.3f} "
              f"Tr(R)*ln(N) = {tr:>+14.10f}")
    
    # CRT projections: every higher -> every lower
    print(f"\n  CRT PROJECTIONS (diagonal trace aggregation):")
    print(f"  {'Source':>8} -> {'Base':>6}  {'Projected Tr*ln(N)':>20}  {'Direct Base':>14}  {'Delta':>12}")
    print(f"  {'--------':>8}    {'------':>6}  {'--------------------':>20}  {'--------------':>14}  {'------------':>12}")
    
    for i, (m_high, _) in enumerate(tower):
        if m_high not in residuals or m_high <= 2:
            continue
        R_high, res_high, _ = residuals[m_high]
        
        for j, (m_base, _) in enumerate(tower):
            if m_base >= m_high or m_base not in residuals or m_base <= 2:
                continue
            if m_high % m_base != 0:
                continue
            
            R_base, res_base, _ = residuals[m_base]
            
            proj_tr, _, counts = crt_project_trace(R_high, res_high, m_high, m_base)
            proj_product = proj_tr * lnN
            direct_product = float(np.trace(R_base)) * lnN
            delta = proj_product - direct_product
            
            # How many fiber copies per base residue?
            fiber_size = int(counts[0]) if len(counts) > 0 else 0
            
            print(f"  {m_high:>8} -> {m_base:>6}  {proj_product:>+20.10f}  "
                  f"{direct_product:>+14.10f}  {delta:>+12.10f}  "
                  f"(fiber={fiber_size})")
    
    # THE KEY LINE: all projections to m=30
    print(f"\n  === ALL PROJECTIONS TO m=30 ===")
    for m_high in [30, 210, 2310]:
        if m_high not in residuals:
            continue
        R_high, res_high, _ = residuals[m_high]
        if m_high == 30:
            val = float(np.trace(R_high)) * lnN
            print(f"    m={m_high:>4} (direct):  {val:>+18.10f}")
        else:
            proj_tr, _, _ = crt_project_trace(R_high, res_high, m_high, 30)
            val = proj_tr * lnN
            print(f"    m={m_high:>4} -> m=30:   {val:>+18.10f}")
    print(f"    -ln(pi):            {-LN_PI:>+18.10f}")
    
    # ALL PROJECTIONS TO m=6
    print(f"\n  === ALL PROJECTIONS TO m=6 ===")
    for m_high in [6, 30, 210, 2310]:
        if m_high not in residuals:
            continue
        R_high, res_high, _ = residuals[m_high]
        if m_high == 6:
            val = float(np.trace(R_high)) * lnN
            print(f"    m={m_high:>4} (direct):  {val:>+18.10f}")
        else:
            proj_tr, _, _ = crt_project_trace(R_high, res_high, m_high, 6)
            val = proj_tr * lnN
            print(f"    m={m_high:>4} -> m=6:    {val:>+18.10f}")
    print(f"    -ln(2):             {-math.log(2):>+18.10f}")

# ===========================================================
# PART 2: CONVERGENCE OF CRT-PROJECTED TRACE vs N
# ===========================================================
print()
print("="*72)
print("PART 2: CONVERGENCE - PROJECTED TRACES vs N")
print("="*72)
print()
print("Does the CRT-projected trace converge at the same rate as the direct?")
print()

for m_base in [6, 30]:
    print(f"  --- Base: m={m_base} ---")
    res_base_list = coprime_res(m_base)
    
    header = f"  {'N':>12}"
    sources = [m_base]
    for m_high in [30, 210, 2310]:
        if m_high > m_base and m_high % m_base == 0:
            sources.append(m_high)
    
    for src in sources:
        if src == m_base:
            header += f"  {'m=' + str(src) + ' direct':>16}"
        else:
            header += f"  {'m=' + str(src) + '->' + str(m_base):>16}"
    print(header)
    print("  " + "-"*(12 + 18*len(sources)))
    
    for exp in range(5, 10):
        N = 10**exp
        p = [x for x in all_primes if x <= N]
        lnN = math.log(N)
        
        row = f"  {N:>12,}"
        for src in sources:
            R_src, res_src, T = compute_residual(p, src, N)
            if src == m_base:
                val = float(np.trace(R_src)) * lnN
            else:
                proj_tr, _, _ = crt_project_trace(R_src, res_src, src, m_base)
                val = proj_tr * lnN
            row += f"  {val:>+16.10f}"
        print(row)
    print()

# ===========================================================
# PART 3: FIBER TRACE - what does each fiber contribute?
# ===========================================================
print("="*72)
print("PART 3: FIBER DECOMPOSITION")
print("="*72)
print()
print("The CRT projection aggregates over the fiber. But what's IN the fiber?")
print()

N = 10**9
p = [x for x in all_primes if x <= N]
lnN = math.log(N)

# For 210 = 30 * 7:
# Each base residue r30 has 6 fiber copies (one for each coprime residue mod 7)
# What does each copy contribute to the diagonal?
R210, res210_list, T = compute_residual(p, 210, N)
res30_list = coprime_res(30)
idx30 = {r: i for i, r in enumerate(res30_list)}
idx210 = {r: i for i, r in enumerate(res210_list)}

print("m=210, fiber = Z/7Z (6 elements)")
print(f"  {'r30':>5} {'sum(fiber diag)':>16} {'individual fiber diagonal entries...':>40}")
for r30 in res30_list:
    fiber_entries = []
    for r210 in res210_list:
        if r210 % 30 == r30:
            i = idx210[r210]
            fiber_entries.append((r210, R210[i, i]))
    total = sum(v for _, v in fiber_entries)
    detail = "  ".join([f"r={r}:{v:+.2e}" for r, v in fiber_entries])
    print(f"  {r30:>5} {total:>+16.10f}  {detail}")

print(f"\n  Total (= Tr projected to m=30): {sum(R210[idx210[r], idx210[r]] for r in res210_list if r%30 == r30 for r30 in res30_list)}")

# Simpler: just show the projected diagonal
proj_tr, proj_diag, counts = crt_project_trace(R210, res210_list, 210, 30)
print(f"\n  Projected diagonal (sum over fiber for each base residue):")
for i, r30 in enumerate(res30_list):
    print(f"    r={r30:>3}: proj_R_diag = {proj_diag[i]:+.10f}, "
          f"direct R30_diag = {float(np.trace(np.diag(np.diag(compute_residual(p, 30, N)[0]))))/len(res30_list):+.10f}")

# Actually let me compare projected diag to direct R30 diag properly
R30_direct, _, _ = compute_residual(p, 30, N)
R30_diag_direct = np.diag(R30_direct)
print(f"\n  Projected vs Direct diagonal comparison:")
print(f"  {'r30':>5} {'Projected':>14} {'Direct R30':>14} {'Ratio':>10}")
for i, r30 in enumerate(res30_list):
    ratio = proj_diag[i] / R30_diag_direct[i] if abs(R30_diag_direct[i]) > 1e-15 else 0
    print(f"  {r30:>5} {proj_diag[i]:>+14.10f} {R30_diag_direct[i]:>+14.10f} {ratio:>10.6f}")

# ===========================================================
# PART 4: THE SCALING LAW
# ===========================================================
print()
print("="*72)
print("PART 4: THE DIMENSIONAL SCALING LAW")
print("="*72)
print()

# If projecting to m=30 always gives -ln(pi), and projecting to m=6 always gives -ln(2),
# then the constant depends on the BASE MODULUS, not the source.
# What is the constant for each base?

N = 10**9
p = [x for x in all_primes if x <= N]
lnN = math.log(N)

base_constants = {}
print("Trace constant at each base level of the primorial tower:")
print()
for m_base in [6, 30]:
    # Direct computation
    R_base, res_base, T = compute_residual(p, m_base, N)
    direct_val = float(np.trace(R_base)) * lnN
    
    # Also get all projections to this base
    proj_vals = []
    for m_high in [30, 210, 2310]:
        if m_high > m_base and m_high % m_base == 0:
            R_high, res_high, _ = compute_residual(p, m_high, N)
            proj_tr, _, _ = crt_project_trace(R_high, res_high, m_high, m_base)
            proj_vals.append((m_high, proj_tr * lnN))
    
    mean_val = np.mean([direct_val] + [v for _, v in proj_vals])
    
    print(f"  Base m={m_base} (phi={len(res_base)}):")
    print(f"    Direct:       {direct_val:+.10f}")
    for m_h, v in proj_vals:
        print(f"    From m={m_h:>4}: {v:+.10f}")
    print(f"    Mean:         {mean_val:+.10f}")
    base_constants[m_base] = mean_val
    
    # What known constant is this?
    candidates = {
        "-ln(pi)": -LN_PI,
        "-ln(2)": -math.log(2),
        "-ln(3)": -math.log(3),
        "-ln(5)": -math.log(5),
        "-2*ln(2)": -2*math.log(2),
        "-ln(2*pi)": -math.log(2*math.pi),
        "-ln(pi)/2": -LN_PI/2,
        "-gamma_EM": -0.5772156649,
    }
    best = min(candidates.items(), key=lambda kv: abs(kv[1] - mean_val))
    print(f"    Best match: {best[0]} = {best[1]:+.10f} (err = {mean_val - best[1]:+.10f})")
    print()

# ===========================================================
# PART 5: THE FULL TOWER FORMULA
# ===========================================================
print("="*72)
print("PART 5: SEEKING THE TOWER FORMULA")
print("="*72)
print()
print("Base constants found:")
for m, c in base_constants.items():
    phi = len(coprime_res(m))
    omega = len([p for p in [2,3,5,7,11,13] if m % p == 0])
    print(f"  m={m:>4} phi={phi:>3} omega={omega} constant={c:+.10f}")

print()
print("Hypothesis: the constant at base m is -ln(p_{omega+1})")
print("where p_{omega+1} is the first prime NOT dividing m")
print()
for m in [6, 30]:
    primes_dividing = [p for p in [2,3,5,7,11,13] if m % p == 0]
    first_not_dividing = [p for p in [2,3,5,7,11,13] if m % p != 0][0]
    print(f"  m={m}: primes dividing = {primes_dividing}, "
          f"first NOT dividing = {first_not_dividing}, "
          f"-ln({first_not_dividing}) = {-math.log(first_not_dividing):+.8f}")
    print(f"      actual constant ~ {base_constants[m]:+.8f}")

print()
print("Hypothesis: constant = -ln(prod of primes dividing m) / something?")
for m in [6, 30]:
    primes_dividing = [p for p in [2,3,5,7,11,13] if m % p == 0]
    prod = 1
    for p in primes_dividing: prod *= p
    print(f"  m={m}: -ln({prod}) = {-math.log(prod):+.8f}, "
          f"-ln({prod})/phi = {-math.log(prod)/len(coprime_res(m)):+.8f}")

print()
# Let's try: -ln(pi) for m=30, -ln(2) for m=6
# pi = 3.14159... ln(pi) = 1.14473
# 2 = ... ln(2) = 0.69315
# ratio: ln(pi)/ln(2) = 1.6515
# phi(30)/phi(6) = 8/2 = 4
# Not that.
# But: at m=6, we said the value is STILL MOVING. Let me check with Richardson.
print("Richardson extrapolation for m=6 convergence:")
vals6 = []
for exp in range(4, 10):
    N_test = 10**exp
    p_test = [x for x in all_primes if x <= N_test]
    R6, _, _ = compute_residual(p_test, 6, N_test)
    product = float(np.trace(R6)) * math.log(N_test)
    vals6.append((math.log(N_test), product))
    
x = np.array([1/v[0] for v in vals6])
y = np.array([v[1] for v in vals6])

# Quadratic Richardson: y = L + a/lnN + b/lnN^2
A = np.column_stack([np.ones_like(x), x, x**2])
coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
print(f"  Quadratic Richardson limit: {coeffs[0]:+.10f}")
print(f"  -ln(2) = {-math.log(2):+.10f}, error = {coeffs[0] - (-math.log(2)):+.10f}")
print(f"  -ln(pi)/2 = {-LN_PI/2:+.10f}, error = {coeffs[0] - (-LN_PI/2):+.10f}")

# Cubic
A3 = np.column_stack([np.ones_like(x), x, x**2, x**3])
coeffs3 = np.linalg.lstsq(A3, y, rcond=None)[0]
print(f"  Cubic Richardson limit: {coeffs3[0]:+.10f}")
print(f"  -ln(2) error = {coeffs3[0] - (-math.log(2)):+.10f}")

# Also: projection from 30, 210, 2310 TO m=6, check convergence
print()
print("CRT projection convergence to m=6:")
for exp in range(5, 10):
    N_test = 10**exp
    p_test = [x for x in all_primes if x <= N_test]
    lnN = math.log(N_test)
    
    row = f"  N=10^{exp}:"
    # Direct m=6
    R6, res6, _ = compute_residual(p_test, 6, N_test)
    row += f"  direct={float(np.trace(R6))*lnN:+.10f}"
    
    # m=30 -> 6
    R30, res30, _ = compute_residual(p_test, 30, N_test)
    proj30, _, _ = crt_project_trace(R30, res30, 30, 6)
    row += f"  30->6={proj30*lnN:+.10f}"
    
    # m=210 -> 6
    R210, res210, _ = compute_residual(p_test, 210, N_test)
    proj210, _, _ = crt_project_trace(R210, res210, 210, 6)
    row += f"  210->6={proj210*lnN:+.10f}"
    
    print(row)

print()
print("="*72)
print("GRAND SUMMARY")
print("="*72)
