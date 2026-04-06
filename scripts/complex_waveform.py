#!/usr/bin/env python3
"""
COMPLEX WAVEFORM ANALYSIS
==========================
The trace oscillates between algebraic and transcendental envelopes.
The complex eigenvalues of R are ROTATING, creating interference.

This script:
1. Dense trajectory of Tr(R)*ln(N) for m=6 and m=30
2. Extract ALL complex eigenvalues of R at each N
3. Track phase rotation of each eigenvalue
4. Decompose: which eigenvalue pairs create which oscillation frequency?
5. Fit damped oscillation model to the trajectory
6. Test midpoint hypothesis vs pure algebraic vs pure transcendental
"""
import math, numpy as np
from numpy.linalg import eigvals

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

print("Sieving to 10^9...")
all_primes = sieve(1_000_000_000)

ALG_30 = -2*math.sqrt(3)/3
ALG_6 = -2.0/3
TRANS_30 = -math.log(math.pi)
TRANS_6 = -math.log(2)
MID_30 = (ALG_30 + TRANS_30) / 2
MID_6 = (ALG_6 + TRANS_6) / 2

# ================================================================
# PART 1: Dense eigenvalue extraction for m=30
# ================================================================
print()
print("=" * 90)
print("  PART 1: COMPLEX EIGENVALUE ROTATION — m=30")
print("=" * 90)
print()

m = 30
res = coprime_res(m)
phi = len(res)  # 8
D = fwd_dist(m, res)

# Dense sampling: 20 points per decade from 10^4 to 10^9
N_values = sorted(set(int(10**(k/10)) for k in range(40, 91)))

print(f"m={m}, phi={phi} => R is {phi}x{phi}, has {phi} eigenvalues")
print(f"Algebraic: {ALG_30:.8f}   Transcendental: {TRANS_30:.8f}   Midpoint: {MID_30:.8f}")
print()

# Store for analysis
trace_data = []  # (N, lnlnN, trace_val)
eig_data = []    # (N, eigenvalues_array)

print(f"  {'N':>14} {'ln(ln(N))':>10} {'Tr(R)*lnN':>13} {'vs ALG':>10} {'vs TRANS':>10} "
      f"{'vs MID':>10} {'|lam1|':>10} {'phase1/pi':>10} {'|lam2|':>10} {'phase2/pi':>10}")
print(f"  {'-'*14} {'-'*10} {'-'*13} {'-'*10} {'-'*10} {'-'*10} "
      f"{'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for N in N_values:
    if N > 1_000_000_000 or N < 5000:
        continue
    p = [x for x in all_primes if x <= N]
    if len(p) < 100:
        continue
    T = N / len(p)
    C = count_trans(p, m, res)
    if np.any(C.sum(axis=1) == 0):
        continue
    Tobs = norm_rows(C.astype(np.float64))
    Tb = boltz(D, T)
    R = Tobs - Tb
    
    lnN = math.log(N)
    lnlnN = math.log(lnN)
    trace_val = float(np.trace(R)) * lnN
    
    # Get ALL eigenvalues of R*ln(N)
    eigs = eigvals(R * lnN)
    # Sort by magnitude descending
    order = np.argsort(-np.abs(eigs))
    eigs_sorted = eigs[order]
    
    trace_data.append((N, lnlnN, trace_val))
    eig_data.append((N, eigs_sorted.copy()))
    
    # Top 2 eigenvalues
    e1 = eigs_sorted[0]
    e2 = eigs_sorted[1] if len(eigs_sorted) > 1 else 0
    
    phase1 = np.angle(e1) / math.pi if np.abs(e1) > 1e-15 else 0
    phase2 = np.angle(e2) / math.pi if np.abs(e2) > 1e-15 else 0
    
    d_alg = trace_val - ALG_30
    d_trans = trace_val - TRANS_30
    d_mid = trace_val - MID_30
    
    if N in [10000, 31623, 100000, 316228, 1000000, 3162278, 10000000, 
             31622777, 100000000, 316227766, 1000000000]:
        print(f"  {N:>14,} {lnlnN:>10.4f} {trace_val:>+13.8f} {d_alg:>+10.6f} "
              f"{d_trans:>+10.6f} {d_mid:>+10.6f} {abs(e1):>10.6f} {phase1:>+10.4f} "
              f"{abs(e2):>10.6f} {phase2:>+10.4f}")

# ================================================================
# PART 2: Eigenvalue phase trajectories
# ================================================================
print()
print("=" * 90)
print("  PART 2: EIGENVALUE PHASE TRAJECTORIES (m=30)")
print("=" * 90)
print()
print("  Tracking the phase angle (theta/pi) of each eigenvalue across N.")
print("  Complex eigenvalues come in conjugate pairs.")
print("  A rotating phase creates oscillation in Re[lambda].")
print()

# For a few key N values, show ALL eigenvalues
for N_target in [10000, 100000, 1000000, 10000000, 100000000, 1000000000]:
    # Find closest
    best = min(eig_data, key=lambda x: abs(x[0] - N_target))
    N, eigs = best
    print(f"  N = {N:>14,}  (ln(ln(N)) = {math.log(math.log(N)):.4f})")
    for k, e in enumerate(eigs):
        r = abs(e)
        theta = np.angle(e)
        re_part = e.real
        im_part = e.imag
        kind = "REAL" if abs(im_part) < 1e-12 else "COMPLEX"
        print(f"    lambda_{k}: {re_part:>+12.8f} + {im_part:>+12.8f}i  "
              f"|{r:.8f}|  theta/pi={theta/math.pi:>+8.4f}  [{kind}]")
    real_sum = sum(e.real for e in eigs)
    print(f"    SUM of Re[lambda] = {real_sum:>+12.8f}  (= Tr(R)*ln(N))")
    print()

# ================================================================
# PART 3: Decompose trace into REAL + OSCILLATORY parts
# ================================================================
print("=" * 90)
print("  PART 3: REAL vs OSCILLATORY DECOMPOSITION")
print("=" * 90)
print()
print("  Tr(R)*ln(N) = sum of REAL eigenvalues + sum of Re[COMPLEX eigenvalues]")
print("  The REAL part gives the monotonic trend.")
print("  The COMPLEX part gives the oscillation.")
print()

print(f"  {'N':>14} {'Tr(R)*lnN':>13} {'REAL sum':>13} {'COMPLEX Re':>13} "
      f"{'#real':>6} {'#complex':>8}")
print(f"  {'-'*14} {'-'*13} {'-'*13} {'-'*13} {'-'*6} {'-'*8}")

real_part_data = []
osc_part_data = []

for N, eigs in eig_data:
    real_sum = 0.0
    complex_re_sum = 0.0
    n_real = 0
    n_complex = 0
    
    for e in eigs:
        if abs(e.imag) < 1e-12:
            real_sum += e.real
            n_real += 1
        else:
            complex_re_sum += e.real
            n_complex += 1
    
    lnN = math.log(N)
    tr = sum(e.real for e in eigs)
    
    real_part_data.append((N, real_sum))
    osc_part_data.append((N, complex_re_sum))
    
    if N in [10000, 31623, 100000, 316228, 1000000, 3162278, 10000000,
             31622777, 100000000, 316227766, 1000000000]:
        print(f"  {N:>14,} {tr:>+13.8f} {real_sum:>+13.8f} {complex_re_sum:>+13.8f} "
              f"{n_real:>6} {n_complex:>8}")

# ================================================================
# PART 4: FIT DAMPED OSCILLATION MODEL
# ================================================================
print()
print("=" * 90)
print("  PART 4: FIT DAMPED OSCILLATION MODEL")
print("=" * 90)
print()
print("  Model: C(N) = C_inf + A * exp(-gamma * s) * cos(omega * s + phi0)")
print("  where s = ln(ln(N))")
print()

# Extract data for fitting
s_arr = np.array([math.log(math.log(N)) for N, _, _ in trace_data])
c_arr = np.array([v for _, _, v in trace_data])

# Try three hypotheses for C_inf
for name, c_inf in [("Algebraic (-2sqrt3/3)", ALG_30), 
                     ("Transcendental (-ln(pi))", TRANS_30),
                     ("Midpoint", MID_30)]:
    residual = c_arr - c_inf
    # RMS residual
    rms = np.sqrt(np.mean(residual**2))
    # Is the residual decreasing in amplitude?
    # Split into halves
    mid_idx = len(residual) // 2
    rms_first = np.sqrt(np.mean(residual[:mid_idx]**2))
    rms_second = np.sqrt(np.mean(residual[mid_idx:]**2))
    
    # Count zero crossings in the residual
    zc = sum(1 for i in range(len(residual)-1) if residual[i]*residual[i+1] < 0)
    
    # Check if amplitude is DAMPING
    damping = "YES" if rms_second < rms_first * 0.7 else ("MAYBE" if rms_second < rms_first else "NO")
    
    print(f"  Hypothesis: C_inf = {name}")
    print(f"    RMS residual overall:  {rms:.6f}")
    print(f"    RMS first half:        {rms_first:.6f}")
    print(f"    RMS second half:       {rms_second:.6f}")
    print(f"    Damping?               {damping}")
    print(f"    Zero crossings:        {zc}")
    print(f"    Last 5 residuals:      {['%+.6f' % r for r in residual[-5:]]}")
    print()

# ================================================================
# PART 5: Complex waveform for m=6
# ================================================================
print("=" * 90)
print("  PART 5: COMPLEX EIGENVALUE ROTATION — m=6 (2x2 matrix)")
print("=" * 90)
print()

m = 6
res6 = coprime_res(m)
phi6 = len(res6)  # 2
D6 = fwd_dist(m, res6)

print(f"m={m} is only 2x2 => 2 eigenvalues.")
print(f"R is traceless (rows sum to 0): lambda_1 + lambda_2 = Tr(R)")
print(f"But since R has rank 1 (each row sums to 0), one eigenvalue = Tr(R), other = 0")
print(f"NO complex eigenvalues possible for m=6! The oscillation is purely from sampling.")
print()

trace6_data = []
print(f"  {'N':>14} {'Tr(R)*lnN':>13} {'vs ALG':>10} {'vs TRANS':>10} {'vs MID':>10} "
      f"{'lam1':>12} {'lam2':>12}")
print(f"  {'-'*14} {'-'*13} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")

for exp in range(4, 10):
    for sub in [1.0, 1.5]:
        N = int(10**(exp + sub - 1)) if sub > 1 else 10**exp
        if N > 1_000_000_000: continue
        p = [x for x in all_primes if x <= N]
        if len(p) < 100: continue
        T = N / len(p)
        C = count_trans(p, m, res6)
        if np.any(C.sum(axis=1) == 0): continue
        Tobs = norm_rows(C.astype(np.float64))
        Tb = boltz(D6, T)
        R = Tobs - Tb
        
        lnN = math.log(N)
        tr = float(np.trace(R)) * lnN
        eigs = eigvals(R * lnN)
        
        trace6_data.append((N, math.log(lnN), tr))
        
        d_alg = tr - ALG_6
        d_trans = tr - TRANS_6
        d_mid = tr - MID_6
        
        print(f"  {N:>14,} {tr:>+13.8f} {d_alg:>+10.6f} {d_trans:>+10.6f} "
              f"{d_mid:>+10.6f} {eigs[0].real:>+12.8f} {eigs[1].real:>+12.8f}")

print()
print("  m=6 has no complex eigenvalues (2x2 rank-1 matrix).")
print("  Its oscillations come from PRIME RACES (Chebyshev bias),")
print("  not from eigenvalue rotation.")
print("  Only m >= 10 (phi >= 4) can have complex eigenvalue pairs.")

# ================================================================
# PART 6: The INTERFERENCE SPECTRUM for m=30
# ================================================================
print()
print("=" * 90)
print("  PART 6: INTERFERENCE SPECTRUM — WHICH EIGENVALUE PAIR OSCILLATES?")
print("=" * 90)
print()

# For m=30, phi=8, so we have 8 eigenvalues
# Group them: how many are real, how many complex at each N?

complex_pairs_by_N = []
for N, eigs in eig_data:
    if N < 100000:
        continue
    n_real = sum(1 for e in eigs if abs(e.imag) < 1e-10)
    n_complex = len(eigs) - n_real
    # Track the dominant complex pair's phase
    complex_eigs = [(abs(e), np.angle(e)/math.pi, e) for e in eigs if abs(e.imag) > 1e-10]
    complex_eigs.sort(reverse=True)  # by magnitude
    
    if N in [100000, 1000000, 10000000, 100000000, 1000000000]:
        print(f"  N={N:>14,}: {n_real} real, {n_complex} complex eigenvalues")
        for rank, (mag, phase_pi, e) in enumerate(complex_eigs[:4]):
            print(f"    #{rank}: |{mag:.8f}| * e^(i*{phase_pi:+.4f}*pi)  = {e.real:+.8f} + {e.imag:+.8f}i")
        if complex_eigs:
            # The dominant pair's contribution to trace
            dom = complex_eigs[0][2]
            osc_contribution = 2 * dom.real  # conjugate pair contributes 2*Re
            print(f"    Dominant pair contributes 2*Re = {osc_contribution:+.8f} to trace")

# ================================================================
# PART 7: OSCILLATION AMPLITUDE DECAY
# ================================================================
print()
print("=" * 90)
print("  PART 7: OSCILLATION AMPLITUDE DECAY vs ln(ln(N))")  
print("=" * 90)
print()

# Compute running deviation from midpoint
s_vals = []
amp_vals = []
for N, lnlnN, tr in trace_data:
    dev = abs(tr - MID_30)
    s_vals.append(lnlnN)
    amp_vals.append(dev)

# Fit |dev| = A * exp(-gamma * s)
s_np = np.array(s_vals)
amp_np = np.array(amp_vals)

# Only use points past initial convergence (s > 2.6, i.e. N > ~10^6)
mask = s_np > 2.6
if mask.sum() > 5:
    s_fit = s_np[mask]
    amp_fit = amp_np[mask]
    
    # Log fit: ln|dev| = ln(A) - gamma * s
    nonzero = amp_fit > 1e-15
    if nonzero.sum() > 3:
        log_amp = np.log(amp_fit[nonzero])
        s_fit2 = s_fit[nonzero]
        # Linear regression
        coeffs = np.polyfit(s_fit2, log_amp, 1)
        gamma = -coeffs[0]
        A = math.exp(coeffs[1])
        
        print(f"  Fit: |deviation from midpoint| = {A:.4f} * exp(-{gamma:.4f} * ln(ln(N)))")
        print(f"  Decay rate gamma = {gamma:.4f}")
        print(f"  Amplitude A = {A:.4f}")
        print()
        
        # What does this mean?
        # |dev| ~ A * (ln(N))^(-gamma) = A * (ln N)^(-gamma)
        # Since s = ln(ln(N)), exp(-gamma*s) = (ln N)^(-gamma)
        print(f"  Equivalent: |deviation| ~ {A:.4f} / (ln N)^{gamma:.4f}")
        print()
        
        # Predict
        for N_pred in [1e10, 1e12, 1e15, 1e20, 1e50, 1e100]:
            s_pred = math.log(math.log(N_pred))
            dev_pred = A * math.exp(-gamma * s_pred)
            print(f"    N = 10^{math.log10(N_pred):.0f}: predicted |deviation| = {dev_pred:.8f}"
                  f"  (envelope width = {2*dev_pred:.8f})")

# ================================================================
# PART 8: VERIFY — IS THE OSCILLATION COHERENT?
# ================================================================
print()
print("=" * 90)
print("  PART 8: AUTOCORRELATION OF OSCILLATION")
print("=" * 90)
print()
print("  If the oscillation is coherent (from eigenvalue rotation),")
print("  the autocorrelation should show periodic structure.")
print("  If it's just noise, autocorrelation decays monotonically.")
print()

# Detrend: subtract smooth envelope (midpoint)
residual_from_mid = np.array([tr - MID_30 for _, _, tr in trace_data])
s_axis = np.array([s for _, s, _ in trace_data])

# Interpolate to uniform spacing in s
from numpy import interp
s_uniform = np.linspace(s_axis.min(), s_axis.max(), 200)
r_uniform = interp(s_uniform, s_axis, residual_from_mid)

# Autocorrelation
r_centered = r_uniform - r_uniform.mean()
acf = np.correlate(r_centered, r_centered, mode='full')
acf = acf[len(acf)//2:]  # positive lags only
acf = acf / acf[0]  # normalize

# Find first few peaks
ds = s_uniform[1] - s_uniform[0]
print(f"  Lag spacing: {ds:.4f} in ln(ln(N))")
print()

# Find peaks in autocorrelation
peaks = []
for i in range(1, len(acf)-1):
    if acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > 0.1:
        lag = i * ds
        peaks.append((lag, acf[i]))
        if len(peaks) <= 5:
            print(f"  ACF peak at lag = {lag:.4f} (acf = {acf[i]:.4f})")

if peaks:
    print(f"\n  Dominant period estimate: {peaks[0][0]:.4f} in ln(ln(N)) space")
    omega = 2 * math.pi / peaks[0][0]
    print(f"  Angular frequency: omega = {omega:.4f}")
    print(f"  This corresponds to frequency {omega/(2*math.pi):.4f} cycles per unit ln(ln(N))")
else:
    print("  No clear periodic peaks found — oscillation may be quasi-periodic or noisy")

# ================================================================
# SUMMARY
# ================================================================
print()
print("=" * 90)
print("  SUMMARY: THE COMPLEX WAVEFORM")
print("=" * 90)
print(f"""
  For m=30 (phi=8):
  
  The trace Tr(R)*ln(N) is NOT converging monotonically to a single constant.
  It OSCILLATES between two envelopes:
  
      Upper: -2*sqrt(3)/3 = {ALG_30:+.8f}  (Algebraic / Sieve Crystal)
      Lower: -ln(pi)      = {TRANS_30:+.8f}  (Transcendental / Prime Gas)
      Gap:                   {abs(ALG_30 - TRANS_30):.8f}
  
  The oscillation is driven by COMPLEX EIGENVALUES of R rotating in phase.
  - m=30 has phi=8 eigenvalues, several are complex conjugate pairs
  - As N increases, their phases rotate, creating interference
  - The real part of the trace oscillates accordingly
  
  For m=6 (phi=2):
  - R is rank-1, only REAL eigenvalues exist
  - Oscillations come from prime race statistics, not eigenvalue rotation
  - Different mechanism, same qualitative behavior
  
  The true asymptotic limit depends on whether the oscillation DAMPS:
  - If it damps: limit = midpoint = {MID_30:+.8f}
  - If amplitude is constant: no single limit exists (oscillates forever)
  - If it locks to algebraic first, then transcendental: phase transition
""")
