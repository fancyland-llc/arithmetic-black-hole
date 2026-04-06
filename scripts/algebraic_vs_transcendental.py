#!/usr/bin/env python3
"""
ALGEBRAIC vs TRANSCENDENTAL: The Final Showdown
=================================================
Gemini claims: C_m = -2/3 * prod(sqrt(p-2)) for p|m, p>3
  C_6  = -2/3         = -0.66667
  C_30 = -2*sqrt(3)/3 = -1.15470

Claude claimed: C_30 = -ln(pi) = -1.14473

At N=10^8: data is near Gemini's values (-1.1520, -0.6655)
At N=10^9: data MOVED toward Claude's values (-1.1453, -0.6982)

CRITICAL TEST: Which direction is convergence heading?
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

def trace_product(primes_list, m, N):
    res = coprime_res(m)
    D = fwd_dist(m, res)
    p = [x for x in primes_list if x <= N]
    if len(p) < 50: return None
    T = N / len(p)
    C = count_trans(p, m, res)
    if np.any(C.sum(axis=1) == 0): return None
    Tobs = norm_rows(C.astype(np.float64))
    Tb = boltz(D, T)
    R = Tobs - Tb
    return float(np.trace(R)) * math.log(N)

print("Sieving to 10^9...")
all_primes = sieve(1_000_000_000)

# Targets
ALGEBRAIC_30 = -2 * math.sqrt(3) / 3      # -1.15470053838
ALGEBRAIC_6  = -2.0 / 3.0                  # -0.66666666667
TRANSCEND_30 = -math.log(math.pi)          # -1.14472988585
TRANSCEND_6  = -math.log(2)                # -0.69314718056

print()
print("=" * 80)
print("  ALGEBRAIC (-2/3 * prod sqrt(p-2))  vs  TRANSCENDENTAL (-ln(pi), -ln(2))")
print("=" * 80)
print()
print(f"  m=30 targets: ALGEBRAIC = {ALGEBRAIC_30:.10f}    TRANSCENDENTAL = {TRANSCEND_30:.10f}")
print(f"  m=6  targets: ALGEBRAIC = {ALGEBRAIC_6:.10f}    TRANSCENDENTAL = {TRANSCEND_6:.10f}")
print()

# =======================================================
# PART 1: Dense convergence data
# =======================================================
print("=" * 80)
print("  PART 1: CONVERGENCE DIRECTION")
print("=" * 80)
print()

for m, alg, trans in [(30, ALGEBRAIC_30, TRANSCEND_30), (6, ALGEBRAIC_6, TRANSCEND_6)]:
    print(f"  --- m = {m} ---")
    alg_name = "-2sqrt(3)/3" if m == 30 else "-2/3"
    trans_name = "-ln(pi)" if m == 30 else "-ln(2)"
    
    print(f"  {'N':>14} {'Empirical':>14} {'err(alg)':>12} {'err(trans)':>12} {'closer_to':>14}")
    print(f"  {'-'*14} {'-'*14} {'-'*12} {'-'*12} {'-'*14}")
    
    vals = []
    for exp in range(4, 10):
        N = 10**exp
        val = trace_product(all_primes, m, N)
        if val is None: continue
        err_a = abs(val - alg)
        err_t = abs(val - trans)
        closer = alg_name if err_a < err_t else trans_name
        vals.append((N, val, err_a, err_t))
        print(f"  {N:>14,} {val:>+14.10f} {err_a:>12.8f} {err_t:>12.8f} {closer:>14}")
    
    # Show the TREND
    if len(vals) >= 2:
        last_two = vals[-2:]
        direction = last_two[1][1] - last_two[0][1]
        print(f"\n  Direction (10^8 -> 10^9): {direction:+.8f}")
        
        if m == 30:
            dist_to_alg = ALGEBRAIC_30 - last_two[1][1]
            dist_to_trans = TRANSCEND_30 - last_two[1][1]
            print(f"  Need to go {dist_to_alg:+.8f} to reach {alg_name}")
            print(f"  Need to go {dist_to_trans:+.8f} to reach {trans_name}")
            if direction * dist_to_alg > 0:
                print(f"  ==> Moving TOWARD {alg_name}")
            else:
                print(f"  ==> Moving AWAY from {alg_name}")
            if direction * dist_to_trans > 0:
                print(f"  ==> Moving TOWARD {trans_name}")
            else:
                print(f"  ==> Moving AWAY from {trans_name}")
        elif m == 6:
            dist_to_alg = ALGEBRAIC_6 - last_two[1][1]
            dist_to_trans = TRANSCEND_6 - last_two[1][1]
            print(f"  Need to go {dist_to_alg:+.8f} to reach {alg_name}")
            print(f"  Need to go {dist_to_trans:+.8f} to reach {trans_name}")
            if direction * dist_to_alg > 0:
                print(f"  ==> Moving TOWARD {alg_name}")
            else:
                print(f"  ==> Moving AWAY from {alg_name}")
            if direction * dist_to_trans > 0:
                print(f"  ==> Moving TOWARD {trans_name}")
            else:
                print(f"  ==> Moving AWAY from {trans_name}")
    print()

# =======================================================
# PART 2: Richardson extrapolation
# =======================================================
print("=" * 80)
print("  PART 2: RICHARDSON EXTRAPOLATION (the N->inf limit)")
print("=" * 80)
print()

for m, alg, trans, alg_name, trans_name in [
    (30, ALGEBRAIC_30, TRANSCEND_30, "-2sqrt(3)/3", "-ln(pi)"),
    (6, ALGEBRAIC_6, TRANSCEND_6, "-2/3", "-ln(2)")
]:
    print(f"  --- m = {m} ---")
    
    vals = []
    for exp in range(4, 10):
        N = 10**exp
        val = trace_product(all_primes, m, N)
        if val is not None:
            vals.append((math.log(N), val))
    
    # Fit: C(N) = L + a/ln(N) + b/ln(N)^2 + c/ln(N)^3
    x = np.array([1/v[0] for v in vals])
    y = np.array([v[1] for v in vals])
    
    for deg in [1, 2, 3, 4]:
        A = np.column_stack([x**k for k in range(deg+1)])
        coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
        L = coeffs[0]
        err_a = abs(L - alg)
        err_t = abs(L - trans)
        closer = alg_name if err_a < err_t else trans_name
        print(f"  Degree {deg}: L = {L:+.12f}  "
              f"|err_alg|={err_a:.8f}  |err_trans|={err_t:.8f}  "
              f"CLOSER TO {closer}")
    
    # Aitken on last 3 decade values
    decade_vals = [v[1] for v in vals if v[0] in [math.log(10**k) for k in range(7,10)]]
    if len(decade_vals) >= 3:
        s0, s1, s2 = decade_vals[-3], decade_vals[-2], decade_vals[-1]
        d = s2 - 2*s1 + s0
        if abs(d) > 1e-15:
            aitken = s0 - (s1-s0)**2 / d
            err_a = abs(aitken - alg)
            err_t = abs(aitken - trans)
            closer = alg_name if err_a < err_t else trans_name
            print(f"  Aitken:    L = {aitken:+.12f}  "
                  f"|err_alg|={err_a:.8f}  |err_trans|={err_t:.8f}  "
                  f"CLOSER TO {closer}")
    print()

# =======================================================
# PART 3: CRT projection A vs T test
# =======================================================
print("=" * 80)
print("  PART 3: CRT PROJECTIONS - ALGEBRAIC vs TRANSCENDENTAL")
print("=" * 80)
print()
print("  The CRT projection from m=210/2310 to m=30 gives the SAME value")
print("  as m=30 direct. The question is: what does THAT value converge to?")
print()

# Project m=210 -> m=30 at various N
res210 = coprime_res(210)
res30 = coprime_res(30)
idx30 = {r: i for i, r in enumerate(res30)}
idx210 = {r: i for i, r in enumerate(res210)}

print(f"  {'N':>12} {'m=30 direct':>14} {'210->30 proj':>14} {'err_alg':>10} {'err_trans':>10}")
for exp in range(5, 10):
    N = 10**exp
    p = [x for x in all_primes if x <= N]
    T = N / len(p)
    
    # Direct m=30
    res = coprime_res(30)
    D30 = fwd_dist(30, res)
    C30 = count_trans(p, 30, res)
    Tobs30 = norm_rows(C30.astype(np.float64))
    Tb30 = boltz(D30, T)
    R30 = Tobs30 - Tb30
    direct = float(np.trace(R30)) * math.log(N)
    
    # CRT projection from 210
    D210 = fwd_dist(210, res210)
    C210 = count_trans(p, 210, res210)
    Tobs210 = norm_rows(C210.astype(np.float64))
    Tb210 = boltz(D210, T)
    R210 = Tobs210 - Tb210
    
    # Project: sum diagonal by r%30 group
    proj = 0.0
    for r in res210:
        i = idx210[r]
        proj += R210[i, i]
    proj *= math.log(N)
    
    err_a = abs(direct - ALGEBRAIC_30)
    err_t = abs(direct - TRANSCEND_30)
    print(f"  {N:>12,} {direct:>+14.10f} {proj:>+14.10f} {err_a:>10.6f} {err_t:>10.6f}")

print()

# =======================================================
# PART 4: The NON-MONOTONIC behavior of m=30
# =======================================================
print("=" * 80)
print("  PART 4: m=30 OVERSHOOTS THEN RETURNS - which target?")
print("=" * 80)
print()
print("  m=30 trace goes: -0.45, -0.97, -1.14, -1.15, -1.15, -1.145")
print("  It OVERSHOOTS around 10^7 then comes back DOWN.")
print()

vals30 = []
for exp in range(4, 10):
    N = 10**exp
    val = trace_product(all_primes, 30, N)
    vals30.append((N, val))

print(f"  {'N':>12} {'Value':>14} {'vs -2sqrt3/3':>14} {'vs -ln(pi)':>14}")
for N, v in vals30:
    print(f"  {N:>12,} {v:>+14.10f} {v - ALGEBRAIC_30:>+14.10f} {v - TRANSCEND_30:>+14.10f}")

print()
print(f"  At N=10^9: value = {vals30[-1][1]:+.10f}")
print(f"  Distance to -2sqrt(3)/3 = {abs(vals30[-1][1] - ALGEBRAIC_30):.10f}")
print(f"  Distance to -ln(pi)     = {abs(vals30[-1][1] - TRANSCEND_30):.10f}")
ratio = abs(vals30[-1][1] - TRANSCEND_30) / abs(vals30[-1][1] - ALGEBRAIC_30)
print(f"  Ratio of distances: {ratio:.6f}")
if ratio < 1:
    print(f"  ==> -ln(pi) is {1/ratio:.1f}x CLOSER")
else:
    print(f"  ==> -2sqrt(3)/3 is {ratio:.1f}x CLOSER")

# =======================================================
# PART 5: THE VERDICT
# =======================================================
print()
print("=" * 80)
print("  THE VERDICT")
print("=" * 80)
print()

# Collect all evidence
m30_val = vals30[-1][1]
m30_err_alg = abs(m30_val - ALGEBRAIC_30)
m30_err_trans = abs(m30_val - TRANSCEND_30)

# m=6 at 10^9
m6_val = trace_product(all_primes, 6, 10**9)
m6_err_alg = abs(m6_val - ALGEBRAIC_6)
m6_err_trans = abs(m6_val - TRANSCEND_6)

print(f"  m=30 at N=10^9: {m30_val:+.10f}")
print(f"    -2sqrt(3)/3 = {ALGEBRAIC_30:+.10f}  error = {m30_err_alg:.10f} ({100*m30_err_alg/abs(ALGEBRAIC_30):.4f}%)")
print(f"    -ln(pi)     = {TRANSCEND_30:+.10f}  error = {m30_err_trans:.10f} ({100*m30_err_trans/abs(TRANSCEND_30):.4f}%)")
print()
print(f"  m=6 at N=10^9: {m6_val:+.10f}")
print(f"    -2/3        = {ALGEBRAIC_6:+.10f}  error = {m6_err_alg:.10f} ({100*m6_err_alg/abs(ALGEBRAIC_6):.4f}%)")
print(f"    -ln(2)      = {TRANSCEND_6:+.10f}  error = {m6_err_trans:.10f} ({100*m6_err_trans/abs(TRANSCEND_6):.4f}%)")
print()

# Evidence summary
print("  EVIDENCE FOR m=30:")
if m30_err_trans < m30_err_alg:
    print(f"    -ln(pi) is {m30_err_alg/m30_err_trans:.1f}x closer than -2sqrt(3)/3")
    print(f"    Convergence direction: TOWARD -ln(pi)")
else:
    print(f"    -2sqrt(3)/3 is {m30_err_trans/m30_err_alg:.1f}x closer than -ln(pi)")
    print(f"    Convergence direction: TOWARD -2sqrt(3)/3")

print()
print("  EVIDENCE FOR m=6:")
if m6_err_trans < m6_err_alg:
    print(f"    -ln(2) is {m6_err_alg/m6_err_trans:.1f}x closer than -2/3")
else:
    print(f"    -2/3 is {m6_err_trans/m6_err_alg:.1f}x closer than -ln(2)")

# Can we distinguish at all?
print()
print("  BOTTOM LINE:")
print(f"    -2sqrt(3)/3 and -ln(pi) differ by only {abs(ALGEBRAIC_30 - TRANSCEND_30):.10f}")
print(f"    Current empirical error at N=10^9: {m30_err_trans:.10f}")
if m30_err_trans < abs(ALGEBRAIC_30 - TRANSCEND_30):
    print(f"    ERROR < GAP => We CAN distinguish them. Winner for m=30: ", end="")
    print("-ln(pi)" if m30_err_trans < m30_err_alg else "-2sqrt(3)/3")
else:
    print(f"    ERROR > GAP => We CANNOT yet distinguish them at N=10^9.")
