#!/usr/bin/env python3
"""
M=6 TURNAROUND CHECK: Does it peak and return to -ln(2)?
==========================================================
m=30 trajectory: rises to -1.155 (algebraic) then falls back to -1.145 (-ln(pi))
m=6 trajectory:  rises to -0.698... and is still going???

If the phase transition picture is correct, m=6 should peak and turn around.
Dense sampling to find the turning point.
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

print("Sieving to 10^9...")
all_primes = sieve(1_000_000_000)

ALGEBRAIC_30 = -2*math.sqrt(3)/3
ALGEBRAIC_6 = -2.0/3
TRANSCEND_30 = -math.log(math.pi)
TRANSCEND_6 = -math.log(2)

# ==============================================
# PART 1: Dense sweep for m=6 and m=30
# ==============================================
print()
print("=" * 80)
print("DENSE SWEEP: looking for m=6 turnaround point")
print("=" * 80)
print()

for m, alg, trans, alg_n, trans_n in [
    (30, ALGEBRAIC_30, TRANSCEND_30, "-2sqrt3/3", "-ln(pi)"),
    (6,  ALGEBRAIC_6,  TRANSCEND_6,  "-2/3",     "-ln(2)")
]:
    res = coprime_res(m)
    D = fwd_dist(m, res)
    
    print(f"--- m = {m} ---")
    print(f"  Algebraic target {alg_n} = {alg:+.10f}")
    print(f"  Transcend target {trans_n}  = {trans:+.10f}")
    print()
    print(f"  {'N':>14} {'C(N)':>14} {'d(alg)':>12} {'d(trans)':>12} {'delta':>12} {'slope':>10}")
    print(f"  {'-'*14} {'-'*14} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    
    prev_val = None
    # Dense logarithmic sampling
    N_values = sorted(set(
        [int(10**(k/4)) for k in range(16, 37)]  # 10^4 to 10^9 in 0.25 steps
    ))
    
    for N in N_values:
        if N > 1_000_000_000:
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
        val = float(np.trace(R)) * math.log(N)
        
        d_alg = val - alg
        d_trans = val - trans
        delta = val - prev_val if prev_val is not None else 0
        slope = "falling" if delta < -0.001 else ("rising" if delta > 0.001 else "flat")
        
        print(f"  {N:>14,} {val:>+14.8f} {d_alg:>+12.8f} {d_trans:>+12.8f} "
              f"{delta:>+12.8f} {slope:>10}")
        
        prev_val = val
    print()

# ==============================================
# PART 2: Decompose: Tr(Tobs)*ln(N) vs Tr(Tboltz)*ln(N)
# ==============================================
print("=" * 80)
print("DECOMPOSITION: Tr(R) = Tr(Tobs) - Tr(Tboltz)")
print("=" * 80)
print()
print("Tr(R)*ln(N) = [Tr(Tobs) - 1]*ln(N) - [Tr(Tboltz) - 1]*ln(N)")
print("            = A(N) - B(N)")
print()
print("B(N) is the Boltzmann diagonal deficit - this is ANALYTICAL and predictable.")
print("A(N) is the observed diagonal deficit - this encodes the prime race.")
print()

for m in [6, 30]:
    res = coprime_res(m)
    phi = len(res)
    D = fwd_dist(m, res)
    
    print(f"m={m} (phi={phi}):")
    print(f"  {'N':>12} {'A(N)':>14} {'B(N)':>14} {'C(N)=A-B':>14} {'A+1/phi*lnN':>14}")
    print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
    
    for exp in range(4, 10):
        N = 10**exp
        p = [x for x in all_primes if x <= N]
        T = N / len(p)
        C = count_trans(p, m, res)
        Tobs = norm_rows(C.astype(np.float64))
        Tb = boltz(D, T)
        
        lnN = math.log(N)
        tr_obs = float(np.trace(Tobs))
        tr_boltz = float(np.trace(Tb))
        
        A = (tr_obs - 1) * lnN
        B = (tr_boltz - 1) * lnN
        C_val = A - B
        
        # The null model trace is phi * (1/phi) = 1
        # So A measures how much the observed self-transition rate
        # differs from 1 (the uniform null), scaled by ln(N)
        
        # At T->inf, Tb->uniform, so B->0
        # What does A approach?
        
        print(f"  {N:>12,} {A:>+14.8f} {B:>+14.8f} {C_val:>+14.8f} {A-(-1/phi)*lnN:>+14.8f}")
    print()

# ==============================================
# PART 3: What does the OBSERVED trace converge to?
# ==============================================
print("=" * 80)
print("PART 3: OBSERVED SELF-TRANSITION RATE")
print("=" * 80)
print()
print("Tr(T_obs)/phi = average prob of self-transition (same residue class)")
print("For mod 6: this is P(p' = p mod 6 | p mod 6)")
print()

for m in [6, 30]:
    res = coprime_res(m)
    phi = len(res)
    D = fwd_dist(m, res)
    
    print(f"m={m} (phi={phi}, null = 1/phi = {1/phi:.6f}):")
    print(f"  {'N':>12} {'Tr(Tobs)/phi':>14} {'Tr(Tb)/phi':>14} {'delta':>12} {'delta*phi*lnN':>14}")
    
    for exp in range(4, 10):
        N = 10**exp
        p = [x for x in all_primes if x <= N]
        T = N / len(p)
        C = count_trans(p, m, res)
        Tobs = norm_rows(C.astype(np.float64))
        Tb = boltz(D, T)
        
        avg_obs = float(np.trace(Tobs)) / phi
        avg_boltz = float(np.trace(Tb)) / phi
        delta = avg_obs - avg_boltz
        
        print(f"  {N:>12,} {avg_obs:>14.10f} {avg_boltz:>14.10f} {delta:>+12.10f} {delta*phi*math.log(N):>+14.8f}")
    print()

# ==============================================
# PART 4: THE CRITICAL QUESTION
# ==============================================
print("=" * 80)
print("PART 4: THE CRITICAL QUESTION - m/T DEPENDENCE")
print("=" * 80)
print()
print("m=30 at N=10^7: m/T=1.99 (near the m/T=2 thawing boundary)")
print("m=30 at N=10^9: m/T=1.52 (well into thawed)")
print("m=6  at N=10^9: m/T=0.31 (superheated)")
print()
print("When m/T < 1, the diagonal is DOMINANT (exp(-m/T) > 0.37).")
print("When m/T > 2, the diagonal is SUPPRESSED (exp(-m/T) < 0.14).")
print()
print("m=30 lives in the SWEET SPOT around m/T~1.5 where the trace converges fast.")
print("m=6 is in the SUPERHEATED regime where m/T~0.3 and convergence is slow.")
print()

# What if the asymptotic expansion is different in the superheated regime?
# For m/T << 1: the diagonal dominates, T_boltz(i,i) is nearly 1/2 for m=6
# The correction terms are larger and oscillatory.

# Let's look at what Tr(T_obs) - 1/phi approaches for m=6
print("m=6: checking if [Tr(Tobs) - 1/phi]*phi*ln(N) converges:")
m = 6
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
    
    lnN = math.log(N)
    
    # The observed self-transition excess over null
    obs_excess = float(np.trace(Tobs)) / phi - 1/phi
    boltz_excess = float(np.trace(Tb)) / phi - 1/phi
    
    # The Chebyshev bias: P(p'=1|p=1) - P(p'=5|p=1)
    # For mod 6, the two classes are 1 and 5
    bias = Tobs[0,0] - Tobs[0,1]  # P(1->1) - P(1->5) 
    
    print(f"  N=10^{exp}: Tr_obs={float(np.trace(Tobs)):.8f}  "
          f"P(1->1)={Tobs[0,0]:.8f}  P(5->5)={Tobs[1,1]:.8f}  "
          f"bias={bias:+.8f}  bias*lnN={bias*lnN:+.8f}")

# For m=6, the 2x2 matrix:
# T(1,1) = P(next prime is 1 mod 6 | current is 1 mod 6)
# T(1,5) = P(next prime is 5 mod 6 | current is 1 mod 6)
# By Chebyshev bias, there should be a preference for p'!=p mod q
# which means T(1,5) > T(1,1), i.e. the diagonal is BELOW 1/2

print()
N = 10**9
p = [x for x in all_primes if x <= N]
T = N / len(p)
C = count_trans(p, m, res)
Tobs = norm_rows(C.astype(np.float64))
Tb = boltz(D, T)
R = Tobs - Tb
print(f"m=6, N=10^9:")
print(f"  T_obs = \n{Tobs}")
print(f"  T_boltz = \n{Tb}")
print(f"  R = T_obs - T_boltz = \n{R}")
print(f"  Tr(R) = {float(np.trace(R)):+.10f}")
print(f"  R has eigenvalues: {np.linalg.eigvals(R)}")

print()
print("=" * 80)
print("OVERALL ASSESSMENT")
print("=" * 80)
print()
print("WHAT IS ROCK SOLID:")
print("  1. CRT projection invariance - PROVEN")
print("  2. m=30: Tr(R)*ln(N) -> -ln(pi) at N=10^9 (0.05% error)")
print("  3. Lattice folds multiplicatively")
print()
print("WHAT IS COMPELLING BUT NOT YET PROVEN:")
print("  4. m=6: target appears to be -ln(2) but value OVERSHOT it")
print("  5. Phase transition from algebraic to transcendental regime")
print("  6. The algebraic law C=-2/3*prod(sqrt(p-2)) as finite-N approximation")
print()
print("WHAT NEEDS MORE DATA:")
print("  7. m=6 at N > 10^9 to confirm turnaround toward -ln(2)")
print("  8. m=30 at N > 10^9 to confirm continued approach to -ln(pi)")
print("  9. Whether the CRT projection trace also shows the phase transition")
