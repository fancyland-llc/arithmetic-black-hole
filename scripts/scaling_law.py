#!/usr/bin/env python3
"""
SCALING LAW: Pin down C_6 and find the true ratio C_30/C_6
=============================================================
Problem: C_30 -> -ln(pi) is well-converged at N=10^9 (error 0.05%).
         C_6 is STILL MOVING at N=10^9: -0.698 and drifting.
         The sqrt(3) ratio claim was from N=10^8 unconverged data.

Strategy:
1. Dense sampling at many N values for m=6 and m=30
2. Aitken/Richardson extrapolation of C_6
3. Track the ratio C_30/C_6 as N grows
4. Test specific hypotheses for C_6
5. Analytical decomposition of the trace
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
    """Compute Tr(R)*ln(N) for modulus m at scale N."""
    res = coprime_res(m)
    D = fwd_dist(m, res)
    p = [x for x in primes_list if x <= N]
    if len(p) < 50:
        return None
    T = N / len(p)
    C = count_trans(p, m, res)
    if np.any(C.sum(axis=1) == 0):
        return None
    Tobs = norm_rows(C.astype(np.float64))
    Tb = boltz(D, T)
    R = Tobs - Tb
    return float(np.trace(R)) * math.log(N)

print("Sieving to 10^9...")
all_primes = sieve(1_000_000_000)
LN_PI = math.log(math.pi)
LN_2 = math.log(2)

# ===========================================================
# PART 1: DENSE SAMPLING
# ===========================================================
print("\n" + "="*72)
print("PART 1: DENSE SAMPLING OF C_6 AND C_30")
print("="*72)
print()

# Sample at many N values
log10_values = list(np.arange(4.0, 9.1, 0.25))
data = {}
for m in [6, 30]:
    data[m] = []
    for log10N in log10_values:
        N = int(10**log10N)
        val = trace_product(all_primes, m, N)
        if val is not None:
            data[m].append((N, math.log(N), val))

print(f"{'N':>14} {'C_6':>14} {'C_30':>14} {'ratio':>10} {'delta_ratio':>12}")
print("-" * 66)
prev_ratio = None
for i in range(len(data[6])):
    N6, _, c6 = data[6][i]
    N30, _, c30 = data[30][i]
    assert N6 == N30
    ratio = c30 / c6 if abs(c6) > 1e-10 else float('inf')
    dr = f"{ratio - prev_ratio:+.6f}" if prev_ratio is not None else ""
    print(f"{N6:>14,} {c6:>+14.8f} {c30:>+14.8f} {ratio:>10.6f} {dr:>12}")
    prev_ratio = ratio

# ===========================================================
# PART 2: EXTRAPOLATION OF C_6
# ===========================================================
print("\n" + "="*72)
print("PART 2: EXTRAPOLATION OF C_6 LIMIT")
print("="*72)

# Use only large-N data where the behavior is cleaner
large_data = [(N, lnN, val) for N, lnN, val in data[6] if N >= 1_000_000]
x = np.array([1/d[1] for d in large_data])  # 1/ln(N)
y = np.array([d[2] for d in large_data])     # C_6(N)
lnN_arr = np.array([d[1] for d in large_data])

print("\nRichardson extrapolation (C_6 = L + a/ln(N) + b/ln(N)^2 + ...):")
for deg in range(1, 5):
    A = np.column_stack([x**k for k in range(deg+1)])
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    L = coeffs[0]
    print(f"  Degree {deg}: L = {L:+.10f}")
    
    # Check against candidates
    for name, val in [("-ln(pi)", -LN_PI), ("-ln(2)", -LN_2), 
                       ("-3/4", -0.75), ("-ln(pi)/2", -LN_PI/2),
                       ("-ln(3)/2", -math.log(3)/2)]:
        print(f"    -> {name:>12} = {val:+.10f}, error = {L - val:+.10f}")

# Aitken on the decade values
print("\nAitken acceleration on decade values:")
decade_vals = [(N, val) for N, _, val in data[6] if N in [10**k for k in range(4,10)]]
for i in range(len(decade_vals) - 2):
    s0, s1, s2 = decade_vals[i][1], decade_vals[i+1][1], decade_vals[i+2][1]
    denom = s2 - 2*s1 + s0
    if abs(denom) > 1e-15:
        aitken = s0 - (s1 - s0)**2 / denom
        print(f"  Aitken(10^{4+i}, 10^{5+i}, 10^{6+i}): {aitken:+.10f}")

# ===========================================================
# PART 3: RATIO EXTRAPOLATION
# ===========================================================
print("\n" + "="*72)
print("PART 3: EXTRAPOLATION OF RATIO C_30/C_6")
print("="*72)

ratios = []
for i in range(len(data[6])):
    N6, lnN6, c6 = data[6][i]
    N30, _, c30 = data[30][i]
    if abs(c6) > 0.01 and N6 >= 100_000:
        ratios.append((lnN6, c30/c6))

x_r = np.array([1/r[0] for r in ratios])
y_r = np.array([r[1] for r in ratios])

for deg in range(1, 4):
    A = np.column_stack([x_r**k for k in range(deg+1)])
    coeffs = np.linalg.lstsq(A, y_r, rcond=None)[0]
    L_ratio = coeffs[0]
    print(f"  Richardson degree {deg}: ratio -> {L_ratio:.8f}")
    for name, val in [("sqrt(3)", math.sqrt(3)), ("ln(pi)/ln(2)", LN_PI/LN_2),
                       ("phi", (1+math.sqrt(5))/2), ("5/3", 5/3),
                       ("e/pi+1", math.e/math.pi + 1),
                       ("ln(5)", math.log(5))]:
        print(f"    -> {name:>14} = {val:.8f}, error = {L_ratio - val:+.8f}")

# ===========================================================
# PART 4: ANALYTICAL DECOMPOSITION
# ===========================================================
print("\n" + "="*72)
print("PART 4: DECOMPOSING Tr(R) INTO COMPONENTS")
print("="*72)
print()
print("Tr(R) = Tr(T_obs) - Tr(T_boltz)")
print("     = sum_i [P_obs(i->i)] - sum_i [P_boltz(i->i)]")
print()

for m in [6, 30]:
    res = coprime_res(m)
    phi = len(res)
    D = fwd_dist(m, res)
    
    print(f"m={m}, phi={phi}")
    print(f"  {'N':>12} {'Tr(Tobs)':>12} {'Tr(Tboltz)':>12} {'Tr(R)':>12} "
          f"{'1-Tr(Tobs)':>12} {'1-Tr(Tb)':>12}")
    
    for exp in range(4, 10):
        N = 10**exp
        p = [x for x in all_primes if x <= N]
        T = N / len(p)
        C = count_trans(p, m, res)
        Tobs = norm_rows(C.astype(np.float64))
        Tb = boltz(D, T)
        R = Tobs - Tb
        
        tr_obs = float(np.trace(Tobs))
        tr_boltz = float(np.trace(Tb))
        tr_R = float(np.trace(R))
        
        print(f"  {N:>12,} {tr_obs:>12.8f} {tr_boltz:>12.8f} {tr_R:>+12.8f} "
              f"{1-tr_obs:>12.8f} {1-tr_boltz:>12.8f}")
    print()

# ===========================================================
# PART 5: T_obs DIAGONAL ANALYSIS
# ===========================================================
print("="*72)
print("PART 5: WHAT CONTROLS Tr(T_obs)?")
print("="*72)
print()
print("For m=6, phi=2: T_obs(1,1) = P(p'=1|p=1), T_obs(5,5) = P(p'=5|p=5)")
print("These are the probability that consecutive primes are in the SAME class mod 6.")
print()

m = 6
res = coprime_res(m)
D = fwd_dist(m, res)
for exp in range(4, 10):
    N = 10**exp
    p = [x for x in all_primes if x <= N]
    T = N / len(p)
    C = count_trans(p, m, res)
    Tobs = norm_rows(C.astype(np.float64))
    Tb = boltz(D, T)
    
    # Tobs diagonal entries
    d1 = Tobs[0,0]  # P(1->1 | mod 6)
    d5 = Tobs[1,1]  # P(5->5 | mod 6)
    
    # Boltz diagonal entries
    b1 = Tb[0,0]
    b5 = Tb[1,1]
    
    # The trace
    tr_obs = d1 + d5
    
    # What is (1 - Tr(Tobs)) * ln(N)?  This is the "escape rate" from self-transitions
    escape = (1 - tr_obs) * math.log(N)
    
    print(f"  N=10^{exp}: P(1->1)={d1:.8f} P(5->5)={d5:.8f} "
          f"Tr(Tobs)={tr_obs:.8f} Tr(Tb)={Tb[0,0]+Tb[1,1]:.8f} "
          f"(1-TrObs)*ln(N)={escape:.6f}")

# ===========================================================
# PART 6: THE KEY INSIGHT - Tr(T_boltz) component
# ===========================================================
print()
print("="*72)
print("PART 6: Tr(T_boltz) AS A FUNCTION OF m/T")
print("="*72)
print()
print("Tr(T_boltz) depends ONLY on the distance matrix and T = N/pi(N).")
print("As N->inf, T->inf, and Tr(T_boltz) -> 1 (uniform).")
print("The RATE of approach matters.")
print()

for m in [6, 30]:
    res = coprime_res(m)
    phi = len(res)
    D = fwd_dist(m, res)
    
    print(f"m={m}:")
    for exp in range(4, 10):
        N = 10**exp
        p = [x for x in all_primes if x <= N]
        T = N / len(p)
        Tb = boltz(D, T)
        tr_b = float(np.trace(Tb))
        
        # First-order expansion: Tr(Tb) ~ phi * exp(-m/T) / (phi*exp(-m/T) + off_diag_exp)
        # As T->inf: Tr(Tb) -> phi * (1/phi) = 1
        # Correction: 1 - Tr(Tb) ~ ?
        
        correction = 1 - tr_b
        correction_lnN = correction * math.log(N)
        
        print(f"  N=10^{exp}: T={T:>8.3f}  Tr(Tb)={tr_b:.10f}  "
              f"1-Tr(Tb)={correction:.10f}  (1-Tr(Tb))*ln(N)={correction_lnN:.8f}")
    print()

# ===========================================================
# PART 7: DECOMPOSITION Tr(R)*ln(N) = [Tr(Tobs)-1]*ln(N) - [Tr(Tb)-1]*ln(N)
# ===========================================================
print("="*72)
print("PART 7: THE TWO-COMPONENT DECOMPOSITION")
print("="*72)
print()
print("Tr(R)*ln(N) = A(N) - B(N)")
print("  A(N) = [Tr(T_obs) - 1] * ln(N)   (observed diagonal deficit)")
print("  B(N) = [Tr(T_boltz) - 1] * ln(N)  (Boltzmann diagonal deficit)")
print()

for m in [6, 30]:
    res = coprime_res(m)
    phi = len(res)
    D = fwd_dist(m, res)
    
    print(f"m={m} (phi={phi}):")
    print(f"  {'N':>12} {'A(N)':>14} {'B(N)':>14} {'A-B=C(N)':>14}")
    print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*14}")
    
    for exp in range(4, 10):
        N = 10**exp
        p = [x for x in all_primes if x <= N]
        T = N / len(p)
        C = count_trans(p, m, res)
        Tobs = norm_rows(C.astype(np.float64))
        Tb = boltz(D, T)
        
        lnN = math.log(N)
        A = (float(np.trace(Tobs)) - 1) * lnN
        B = (float(np.trace(Tb)) - 1) * lnN
        C_val = A - B
        
        print(f"  {N:>12,} {A:>+14.8f} {B:>+14.8f} {C_val:>+14.8f}")
    print()

print()
print("="*72)
print("PART 8: THE BOLTZMANN TRACE IS ANALYTICAL")
print("="*72)
print()
print("Tr(T_boltz) can be computed exactly from the distance matrix.")
print("Let's find [1 - Tr(T_boltz)] * ln(N) analytically.")
print()

for m in [6, 30]:
    res = coprime_res(m)
    phi = len(res)
    D = fwd_dist(m, res)
    
    print(f"m={m}: distance matrix D =")
    print(D)
    print(f"Row sums of D: {D.sum(axis=1)}")
    print(f"Diagonal: {np.diag(D)}")
    print()
    
    # For T -> infinity, expand Tr(T_boltz) in 1/T:
    # T_boltz(i,i) = exp(-D_ii/T) / sum_j exp(-D_ij/T)
    # = exp(-m/T) / sum_j exp(-D_ij/T)
    #
    # For large T: exp(-d/T) ~ 1 - d/T + d^2/2T^2
    # Numerator: 1 - m/T + m^2/2T^2
    # Denominator: sum_j [1 - D_ij/T + D_ij^2/2T^2] = phi - S1_i/T + S2_i/2T^2
    # where S1_i = sum_j D_ij, S2_i = sum_j D_ij^2
    
    S1 = D.sum(axis=1)    # sum of distances from each row
    S2 = (D**2).sum(axis=1)  # sum of squared distances
    
    print(f"S1 (row sums): {S1}")
    print(f"S2 (row sum of D^2): {S2}")
    print()
    
    # Tr(T_boltz) = sum_i exp(-m/T) / [sum_j exp(-D_ij/T)]
    # For large T, to first order:
    # T_boltz(i,i) ~ (1/phi) * [1 - m/T + 1/(phi*T) * S1_i]
    #              = (1/phi) * [1 + (S1_i/phi - m)/T + ...]
    #              = 1/phi + (S1_i/phi - m)/(phi*T) + ...
    #
    # Tr(T_boltz) ~ 1 + sum_i [(S1_i/phi - m)/(phi*T)]
    #             = 1 + [sum_i S1_i/phi - phi*m]/(phi*T)
    #             = 1 + [total_sum/phi - phi*m]/(phi*T)
    
    total_sum = float(np.sum(D))
    first_order_coeff = (total_sum/phi - phi*m) / phi
    print(f"First-order: 1 - Tr(Tb) ~ {-first_order_coeff:.6f}/T")
    print(f"  => [1-Tr(Tb)]*ln(N) ~ {-first_order_coeff:.6f} * ln(N)/T")
    print(f"  Since T ~ ln(N), this gives ~ {-first_order_coeff:.6f} (constant!)")
    print()
    
    # Let's verify numerically
    print(f"  Numerical check:")
    for exp in [7, 8, 9]:
        N = 10**exp
        p = [x for x in all_primes if x <= N]
        T = N / len(p)
        Tb = boltz(D, T)
        actual = (1 - float(np.trace(Tb))) * math.log(N)
        predicted = -first_order_coeff * math.log(N) / T
        # Better: compute exactly
        print(f"    N=10^{exp}: actual [1-Tr(Tb)]*lnN = {actual:.8f}, "
              f"1st-order pred = {predicted:.8f}")
    print()

print()
print("="*72)
print("GRAND SYNTHESIS")
print("="*72)
