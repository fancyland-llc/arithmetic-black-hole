#!/usr/bin/env python3
"""Check how Tr(R)*ln(N) depends on modulus for m = 6, 10, 30."""
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

primes = sieve(1_000_000_000)
LN_PI = math.log(math.pi)

for m in [6, 10, 30]:
    res = coprime_res(m)
    phi = len(res)
    D = fwd_dist(m, res)
    
    print(f"m={m} phi={phi}")
    header = f"  {'N':>12}  {'Tr(R)*ln(N)':>14}  {'/phi':>10}  {'/ln(m)':>10}"
    print(header)
    print(f"  {'-'*12}  {'-'*14}  {'-'*10}  {'-'*10}")
    
    for exp in range(4, 10):
        N = 10**exp
        p = [x for x in primes if x <= N]
        T = N/len(p)
        C = count_trans(p, m, res)
        Tobs = norm_rows(C.astype(np.float64))
        Tb = boltz(D, T)
        R = Tobs - Tb
        tr = float(np.trace(R))
        lnN = math.log(N)
        product = tr * lnN
        per_phi = product / phi
        per_lnm = product / math.log(m)
        print(f"  {N:>12,}  {product:>+14.8f}  {per_phi:>+10.6f}  {per_lnm:>+10.6f}")
    
    # Converged value and candidate matches
    N = 10**9
    p = [x for x in primes if x <= N]
    T = N/len(p)
    C = count_trans(p, m, res)
    Tobs = norm_rows(C.astype(np.float64))
    Tb = boltz(D, T)
    R = Tobs - Tb
    val = float(np.trace(R)) * math.log(N)
    
    print(f"  Converged: {val:+.10f}")
    checks = {
        "-ln(pi)": -LN_PI,
        "-ln(2)": -math.log(2),
        "-ln(3)": -math.log(3),
        "-ln(5)": -math.log(5),
        "-2*ln(2)": -2*math.log(2),
        "-ln(6)": -math.log(6),
        f"-ln(m/{phi})": -math.log(m/phi),
        f"-ln(pi)*{phi}/8": -LN_PI * phi / 8,
        "-1": -1.0,
        "-gamma_EM": -0.5772156649,
        "-2*gamma_EM": -2*0.5772156649,
        "-ln(pi)/2": -LN_PI/2,
        "-ln(pi)/4": -LN_PI/4,
    }
    print(f"  Best matches:")
    ranked = sorted(checks.items(), key=lambda kv: abs(kv[1] - val))
    for name, cval in ranked[:5]:
        print(f"    {name:>20} = {cval:+.8f}  err = {val - cval:+.8f}")
    print()
