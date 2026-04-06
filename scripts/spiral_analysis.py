#!/usr/bin/env python3
"""Quick analysis of the leading eigenvalue's spiral trajectory."""
import math

data = [
    (1e4,  8.1367,  +0.05836640, +0.12086526),
    (1e5,  10.4254, +0.00097965, +0.08008025),
    (1e6,  12.7392, -0.00739828, +0.04694098),
    (1e7,  15.0471, -0.00971165, +0.02935627),
    (1e8,  17.3567, -0.01283326, +0.01843810),
    (1e9,  19.6666, -0.01230897, +0.01386637),
]

print("=" * 80)
print("  LOGARITHMIC SPIRAL ANALYSIS: leading eigenvalue of R")
print("=" * 80)
print()
hdr = f"  {'N':>10}  {'T':>8}  {'ln(N)':>8}  {'Re':>10}  {'Im':>10}  {'|lam|':>10}  {'phase/pi':>10}"
print(hdr)
print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
for N, T, re, im in data:
    mod = math.sqrt(re**2 + im**2)
    phase = math.atan2(im, re) / math.pi
    lnN = math.log(N)
    print(f"  {N:>10.0f}  {T:>8.4f}  {lnN:>8.4f}  {re:>+10.6f}  {im:>+10.6f}  {mod:>10.6f}  {phase:>+10.6f}")

print()
print("  HYPOTHESIS 1: phase = gamma * ln(N)")
gammas = []
for N, T, re, im in data:
    phase = math.atan2(im, re)
    lnN = math.log(N)
    gamma = phase / lnN
    gammas.append(gamma)
    print(f"  N = {N:>10.0f}  gamma = {gamma:.6f}")
mean_g = sum(gammas) / len(gammas)
std_g = (sum((g - mean_g)**2 for g in gammas) / len(gammas))**0.5
print(f"  mean(gamma) = {mean_g:.6f}  std = {std_g:.6f}  CoV = {std_g/abs(mean_g):.3f}")

print()
print("  HYPOTHESIS 2: |lambda| ~ A / T^alpha")
for i in range(1, len(data)):
    N1, T1, r1, i1 = data[i-1]
    N2, T2, r2, i2 = data[i]
    m1 = math.sqrt(r1**2 + i1**2)
    m2 = math.sqrt(r2**2 + i2**2)
    alpha = (math.log(m1) - math.log(m2)) / (math.log(T2) - math.log(T1))
    A = m1 * T1**alpha
    print(f"  T: {T1:.2f} -> {T2:.2f}  alpha = {alpha:.4f}  A = {A:.6f}")

print()
print("  HYPOTHESIS 3: |lambda| * T^n products")
for N, T, re, im in data:
    mod = math.sqrt(re**2 + im**2)
    print(f"  N={N:>10.0f}  |l|*T={mod*T:.6f}  |l|*T^1.5={mod*T**1.5:.5f}  |l|*T^2={mod*T**2:.4f}")

print()
print("  HYPOTHESIS 4: phase vs pi*k/phi(m) with k = round(phase*phi/pi)")
phi = 8
for N, T, re, im in data:
    phase = math.atan2(im, re)
    k_est = phase * phi / math.pi
    print(f"  N={N:>10.0f}  phase*phi/pi = {k_est:.4f}  (nearest: {round(k_est)}, frac: {k_est - round(k_est):+.4f})")

print()
print("  HYPOTHESIS 5: Re(lam) convergence (real part stabilizes?)")
for N, T, re, im in data:
    print(f"  N={N:>10.0f}  Re={re:>+.8f}  Re*T={re*T:>+.6f}  Re*sqrt(T)={re*math.sqrt(T):>+.6f}")

# KEY: the whole eigenvalue as r*exp(i*theta) = r*exp(i*gamma*ln(N))
# This means lambda_0 = A * N^(sigma + i*gamma) for some real sigma, gamma
# So ln|lambda| = ln(A) + sigma*ln(N) and arg(lambda) = gamma*ln(N)
print()
print("  POWER LAW FIT: lambda_0 ~ A * N^(sigma + i*gamma)")
print("  Extracting sigma from |lambda| vs N:")
sigmas = []
for i in range(1, len(data)):
    N1, T1, r1, i1 = data[i-1]
    N2, T2, r2, i2 = data[i]
    m1 = math.sqrt(r1**2 + i1**2)
    m2 = math.sqrt(r2**2 + i2**2)
    sigma = (math.log(m2) - math.log(m1)) / (math.log(N2) - math.log(N1))
    sigmas.append(sigma)
    print(f"  N: {N1:.0f} -> {N2:.0f}  sigma = {sigma:.6f}")
mean_sig = sum(sigmas) / len(sigmas)
print(f"  mean(sigma) = {mean_sig:.6f}")

print()
print("  *** COMBINED: lambda_0 ~ A * N^(sigma + i*gamma)")
print(f"  *** sigma = {mean_sig:.6f}  (governs decay rate)")
print(f"  *** gamma = {mean_g:.6f}  (governs rotation rate)")
print(f"  *** Complex exponent: s = {mean_sig:.6f} + {mean_g:.6f}i")
print(f"  *** |Im/Re| = {abs(mean_g/mean_sig):.4f}")

# Check the Riemann zeta zero hypothesis
print()
print("  ZETA ZERO COMPARISON:")
gamma1 = 14.134725  # first zeta zero
print(f"  gamma_1 = {gamma1:.6f} (first Riemann zeta zero imaginary part)")
print(f"  Our gamma = {mean_g:.6f}")
print(f"  Ratio: gamma_1 / our_gamma = {gamma1 / mean_g:.4f}")
print(f"  Our gamma * 2*pi = {mean_g * 2 * math.pi:.6f}")
print(f"  gamma_1 / (2*pi) = {gamma1 / (2*math.pi):.6f}")

# Ah but the eigenvalue is NOT directly x^rho.  It's a residual of
# a transition matrix.  The connection is more subtle.
# sigma = -1/2 would correspond to the Riemann Hypothesis (error term ~ N^{-1/2})
print(f"\n  sigma = {mean_sig:.6f} vs -1/2 = -0.500000")
print(f"  Deviation from RH prediction: {abs(mean_sig + 0.5):.6f}")
