#!/usr/bin/env python3
"""One-shot dump of scrambler sweep results."""
import json

d = json.load(open('hawking_page_results/scrambler_sweep.json'))

print(f"{'m':>6} {'gamma':>6} {'H_omega':>8} {'CA_omega':>9} {'H_class':>10} {'CA_class':>10}")
print("=" * 60)

for m_str in ['30', '210', '2310']:
    sweep = d[m_str]['sweep']
    for r in sweep[::5]:  # every 5th point
        g = r['gamma']
        h_w = r['brody_omega']
        h_c = r['classification']
        ca_w = r.get('ca_brody_omega', None)
        ca_c = r.get('ca_classification', '—')
        ca_ws = f"{ca_w:.3f}" if ca_w is not None else "  —  "
        print(f"{m_str:>6} {g:>6.2f} {h_w:>8.3f} {ca_ws:>9} {h_c:>10} {ca_c:>10}")
    print()

# Peak values
print("\nPEAK VALUES:")
print(f"{'m':>6} {'H_peak':>8} {'H_at_g':>7} {'CA_peak':>9} {'CA_at_g':>8}")
print("-" * 45)
for m_str in ['30', '210', '2310']:
    sweep = d[m_str]['sweep']
    h_omegas = [(r['brody_omega'], r['gamma']) for r in sweep]
    h_best = max(h_omegas, key=lambda x: x[0])
    
    ca_omegas = [(r.get('ca_brody_omega', 0), r['gamma']) for r in sweep if 'ca_brody_omega' in r]
    ca_best = max(ca_omegas, key=lambda x: x[0]) if ca_omegas else (None, None)
    
    ca_s = f"{ca_best[0]:.3f}" if ca_best[0] is not None else "  —  "
    ca_g = f"{ca_best[1]:.2f}" if ca_best[1] is not None else " — "
    print(f"{m_str:>6} {h_best[0]:>8.3f} {h_best[1]:>7.2f} {ca_s:>9} {ca_g:>8}")
