#!/usr/bin/env python3
"""Generate compact JS SCRAMBLER_SWEEP with both H and C_A spectra."""
import json

d = json.load(open('hawking_page_results/scrambler_sweep.json'))

print('// Updated SCRAMBLER_SWEEP with both H-spectrum and C_A-spectrum')
print('const SCRAMBLER_SWEEP = {')
for m_str in sorted(d.keys(), key=lambda x: int(x)):
    sweep = d[m_str]['sweep']
    # Downsample 2x for compactness
    pts = sweep[::2]
    g = [round(r['gamma'], 4) for r in pts]
    ho = [round(r['brody_omega'], 3) for r in pts]
    hkp = [round(r['kl_poisson'], 3) for r in pts]
    hkg = [round(r['kl_goe'], 3) for r in pts]
    
    has_ca = any('ca_brody_omega' in r for r in pts)
    
    print(f'    {m_str}: {{')
    print(f'        gamma: {json.dumps(g)},')
    print(f'        omega: {json.dumps(ho)},')
    print(f'        kl_p:  {json.dumps(hkp)},')
    print(f'        kl_g:  {json.dumps(hkg)},')
    
    if has_ca:
        cao = [round(r.get('ca_brody_omega', 0), 3) for r in pts]
        cakp = [round(r.get('ca_kl_poisson', 0), 3) for r in pts]
        cakg = [round(r.get('ca_kl_goe', 0), 3) for r in pts]
        print(f'        ca_omega: {json.dumps(cao)},')
        print(f'        ca_kl_p:  {json.dumps(cakp)},')
        print(f'        ca_kl_g:  {json.dumps(cakg)},')
    
    print(f'    }},')

print('};')
