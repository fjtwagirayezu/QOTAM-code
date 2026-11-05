#!/usr/bin/env python3
"""Compare full Gibbs minimization with explicitly re-expanded O(lambda) and
O(lambda^2) observables for the revised QOTAM paper's three-site model.

All branches use hbar=m^2=1, kappa=0.7 and the same field cutoff [-6,6]^3.
Reference: independent nested adaptive quadrature; no fitted response formula.
Run: python compare_full_and_perturbative.py --output-dir results
Dependencies: numpy, scipy.  Outputs: CSV and JSON, no plots.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import platform
import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
from scipy.integrate import quad_vec

KAPPA=0.7
CUTOFF=6.0
COUPLINGS=(0.5,2.0,4.0)

def grid(n: int):
    if n < 8:
        raise ValueError('Use at least eight nodes per field coordinate.')
    x,w=leggauss(n)
    x=x*CUTOFF; w=w*CUTOFF
    a,b,c=x[:,None,None],x[None,:,None],x[None,None,:]
    s0=(a*a+b*b+c*c)/2+KAPPA*((b-a)**2+(c-b)**2)/2
    u=(a**4+b**4+c**4)/24
    lr=np.log(w[:,None,None])+np.log(w[None,:,None])+np.log(w[None,None,:])-s0
    lr-=logsumexp(lr)
    return x,lr,u

def expansion(n: int):
    x,lr,u=grid(n)
    p=np.exp(lr)
    eu=float(np.sum(p*u)); eu2=float(np.sum(p*u*u))
    obs={'variance_middle':x[None,:,None]**2,
         'fourth_moment_middle':x[None,:,None]**4,
         'endpoint_correlation':x[:,None,None]*x[None,None,:]}
    coeff={}
    for name,o in obs.items():
        a0=float(np.sum(p*o)); ou=float(np.sum(p*o*u)); ou2=float(np.sum(p*o*u*u))
        cov=ou-a0*eu
        coeff[name]=[a0,-cov,0.5*(ou2-a0*eu2)-eu*cov]
    v=np.array(coeff['variance_middle']); m4=np.array(coeff['fourth_moment_middle'])
    coeff['connected_fourth_middle']=(m4-3*np.convolve(v,v)[:3]).tolist()
    coeff['free_energy']=[0.,eu,-0.5*(eu2-eu*eu)]
    return coeff

def full_minimization(n: int, coupling: float):
    x,lr,u=grid(n); v=coupling*u; lq=lr.copy()
    for _ in range(100):
        grad=v+lq-lr+1
        lq-=0.25*grad
        lq-=logsumexp(lq)
    q=np.exp(lq); middle=q.sum(axis=(0,2)); mu=float(middle@x)
    var=float(middle@((x-mu)**2))
    grad=v+lq-lr+1; grad-=np.sum(q*grad)
    return {'variance_middle':var,
            'connected_fourth_middle':float(middle@((x-mu)**4)-3*var*var),
            'endpoint_correlation':float(np.sum(q.sum(axis=1)*x[:,None]*x[None,:])),
            'free_energy':float(np.sum(q*(v+lq-lr))),
            'normalization_error':float(abs(q.sum()-1)),
            'centered_gradient_rms':float(np.sqrt(np.sum(q*grad**2)))}

def adaptive(coupling: float):
    def onsite(t): return t*t/2+coupling*t**4/24
    def outer(y):
        def inner(x):
            v=np.exp(-onsite(x)-KAPPA*(x-y)**2/2)
            return np.array([v,x*v])
        b,err=quad_vec(inner,-CUTOFF,CUTOFF,epsabs=2e-12,epsrel=2e-12)
        return np.exp(-onsite(y))*np.array([b[0]**2,y*y*b[0]**2,y**4*b[0]**2,b[1]**2])
    vals,err=quad_vec(outer,-CUTOFF,CUTOFF,epsabs=2e-11,epsrel=2e-11)
    z=float(vals[0]); var=float(vals[1]/z)
    return {'partition':z,'variance_middle':var,
            'connected_fourth_middle':float(vals[2]/z-3*var*var),
            'endpoint_correlation':float(vals[3]/z),'adaptive_error_estimate':float(err)}

def main(output: Path):
    output.mkdir(parents=True,exist_ok=True)
    coef=expansion(80); checkcoef=expansion(96)
    coeff_difference=max(abs(np.array(coef[k])-np.array(checkcoef[k])).max() for k in coef)
    ref0=adaptive(0.)
    rows=[]; results={}
    for lam in COUPLINGS:
        full=full_minimization(80,lam); ref=adaptive(lam)
        ref['free_energy']=-float(np.log(ref['partition']/ref0['partition']))
        results[str(lam)]={'full_minimization':full,'adaptive_reference':ref}
        for key in ['free_energy','variance_middle','endpoint_correlation','connected_fourth_middle']:
            a0,a1,a2=coef[key]
            pt1=a0+lam*a1; pt2=pt1+lam**2*a2
            rows.append({'coupling':lam,'observable':key,'full_minimization':full[key],
                         'adaptive_reference':ref[key],'perturbative_order_1':pt1,
                         'perturbative_order_2':pt2,
                         'relative_error_full':abs(full[key]-ref[key])/abs(ref[key]),
                         'relative_error_order_1':abs(pt1-ref[key])/abs(ref[key]),
                         'relative_error_order_2':abs(pt2-ref[key])/abs(ref[key])})
    with (output/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={'scope':'three-site regulated Euclidean scalar model; no continuum or scattering claim',
            'parameters':{'mass_squared':1,'hbar':1,'kappa':KAPPA,'cutoff':CUTOFF,'nodes':80,
                          'mirror_steps':100,'mirror_step_size':.25},
            'series_convention':'Taylor coefficients at zero coupling, with the normalized ratio explicitly re-expanded; no resummation',
            'perturbative_coefficients_80':coef,'coefficient_refinement_80_to_96_max_abs_difference':float(coeff_difference),
            'results':results,'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}}
    (output/'comparison.json').write_text(json.dumps(result,indent=2))
    assert coeff_difference<1e-10
    assert max(row['relative_error_full'] for row in rows)<1e-9
    for row in rows:
        print(json.dumps(row))
    print('Maximum PT coefficient change, n=80 to n=96:',coeff_difference)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
    main(p.parse_args().output_dir)
