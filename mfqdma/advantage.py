"""Tests targeting the regimes where quantile detrending should matter:
   (1) outlier / heavy-tail contamination of a cascade with KNOWN tau(q)
   (2) alpha-stable (infinite-variance) signals
   (3) rank-preserving monotonic transforms  (invariance probe)
   (4) amplitude-asymmetric cascades        (does tau expose structure?)
"""
import sys, os, json
import numpy as np
from .generators import (binomial_measure, binomial_tau_theory, contaminate,
                        levy_stable, lognormal_cascade, monotonic_family,
                        asymmetric_cascade)
from .methods import mfdfa, mfdma, mfqdma

from . import paths
ROOT = str(paths.REPO)
Q = np.arange(-4, 4.5, 0.5)
SM = np.unique(np.logspace(np.log10(16), np.log10(2048), 14).astype(int))
SD = np.unique(np.logspace(np.log10(64), np.log10(8192), 14).astype(int))
TAUS = [0.10, 0.25, 0.50, 0.75, 0.90]


def rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() else np.nan


def methods_1d(x, demean):
    out = {'MFDFA': mfdfa(x, Q, scales=SD, demean=demean),
           'MFDMA': mfdma(x, Q, scales=SM, theta=0.0, demean=demean)}
    for t in TAUS:
        out[f'MFQDMA_tau{t}'] = mfqdma(x, Q, scales=SM, theta=0.0, tau=t,
                                       demean=demean)
    return out


# ---------------------------------------------------- (1) contamination
def test_contamination(n_rep=5):
    clean = binomial_measure(0.30, 16)
    th = binomial_tau_theory(Q, 0.30)
    rows = []
    for kind in ('outlier', 'student_t', 'levy'):
        for frac in (0.0, 0.01, 0.02, 0.05, 0.10):
            for rep in range(1 if frac == 0 else n_rep):
                x = clean if frac == 0 else contaminate(
                    clean, frac, 20.0, kind, seed=500 + rep)[0]
                for nm, r in methods_1d(x, demean=False).items():
                    rows.append(dict(test='contamination', kind=kind,
                                     frac=frac, rep=rep, method=nm,
                                     rmse=rmse(r['tau_q'], th),
                                     r2=float(np.nanmean(r['r2']))))
    return rows


# ---------------------------------------------------- (2) alpha-stable
def test_levy(n_rep=5):
    rows = []
    for a in (1.2, 1.5, 1.8, 2.0):
        for rep in range(n_rep):
            x = levy_stable(2 ** 15, a, 0.0, seed=600 + rep)
            for nm, r in methods_1d(x, demean=True).items():
                hq = r['hq']
                rows.append(dict(test='levy', alpha=a, rep=rep, method=nm,
                                 h2=float(hq[np.argmin(np.abs(Q - 2.0))]),
                                 width=float(np.nanmax(hq) - np.nanmin(hq)),
                                 r2=float(np.nanmean(r['r2']))))
    return rows


# ---------------------------------------------------- (3) monotonic maps
def test_monotonic(n_rep=3):
    rows = []
    for rep in range(n_rep):
        base = lognormal_cascade(15, 0.4, seed=700 + rep)
        fam = monotonic_family(base)
        ref = {nm: r['hq'] for nm, r in methods_1d(fam['identity'],
                                                   demean=False).items()}
        for tname, v in fam.items():
            for nm, r in methods_1d(v, demean=False).items():
                d = np.abs(r['hq'] - ref[nm])
                rows.append(dict(test='monotonic', transform=tname, rep=rep,
                                 method=nm,
                                 max_dev=float(np.nanmax(d)),
                                 mean_dev=float(np.nanmean(d))))
    return rows


# ---------------------------------------------------- (4) asymmetry
def test_asymmetry(n_rep=5):
    rows = []
    for (pu, pdn) in [(0.5, 0.5), (0.35, 0.65), (0.30, 0.50), (0.25, 0.75)]:
        for rep in range(n_rep):
            x = asymmetric_cascade(15, pu, pdn, seed=800 + rep)
            res = methods_1d(x, demean=False)
            hs = {t: res[f'MFQDMA_tau{t}']['hq'] for t in TAUS}
            M = np.column_stack([hs[t] for t in TAUS])
            spread = np.nanmax(M, axis=1) - np.nanmin(M, axis=1)
            rows.append(dict(test='asymmetry', p_up=pu, p_dn=pdn, rep=rep,
                             mean_tau_spread=float(np.nanmean(spread)),
                             max_tau_spread=float(np.nanmax(spread)),
                             dH_lo_hi=float(np.nanmean(
                                 np.abs(hs[0.10] - hs[0.90])))))
    return rows


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    out = {}
    if which in ('all', 'contam'):
        out['contamination'] = test_contamination()
        print('contamination done', flush=True)
    if which in ('all', 'levy'):
        out['levy'] = test_levy()
        print('levy done', flush=True)
    if which in ('all', 'mono'):
        out['monotonic'] = test_monotonic()
        print('monotonic done', flush=True)
    if which in ('all', 'asym'):
        out['asymmetry'] = test_asymmetry()
        print('asymmetry done', flush=True)
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    with open(os.path.join(ROOT, 'results', f'advantage_{which}.json'), 'w') as f:
        json.dump(out, f)
    print('saved')
