"""Comprehensive 2D MFQDMA analysis:
  A  theta x tau grid on multiplicative cascades (accuracy vs analytic tau(q))
  B  monofractal fBm surfaces (h(q) must be flat at H)
  C  contamination robustness in 2D
  D  real 2D fields (terrain DEM, photograph, medical image)
"""
import sys, os, json, time, gzip
import numpy as np
from .generators import cascade_2d, fbm_surface, contaminate
from .methods import mfdma_2d
from .theory_2d import cascade2d_tau, fbm_surface_tau

from . import paths
ROOT = str(paths.REPO)
Q = np.arange(-4, 4.5, 0.5)
THETAS = [0.0, 0.5, 1.0]
TAUS = [0.25, 0.50, 0.75]
PSETS = {'A': (0.05, 0.15, 0.25, 0.55),
         'B': (0.10, 0.20, 0.30, 0.40),
         'C': (0.02, 0.18, 0.30, 0.50)}


def sc2d(N, lo=6, hi=None, n=10):
    hi = hi or max(lo + 2, N // 6)
    s = np.unique(np.logspace(np.log10(lo), np.log10(hi), n).astype(int))
    return s[s >= lo]


def rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() else np.nan


def run_methods(Z, scales, demean, full_theta=False):
    """MFDMA at every theta (cheap, integral image) plus MFQDMA.
    The MFQDMA theta sweep is only run where explicitly requested, because
    its cost is O(R1 R2 n^2) per scale."""
    out = {}
    for th in THETAS:
        out[f'MFDMA2D_th{th}'] = mfdma_2d(Z, Q, scales=scales, theta=th,
                                          tau=None, demean=demean)
    th_list = THETAS if full_theta else [0.0]
    for th in th_list:
        for tt in TAUS:
            out[f'MFQDMA2D_th{th}_tau{tt}'] = mfdma_2d(
                Z, Q, scales=scales, theta=th, tau=tt, demean=demean)
    return out


# ------------------------------------------------------------------ A
def part_A():
    rows = []
    for tag, p in PSETS.items():
        for k in (8, 9):
            Z = cascade_2d(p, k)
            th = cascade2d_tau(Q, p)
            s = sc2d(2 ** k, hi=2 ** k // 8, n=8)
            for nm, r in run_methods(Z, s, demean=False,
                                     full_theta=(k == 8)).items():
                rows.append(dict(part='A', pset=tag, k=k, method=nm,
                                 rmse=rmse(r['tau_q'], th),
                                 r2=float(np.nanmean(r['r2'])),
                                 est=r['tau_q'].tolist(),
                                 theory=th.tolist()))
            print(f'  A {tag} k={k} done', flush=True)
    return rows


# ------------------------------------------------------------------ B
def part_B():
    rows = []
    for H in (0.3, 0.5, 0.7, 0.9):
        for sd in range(1):
            Z = fbm_surface(256, H, seed=4100 + sd)
            th = fbm_surface_tau(Q, H)
            s = sc2d(256, hi=32, n=8)
            for nm, r in run_methods(Z, s, demean=True).items():
                hq = r['hq']
                rows.append(dict(part='B', H=H, seed=sd, method=nm,
                                 rmse=rmse(r['tau_q'], th),
                                 h_flatness=float(np.nanmax(hq) - np.nanmin(hq)),
                                 h2=float(hq[np.argmin(np.abs(Q - 2.0))]),
                                 r2=float(np.nanmean(r['r2']))))
            print(f'  B H={H} s={sd} done', flush=True)
    return rows


# ------------------------------------------------------------------ C
def part_C():
    rows = []
    p = PSETS['A']
    base = cascade_2d(p, 8)
    th = cascade2d_tau(Q, p)
    s = sc2d(256, hi=32, n=8)
    for kind in ('outlier', 'student_t'):
        for frac in (0.0, 0.01, 0.02, 0.05, 0.10):
            for rep in range(1 if frac == 0 else 2):
                if frac == 0:
                    Z = base
                else:
                    flat, _ = contaminate(base.ravel(), frac, 20.0, kind,
                                          seed=900 + rep)
                    Z = flat.reshape(base.shape)
                for nm, r in run_methods(Z, s, demean=False).items():
                    rows.append(dict(part='C', kind=kind, frac=frac, rep=rep,
                                     method=nm, rmse=rmse(r['tau_q'], th),
                                     r2=float(np.nanmean(r['r2']))))
            print(f'  C {kind} f={frac} done', flush=True)
    return rows


# ------------------------------------------------------------------ D
def load_real_fields():
    import matplotlib.cbook as cb
    d = os.path.join(os.path.dirname(cb.__file__), 'mpl-data', 'sample_data')
    out = {}
    z = np.load(os.path.join(d, 'jacksboro_fault_dem.npz'))
    out['dem_terrain'] = z['elevation'].astype(float)
    try:
        with gzip.open(os.path.join(d, 's1045.ima.gz'), 'rb') as f:
            raw = np.frombuffer(f.read(), dtype='>u2')
        n = int(np.sqrt(raw.size))
        if n * n == raw.size:
            out['mri_slice'] = raw.reshape(n, n).astype(float)
    except Exception as e:
        print('   mri skipped:', e)
    return out


def part_D():
    rows = {}
    for name, F in load_real_fields().items():
        N = min(F.shape)
        s = sc2d(N, lo=6, hi=max(12, N // 8), n=8)
        res = run_methods(F, s, demean=True)
        rows[name] = {'shape': list(F.shape), 'scales': s.tolist(),
                      'methods': {nm: dict(hq=r['hq'].tolist(),
                                           tau_q=r['tau_q'].tolist(),
                                           alpha=r['alpha'].tolist(),
                                           f_alpha=r['f_alpha'].tolist(),
                                           r2=float(np.nanmean(r['r2'])))
                                  for nm, r in res.items()}}
        print(f'  D {name} {F.shape} done', flush=True)
    return rows


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    out = {}
    t0 = time.time()
    if which in ('all', 'A'):
        out['A'] = part_A(); print(f'A done {time.time()-t0:.0f}s', flush=True)
    if which in ('all', 'B'):
        out['B'] = part_B(); print(f'B done {time.time()-t0:.0f}s', flush=True)
    if which in ('all', 'C'):
        out['C'] = part_C(); print(f'C done {time.time()-t0:.0f}s', flush=True)
    if which in ('all', 'D'):
        out['D'] = part_D(); print(f'D done {time.time()-t0:.0f}s', flush=True)
    os.makedirs(f'{ROOT}/results', exist_ok=True)
    json.dump(out, open(f'{ROOT}/results/analysis_2d_{which}.json', 'w'))
    print('saved', f'{time.time()-t0:.0f}s')
