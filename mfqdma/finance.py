"""Financial-markets study for the MFQDMA paper (Sec. VI C of the manuscript).

Data: qrmdata (Hofert, Hornik & McNeil, CRAN) -- adjusted daily closes from
Yahoo Finance: 12 stock indices, 3 FX rates, gold, Brent oil, Bitcoin, and the
30 Dow Jones constituents.  Read with our own pure-data .rda parser.

Parts
  A  descriptive statistics of daily log returns
  B  multifractal spectra + shuffled / phase-randomised surrogates
  C  influence of the largest genuine extreme days (leave-k-out)
  D  injected data errors: isolated bad ticks, return outliers, microstructure
     noise -> deviation of h(q) from the uncontaminated estimate
  E  same as D for the 29 DJIA constituents with complete 2000-2015 histories
  F  rolling-window Delta-alpha of the S&P 500, clean vs 0.5% bad ticks
"""
import sys, os, json, time
import numpy as np
from .rdata_reader import read_rda, to_series
from .methods import mfdfa, mfdma, mfqdma_multi

from . import paths
RAW = str(paths.QRM_DATA)
OUT = str(paths.RESULTS / 'finance.json')
Q = np.arange(-4, 4.5, 0.5)
LAMS = (0.1, 0.25, 0.5, 0.75, 0.9)
SERIES = ['SP500', 'DJ', 'NASDAQ', 'FTSE', 'DAX', 'CAC', 'SMI', 'EURSTOXX',
          'NIKKEI', 'HSI', 'SSEC', 'CSI', 'EUR_USD', 'GBP_USD', 'JPY_USD',
          'GOLD', 'OIL_Brent', 'BTC']
LABEL = {'SP500': 'S&P 500', 'DJ': 'DJIA', 'NASDAQ': 'NASDAQ-100', 'FTSE': 'FTSE 100',
         'DAX': 'DAX', 'CAC': 'CAC 40', 'SMI': 'SMI', 'EURSTOXX': 'Euro Stoxx 50',
         'NIKKEI': 'Nikkei 225', 'HSI': 'Hang Seng', 'SSEC': 'SSE Composite',
         'CSI': 'CSI 300', 'EUR_USD': 'EUR/USD', 'GBP_USD': 'GBP/USD',
         'JPY_USD': 'JPY/USD', 'GOLD': 'Gold', 'OIL_Brent': 'Brent oil', 'BTC': 'Bitcoin'}


# ------------------------------------------------------------------ data
def load_prices(name):
    paths.ensure_qrmdata()
    if name == 'BTC':
        d, x, c = to_series(read_rda(f'{RAW}/crypto.rda')['crypto'])
        p = x[:, c.index('BTC')]
        weekdays_only = False                      # crypto trades 7 days
    else:
        d, x, c = to_series(read_rda(f'{RAW}/{name}.rda')[name])
        p = x[:, 0]
        weekdays_only = True
    ok = np.isfinite(p) & (p > 0)
    if weekdays_only:
        dow = (d.astype('datetime64[D]').astype('int64') + 3) % 7   # 0=Mon
        ok &= dow < 5
    return d[ok], p[ok]


def returns(name):
    d, p = load_prices(name)
    r = np.diff(np.log(p))
    return d[1:], r


def dj_constituents(start='2000-01-01', end='2015-12-31'):
    paths.ensure_qrmdata()
    d, x, c = to_series(read_rda(f'{RAW}/DJ_const.rda')['DJ_const'])
    m = (d >= np.datetime64(start)) & (d <= np.datetime64(end))
    d, x = d[m], x[m]
    out = {}
    for j, tick in enumerate(c):
        p = x[:, j]
        if np.isfinite(p).all() and (p > 0).all():
            out[tick] = np.diff(np.log(p))
    return out


# ------------------------------------------------------------------ estimation
def scales(N):
    hi = min(500, N // 10)
    sm = np.unique(np.logspace(1, np.log10(hi), 14).astype(int))
    sd = np.unique(np.logspace(np.log10(20), np.log10(2 * hi), 14).astype(int))
    return sm, sd


def width(a):
    a = np.asarray(a)[np.isfinite(a)]
    return float(a.max() - a.min()) if len(a) else np.nan


def analyse(r):
    sm, sd = scales(len(r))
    res = {'MFDFA': mfdfa(r, Q, scales=sd), 'MFDMA': mfdma(r, Q, scales=sm, theta=0.0)}
    for lam, v in mfqdma_multi(r, Q, scales=sm, theta=0.0, taus=LAMS).items():
        res[f'MFQDMA_{lam}'] = v
    return {k: dict(hq=v['hq'].tolist(), alpha=v['alpha'].tolist(),
                    f_alpha=v['f_alpha'].tolist(), dalpha=width(v['alpha']),
                    h2=float(v['hq'][np.argmin(np.abs(Q - 2))]),
                    r2=float(np.nanmean(v['r2'])),
                    Fq=v['Fq'].tolist(), scales=v['scales'].tolist())
            for k, v in res.items()}


def slim(a):
    return {k: dict(hq=v['hq'], dalpha=v['dalpha'], h2=v['h2'], r2=v['r2'])
            for k, v in a.items()}


def shuffled(r, rng):
    return rng.permutation(r)


def phase_rand(r, rng):
    F = np.fft.rfft(r - r.mean())
    ph = rng.uniform(0, 2 * np.pi, len(F)); ph[0] = 0
    if len(r) % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=len(r)) + r.mean()


# ------------------------------------------------------------------ contamination
def contaminate(r, kind, level, rng, kappa=10.0):
    """kind: 'badtick' (isolated mis-printed price -> return pair +s,-s),
             'outlier' (erroneous return -> permanent level shift),
             'noise'   (microstructure noise on log price, std = level*sigma)."""
    r = r.copy()
    s = r.std()
    if kind == 'noise':
        xi = rng.randn(len(r) + 1) * level * s
        return r + np.diff(xi)
    m = max(1, int(round(level * len(r))))
    idx = rng.choice(np.arange(1, len(r) - 1), m, replace=False)
    sgn = rng.choice([-1.0, 1.0], m)
    if kind == 'badtick':
        r[idx] += sgn * kappa * s
        r[idx + 1] -= sgn * kappa * s
    elif kind == 'outlier':
        r[idx] += sgn * kappa * s
    return r


def hdev(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


LEVELS = {'badtick': (0.001, 0.0025, 0.005, 0.01),
          'outlier': (0.001, 0.0025, 0.005, 0.01),
          'noise': (0.1, 0.25, 0.5, 1.0)}


def contamination_study(r, n_rep, seed):
    base = slim(analyse(r))
    rows = []
    rng = np.random.RandomState(seed)
    for kind, levels in LEVELS.items():
        for lev in levels:
            for rep in range(n_rep):
                a = slim(analyse(contaminate(r, kind, lev, rng)))
                for m in a:
                    rows.append(dict(kind=kind, level=lev, rep=rep, method=m,
                                     dev_h=hdev(a[m]['hq'], base[m]['hq']),
                                     d_dalpha=a[m]['dalpha'] - base[m]['dalpha'],
                                     d_h2=a[m]['h2'] - base[m]['h2']))
    return base, rows


# ------------------------------------------------------------------ parts
def part_A_B():
    out = {}
    for nm in SERIES:
        d, r = returns(nm)
        z = (r - np.median(r)) / (1.4826 * np.median(np.abs(r - np.median(r))))
        desc = dict(label=LABEL[nm], start=str(d[0]), end=str(d[-1]), N=len(r),
                    mean=float(r.mean()), std=float(r.std()),
                    skew=float(((r - r.mean()) ** 3).mean() / r.std() ** 3),
                    exkurt=float(((r - r.mean()) ** 4).mean() / r.std() ** 4 - 3),
                    zero_frac=float(np.mean(r == 0)), max_absz=float(np.abs(z).max()),
                    acf1_abs=float(np.corrcoef(np.abs(r[1:]), np.abs(r[:-1]))[0, 1]))
        orig = analyse(r)
        rng = np.random.RandomState(2024)
        sh = [slim(analyse(shuffled(r, rng))) for _ in range(5)]
        pr = [slim(analyse(phase_rand(r, rng))) for _ in range(5)]
        surr = {m: dict(shuf_dalpha=float(np.mean([s[m]['dalpha'] for s in sh])),
                        shuf_dalpha_sd=float(np.std([s[m]['dalpha'] for s in sh])),
                        phase_dalpha=float(np.mean([s[m]['dalpha'] for s in pr])),
                        phase_dalpha_sd=float(np.std([s[m]['dalpha'] for s in pr])),
                        shuf_h2=float(np.mean([s[m]['h2'] for s in sh])))
                for m in orig}
        out[nm] = dict(desc=desc, orig=orig if nm in ('SP500', 'SSEC', 'BTC') else slim(orig),
                       surr=surr)
        print(f'  A/B {nm} N={len(r)} done', flush=True)
    return out


def part_C():
    """Leave-k-out of the largest |r| days (genuine extreme events)."""
    out = {}
    for nm in SERIES:
        d, r = returns(nm)
        base = slim(analyse(r))
        order = np.argsort(-np.abs(r - np.median(r)))
        rec = {}
        for k in (1, 5, 10, 20, 50):
            keep = np.ones(len(r), bool); keep[order[:k]] = False
            a = slim(analyse(r[keep]))
            rec[k] = {m: dict(dev_h=hdev(a[m]['hq'], base[m]['hq']),
                              d_dalpha=a[m]['dalpha'] - base[m]['dalpha'])
                      for m in a}
        out[nm] = dict(top_days=[str(d[i]) for i in order[:5]],
                       top_r=[float(r[i]) for i in order[:5]], rec=rec)
        print(f'  C {nm} done', flush=True)
    return out


def part_D(n_rep=5):
    out = {}
    for i, nm in enumerate(SERIES):
        d, r = returns(nm)
        base, rows = contamination_study(r, n_rep, seed=100 + i)
        out[nm] = dict(rows=rows)
        print(f'  D {nm} done', flush=True)
    return out


def part_E(n_rep=3):
    out = {}
    for i, (tick, r) in enumerate(dj_constituents().items()):
        base, rows = contamination_study(r, n_rep, seed=500 + i)
        out[tick] = dict(N=len(r), base={m: dict(dalpha=base[m]['dalpha'], h2=base[m]['h2'],
                                                 r2=base[m]['r2']) for m in base},
                         rows=rows)
        print(f'  E {tick} done', flush=True)
    return out


def part_F(win=2500, step=125):
    d, r = returns('SP500')
    rng = np.random.RandomState(77)
    rc = contaminate(r, 'badtick', 0.005, rng)
    out = dict(t=[], clean={}, dirty={})
    for s in range(0, len(r) - win + 1, step):
        out['t'].append(str(d[s + win - 1]))
        for tag, x in (('clean', r), ('dirty', rc)):
            a = slim(analyse(x[s:s + win]))
            for m in ('MFDFA', 'MFDMA', 'MFQDMA_0.1', 'MFQDMA_0.25', 'MFQDMA_0.5'):
                out[tag].setdefault(m, []).append(a[m]['dalpha'])
    print('  F done', flush=True)
    return out


if __name__ == '__main__':
    which = sys.argv[1:] or ['AB', 'C', 'D', 'E', 'F']
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    t0 = time.time()
    for w in which:
        fn = {'AB': part_A_B, 'C': part_C, 'D': part_D, 'E': part_E, 'F': part_F}[w]
        res[w] = fn()
        json.dump(res, open(OUT, 'w'))
        print(f'{w} saved {time.time() - t0:.0f}s', flush=True)
