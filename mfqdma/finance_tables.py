"""Generate LaTeX rows and summary numbers for the finance section directly
from results/finance.json (no hand transcription)."""
import json, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from . import paths
J = {}


def load():
    """(Re)load results/finance.json into the module-level J."""
    J.clear()
    J.update(json.load(open(paths.RESULTS / 'finance.json')))
    return J


load()
SER = ['SP500', 'DJ', 'NASDAQ', 'FTSE', 'DAX', 'CAC', 'SMI', 'EURSTOXX', 'NIKKEI', 'HSI',
       'SSEC', 'CSI', 'EUR_USD', 'GBP_USD', 'JPY_USD', 'GOLD', 'OIL_Brent', 'BTC']
M = ['MFDFA', 'MFDMA', 'MFQDMA_0.1', 'MFQDMA_0.25', 'MFQDMA_0.5', 'MFQDMA_0.75', 'MFQDMA_0.9']
MLAB = {'MFDFA': 'MFDFA', 'MFDMA': 'MFDMA ($\\theta=0$)'}
MLAB.update({f'MFQDMA_{t}': f'MFQDMA $\\lam={t}$' for t in (0.1, 0.25, 0.5, 0.75, 0.9)})
out = {}


def table_markets():
    AB = J['AB']
    lines = []
    for s in SER:
        d = AB[s]['desc']; o = AB[s]['orig']; sr = AB[s]['surr']
        lab = d['label'].replace('&', '\\&')
        yrs = f"{d['start'][:4]}--{d['end'][2:4]}"
        da = ' & '.join(f"{o[m]['dalpha']:.2f}" for m in ('MFDFA', 'MFDMA', 'MFQDMA_0.1', 'MFQDMA_0.25', 'MFQDMA_0.5'))
        sh = ' & '.join(f"{100 * sr[m]['shuf_dalpha'] / o[m]['dalpha']:.0f}" for m in ('MFDMA', 'MFQDMA_0.1'))
        ph = ' & '.join(f"{100 * sr[m]['phase_dalpha'] / o[m]['dalpha']:.0f}" for m in ('MFDMA', 'MFQDMA_0.1'))
        lines.append(f"{lab} & {yrs} & {d['N']} & {d['exkurt']:.1f} & {da} & {sh} & {ph} \\\\")
    return '\n'.join(lines)


def long_rows(part, key='series'):
    rows = []
    for s, v in J[part].items():
        for r in v['rows']:
            r = dict(r); r[key] = s; rows.append(r)
    return pd.DataFrame(rows)


def robust_summary():
    C = J['C']
    D = long_rows('D')
    E = long_rows('E', 'stock')
    Fr = J['F']
    res = {}
    for m in M:
        r = {}
        r['C5'] = np.median([C[s]['rec']['5'][m]['dev_h'] for s in SER])
        r['C5da'] = np.median([abs(C[s]['rec']['5'][m]['d_dalpha']) for s in SER])
        for kind, lev in (('badtick', 0.005), ('outlier', 0.005), ('noise', 0.5)):
            g = D[(D.kind == kind) & (D.level == lev) & (D.method == m)].groupby('series').dev_h.mean()
            r[f'D_{kind}'] = g.median()
            g2 = E[(E.kind == kind) & (E.level == lev) & (E.method == m)].groupby('stock').dev_h.mean()
            r[f'E_{kind}'] = g2.median()
        if m in Fr['clean']:
            r['F'] = float(np.mean(np.abs(np.array(Fr['dirty'][m]) - np.array(Fr['clean'][m]))))
        res[m] = r
    # paired tests vs MFDMA
    tests = {}
    for kind, lev in (('badtick', 0.005), ('outlier', 0.005), ('noise', 0.5)):
        for part, df, key in (('D', D, 'series'), ('E', E, 'stock')):
            S = df[(df.kind == kind) & (df.level == lev)].groupby([key, 'method']).dev_h.mean().unstack()
            for m in M:
                if m == 'MFDMA':
                    continue
                w = int((S[m] < S['MFDMA']).sum())
                p = wilcoxon(S[m], S['MFDMA']).pvalue
                tests[(part, kind, m)] = (w, len(S), p, float((S[m] / S['MFDMA']).median()))
    return res, tests


def levels_table():
    D = long_rows('D')
    out = {}
    for kind in ('badtick', 'outlier', 'noise'):
        for m in M:
            g = D[(D.kind == kind) & (D.method == m)].groupby(['series', 'level']).dev_h.mean().groupby('level').median()
            out[(kind, m)] = g.to_dict()
    return out


if __name__ == '__main__':
    print(table_markets())
    if 'D' in J and 'E' in J:
        res, tests = robust_summary()
        print(pd.DataFrame(res).T.round(4))
        for k, v in tests.items():
            print(k, 'wins %d/%d p=%.2g medratio=%.2f' % v)
        lt = levels_table()
        for k, v in lt.items():
            print(k, {kk: round(vv, 4) for kk, vv in v.items()})
