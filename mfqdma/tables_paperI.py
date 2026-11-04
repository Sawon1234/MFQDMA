"""Data tables of Paper I, generated from the stored results in results/.

Every function returns a pandas DataFrame laid out like the manuscript table
and writes the LaTeX body rows to tables_out/<name>.tex, so that no number in
the paper is transcribed by hand. Table I (corpus) and Table XIII (guidelines)
contain no computed numbers and are not generated.
"""
import json

import numpy as np
import pandas as pd

from . import paths

RES = paths.RESULTS
OUT = paths.REPO / 'tables_out'
OUT.mkdir(exist_ok=True)
TAUS = [0.1, 0.25, 0.5, 0.75, 0.9]


def _J(name):
    with open(RES / name) as f:
        return json.load(f)


def _write(df, name, index=True):
    """Write the table body as LaTeX rows (cells joined by &)."""
    lines = []
    for idx, row in df.iterrows():
        cells = ([str(idx) if not isinstance(idx, tuple) else ' & '.join(map(str, idx))] if index else []) \
            + [str(v) for v in row.values]
        lines.append(' & '.join(cells) + r' \\')
    (OUT / f'{name}.tex').write_text('\n'.join(lines) + '\n')
    return df


def _pm(v):
    return f'{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}'


# ---------------------------------------------------------------- Tables II, III
def table2_binomial():
    """Table II: RMSE of tau(q) over the eight binomial measures."""
    D = pd.DataFrame(_J('bench_1d.json')['auto_1d'])
    D = D[D.group == '01_binomial_1d']
    order = [('MFDFA', 'MFDFA', '---'), ('MFDMA_th0.0', 'MFDMA', '0'), ('MFDMA_th0.5', 'MFDMA', '1/2'),
             ('MFDMA_th1.0', 'MFDMA', '1')] + \
        [(f'MFQDMA_th0_tau{t}', f'MFQDMA, λ={t:.2f}', '0') for t in TAUS] + \
        [('MFQDMA_th1.0_tau0.5', 'MFQDMA, λ=0.50', '1'), ('MFQDMA_th0.5_tau0.5', 'MFQDMA, λ=0.50', '1/2')]
    rows = [(lb, th, _pm(D[D.method == m].rmse)) for m, lb, th in order]
    df = pd.DataFrame(rows, columns=['Method', 'θ', 'RMSE_τ']).set_index('Method')
    return _write(df, 'table02_binomial1d')


def table3_fgn():
    """Table III: RMSE of h(q) versus H over 18 fGn series."""
    D = pd.DataFrame(_J('bench_1d.json')['auto_1d'])
    D = D[D.group == '06_monofractal']
    order = [('MFDFA', 'MFDFA', '---'), ('MFDMA_th0.0', 'MFDMA', '0'), ('MFDMA_th0.5', 'MFDMA', '1/2'),
             ('MFDMA_th1.0', 'MFDMA', '1')] + \
        [(f'MFQDMA_th0_tau{t}', f'MFQDMA, λ={t:.2f}', '0') for t in TAUS] + \
        [('MFQDMA_th0.5_tau0.5', 'MFQDMA, λ=0.50', '1/2')]
    rows = [(lb, th, _pm(D[D.method == m].rmse)) for m, lb, th in order]
    df = pd.DataFrame(rows, columns=['Method', 'θ', 'RMSE_h']).set_index('Method')
    return _write(df, 'table03_fgn')


# ---------------------------------------------------------------- Table IV
def table4_contamination():
    """Table IV: RMSE of tau(q), contaminated binomial measure (mean of 5 replicates)."""
    D = pd.DataFrame(_J('advantage_all.json')['contamination'])
    meths = [('MFDFA', 'MFDFA'), ('MFDMA', 'MFDMA (θ=0)')] + \
        [(f'MFQDMA_tau{t}', f'MFQDMA λ={t:.2f}') for t in TAUS]
    cols = [(k, f) for k in ('outlier', 'student_t', 'levy') for f in (0.01, 0.02, 0.05, 0.10)]
    P = D.pivot_table(index='method', columns=['kind', 'frac'], values='rmse', aggfunc='mean')
    df = pd.DataFrame({f'{k} {int(100*f)}%': [f'{P.loc[m, (k, f)]:.3f}' for m, _ in meths] for k, f in cols},
                      index=[lb for _, lb in meths])
    clean = P[('outlier', 0.0)]
    print('clean values (ε=0):', ', '.join(f'{lb} {clean[m]:.3f}' for m, lb in meths))
    return _write(df, 'table04_contamination')


# ---------------------------------------------------------------- Table V
def table5_cascade2d():
    """Table V: RMSE of tau(q) of 2D cascades, backward windows."""
    A = pd.DataFrame(_J('analysis_2d_all.json')['A'])
    meths = [('MFDMA2D_th0.0', 'Moving mean')] + \
        [(f'MFQDMA2D_th0.0_tau{t}', f'λ={t:.2f}') for t in (0.25, 0.5, 0.75)]
    cols = [(k, p) for k in (8, 9) for p in ('A', 'B', 'C')]
    df = pd.DataFrame({f'{2**k}² {p}': [f"{A[(A.method == m) & (A.k == k) & (A.pset == p)].rmse.mean():.3f}"
                                         for m, _ in meths] for k, p in cols},
                      index=[lb for _, lb in meths])
    return _write(df, 'table05_cascade2d')


# ---------------------------------------------------------------- Table VI
def table6_mono2d():
    """Table VI: h(2) of the mixed increments of fBm surfaces, mean(sd) of 3 realizations."""
    D = pd.DataFrame(_J('paper_2d_mono.json')['rows'])
    meths = [('MFDMA2D_th0', '2D-MFDMA'), ('MFQDMA2D_th0_tau0.25', 'λ=0.25'),
             ('MFQDMA2D_th0_tau0.5', 'λ=0.5'), ('MFQDMA2D_th0_tau0.75', 'λ=0.75')]

    def cell(v):
        return f'{v.mean():.3f}({int(round(1000 * v.std(ddof=1)))})'
    rows = {H: [cell(D[(D.H == H) & (D.method == m)].h2) for m, _ in meths] for H in sorted(D.H.unique())}
    df = pd.DataFrame(rows, index=[lb for _, lb in meths]).T
    df.index.name = 'H'
    return _write(df, 'table06_mono2d')


# ---------------------------------------------------------------- Tables VII, VIII
def table7_markets():
    """Table VII: 18 assets, widths of the clean data and retention by surrogates."""
    from .finance_tables import SER, load
    AB = load()['AB']
    rows = []
    for s in SER:
        d = AB[s]['desc']; o = AB[s]['orig']; sr = AB[s]['surr']
        rows.append([d['label'], f"{d['start'][:4]}--{d['end'][2:4]}", d['N'], f"{d['exkurt']:.1f}"]
                    + [f"{o[m]['dalpha']:.2f}" for m in ('MFDFA', 'MFDMA', 'MFQDMA_0.1', 'MFQDMA_0.25', 'MFQDMA_0.5')]
                    + [f"{100 * sr[m]['shuf_dalpha'] / o[m]['dalpha']:.0f}" for m in ('MFDMA', 'MFQDMA_0.1')]
                    + [f"{100 * sr[m]['phase_dalpha'] / o[m]['dalpha']:.0f}" for m in ('MFDMA', 'MFQDMA_0.1')])
    df = pd.DataFrame(rows, columns=['Asset', 'Period', 'N', 'κ', 'Δα MFDFA', 'Δα MFDMA', 'Δα λ=0.1',
                                     'Δα λ=0.25', 'Δα λ=0.5', 'shuf% MFDMA', 'shuf% λ=0.1',
                                     'phase% MFDMA', 'phase% λ=0.1']).set_index('Asset')
    return _write(df, 'table07_markets')


def table8_finrobust():
    """Table VIII: sensitivity D_h (x 10^3) to extremes and injected errors."""
    from .finance_tables import robust_summary, M, MLAB, load
    load()
    res, _ = robust_summary()
    rows = []
    for m in M:
        r = res[m]
        rows.append([MLAB[m].replace('$\\lam=', 'λ=').replace('$\\theta=0$', 'θ=0').replace('$', ''),
                     f"{1e3 * r['C5']:.1f}", f"{r['C5da']:.3f}",
                     f"{1e3 * r['D_badtick']:.1f}", f"{1e3 * r['E_badtick']:.1f}",
                     f"{1e3 * r['D_outlier']:.1f}", f"{1e3 * r['E_outlier']:.1f}",
                     f"{1e3 * r['D_noise']:.1f}", f"{1e3 * r['E_noise']:.1f}",
                     f"{r['F']:.3f}" if 'F' in r else '---'])
    df = pd.DataFrame(rows, columns=['Method', 'Top5 D_h', 'Top5 |δΔα|', 'Bad ticks Mkts', 'Bad ticks Stocks',
                                     'Outliers Mkts', 'Outliers Stocks', 'Noise Mkts', 'Noise Stocks',
                                     'Rolling |δΔα|']).set_index('Method')
    return _write(df, 'table08_finrobust')


# ---------------------------------------------------------------- Tables IX-XI (EEG)
def table9_preprocessing():
    """Table IX: mean |Δh(q,1/2)| caused by each preprocessing step."""
    from .experiments_paperI import PREPROC
    d = _J('eeg_preproc.json')['mean_abs_dev']
    df = pd.DataFrame([(lb, f"{d[m]['0.5']:.4f}") for m, lb in PREPROC],
                      columns=['Operation', 'mean |Δh(q,1/2)|']).set_index('Operation')
    return _write(df, 'table09_preprocessing')


def table10_eeg():
    """Table X: Bonn EEG widths of the originals and retention by surrogates."""
    D = _J('real_eeg.json')
    recs = D['records']; sets = ['Z', 'O', 'N', 'F', 'S']
    meths = [('MFDFA', 'MFDFA'), ('MFDMA_th0', 'MFDMA')] + \
        [(f'MFQDMA_tau{t}', f'MFQDMA λ={t:.2f}') for t in TAUS]

    def W(r, c, m):
        return r['original'][m]['width'] if c == 'original' else float(np.mean([s[m]['width'] for s in r[c]]))
    rows = []
    for m, lb in meths:
        orig = [np.mean([W(r, 'original', m) for r in recs if r['set'] == s]) for s in sets]
        cells = [f'{v:.3f}' for v in orig]
        for cond in ('shuffled', 'phase_rand'):
            cells += [f"{100 * np.mean([W(r, cond, m) for r in recs if r['set'] == s]) / o:.0f}"
                      for s, o in zip(sets, orig)]
        rows.append([lb] + cells)
    cols = ['Method'] + [f'Δα {s}' for s in sets] + [f'shuf% {s}' for s in sets] + [f'phase% {s}' for s in sets]
    return _write(pd.DataFrame(rows, columns=cols).set_index('Method'), 'table10_eeg_surrogates')


def table11_auc():
    """Table XI: cross-validated AUC, MFDMA h(q) vs best single-lambda MFQDMA h(q, lambda)."""
    A = _J('eeg_auc.json')
    rows = []
    for key in ('F vs N', 'S vs F', 'Z vs S', 'Z vs O'):
        r = A[key]; pl = {float(k): v for k, v in r['per_lambda'].items()}
        best = max(pl.values())
        lams = [k for k, v in pl.items() if round(v, 3) == round(best, 3)]
        rows.append([key, f"{r['base']:.3f}", f'{best:.3f}', '/'.join(f'{t:.2f}' for t in sorted(lams))])
    df = pd.DataFrame(rows, columns=['Contrast', 'MFDMA h(q)', 'MFQDMA h(q,λ*)', 'λ*']).set_index('Contrast')
    return _write(df, 'table11_auc')


# ---------------------------------------------------------------- Table XII
def table12_dem():
    """Table XII: terrain elevation model (raw surface, increments, surrogates)."""
    d = _J('dem.json')['methods']
    rows = []
    for m, lb in (('MFDMA2D_th0', 'MFDMA'), ('MFDMA2D_th0.5', 'MFDMA θ=1/2'), ('MFQDMA2D_tau0.25', 'λ=0.25'),
                  ('MFQDMA2D_tau0.5', 'λ=0.50'), ('MFQDMA2D_tau0.75', 'λ=0.75')):
        r = d[m]
        rows.append([lb, f"{r['h2_raw']:.3f}", f"{r['h2_incr']:.3f}", f"{r['dalpha_incr']:.3f}",
                     f"{r['dalpha_shuf']:.3f}" if 'dalpha_shuf' in r else '---',
                     f"{r['dalpha_phase']:.3f}" if 'dalpha_phase' in r else '---', f"{r['r2_incr']:.3f}"])
    df = pd.DataFrame(rows, columns=['Trend', 'h(2) raw', 'h(2) incr.', 'Δα orig.', 'Δα shuf.',
                                     'Δα phase', 'R²']).set_index('Trend')
    return _write(df, 'table12_dem')


ALL = [('Table II', table2_binomial), ('Table III', table3_fgn), ('Table IV', table4_contamination),
       ('Table V', table5_cascade2d), ('Table VI', table6_mono2d), ('Table VII', table7_markets),
       ('Table VIII', table8_finrobust), ('Table IX', table9_preprocessing), ('Table X', table10_eeg),
       ('Table XI', table11_auc), ('Table XII', table12_dem)]
