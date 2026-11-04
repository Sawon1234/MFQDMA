"""Experiments of Paper I. Each function recomputes one file in results/.

The notebook MFQDMA_PaperI.ipynb calls these functions when the corresponding
RECOMPUTE flag is set. Run times are rough estimates for a free Colab runtime. Seeds are fixed, so the stored results are reproduced.
"""
import json
import time

import numpy as np

from . import paths

RES = paths.RESULTS


def _dump(obj, name):
    with open(RES / name, 'w') as f:
        json.dump(obj, f)
    print('saved results/' + name, flush=True)


# ------------------------------------------------------------- Tables II, III
def run_corpus_1d():
    """Binomial measures (8) and fGn (18) of the benchmark corpus, every
    estimator of Tables II and III. Writes results/bench_1d.json (~10 min)."""
    from .generators import binomial_measure, binomial_tau_theory, fgn
    from .methods import mfdfa, mfdma, mfqdma, mfqdma_multi

    qsel = np.arange(-4, 4.5, 0.5)

    def scales_for(N, family):         # DFA prefers larger scales than DMA [Jiang & Zhou 2011]
        lo, hi = (64, min(N // 4, 8192)) if family == 'dfa' else (16, min(N // 8, 2048))
        s = np.unique(np.logspace(np.log10(lo), np.log10(hi), 14).astype(int))
        return s[s >= lo]
    taus = [0.10, 0.25, 0.50, 0.75, 0.90]
    thetas = [0.0, 0.5, 1.0]

    def rmse(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() else np.nan

    def estimators(x, dem):
        sD, sM = scales_for(len(x), 'dfa'), scales_for(len(x), 'dma')
        m0 = mfqdma_multi(x, qsel, scales=sM, theta=0.0, taus=taus, demean=dem)
        yield 'MFDFA', mfdfa(x, qsel, scales=sD, demean=dem)
        for t in thetas:
            yield f'MFDMA_th{t}', mfdma(x, qsel, scales=sM, theta=t, demean=dem)
        for tt in taus:
            yield f'MFQDMA_th0_tau{tt}', m0[tt]
        for t in (0.5, 1.0):
            yield f'MFQDMA_th{t}_tau0.5', mfqdma(x, qsel, scales=sM, theta=t, tau=0.5, demean=dem)

    rows, t0 = [], time.time()
    for p in (0.25, 0.30, 0.35, 0.40):                       # 01 binomial measures
        for k in (14, 16):
            x = binomial_measure(p, k)
            th = binomial_tau_theory(qsel, p)
            for nm, r in estimators(x, False):
                rows.append(dict(group='01_binomial_1d', file=f'binomial_p{int(p*100)}_k{k}',
                                 p=p, k=k, method=nm, metric='tau', rmse=rmse(r['tau_q'], th),
                                 r2=float(np.nanmean(r['r2'])), est=r['tau_q'].tolist()))
        print(f'binomial p={p} done {time.time()-t0:.0f}s', flush=True)
    for H in (0.2, 0.3, 0.5, 0.7, 0.8, 0.9):                 # 06 fractional Gaussian noise
        for sd in range(3):
            x = fgn(2 ** 16, H, seed=3000 + sd)
            th = np.full(len(qsel), H)
            for nm, r in estimators(x, True):
                rows.append(dict(group='06_monofractal', file=f'fgn_H{int(H*10)}_s{sd}',
                                 H=H, seed=sd, method=nm, metric='h', rmse=rmse(r['hq'], th),
                                 r2=float(np.nanmean(r['r2'])), est=r['hq'].tolist()))
        print(f'fGn H={H} done {time.time()-t0:.0f}s', flush=True)
    _dump({'q': qsel.tolist(), 'auto_1d': rows}, 'bench_1d.json')


# ------------------------------------------------------- Table IV, Figs. 5, 6
def run_advantage():
    """Contamination, alpha-stable noise, monotone maps, asymmetric cascades.
    Writes results/advantage_all.json (~45 min)."""
    from . import advantage as A
    out = {}
    out['contamination'] = A.test_contamination(); print('contamination done', flush=True)
    out['levy'] = A.test_levy(); print('levy done', flush=True)
    out['monotonic'] = A.test_monotonic(); print('monotonic done', flush=True)
    out['asymmetry'] = A.test_asymmetry(); print('asymmetry done', flush=True)
    _dump(out, 'advantage_all.json')


# --------------------------------------------------- Table V, Figs. 8, 9
def run_analysis_2d():
    """2D cascades (theta x lambda), raw fBm surfaces, contaminated 2D cascades.
    Writes results/analysis_2d_all.json (~25 min)."""
    from . import analysis_2d as A
    out, t0 = {}, time.time()
    for part in ('A', 'B', 'C'):
        out[part] = getattr(A, 'part_' + part)()
        print(f'{part} done {time.time()-t0:.0f}s', flush=True)
    _dump(out, 'analysis_2d_all.json')


# ------------------------------------------------------- Table VI, Fig. 9
def run_2d_mono():
    """Mixed increments of fractional Brownian surfaces (H = 0.2..0.9).
    Writes results/paper_2d_mono.json (~5 min)."""
    from .generators import fbm_surface
    from .methods import mfdma_2d
    from .analysis_2d import sc2d
    Q = np.arange(-4, 4.5, 0.5)
    methods = [('MFDMA2D_th0', dict(theta=0.0, tau=None)),
               ('MFQDMA2D_th0_tau0.25', dict(theta=0.0, tau=0.25)),
               ('MFQDMA2D_th0_tau0.5', dict(theta=0.0, tau=0.5)),
               ('MFQDMA2D_th0_tau0.75', dict(theta=0.0, tau=0.75))]
    N = 256
    rows, timing = [], {}
    s = sc2d(N - 1, lo=6, hi=32, n=8)
    for H in (0.2, 0.3, 0.5, 0.7, 0.9):
        for sd in (0, 1, 2):
            X = fbm_surface(N, H, seed=7100 + sd)
            G = np.diff(np.diff(X, axis=0), axis=1)
            for nm, kw in methods:
                t0 = time.time()
                r = mfdma_2d(G, Q, scales=s, demean=True, **kw)
                timing.setdefault(nm, []).append(time.time() - t0)
                hq = r['hq']
                rows.append(dict(H=H, seed=sd, method=nm, hq=hq.tolist(),
                                 h2=float(hq[np.argmin(np.abs(Q - 2))]),
                                 rmse_h=float(np.sqrt(np.nanmean((hq - H) ** 2))),
                                 flat=float(np.nanmax(hq) - np.nanmin(hq)),
                                 r2=float(np.nanmean(r['r2'])),
                                 Fq=(r['Fq'].tolist() if sd == 0 else None),
                                 scales=s.tolist()))
            print(f'H={H} done', flush=True)
    _dump(dict(rows=rows, timing={k: float(np.median(v)) for k, v in timing.items()},
               q=Q.tolist(), N=N), 'paper_2d_mono.json')


# -------------------------------------------- Tables VII, VIII, Figs. 10-13
def run_finance(parts=('AB', 'C', 'F', 'D', 'E')):
    """18 assets and 29 DJIA stocks (qrmdata, cloned on first use).
    Writes results/finance.json (~15 min)."""
    from . import finance as Fn
    paths.ensure_qrmdata()
    out_file = RES / 'finance.json'
    res = json.load(open(out_file)) if out_file.exists() else {}
    fns = {'AB': Fn.part_A_B, 'C': Fn.part_C, 'D': Fn.part_D, 'E': Fn.part_E, 'F': Fn.part_F}
    t0 = time.time()
    for w in parts:
        res[w] = fns[w]()
        json.dump(res, open(out_file, 'w'))
        print(f'{w} saved {time.time() - t0:.0f}s', flush=True)


# ------------------------------------------------------- Bonn EEG (data needed)
SETS = ['Z', 'O', 'N', 'F', 'S']
EEG_SCALES = np.unique(np.logspace(np.log10(12), np.log10(128), 12).astype(int))


def run_eeg_surrogates():
    """Originals vs shuffled and phase-randomized surrogates, 20 segments per set.
    Writes results/real_eeg.json (Table X, Fig. 14; ~2 min)."""
    from .eeg import load_bonn
    from .realworld import run_series_set, Q, TAUS
    SD = np.unique(np.logspace(np.log10(16), np.log10(256), 12).astype(int))
    d = load_bonn()
    out, t0 = [], time.time()
    for s in SETS:
        out += run_series_set(s, d[s], EEG_SCALES, SD, demean=True, n_sig=20, n_surr=1, seed0=11)
        print(f'{s} done {time.time()-t0:.0f}s', flush=True)
    _dump({'q': Q.tolist(), 'taus': TAUS, 'scales_dma': EEG_SCALES.tolist(),
           'scales_dfa': SD.tolist(), 'records': out}, 'real_eeg.json')


def run_eeg_main():
    """MFDMA h(q) and MFQDMA h(q, lambda) of all 500 segments.
    Writes results/eeg_main.npz (Figs. 15, 16; ~3 min)."""
    from .eeg import load_bonn, mfqdma, mfdma, preprocess
    d = load_bonn()
    Q = np.arange(-5, 5.5, 0.5)
    TAUS = [0.10, 0.25, 0.50, 0.75, 0.90]
    H = {s: {t: np.full((100, len(Q)), np.nan) for t in TAUS} for s in SETS}
    R2 = {s: {t: np.full((100, len(Q)), np.nan) for t in TAUS} for s in SETS}
    AL = {s: {t: np.full((100, len(Q)), np.nan) for t in TAUS} for s in SETS}
    HM = {s: np.full((100, len(Q)), np.nan) for s in SETS}
    AM = {s: np.full((100, len(Q)), np.nan) for s in SETS}
    t0 = time.time()
    for s in SETS:
        for i in range(100):
            x = preprocess(d[s][i], 'minimal')
            rm = mfdma(x, q_list=Q, scales=EEG_SCALES)
            HM[s][i] = rm['hq']; AM[s][i] = rm['alpha']
            for t in TAUS:
                r = mfqdma(x, tau=t, q_list=Q, scales=EEG_SCALES)
                H[s][t][i] = r['hq']; R2[s][t][i] = r['r2']; AL[s][t][i] = r['alpha']
        print(f'{s} done {time.time()-t0:.0f}s', flush=True)
    np.savez_compressed(RES / 'eeg_main.npz', Q=Q, SC=EEG_SCALES, TAUS=np.array(TAUS),
                        **{f'H_{s}_{t}': H[s][t] for s in SETS for t in TAUS},
                        **{f'R2_{s}_{t}': R2[s][t] for s in SETS for t in TAUS},
                        **{f'AL_{s}_{t}': AL[s][t] for s in SETS for t in TAUS},
                        **{f'HM_{s}': HM[s] for s in SETS},
                        **{f'AM_{s}': AM[s] for s in SETS})
    print('saved results/eeg_main.npz', flush=True)


def run_eeg_classification():
    """Logistic regression on 21 exponents, 10x10 repeated stratified CV (seed 1).
    Reads results/eeg_main.npz, writes results/eeg_auc.json (Table XI; ~1 min)."""
    from scipy.stats import wilcoxon
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import roc_auc_score

    z = np.load(RES / 'eeg_main.npz')
    TAUS = [float(t) for t in z['TAUS']]
    H = {s: {t: z[f'H_{s}_{t}'] for t in TAUS} for s in SETS}
    HM = {s: z[f'HM_{s}'] for s in SETS}
    y = np.r_[np.ones(100, int), np.zeros(100, int)]

    def fold_aucs(X, seed=1):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=seed)
        out = []
        for tr, te in cv.split(X, y):
            p = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                              LogisticRegression(max_iter=5000))
            p.fit(X[tr], y[tr])
            out.append(roc_auc_score(y[te], p.predict_proba(X[te])[:, 1]))
        return np.array(out)

    def feats(s, kind, rng=None):
        if kind == 'base':
            return HM[s]
        if kind == 'all5':
            return np.hstack([H[s][t] for t in TAUS])
        if kind == 'tau3':
            return np.hstack([H[s][t] for t in (0.25, 0.5, 0.75)])
        if kind == 'noise5':
            b = HM[s]
            return np.hstack([b] + [b + rng.normal(0, b.std() * 0.05, b.shape) for _ in range(4)])
        return H[s][kind]

    res = {}
    for a, b in [('F', 'N'), ('S', 'F'), ('Z', 'S'), ('Z', 'O')]:
        base = fold_aucs(np.vstack([feats(a, 'base'), feats(b, 'base')]))
        r = {'base': float(base.mean()), 'per_lambda': {}, 'paired': {}}
        for t in TAUS:
            r['per_lambda'][str(t)] = float(fold_aucs(np.vstack([feats(a, t), feats(b, t)])).mean())
        for kind in ('all5', 'tau3', 'noise5'):
            rng = np.random.RandomState(0)
            f = fold_aucs(np.vstack([feats(a, kind, rng), feats(b, kind, rng)]))
            r['paired'][kind] = dict(auc=float(f.mean()), gain=float(f.mean() - base.mean()),
                                     p=float(wilcoxon(f, base).pvalue))
        res[f'{a} vs {b}'] = r
        print(f'{a} vs {b}: MFDMA {r["base"]:.3f}', flush=True)
    with open(RES / 'eeg_auc.json', 'w') as f:
        json.dump(res, f, indent=1)
    print('saved results/eeg_auc.json', flush=True)


PREPROC = [('zscore', '$z$ scoring'), ('notch50', '50 Hz notch'),
           ('detrend', 'Linear detrending'), ('downsample', 'Downsampling by 2'),
           ('bp_0p5_70', 'FIR high-pass, 0.5 Hz'), ('bp_1_40', 'FIR band-pass, 1-40 Hz')]


def run_eeg_preprocessing():
    """Sensitivity of h(q, lambda) to preprocessing, 20 segments per set.
    Writes results/eeg_preproc.json (Table IX; ~2 min)."""
    from .eeg import load_bonn, mfqdma, preprocess
    d = load_bonn()
    q = np.arange(-5, 5.5, 0.5)
    taus = [0.25, 0.5, 0.75]
    res = {}
    for mode in ['minimal'] + [m for m, _ in PREPROC]:
        acc = {t: [] for t in taus}
        for s in SETS:
            for i in range(0, 100, 5):
                xp = preprocess(d[s][i], mode)
                sc = EEG_SCALES if mode != 'downsample' else np.unique((EEG_SCALES / 2).astype(int))
                sc = sc[sc >= 6]
                for t in taus:
                    acc[t].append(mfqdma(xp, tau=t, q_list=q, scales=sc)['hq'])
        res[mode] = {str(t): np.array(acc[t]) for t in taus}
        print(f'{mode} done', flush=True)
    ref = res['minimal']
    out = {mode: {t: float(np.nanmean(np.abs(res[mode][t] - ref[t]))) for t in ref}
           for mode in res}
    _dump({'mean_abs_dev': out, 'q': q.tolist(), 'n_segments': 100}, 'eeg_preproc.json')


# ------------------------------------------------------- Table XII, Fig. 17
def run_dem():
    """Terrain elevation model shipped with Matplotlib: raw surface, mixed
    increments, and shuffled / phase-randomized surrogates of the increments.
    Writes results/dem.json (~3 min)."""
    from .methods import mfdma_2d
    from .analysis_2d import sc2d, load_real_fields
    Q = np.arange(-4, 4.5, 0.5)
    i2 = int(np.argmin(np.abs(Q - 2)))
    F = load_real_fields()['dem_terrain']
    G = np.diff(np.diff(F, axis=0), axis=1)
    s_raw = sc2d(min(F.shape), lo=6, hi=max(12, min(F.shape) // 8), n=8)
    s = sc2d(min(G.shape), lo=6, hi=max(12, min(G.shape) // 8), n=8)
    rng = np.random.RandomState(5)                     # same surrogates as Fig. 17
    Gs = rng.permutation(G.ravel()).reshape(G.shape)
    Fg = np.fft.fft2(G - G.mean())
    ph = np.angle(np.fft.fft2(rng.randn(*G.shape)))
    Gp = np.real(np.fft.ifft2(np.abs(Fg) * np.exp(1j * ph)))

    def width(a):
        a = np.asarray(a)[np.isfinite(a)]
        return float(a.max() - a.min()) if len(a) else float('nan')

    methods = [('MFDMA2D_th0', dict(theta=0.0, tau=None)),
               ('MFDMA2D_th0.5', dict(theta=0.5, tau=None)),
               ('MFQDMA2D_tau0.25', dict(theta=0.0, tau=0.25)),
               ('MFQDMA2D_tau0.5', dict(theta=0.0, tau=0.5)),
               ('MFQDMA2D_tau0.75', dict(theta=0.0, tau=0.75))]
    out = {}
    for nm, kw in methods:
        rr = mfdma_2d(F, Q, scales=s_raw, demean=True, **kw)
        ri = mfdma_2d(G, Q, scales=s, demean=True, **kw)
        rec = dict(h2_raw=float(rr['hq'][i2]), h2_incr=float(ri['hq'][i2]),
                   dalpha_incr=width(ri['alpha']), r2_incr=float(np.nanmean(ri['r2'])))
        if kw['theta'] == 0.0:
            rec['dalpha_shuf'] = width(mfdma_2d(Gs, Q, scales=s, demean=True, **kw)['alpha'])
            rec['dalpha_phase'] = width(mfdma_2d(Gp, Q, scales=s, demean=True, **kw)['alpha'])
        out[nm] = rec
        print(nm, {k: round(v, 3) for k, v in rec.items()}, flush=True)
    _dump({'shape_raw': list(F.shape), 'shape_incr': list(G.shape), 'scales': s.tolist(),
           'methods': out}, 'dem.json')
