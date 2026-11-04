"""Real-world multifractal analysis in the PRE house style.

Follows the real-data sections of
  [P1] Gu & Zhou PRE 82, 011136 (2010)  Sec. VI  (SSEC index)
  [P3] Jiang & Zhou PRE 84, 016106 (2011) Sec. IV (DJIA / NASDAQ)

i.e. for each series: log-log fluctuation functions, h(q), tau(q), f(alpha),
method comparison, and the SOURCE-OF-MULTIFRACTALITY surrogate tests:

  shuffled   -> destroys all temporal correlations, preserves the marginal
                distribution.  Residual multifractality is tail-driven.
  phase-rand -> preserves the linear (second-order) correlations, destroys
                nonlinear ones and Gaussianises the marginal.

Works on any 1-D series, so the same driver handles EEG and, when the CSVs
are supplied, the stock indices of [P1]/[P3].
"""
import sys, os, json
import numpy as np
from .methods import mfdfa, mfdma, mfqdma_multi

Q = np.arange(-5, 5.5, 0.5)
TAUS = [0.10, 0.25, 0.50, 0.75, 0.90]


# ------------------------------------------------------------- surrogates
def shuffled(x, seed=0):
    r = np.random.RandomState(seed)
    y = np.asarray(x, float).copy()
    r.shuffle(y)
    return y


def phase_randomised(x, seed=0):
    """Fourier phase randomisation (FT surrogate; preserves the power spectrum, Gaussianises amplitudes)."""
    r = np.random.RandomState(seed)
    x = np.asarray(x, float)
    n = len(x)
    F = np.fft.rfft(x)
    ph = r.uniform(0, 2 * np.pi, len(F))
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=n)


def spectrum_width(alpha):
    a = alpha[np.isfinite(alpha)]
    return float(a.max() - a.min()) if len(a) else np.nan


# ------------------------------------------------------------- one series
def analyse(x, scales_dma, scales_dfa, demean=True):
    """Return {method: result} for one series."""
    out = {'MFDFA': mfdfa(x, Q, scales=scales_dfa, demean=demean),
           'MFDMA_th0': mfdma(x, Q, scales=scales_dma, theta=0.0, demean=demean)}
    multi = mfqdma_multi(x, Q, scales=scales_dma, theta=0.0, taus=TAUS,
                         demean=demean)
    for t in TAUS:
        out[f'MFQDMA_tau{t}'] = multi[t]
    return out


def summarise(res):
    return {m: dict(hq=r['hq'].tolist(),
                    tau_q=r['tau_q'].tolist(),
                    alpha=r['alpha'].tolist(),
                    f_alpha=r['f_alpha'].tolist(),
                    width=spectrum_width(r['alpha']),
                    r2=float(np.nanmean(r['r2'])))
            for m, r in res.items()}


def run_series_set(name, X, scales_dma, scales_dfa, demean=True,
                   n_sig=None, n_surr=1, seed0=0):
    """X: (n_signals, N). Returns a list of per-signal records with the
    original and its surrogates."""
    recs = []
    idx = range(len(X)) if n_sig is None else range(min(n_sig, len(X)))
    for i in idx:
        x = X[i]
        rec = {'set': name, 'i': int(i),
               'original': summarise(analyse(x, scales_dma, scales_dfa, demean))}
        sh, pr = [], []
        for s in range(n_surr):
            sh.append(summarise(analyse(shuffled(x, seed0 + 100 * i + s),
                                        scales_dma, scales_dfa, demean)))
            pr.append(summarise(analyse(phase_randomised(x, seed0 + 100 * i + s),
                                        scales_dma, scales_dfa, demean)))
        rec['shuffled'] = sh
        rec['phase_rand'] = pr
        recs.append(rec)
    return recs


# ------------------------------------------------------------- CSV loader
def load_price_csv(path, date_col=None, price_col=None):
    """Load a stock-index CSV and return (dates, log-returns, volatility).

    Accepts the usual layouts (Date,Open,High,Low,Close,... or Date,Value).
    Returns are r(t) = ln P(t) - ln P(t-1); volatility is |r(t)|, exactly the
    definitions used in [P3] Sec. IV.
    """
    import csv
    rows = list(csv.reader(open(path)))
    hdr = [h.strip().lower() for h in rows[0]]
    if price_col is None:
        for cand in ('adj close', 'adjclose', 'close', 'value', 'price',
                     'djia', 'nasdaqcom', 'sp500'):
            if cand in hdr:
                price_col = hdr.index(cand); break
        else:
            price_col = len(hdr) - 1
    else:
        price_col = hdr.index(price_col.lower())
    if date_col is None:
        date_col = 0 if ('date' in hdr[0] or 'time' in hdr[0]) else None

    d, p = [], []
    for r in rows[1:]:
        if not r or len(r) <= price_col:
            continue
        v = r[price_col].strip()
        if v in ('', '.', 'null', 'NA', 'NaN'):
            continue
        try:
            val = float(v)
        except ValueError:
            continue
        if val <= 0:
            continue
        p.append(val)
        d.append(r[date_col] if date_col is not None else len(p))
    p = np.asarray(p, float)
    ret = np.diff(np.log(p))
    return d[1:], ret, np.abs(ret)
