"""
MFQDMA for EEG + MNE-equivalent preprocessing (scipy implementation).

Dataset: Andrzejak et al., Phys. Rev. E 64, 061907 (2001) -- "Bonn" dataset.
  Z (A): healthy, eyes open,  surface EEG
  O (B): healthy, eyes closed, surface EEG
  N (C): interictal, hippocampal formation CONTRALATERAL to epileptogenic zone
  F (D): interictal, WITHIN epileptogenic zone
  S (E): ictal (seizure)
  fs = 173.61 Hz, 4097 samples (23.6 s), 12-bit, already band-pass 0.53-40 Hz.

WHY NOT MNE/EEGLAB/FieldTrip HERE
  Those toolboxes exist for multichannel continuous recordings: montage
  handling, re-referencing, ICA artifact rejection, epoching, EDF/FIF I/O.
  This dataset is single-channel, pre-segmented, pre-filtered ASCII, so none
  of that machinery applies. What remains is FIR filtering, which MNE itself
  implements on top of scipy.signal.firwin. We reproduce MNE's exact default
  filter design below so the pipeline is auditable and reproducible without
  the dependency -- and, more importantly, so filtering can be treated as a
  TESTED VARIABLE rather than an unexamined default (filtering imposes a
  characteristic scale and is known to distort power-law scaling exponents).
"""
import numpy as np
from scipy.signal import firwin, filtfilt, iirnotch, resample_poly

FS = 173.61


# ----------------------------------------------------------------- loading
def load_bonn(root=None):
    """Return {set_name: ndarray (100, 4097)} from data/bonn/{Z,O,N,F,S}."""
    if root is None:
        from .paths import ensure_bonn
        root = str(ensure_bonn())
    import glob, os
    out = {}
    for s in ["Z", "O", "N", "F", "S"]:
        # set N ships with uppercase .TXT -> glob case-insensitively
        files = sorted(glob.glob(os.path.join(root, s, "*.txt")) +
                       glob.glob(os.path.join(root, s, "*.TXT")))
        if not files:
            raise FileNotFoundError(f"no files found for set {s} in {root}")
        arr = np.array([np.loadtxt(f) for f in files], dtype=float)
        if arr.shape != (100, 4097):
            raise ValueError(f"set {s}: expected (100,4097), got {arr.shape}")
        out[s] = arr
    return out


# ------------------------------------------------- MNE-equivalent FIR design
def _mne_trans_bw(l_freq, h_freq, sfreq):
    """MNE's default transition bandwidth rules (mne.filter.create_filter)."""
    l_trans = None if l_freq is None else min(max(l_freq * 0.25, 2.0), l_freq)
    h_trans = None if h_freq is None else min(max(h_freq * 0.25, 2.0),
                                              sfreq / 2.0 - h_freq)
    return l_trans, h_trans


def _mne_filter_length(sfreq, l_trans, h_trans):
    """MNE 'auto' length: 3.3/trans_bw (hamming), odd, >= 1."""
    trans = [t for t in (l_trans, h_trans) if t is not None]
    if not trans:
        return 1
    n = int(round(3.3 * sfreq / min(trans)))
    return n + 1 if n % 2 == 0 else n


def mne_fir_bandpass(x, sfreq=FS, l_freq=None, h_freq=None):
    """Zero-phase FIR filter reproducing MNE's default design
    (hamming window, firwin, filtfilt for zero phase)."""
    l_trans, h_trans = _mne_trans_bw(l_freq, h_freq, sfreq)
    n = _mne_filter_length(sfreq, l_trans, h_trans)
    n = min(n, (len(x) - 1) // 3)          # keep filtfilt stable on 4097 pts
    if n % 2 == 0:
        n += 1
    if n < 9:
        return x.copy()
    nyq = sfreq / 2.0
    if l_freq is not None and h_freq is not None:
        taps = firwin(n, [l_freq / nyq, h_freq / nyq], window="hamming",
                      pass_zero=False)
    elif h_freq is not None:
        taps = firwin(n, h_freq / nyq, window="hamming", pass_zero=True)
    elif l_freq is not None:
        taps = firwin(n, l_freq / nyq, window="hamming", pass_zero=False)
    else:
        return x.copy()
    return filtfilt(taps, [1.0], x)


def notch(x, sfreq=FS, f0=50.0, Q=30.0):
    if f0 >= sfreq / 2:
        return x.copy()
    b, a = iirnotch(f0 / (sfreq / 2), Q)
    return filtfilt(b, a, x)


# ------------------------------------------------------- preprocessing menu
def preprocess(x, mode="minimal", sfreq=FS):
    """Preprocessing variants, all returning a 1-D float array.

    minimal     : as distributed (already 0.53-40 Hz by the original authors)
    zscore      : minimal + z-score  (affine -> MFQDMA-invariant by theory)
    detrend     : minimal + linear detrend
    notch50     : minimal + 50 Hz notch
    bp_1_40     : MNE-equivalent FIR band-pass 1-40 Hz
    bp_0p5_70   : MNE-equivalent FIR high-pass 0.5 Hz (h_freq above Nyquist->HP)
    downsample  : decimate to ~86.8 Hz (factor 2)
    """
    x = np.asarray(x, float)
    if mode == "minimal":
        return x
    if mode == "zscore":
        s = x.std()
        return (x - x.mean()) / (s if s > 0 else 1.0)
    if mode == "detrend":
        t = np.arange(len(x))
        return x - np.polyval(np.polyfit(t, x, 1), t)
    if mode == "notch50":
        return notch(x, sfreq)
    if mode == "bp_1_40":
        return mne_fir_bandpass(x, sfreq, 1.0, 40.0)
    if mode == "bp_0p5_70":
        return mne_fir_bandpass(x, sfreq, 0.5, None)
    if mode == "downsample":
        return resample_poly(x, 1, 2)
    raise ValueError(mode)


PREPROC_MODES = ["minimal", "zscore", "detrend", "notch50",
                 "bp_1_40", "bp_0p5_70", "downsample"]


# ------------------------------------------------------------------ MFQDMA
def _moving_quantile(y, n, tau, theta=0.0):
    """tau-th moving quantile in a theta-parametrised window (Gu & Zhou Eq. 2
    with the arithmetic mean replaced by the tau-quantile).

    Vectorised via a strided view -> O(N n log n) but with numpy inner loop.
    """
    N = len(y)
    nb = int(np.ceil((n - 1) * (1 - theta)))
    nf = int(np.floor((n - 1) * theta))
    lo, hi = nb, N - nf
    if hi <= lo:
        return None, lo, hi
    win = np.lib.stride_tricks.sliding_window_view(y, n)   # (N-n+1, n)
    q = np.quantile(win, tau, axis=1)
    # window ending at t=lo+k starts at lo+k-nb  -> index k in sliding view
    return q[: hi - lo], lo, hi


def mfqdma(x, tau=0.5, q_list=None, scales=None, theta=0.0, demean=True):
    """Multifractal Quantile-based Detrending Moving Average.

    demean : True for SIGNALS (standard MFDFA/MFDMA convention for time
             series, e.g. EEG).  False for positive MULTIFRACTAL MEASURES
             (e.g. the p-model cascade), where subtracting the mean destroys
             the multiplicative structure and biases tau(q) at large |q|.

    Returns dict with h(q,tau), tau(q), alpha, f(alpha), and fit R^2.
    """
    x = np.asarray(x, float)
    N = len(x)
    if q_list is None:
        q_list = np.arange(-5, 5.5, 0.5)
    q_list = np.asarray(q_list, float)
    y = np.cumsum(x - x.mean()) if demean else np.cumsum(x)

    if scales is None:
        scales = np.unique(np.logspace(np.log10(16), np.log10(N // 8),
                                       16).astype(int))
        scales = scales[scales >= 16]

    Fq = np.full((len(q_list), len(scales)), np.nan)
    for si, n in enumerate(scales):
        qf, lo, hi = _moving_quantile(y, n, tau, theta)
        if qf is None:
            continue
        eps = y[lo:hi] - qf
        eps = eps[np.isfinite(eps)]
        Nn = len(eps) // n
        if Nn < 4:
            continue
        seg = eps[: Nn * n].reshape(Nn, n)
        Fv = np.sqrt((seg ** 2).mean(axis=1))
        Fv = Fv[Fv > 1e-12]
        if len(Fv) < 4:
            continue
        lnF = np.log(Fv)
        for qi, q in enumerate(q_list):
            if abs(q) < 1e-12:
                Fq[qi, si] = np.exp(lnF.mean())
            else:
                # log-sum-exp for numerical stability at large |q|
                m = (q * lnF).max()
                Fq[qi, si] = np.exp((m + np.log(np.exp(q * lnF - m).mean())) / q)

    ls = np.log(scales.astype(float))
    hq = np.full(len(q_list), np.nan)
    r2 = np.full(len(q_list), np.nan)
    for qi in range(len(q_list)):
        f = Fq[qi]
        ok = np.isfinite(f) & (f > 0)
        if ok.sum() < 5:
            continue
        xx, yy = ls[ok], np.log(f[ok])
        A = np.vstack([xx, np.ones(ok.sum())]).T
        coef, res, *_ = np.linalg.lstsq(A, yy, rcond=None)
        hq[qi] = coef[0]
        pred = A @ coef
        ss_res = ((yy - pred) ** 2).sum()
        ss_tot = ((yy - yy.mean()) ** 2).sum()
        r2[qi] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    tq = q_list * hq - 1.0
    alpha = np.full(len(q_list), np.nan)
    falpha = np.full(len(q_list), np.nan)
    ok = np.isfinite(tq)
    if ok.sum() >= 3:
        qv, tv = q_list[ok], tq[ok]
        a = np.gradient(tv, qv)
        f_ = qv * a - tv
        alpha[ok], falpha[ok] = a, f_

    return dict(q=q_list, scales=scales, Fq=Fq, hq=hq, r2=r2,
                tau_q=tq, alpha=alpha, f_alpha=falpha, tau=tau)


def mfdma(x, **kw):
    """Standard MFDMA = arithmetic moving average (reference method).
    See mfqdma() for the meaning of `demean`."""
    x = np.asarray(x, float)
    N = len(x)
    q_list = kw.get("q_list", np.arange(-5, 5.5, 0.5))
    q_list = np.asarray(q_list, float)
    scales = kw.get("scales")
    demean = kw.get("demean", True)
    y = np.cumsum(x - x.mean()) if demean else np.cumsum(x)
    if scales is None:
        scales = np.unique(np.logspace(np.log10(16), np.log10(N // 8),
                                       16).astype(int))
        scales = scales[scales >= 16]
    Fq = np.full((len(q_list), len(scales)), np.nan)
    for si, n in enumerate(scales):
        win = np.lib.stride_tricks.sliding_window_view(y, n)
        mavg = win.mean(axis=1)
        lo, hi = n - 1, N
        eps = y[lo:hi] - mavg[: hi - lo]
        Nn = len(eps) // n
        if Nn < 4:
            continue
        seg = eps[: Nn * n].reshape(Nn, n)
        Fv = np.sqrt((seg ** 2).mean(axis=1))
        Fv = Fv[Fv > 1e-12]
        if len(Fv) < 4:
            continue
        lnF = np.log(Fv)
        for qi, q in enumerate(q_list):
            if abs(q) < 1e-12:
                Fq[qi, si] = np.exp(lnF.mean())
            else:
                m = (q * lnF).max()
                Fq[qi, si] = np.exp((m + np.log(np.exp(q * lnF - m).mean())) / q)
    ls = np.log(scales.astype(float))
    hq = np.full(len(q_list), np.nan)
    for qi in range(len(q_list)):
        f = Fq[qi]
        ok = np.isfinite(f) & (f > 0)
        if ok.sum() < 5:
            continue
        A = np.vstack([ls[ok], np.ones(ok.sum())]).T
        hq[qi] = np.linalg.lstsq(A, np.log(f[ok]), rcond=None)[0][0]
    tq = q_list * hq - 1.0
    alpha = np.full(len(q_list), np.nan)
    falpha = np.full(len(q_list), np.nan)
    ok = np.isfinite(tq)
    if ok.sum() >= 3:
        a = np.gradient(tq[ok], q_list[ok])
        alpha[ok] = a
        falpha[ok] = q_list[ok] * a - tq[ok]
    return dict(q=q_list, hq=hq, tau_q=tq, alpha=alpha, f_alpha=falpha)


# ------------------------------------------------------- asymmetry measures
def asymmetry_index(x, tau_lo=0.25, q_list=None, scales=None, theta=0.0):
    """Delta-H(q,tau) = h(q,tau) - h(q,1-tau)."""
    r_lo = mfqdma(x, tau=tau_lo, q_list=q_list, scales=scales, theta=theta)
    r_hi = mfqdma(x, tau=1 - tau_lo, q_list=q_list, scales=scales, theta=theta)
    return r_lo["hq"] - r_hi["hq"], r_lo, r_hi


def spectrum_width(alpha):
    a = alpha[np.isfinite(alpha)]
    return float(a.max() - a.min()) if len(a) else np.nan
