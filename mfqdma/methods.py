"""Full method set for the corpus benchmark:
   MFDFA, MFDMA, MFQDMA            (1D auto)
   MFDMA-2D, MFQDMA-2D             (2D auto)
   MF-X-DFA, MF-X-DMA, MF-X-QDMA   (1D cross-correlation)

theta convention follows Gu & Zhou PRE 82, 011136: theta=0 backward,
0.5 centred, 1 forward.
"""
import numpy as np

EPS = 1e-300


# ----------------------------------------------------------------- helpers
def _profile(x, demean=True):
    x = np.asarray(x, float)
    return np.cumsum(x - x.mean()) if demean else np.cumsum(x)


def _default_scales(N, lo=16, hi=None, n=16):
    hi = hi or N // 8
    s = np.unique(np.logspace(np.log10(lo), np.log10(hi), n).astype(int))
    return s[s >= max(4, lo)]


def _qmoments(Fv, q_list):
    """q-th order moments of a set of per-box fluctuations (log-sum-exp)."""
    Fv = np.asarray(Fv, float)
    Fv = Fv[np.isfinite(Fv) & (Fv > EPS)]
    if len(Fv) < 4:
        return None
    lnF = np.log(Fv)
    out = np.empty(len(q_list))
    for i, q in enumerate(q_list):
        if abs(q) < 1e-12:
            out[i] = np.exp(lnF.mean())
        else:
            m = (q * lnF).max()
            out[i] = np.exp((m + np.log(np.exp(q * lnF - m).mean())) / q)
    return out


def _fit(Fq, scales, q_list, Df=1.0):
    ls = np.log(scales.astype(float))
    hq = np.full(len(q_list), np.nan)
    r2 = np.full(len(q_list), np.nan)
    for i in range(len(q_list)):
        f = Fq[i]
        ok = np.isfinite(f) & (f > 0)
        if ok.sum() < 5:
            continue
        xx, yy = ls[ok], np.log(f[ok])
        A = np.vstack([xx, np.ones(ok.sum())]).T
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        hq[i] = coef[0]
        pred = A @ coef
        sst = ((yy - yy.mean()) ** 2).sum()
        r2[i] = 1 - ((yy - pred) ** 2).sum() / sst if sst > 0 else np.nan
    tau = q_list * hq - Df
    alpha = np.full(len(q_list), np.nan)
    falpha = np.full(len(q_list), np.nan)
    ok = np.isfinite(tau)
    if ok.sum() >= 3:
        a = np.gradient(tau[ok], q_list[ok])
        alpha[ok] = a
        falpha[ok] = q_list[ok] * a - tau[ok]
    return dict(hq=hq, r2=r2, tau_q=tau, alpha=alpha, f_alpha=falpha,
                Fq=Fq, scales=scales, q=q_list)


def _win_bounds(n, theta):
    nb = int(np.ceil((n - 1) * (1 - theta)))
    nf = int(np.floor((n - 1) * theta))
    return nb, nf


def _detrended_residual(y, n, theta, tau=None):
    """Residual of the profile after removing the moving mean (tau=None)
    or the moving tau-quantile."""
    N = len(y)
    nb, nf = _win_bounds(n, theta)
    lo, hi = nb, N - nf
    if hi <= lo:
        return None
    win = np.lib.stride_tricks.sliding_window_view(y, n)
    trend = win.mean(axis=1) if tau is None else np.quantile(win, tau, axis=1)
    return y[lo:hi] - trend[: hi - lo]


# ----------------------------------------------------------------- 1D auto
def mfdfa(x, q_list, scales=None, order=1, demean=True):
    x = np.asarray(x, float); N = len(x)
    y = _profile(x, demean)
    scales = _default_scales(N) if scales is None else np.asarray(scales)
    Fq = np.full((len(q_list), len(scales)), np.nan)
    for si, s in enumerate(scales):
        Ns = N // s
        if Ns < 4:
            continue
        seg = y[: Ns * s].reshape(Ns, s)
        t = np.arange(s)
        V = np.vander(t, order + 1)
        coef, *_ = np.linalg.lstsq(V, seg.T, rcond=None)
        res = seg.T - V @ coef
        Fv = np.sqrt((res ** 2).mean(axis=0))
        m = _qmoments(Fv, q_list)
        if m is not None:
            Fq[:, si] = m
    return _fit(Fq, scales, q_list)


def mfdma(x, q_list, scales=None, theta=0.0, demean=True):
    x = np.asarray(x, float); N = len(x)
    y = _profile(x, demean)
    scales = _default_scales(N) if scales is None else np.asarray(scales)
    Fq = np.full((len(q_list), len(scales)), np.nan)
    for si, n in enumerate(scales):
        eps = _detrended_residual(y, n, theta, None)
        if eps is None or len(eps) < n:
            continue
        Nn = len(eps) // n
        if Nn < 4:
            continue
        seg = eps[: Nn * n].reshape(Nn, n)
        m = _qmoments(np.sqrt((seg ** 2).mean(axis=1)), q_list)
        if m is not None:
            Fq[:, si] = m
    return _fit(Fq, scales, q_list)


def _moving_quantiles_multi(y, n, theta, taus, mem_budget=4e7):
    """All requested moving tau-quantiles of the profile in ONE partition pass.

    np.partition accepts a list of kth indices, so every tau (and both
    neighbours needed for linear interpolation) is obtained from a single
    O(len) selection per chunk instead of one full quantile call per tau.
    Chunked along position to bound peak memory.
    """
    N = len(y)
    nb, nf = _win_bounds(n, theta)
    lo_i, hi_i = nb, N - nf
    if hi_i <= lo_i:
        return None
    M = hi_i - lo_i

    pos = [t * (n - 1) for t in taus]
    los = [int(np.floor(p)) for p in pos]
    his = [min(l + 1, n - 1) for l in los]
    frac = [p - l for p, l in zip(pos, los)]
    ks = sorted(set(los) | set(his))

    out = {t: np.empty(M) for t in taus}
    chunk = max(1, int(mem_budget // max(1, n * 8)))
    win_all = np.lib.stride_tricks.sliding_window_view(y, n)
    for s in range(0, M, chunk):
        e = min(M, s + chunk)
        blk = np.ascontiguousarray(win_all[s:e])
        part = np.partition(blk, ks, axis=1)
        for t, l, h, f in zip(taus, los, his, frac):
            out[t][s:e] = part[:, l] * (1.0 - f) + part[:, h] * f
    return out, lo_i, hi_i


def mfqdma_multi(x, q_list, scales=None, theta=0.0, taus=(0.5,), demean=True):
    """MFQDMA evaluated at several tau values simultaneously.
    Returns {tau: result_dict}."""
    x = np.asarray(x, float); N = len(x)
    y = _profile(x, demean)
    scales = _default_scales(N) if scales is None else np.asarray(scales)
    taus = list(taus)
    Fq = {t: np.full((len(q_list), len(scales)), np.nan) for t in taus}
    for si, n in enumerate(scales):
        got = _moving_quantiles_multi(y, n, theta, taus)
        if got is None:
            continue
        qd, lo_i, hi_i = got
        for t in taus:
            eps = y[lo_i:hi_i] - qd[t]
            Nn = len(eps) // n
            if Nn < 4:
                continue
            seg = eps[: Nn * n].reshape(Nn, n)
            m = _qmoments(np.sqrt((seg ** 2).mean(axis=1)), q_list)
            if m is not None:
                Fq[t][:, si] = m
    return {t: _fit(Fq[t], scales, q_list) for t in taus}


def mfqdma(x, q_list, scales=None, theta=0.0, tau=0.5, demean=True):
    x = np.asarray(x, float); N = len(x)
    y = _profile(x, demean)
    scales = _default_scales(N) if scales is None else np.asarray(scales)
    Fq = np.full((len(q_list), len(scales)), np.nan)
    for si, n in enumerate(scales):
        eps = _detrended_residual(y, n, theta, tau)
        if eps is None or len(eps) < n:
            continue
        Nn = len(eps) // n
        if Nn < 4:
            continue
        seg = eps[: Nn * n].reshape(Nn, n)
        m = _qmoments(np.sqrt((seg ** 2).mean(axis=1)), q_list)
        if m is not None:
            Fq[:, si] = m
    return _fit(Fq, scales, q_list)


# ----------------------------------------------------------------- 2D auto
def _profile_2d(Z, demean=True):
    Z = np.asarray(Z, float)
    if demean:
        Z = Z - Z.mean()
    return np.cumsum(np.cumsum(Z, axis=0), axis=1)


def _box_sums_2d(Y, n):
    """Sum of the original field over every n x n box via the integral image."""
    N1, N2 = Y.shape
    P = np.zeros((N1 + 1, N2 + 1))
    P[1:, 1:] = Y
    return P


def mfdma_2d(Z, q_list, scales=None, theta=0.0, tau=None, demean=True):
    """2D MFDMA (tau=None) / 2D MFQDMA (tau given).

    Follows the d-dimensional formulation of [P1] Sec. III / [P3] Appendix:
    for each n x n window, Q is the TOTAL window sum and the trend is built
    from the WITHIN-WINDOW cumulative sums Z'(k1,k2) (Eqs. A5-A8), i.e.

        eps(i1,i2) = Q(i1,i2) - mean_{k1,k2} Z'(k1,k2).

    In 1D the within-window and global-cumulative-sum formulations coincide
    (they differ by a per-window constant that cancels), but in 2D they do
    NOT: the inclusion-exclusion correction varies across the window.  Using a
    global 2D cumsum here is therefore wrong and badly biases tau(q).

    Because mean_{k} Z'(k1,k2) = sum_{l1,l2} z(l1,l2)(n-l1+1)(n-l2+1)/n^2 is a
    SEPARABLE ramp-weighted sum, the mean trend is computed with two 1-D
    convolutions.  For the quantile trend the n^2 within-window cumulative
    sums are materialised in row chunks.
    """
    Z = np.asarray(Z, float)
    N1, N2 = Z.shape
    if demean:
        Z = Z - Z.mean()
    Y = Z          # the d-dimensional algorithm works on the raw field
    if scales is None:
        scales = _default_scales(min(N1, N2), lo=8, hi=min(N1, N2) // 8, n=10)
    scales = np.asarray(scales)
    Fq = np.full((len(q_list), len(scales)), np.nan)
    for si, n in enumerate(scales):
        nb, nf = _win_bounds(n, theta)
        if N1 - nf <= nb or N2 - nf <= nb:
            continue
        R1, R2 = N1 - n + 1, N2 - n + 1
        if R1 < 2 * n or R2 < 2 * n:
            continue
        # Q(i1,i2): total sum over the BACKWARD n x n window ([P3] Eq. A7).
        # The trend window is shifted forward by floor((n-1)*theta) relative
        # to it ([P3] Eq. A5-A6), so theta slides the trend against Q.
        shift = int(np.floor((n - 1) * theta))
        P = np.zeros((N1 + 1, N2 + 1))
        P[1:, 1:] = np.cumsum(np.cumsum(Y, axis=0), axis=1)
        Qbox = P[n:, n:] - P[:-n, n:] - P[n:, :-n] + P[:-n, :-n]

        if tau is None:
            # mean of within-window cumulative sums == separable ramp weights
            ramp = np.arange(n, 0, -1, dtype=float)        # (n-l+1), l=1..n
            tmp = np.apply_along_axis(
                lambda v: np.convolve(v, ramp[::-1], mode='valid'), 0, Y)
            wsum = np.apply_along_axis(
                lambda v: np.convolve(v, ramp[::-1], mode='valid'), 1, tmp)
            trend = wsum / (n * n)
        else:
            # tau-quantile of the n^2 within-window cumulative sums.
            # Z'(a,b) is obtained from the GLOBAL integral image by
            # inclusion-exclusion, which avoids re-running a 2-D cumsum
            # inside every window:
            #   Z'(a,b) = P[i1+a+1, i2+b+1] - P[i1, i2+b+1]
            #           - P[i1+a+1, i2]     + P[i1, i2]
            trend = np.empty((R1, R2))
            k = int(np.floor(tau * (n * n - 1)))
            k2 = min(k + 1, n * n - 1)
            fr = tau * (n * n - 1) - k
            budget = max(1, int(2.0e7 // max(1, R2 * n * n)))
            for r0 in range(0, R1, budget):
                r1 = min(R1, r0 + budget)
                W = np.lib.stride_tricks.sliding_window_view(
                    P[r0:r1 + n, :], (n + 1, n + 1))
                Zp = (W[:, :, 1:, 1:] - W[:, :, :1, 1:]
                      - W[:, :, 1:, :1] + W[:, :, :1, :1])
                flat = Zp.reshape(r1 - r0, R2, -1)
                part = np.partition(flat, [k, k2], axis=2)
                trend[r0:r1] = part[:, :, k] * (1.0 - fr) + part[:, :, k2] * fr
        # align: eps(i) = Q(i) - trend(i + shift)
        if shift > 0:
            if trend.shape[0] <= shift or trend.shape[1] <= shift:
                continue
            eps = Qbox[:-shift, :-shift] - trend[shift:, shift:]
        else:
            eps = Qbox - trend
        M1, M2 = eps.shape
        B1, B2 = M1 // n, M2 // n
        if B1 < 2 or B2 < 2:
            continue
        blocks = eps[:B1 * n, :B2 * n].reshape(B1, n, B2, n).transpose(0, 2, 1, 3)
        Fv = np.sqrt((blocks ** 2).mean(axis=(2, 3))).ravel()
        m = _qmoments(Fv, q_list)
        if m is not None:
            Fq[:, si] = m
    return _fit(Fq, scales, q_list, Df=2.0)


# ------------------------------------------------------- 1D cross-correlation
def _signed_qmoments_cross(Fv_signed, q_list):
    """q-th order moments of a signed cross-covariance set.
    Uses |F|^{q/2} as in [P2] Eq.(7)/[P3] Eq.(7); q=0 via the log rule."""
    a = np.abs(np.asarray(Fv_signed, float))
    a = a[np.isfinite(a) & (a > EPS)]
    if len(a) < 4:
        return None
    ln = np.log(a)
    out = np.empty(len(q_list))
    for i, q in enumerate(q_list):
        if abs(q) < 1e-12:
            out[i] = np.exp(0.5 * ln.mean())
        else:
            m = (0.5 * q * ln).max()
            out[i] = np.exp((m + np.log(np.exp(0.5 * q * ln - m).mean())) / q)
    return out


def mfx_qdma_multi(x, y, q_list, scales=None, theta=0.0, taus=(0.5,),
                   demean=True):
    """MF-X-QDMA at several tau values in one pass. Returns {tau: result}."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    N = min(len(x), len(y))
    Yx, Yy = _profile(x[:N], demean), _profile(y[:N], demean)
    scales = _default_scales(N) if scales is None else np.asarray(scales)
    taus = list(taus)
    Fq = {t: np.full((len(q_list), len(scales)), np.nan) for t in taus}
    for si, n in enumerate(scales):
        gx = _moving_quantiles_multi(Yx, n, theta, taus)
        gy = _moving_quantiles_multi(Yy, n, theta, taus)
        if gx is None or gy is None:
            continue
        qx, lo_i, hi_i = gx
        qy, _, _ = gy
        for t in taus:
            ex = Yx[lo_i:hi_i] - qx[t]
            ey = Yy[lo_i:hi_i] - qy[t]
            L = min(len(ex), len(ey))
            Nn = L // n
            if Nn < 4:
                continue
            sx = ex[: Nn * n].reshape(Nn, n)
            sy = ey[: Nn * n].reshape(Nn, n)
            m = _signed_qmoments_cross((sx * sy).mean(axis=1), q_list)
            if m is not None:
                Fq[t][:, si] = m
    return {t: _fit(Fq[t], scales, q_list) for t in taus}


def mfx(x, y, q_list, scales=None, method='dma', theta=0.0, tau=0.5,
        order=1, demean=True):
    """Cross-correlation multifractal analysis.
    method: 'dfa' -> MF-X-DFA [P2];  'dma' -> MF-X-DMA [P3];
            'qdma'-> MF-X-QDMA (quantile detrending, this work)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    N = min(len(x), len(y))
    Yx, Yy = _profile(x[:N], demean), _profile(y[:N], demean)
    scales = _default_scales(N) if scales is None else np.asarray(scales)
    Fq = np.full((len(q_list), len(scales)), np.nan)
    for si, n in enumerate(scales):
        if method == 'dfa':
            Ns = N // n
            if Ns < 4:
                continue
            sx = Yx[: Ns * n].reshape(Ns, n)
            sy = Yy[: Ns * n].reshape(Ns, n)
            t = np.arange(n)
            V = np.vander(t, order + 1)
            cx, *_ = np.linalg.lstsq(V, sx.T, rcond=None)
            cy, *_ = np.linalg.lstsq(V, sy.T, rcond=None)
            rx = sx.T - V @ cx
            ry = sy.T - V @ cy
            Fv = (rx * ry).mean(axis=0)
        else:
            tt = None if method == 'dma' else tau
            ex = _detrended_residual(Yx, n, theta, tt)
            ey = _detrended_residual(Yy, n, theta, tt)
            if ex is None or ey is None or len(ex) < n:
                continue
            L = min(len(ex), len(ey))
            Nn = L // n
            if Nn < 4:
                continue
            sx = ex[: Nn * n].reshape(Nn, n)
            sy = ey[: Nn * n].reshape(Nn, n)
            Fv = (sx * sy).mean(axis=1)
        m = _signed_qmoments_cross(Fv, q_list)
        if m is not None:
            Fq[:, si] = m
    return _fit(Fq, scales, q_list)
