"""
Dataset generators replicating the synthetic benchmarks of:
  [P1] Gu & Zhou,   Phys. Rev. E 82, 011136 (2010)  -- MFDMA
  [P2] Zhou,        Phys. Rev. E 77, 066211 (2008)  -- MF-DCCA (MF-X-DFA)
  [P3] Jiang & Zhou,Phys. Rev. E 84, 016106 (2011)  -- MF-X-DMA

plus MFQDMA-targeted additions (heavy tails, contamination, asymmetry,
monotonic-transform family) where the quantile parameter tau should matter.

Every generator that has a closed-form multifractal solution also exposes it,
so accuracy can be measured rather than asserted.
"""
import numpy as np
from scipy.special import gammaln


# ===================================================================== 1D p-model
def binomial_measure(p=0.3, k=20, seed=None, shuffle=False):
    """1D binomial multiplicative cascade (p-model). [P1] Sec. IV, [P3] Sec. III C.

    Deterministic construction: at each of k iterations every interval is split
    and the measure redistributed with weights (p, 1-p).  Length N = 2^k.
    """
    N = 2 ** k
    m = np.ones(N)
    for i in range(k):
        s = N // (2 ** (i + 1))
        for j in range(2 ** i):
            v = m[2 * j * s]
            m[2 * j * s:(2 * j + 1) * s] = v * p
            m[(2 * j + 1) * s:(2 * j + 2) * s] = v * (1 - p)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(m)
    return m


def binomial_tau_theory(q, p=0.3):
    """tau(q) = -log2(p^q + (1-p)^q).  [P1] Eq. (31), [P3] Eq. (25)."""
    q = np.asarray(q, float)
    return -np.log2(p ** q + (1 - p) ** q)


def binomial_h_theory(q, p=0.3):
    """h(q) = [tau(q) + 1]/q.

    At q = 0 this is 0/0; L'Hopital gives h(0) = tau'(0) = alpha(0)
        tau'(0) = -[ln p + ln(1-p)] / (2 ln 2).
    (Note this is NOT the entropy-weighted expression -[p ln p + (1-p)ln(1-p)]
    /ln2, which is a different quantity and gives a visibly wrong h(0).)
    """
    q = np.asarray(q, float)
    out = np.empty_like(q)
    small = np.abs(q) < 1e-9
    out[~small] = (binomial_tau_theory(q[~small], p) + 1.0) / q[~small]
    p2 = 1 - p
    out[small] = -(np.log(p) + np.log(p2)) / (2.0 * np.log(2))
    return out


# ===================================================================== 2D cascade
def cascade_2d(p=(0.05, 0.15, 0.25, 0.55), k=9):
    """2D multiplicative cascade on a 2^k x 2^k grid. [P1] Sec. V.
    Four multipliers assigned to the four quadrants at every iteration."""
    p = np.asarray(p, float)
    assert abs(p.sum() - 1.0) < 1e-12, "multipliers must sum to 1"
    N = 2 ** k
    m = np.ones((N, N))
    for i in range(k):
        s = N // (2 ** (i + 1))
        for a in range(2 ** i):
            for b in range(2 ** i):
                r0, c0 = 2 * a * s, 2 * b * s
                v = m[r0, c0]
                m[r0:r0 + s,         c0:c0 + s]         = v * p[0]
                m[r0:r0 + s,         c0 + s:c0 + 2 * s] = v * p[1]
                m[r0 + s:r0 + 2 * s, c0:c0 + s]         = v * p[2]
                m[r0 + s:r0 + 2 * s, c0 + s:c0 + 2 * s] = v * p[3]
    return m


def cascade_2d_tau_theory(q, p=(0.05, 0.15, 0.25, 0.55)):
    """tau(q) = -log2(sum_i p_i^q). [P1] Eq. (36)."""
    q = np.asarray(q, float)[..., None]
    p = np.asarray(p, float)
    return -np.log2((p ** q).sum(axis=-1))


# ===================================================================== fGn / fBm
def _circulant_eigs(acov_fn, n):
    acov = acov_fn(n)
    c = np.concatenate([acov, acov[n - 1:0:-1]])
    return np.fft.fft(c).real


def _davies_harte_adaptive(acov_fn, N, rng, max_pow=4):
    """Circulant embedding with ADAPTIVE PADDING.

    For strongly persistent processes (H -> 1) the minimal circulant embedding
    is not positive definite; truncating the negative eigenvalues biases the
    realised autocovariance downward.  We grow the embedding until it is PD
    (Dietrich & Newsam), which removes that bias.
    """
    n = 1
    while n < N:
        n *= 2
    for _ in range(max_pow + 1):
        lam = _circulant_eigs(acov_fn, n)
        if lam.min() >= -1e-10 * max(1.0, lam.max()):
            break
        n *= 2
    lam = np.maximum(lam, 0.0)
    return _dh_sample(lam, N, rng)


def _dh_sample(lam, N, rng):
    m = len(lam)
    V = np.zeros(m, dtype=complex)
    half = m // 2
    V[0] = np.sqrt(lam[0]) * rng.randn()
    V[half] = np.sqrt(lam[half]) * rng.randn()
    a = rng.randn(half - 1)
    b = rng.randn(half - 1)
    V[1:half] = np.sqrt(lam[1:half] / 2.0) * (a + 1j * b)
    V[half + 1:] = np.conj(V[1:half][::-1])
    x = np.fft.fft(V).real / np.sqrt(m)
    return x[:N]


def fgn_acov(n, H):
    """Autocovariance of fractional Gaussian noise, lags 0..n."""
    k = np.arange(0, n + 1, dtype=float)
    return 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H)
                  + np.abs(k + 1) ** (2 * H))


def fgn(N, H=0.7, seed=0):
    """Fractional Gaussian noise, exact via Davies-Harte circulant embedding."""
    rng = np.random.RandomState(seed)
    return _davies_harte_adaptive(lambda n: fgn_acov(n, H), N, rng)


def arfima_acov(n, d):
    """Exact ARFIMA(0,d,0) autocovariance (unit innovation variance):
       gamma(k) = Gamma(1-2d) Gamma(k+d) / [Gamma(d) Gamma(1-d) Gamma(k+1-d)]."""
    k = np.arange(0, n + 1, dtype=float)
    log_g = (gammaln(1 - 2 * d) + gammaln(k + d)
             - gammaln(d) - gammaln(1 - d) - gammaln(k + 1 - d))
    return np.exp(log_g)


def fbm_surface(N=512, H=0.7, seed=0):
    """Isotropic fractional Brownian surface by spectral synthesis."""
    rng = np.random.RandomState(seed)
    kx = np.fft.fftfreq(N)[:, None]
    ky = np.fft.fftfreq(N)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    k[0, 0] = 1.0
    amp = k ** (-(H + 1.0))
    amp[0, 0] = 0.0
    ph = rng.randn(N, N) + 1j * rng.randn(N, N)
    return np.fft.ifft2(amp * ph).real


# ======================================================== bivariate fBm  [P3] III A
def bfbm(N, Hxx=0.1, Hyy=0.5, rho=0.3, seed=0):
    """Bivariate fractional Gaussian noise with prescribed marginal Hurst
    exponents and cross-correlation rho, by correlating the Fourier phases of
    two spectrally-shaped Gaussian fields.

    Theory (Lavancier/Coeurjolly/Amblard; [P3] Eq. 13):  H_xy = (H_xx+H_yy)/2.

    Each component is generated EXACTLY (Davies-Harte) from its own fGn
    autocovariance, driven by a pair of correlated Gaussian innovation fields,
    so the marginal Hurst exponents are exact and the cross-correlation is
    monotone in rho.
    """
    rng = np.random.RandomState(seed)
    n = 1
    while n < N:
        n *= 2
    # grow the embedding until BOTH components are positive definite
    for _ in range(5):
        lx = _circulant_eigs(lambda k: fgn_acov(k, Hxx), n)
        ly = _circulant_eigs(lambda k: fgn_acov(k, Hyy), n)
        tol = -1e-10 * max(1.0, lx.max(), ly.max())
        if lx.min() >= tol and ly.min() >= tol:
            break
        n *= 2
    lx = np.maximum(lx, 0.0)
    ly = np.maximum(ly, 0.0)
    m = len(lx)
    half = m // 2

    # correlated innovations, shared across the two spectral shapings
    def draws():
        return rng.randn(half - 1), rng.randn(half - 1), rng.randn(), rng.randn()

    a1, b1, s01, sh1 = draws()
    a2, b2, s02, sh2 = draws()
    r = float(np.clip(rho, -1.0, 1.0))
    w = np.sqrt(max(0.0, 1 - r ** 2))
    a2 = r * a1 + w * a2
    b2 = r * b1 + w * b2
    s02 = r * s01 + w * s02
    sh2 = r * sh1 + w * sh2

    def build(lam, a, b, s0, sh):
        V = np.zeros(m, dtype=complex)
        V[0] = np.sqrt(lam[0]) * s0
        V[half] = np.sqrt(lam[half]) * sh
        V[1:half] = np.sqrt(lam[1:half] / 2.0) * (a + 1j * b)
        V[half + 1:] = np.conj(V[1:half][::-1])
        return (np.fft.fft(V).real / np.sqrt(m))[:N]

    return build(lx, a1, b1, s01, sh1), build(ly, a2, b2, s02, sh2)


def bfbm_hxy_theory(Hxx, Hyy):
    """[P3] Eq. (13)/(14)."""
    return 0.5 * (Hxx + Hyy)


# ================================================== two-component ARFIMA  [P3] III B
def _arfima_weights(d, M):
    """a_n(d) = d*Gamma(n-d)/[Gamma(1-d)Gamma(n+1)]  -- [P3] Eq. (18)."""
    n = np.arange(1, M + 1)
    log_a = np.log(abs(d) + 1e-300) + gammaln(n - d) - gammaln(1 - d) - gammaln(n + 1)
    return np.sign(d) * np.exp(log_a)


def arfima(N, d=0.3, M=None, seed=0, burn=None, method="exact"):
    """ARFIMA(0,d,0) with H = 0.5 + d.

    method='exact'    : circulant embedding of the exact autocovariance
                        (no truncation bias -- recommended).
    method='truncated': the literal weight sum of [P3] Eq. (17)-(18),
                        kept for reference; biased low for large d because
                        the weights decay only as n^(d-1).
    """
    rng = np.random.RandomState(seed)
    if method == "exact":
        Z = _davies_harte_adaptive(lambda n: arfima_acov(n, d), N, rng)
        return Z, rng.randn(N)
    if M is None:
        M = min(2 ** 15, 8 * N)
    if burn is None:
        burn = M
    eps = rng.randn(N + burn)
    a = _arfima_weights(d, M)
    Z = np.convolve(eps, np.concatenate([[0.0], a]))[:N + burn]
    return Z[burn:], eps[burn:]


def arfima_two_component(N, d1=0.4, d2=0.4, W=0.75, seed=0, shared_noise=False):
    """[P3] Eq. (20)  (W in [0.5,1], independent noises)
       [P3] Eq. (22)  (W = 1 with a single shared noise eps)

    H_xx = H_yy = 0.5 + d when d1 = d2 = d   ([P3] Eq. 21).
    """
    Z1, e1 = arfima(N, d1, seed=seed)
    Z2, e2 = arfima(N, d2, seed=seed + 1)
    if shared_noise:
        eps = np.random.RandomState(seed + 2).randn(N)
        x = Z1 + eps
        y = Z2 + eps
    else:
        x = W * Z1 + (1 - W) * Z2 + e1
        y = (1 - W) * Z1 + W * Z2 + e2
    return x, y


# ============================================ correlated binomial pair  [P3] III C
def binomial_pair(px=0.3, py=0.4, k=16):
    """Two binomial measures built with the SAME split pattern, so they are
    strongly cross-correlated ([P3] reports r ~ 0.82 for px=0.3, py=0.4)."""
    return binomial_measure(px, k), binomial_measure(py, k)


def binomial_H_theory(q, p):
    """H_zz(q) = 1/q - log2[p^q + (1-p)^q]/q   -- [P3] Eq. (24)."""
    q = np.asarray(q, float)
    out = np.empty_like(q)
    small = np.abs(q) < 1e-9
    qq = q[~small]
    out[~small] = 1.0 / qq - np.log2(p ** qq + (1 - p) ** qq) / qq
    p2 = 1 - p
    # q -> 0 limit (L'Hopital): tau'(0) = -[ln p + ln(1-p)]/(2 ln 2)
    out[small] = -(np.log(p) + np.log(p2)) / (2.0 * np.log(2))
    return out


def binomial_pair_Hxy_theory(q, px=0.3, py=0.4):
    """H_xy(q) = [H_xx(q) + H_yy(q)]/2   -- [P3] Eq. (27)."""
    return 0.5 * (binomial_H_theory(q, px) + binomial_H_theory(q, py))


# ===================================================== MFQDMA-targeted additions
def lognormal_cascade(k=16, sigma=0.4, seed=0):
    """Continuous (tie-free) multiplicative cascade with log-normal weights.
    Unlike the binomial p-model this has no repeated values, so ordinal /
    quantile operations are well defined."""
    rng = np.random.RandomState(seed)
    N = 2 ** k
    m = np.ones(N)
    for i in range(k):
        s = N // (2 ** (i + 1))
        for j in range(2 ** i):
            w = np.exp(rng.randn() * sigma)
            w1, w2 = w, 2.0 - w
            if w2 <= 0:
                w1, w2 = 1.5, 0.5
            tot = w1 + w2
            w1, w2 = 2 * w1 / tot, 2 * w2 / tot
            v = m[2 * j * s]
            m[2 * j * s:(2 * j + 1) * s] = v * w1
            m[(2 * j + 1) * s:(2 * j + 2) * s] = v * w2
    return np.abs(m)


def asymmetric_cascade(k=16, p_up=0.35, p_dn=0.65, seed=0):
    """Cascade whose split weight depends on a random branch choice, producing
    genuinely asymmetric amplitude structure (target for the tau dimension)."""
    rng = np.random.RandomState(seed)
    N = 2 ** k
    m = np.ones(N)
    for i in range(k):
        s = N // (2 ** (i + 1))
        for j in range(2 ** i):
            pp = p_up if rng.rand() < 0.5 else p_dn
            v = m[2 * j * s]
            m[2 * j * s:(2 * j + 1) * s] = v * pp
            m[(2 * j + 1) * s:(2 * j + 2) * s] = v * (1 - pp)
    return m


def levy_stable(N, alpha=1.5, beta=0.0, seed=0):
    """Symmetric/skewed alpha-stable noise (Chambers-Mallows-Stuck).
    Infinite variance for alpha<2 -- the canonical stress test for
    second-moment (RMS / mean) based estimators."""
    rng = np.random.RandomState(seed)
    U = (rng.rand(N) - 0.5) * np.pi
    W = rng.exponential(1.0, N)
    if abs(alpha - 1.0) < 1e-9:
        X = (2 / np.pi) * ((np.pi / 2 + beta * U) * np.tan(U)
                           - beta * np.log((np.pi / 2) * W * np.cos(U)
                                           / (np.pi / 2 + beta * U)))
    else:
        zeta = -beta * np.tan(np.pi * alpha / 2)
        xi = np.arctan(-zeta) / alpha
        X = ((1 + zeta ** 2) ** (1 / (2 * alpha))
             * np.sin(alpha * (U + xi)) / np.cos(U) ** (1 / alpha)
             * (np.cos(U - alpha * (U + xi)) / W) ** ((1 - alpha) / alpha))
    return X


def contaminate(x, frac=0.05, magnitude=20.0, kind="outlier", seed=0):
    """Additive outliers / heavy-tailed replacement contamination."""
    rng = np.random.RandomState(seed)
    y = np.asarray(x, float).copy()
    n = max(1, int(frac * len(y)))
    idx = rng.choice(len(y), n, replace=False)
    s = np.std(y)
    if kind == "outlier":
        y[idx] = rng.choice([-1.0, 1.0], n) * magnitude * s
    elif kind == "student_t":
        y[idx] = rng.standard_t(2, n) * s * magnitude / 5.0
    elif kind == "levy":
        y[idx] = levy_stable(n, 1.2, seed=seed) * s
    return y, idx


def monotonic_family(x):
    """Strictly increasing maps that preserve ALL rank order but destroy the
    amplitude distribution -- used to probe transform (in)variance."""
    from scipy.stats import rankdata, norm
    x = np.asarray(x, float)
    xp = x - x.min() + 1e-9          # make positive for power/log maps
    r = rankdata(x, method="ordinal")
    return {
        "identity":    x.copy(),
        "cube":        xp ** 3,
        "cuberoot":    xp ** (1 / 3),
        "sqrt":        np.sqrt(xp),
        "log":         np.log(xp),
        "gaussianize": norm.ppf(r / (len(x) + 1.0)),
    }
