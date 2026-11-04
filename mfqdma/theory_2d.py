"""Analytical multifractal theory for the 2D benchmarks.

MEASURE ON A 2^k x 2^k GRID (four-multiplier cascade, [P1] Sec. V)
-----------------------------------------------------------------
At every iteration the measure in a square is split among its four quadrants
with weights p1..p4 (sum 1).  The partition function of the box measure obeys

    chi(q, s) = sum_boxes mu^q ~ s^{tau(q)},

    tau(q) = -log_2 ( sum_i p_i^q ).                                   (T1)

Consistency checks that any implementation must satisfy:
    tau(0)  = -log_2(4) = -2 = -D_f      (D_f = 2 for a plane-filling support)
    tau(1)  = -log_2(sum p_i) = 0        (measure normalisation)

Generalised Hurst exponent and spectrum, with D_f = 2:

    h(q)      = [tau(q) + 2] / q                                        (T2)
    alpha(q)  = d tau / dq
              = -[ sum_i p_i^q ln p_i ] / [ (sum_i p_i^q) ln 2 ]        (T3)
    f(alpha)  = q alpha(q) - tau(q)                                     (T4)

q -> 0 limits (both 0/0; L'Hopital gives the SAME value):

    h(0) = alpha(0) = tau'(0) = -(1/4) sum_i ln p_i / ln 2              (T5)

Note the 1/4 (= 1/number of multipliers).  Using the entropy-weighted form
-sum p_i ln p_i / ln2 here is a common slip and gives a visibly wrong h(0);
the same error in 1D was caught earlier in this project.

MONOFRACTAL SURFACE (fractional Brownian surface, Hurst H)
----------------------------------------------------------
    h(q) = H for all q,     tau(q) = qH - 2,     alpha = H,  f(alpha) = 2.
"""
import numpy as np


def cascade2d_tau(q, p=(0.05, 0.15, 0.25, 0.55)):
    """T1."""
    q = np.asarray(q, float)[..., None]
    p = np.asarray(p, float)
    return -np.log2((p ** q).sum(axis=-1))


def cascade2d_h(q, p=(0.05, 0.15, 0.25, 0.55)):
    """T2 with the T5 limit at q = 0."""
    q = np.asarray(q, float)
    p = np.asarray(p, float)
    out = np.empty_like(q)
    small = np.abs(q) < 1e-9
    if np.any(~small):
        out[~small] = (cascade2d_tau(q[~small], p) + 2.0) / q[~small]
    if np.any(small):
        out[small] = -np.sum(np.log(p)) / (len(p) * np.log(2))
    return out


def cascade2d_alpha(q, p=(0.05, 0.15, 0.25, 0.55)):
    """T3 (equals T5 at q = 0)."""
    q = np.asarray(q, float)
    p = np.asarray(p, float)
    out = np.empty_like(q)
    for i, qq in enumerate(q):
        pq = p ** qq
        out[i] = -np.sum(pq * np.log(p)) / (np.sum(pq) * np.log(2))
    return out


def cascade2d_falpha(q, p=(0.05, 0.15, 0.25, 0.55)):
    """T4."""
    q = np.asarray(q, float)
    return q * cascade2d_alpha(q, p) - cascade2d_tau(q, p)


def fbm_surface_tau(q, H):
    return np.asarray(q, float) * H - 2.0


def self_check(verbose=True):
    """Verify the identities the theory must satisfy."""
    ok = True
    for p in [(0.05, 0.15, 0.25, 0.55), (0.10, 0.20, 0.30, 0.40),
              (0.02, 0.18, 0.30, 0.50), (0.25, 0.25, 0.25, 0.25)]:
        t0 = float(cascade2d_tau(np.array([0.0]), p)[0])
        t1 = float(cascade2d_tau(np.array([1.0]), p)[0])
        h0 = float(cascade2d_h(np.array([0.0]), p)[0])
        a0 = float(cascade2d_alpha(np.array([0.0]), p)[0])
        # numerical derivative of tau at 0 as an independent check of T5
        e = 1e-5
        dnum = float((cascade2d_tau(np.array([e]), p)[0]
                      - cascade2d_tau(np.array([-e]), p)[0]) / (2 * e))
        c = (abs(t0 + 2) < 1e-12 and abs(t1) < 1e-12
             and abs(h0 - a0) < 1e-12 and abs(h0 - dnum) < 1e-6)
        ok &= c
        if verbose:
            print(f"  p={p}: tau(0)={t0:+.6f} (want -2)  tau(1)={t1:+.3e} (want 0)"
                  f"  h(0)={h0:.6f}  alpha(0)={a0:.6f}  tau'(0)_num={dnum:.6f}"
                  f"  {'OK' if c else 'FAIL'}")
    # monofractal limit: equal multipliers -> tau(q) = 2(q-1) -> h(q) = 2...
    p = (0.25,) * 4
    q = np.arange(-4, 4.5, 1.0)
    t = cascade2d_tau(q, p)
    expect = 2.0 * (q - 1.0)
    c = np.allclose(t, expect)
    ok &= c
    if verbose:
        print(f"  uniform cascade p=1/4: tau(q) == 2(q-1)? {'OK' if c else 'FAIL'}")
    return ok


if __name__ == '__main__':
    print("2D theory self-check")
    print("OK" if self_check() else "FAILED")
