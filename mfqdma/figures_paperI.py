"""Figures of Paper I (MFQDMA, univariate 1D + 2D). Each function writes one
manuscript figure to figures_out/. Results are read from results/; light
computations are done on the fly and cached in cache/."""
import os, json, gzip, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .generators import (binomial_measure, binomial_tau_theory, cascade_2d,
                         fgn, contaminate, lognormal_cascade)
from .methods import (mfdfa, mfdma, mfqdma, mfqdma_multi, mfdma_2d,
                      _detrended_residual, _win_bounds)
from .theory_2d import cascade2d_tau, cascade2d_alpha, cascade2d_falpha
from .plotting import (RES, OUT, CACHE, W1, W2, Q, C_DFA, C_DMA, TAUS, TCOL, FIGNAME,
                       lab, logticks, save, sc, SM, SD, rmse, cached)

# ------------------------------------------------------------------ Fig 1
def fig1():
    """The tau-family of detrending trends, residuals and fluctuation functions."""
    x = fgn(2 ** 14, 0.7, seed=11)
    y = np.cumsum(x - x.mean())
    n, th = 128, 0.0
    nb, nf = _win_bounds(n, th)
    i = np.arange(nb, len(y) - nf)
    win = np.lib.stride_tricks.sliding_window_view(y, n)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.0))
    sl = slice(2000, 3200)
    ii = np.arange(len(y))[sl]
    ax[0].plot(ii, y[sl], color='k', lw=0.9, label='profile $y$')
    ax[0].plot(ii, win.mean(1)[sl.start - nb:sl.stop - nb], color=C_DMA, lw=1.0,
               label='moving mean')
    for t in (0.1, 0.5, 0.9):
        ax[0].plot(ii, np.quantile(win[sl.start - nb:sl.stop - nb], t, axis=1),
                   color=TCOL[t], lw=1.0, ls='--', label=rf'$\widetilde y_\lambda$, $\lambda={t}$')
    ax[0].set(xlabel='$i$', ylabel=r'$y(i)$ and trend')
    ax[0].legend(frameon=False, fontsize=5.4, loc='upper right')
    lab(ax[0], '(a)')
    ax[1].plot(ii, (y[sl] - win.mean(1)[sl.start - nb:sl.stop - nb]), color=C_DMA, lw=0.7,
               label=r'$\epsilon$ (mean)')
    for t in (0.1, 0.9):
        ax[1].plot(ii, y[sl] - np.quantile(win[sl.start - nb:sl.stop - nb], t, axis=1),
                   color=TCOL[t], lw=0.7, label=rf'$\epsilon_\lambda$, $\lambda={t}$')
    ax[1].axhline(0, color='k', lw=0.5)
    ax[1].set(xlabel='$i$', ylabel='residual')
    ax[1].legend(frameon=False, fontsize=5.6, ncol=3, loc='upper center')
    lab(ax[1], '(b)')
    r = mfdma(x, Q, scales=SM, theta=0.0)
    q2 = int(np.argmin(np.abs(Q - 2)))
    ax[2].loglog(r['scales'], r['Fq'][q2], 'o', color=C_DMA, mfc='none',
                 label=f'MFDMA, $h(2)$={r["hq"][q2]:.3f}')
    for t, rr in mfqdma_multi(x, Q, scales=SM, theta=0.0, taus=TAUS).items():
        ax[2].loglog(rr['scales'], rr['Fq'][q2], 's', ms=2.4, color=TCOL[t],
                     label=rf'$\lambda={t}$, $h(2)$={rr["hq"][q2]:.3f}')
    ax[2].set(xlabel='$n$', ylabel='$F_2(n)$')
    ax[2].legend(frameon=False, fontsize=5.2, loc='lower right')
    lab(ax[2], '(c)')
    fig.tight_layout(w_pad=0.5)
    save(fig, 'fig01_schematic')


# ------------------------------------------------------------------ Fig 2
def fig2():
    """Box-sum identity (Proposition 1) and the degeneracy locus theta+tau=1."""
    x = binomial_measure(0.3, 14)
    y = np.cumsum(x)
    def ratio():
        out = {}
        n = 65
        base = _detrended_residual(y, n, 0.0, None).std()
        for t in (0.25, 0.5, 0.75):
            out[t] = [(_detrended_residual(y, n, th, t).std() / base)
                      for th in np.linspace(0, 1, 41)]
        out['mean'] = [(_detrended_residual(y, n, th, None).std() / base)
                       for th in np.linspace(0, 1, 41)]
        return out
    R = cached('fig2_ratio', ratio)
    def acc():
        th_q = binomial_tau_theory(Q, 0.3)
        xx = binomial_measure(0.3, 16)
        out = {}
        for th in np.linspace(0, 1, 11):
            row = {}
            for t in (0.25, 0.5, 0.75):
                r = mfqdma(xx, Q, scales=SM, theta=th, tau=t, demean=False)
                row[t] = rmse(r['tau_q'], th_q)
            r = mfdma(xx, Q, scales=SM, theta=th, demean=False)
            row['mean'] = rmse(r['tau_q'], th_q)
            out[round(th, 2)] = row
        return out
    A = cached('fig2_acc', acc)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.05))
    # (a) geometry
    n, th, t = 33, 0.0, 0.25
    i0 = 4000
    seg = np.arange(i0 - 45, i0 + 6)
    ax[0].plot(seg, y[seg], 'k.-', lw=0.7, ms=2)
    nb, nf = _win_bounds(n, th)
    w = np.arange(i0 - nb, i0 + nf + 1)
    qv = np.quantile(y[w], t)
    ell = nb - t * (n - 1)
    ax[0].axvspan(w[0], w[-1], color='0.9')
    ax[0].axhline(qv, color=TCOL[0.25], ls='--', lw=0.9)
    ax[0].plot([i0], [y[i0]], 'o', color='k', ms=4)
    ax[0].annotate('', xy=(i0, qv), xytext=(i0, y[i0]),
                   arrowprops=dict(arrowstyle='<->', lw=0.8, color='crimson'))
    ax[0].annotate('', xy=(i0 - ell, qv - 0.0004), xytext=(i0, qv - 0.0004),
                   arrowprops=dict(arrowstyle='<->', lw=0.8, color=TCOL[0.25]))
    ax[0].text(i0 - ell / 2, qv - 0.0007, r'$\ell=(1-\theta-\lambda)(n-1)$',
               ha='center', va='top', fontsize=6.5)
    ax[0].text(i0 + 1, (y[i0] + qv) / 2, r'$\epsilon_\lambda(i)=\mu(i-\ell,i]$',
               fontsize=6.5, color='crimson', va='center')
    ax[0].set(xlabel='$i$', ylabel=r'$y(i)$')
    lab(ax[0], '(a)')
    th_grid = np.linspace(0, 1, 41)
    ax[1].plot(th_grid, R['mean'], color=C_DMA, label='moving mean')
    for t in (0.25, 0.5, 0.75):
        ax[1].plot(th_grid, R[t], color=TCOL[t], label=rf'$\lambda={t}$')
        ax[1].axvline(1 - t, color=TCOL[t], ls=':', lw=0.7)
    ax[1].set(xlabel=r'$\theta$', ylabel=r'std$(\epsilon)$ / std$(\epsilon_{\rm MA,\theta=0})$',
              yscale='log', ylim=(1e-5, 30))
    ax[1].legend(frameon=False, loc='lower center', fontsize=5.2, ncol=2)
    lab(ax[1], '(b)')
    ths = sorted(A)
    ax[2].plot(ths, [A[k]['mean'] for k in ths], 'o-', color=C_DMA, label='MFDMA')
    for t in (0.25, 0.5, 0.75):
        ax[2].plot(ths, [A[k][t] for k in ths], 's-', color=TCOL[t],
                   label=rf'MFQDMA $\lambda={t}$')
    ax[2].set(xlabel=r'$\theta$', ylabel=r'RMSE of $\tau(q)$', yscale='log', ylim=(0.01, 40))
    ax[2].legend(frameon=False, loc='upper left', fontsize=5.4, ncol=2)
    lab(ax[2], '(c)', x=0.88)
    fig.tight_layout(w_pad=0.6)
    save(fig, 'fig02_boxsum_degeneracy')


# ------------------------------------------------------------------ Fig 3
def run_binomial():
    x = binomial_measure(0.3, 16)
    out = {'MFDFA': mfdfa(x, Q, scales=SD, demean=False),
           'MFDMA': mfdma(x, Q, scales=SM, theta=0.0, demean=False)}
    for t, r in mfqdma_multi(x, Q, scales=SM, theta=0.0, taus=TAUS,
                             demean=False).items():
        out[t] = r
    return out


def fig3():
    R = cached('binomial_p03_k16', run_binomial)
    th = binomial_tau_theory(Q, 0.3)
    p = 0.3
    qq = np.linspace(-4, 4, 161)
    pq = p ** qq; rq = (1 - p) ** qq
    a_th = -(pq * np.log2(p) + rq * np.log2(1 - p)) / (pq + rq)
    f_th = qq * a_th - binomial_tau_theory(qq, p)
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.95))
    r = R[0.25]
    for qi, qv in enumerate(Q):
        if qv in (-4, -2, 0, 2, 4):
            F = r['Fq'][qi]
            k = np.isfinite(F)
            c = plt.cm.viridis((qv + 4) / 8.5)
            ax[0].loglog(r['scales'][k], F[k] * 10 ** (0.6 * qv / 2), 'o', color=c,
                         ms=2.5, label=f'$q={int(qv)}$')
            ax[0].loglog(r['scales'][k], np.exp(np.polyval(np.polyfit(
                np.log(r['scales'][k]), np.log(F[k]), 1), np.log(r['scales'][k])))
                         * 10 ** (0.6 * qv / 2), '-', color=c, lw=0.7)
    ax[0].set(xlabel='$n$', ylabel='$F_q(n)$ (shifted)')
    ax[0].set_ylim(top=ax[0].get_ylim()[1] * 1e3)
    ax[0].legend(frameon=False, fontsize=5.2, loc='upper left', ncol=3)
    lab(ax[0], '(a)')
    ax[1].plot(qq, binomial_tau_theory(qq, p), 'k-', lw=1.6, label='theory')
    ax[1].plot(Q, R['MFDFA']['tau_q'], 'v', color=C_DFA, mfc='none', label='MFDFA')
    ax[1].plot(Q, R['MFDMA']['tau_q'], 'o', color=C_DMA, mfc='none', label='MFDMA')
    ax[1].plot(Q, R[0.25]['tau_q'], 's', color=TCOL[0.25], ms=2.5,
               label=r'MFQDMA $\lambda=0.25$')
    ax[1].set(xlabel='$q$', ylabel=r'$\tau(q)$')
    ax[1].legend(frameon=False, loc='upper left', fontsize=5.8)
    lab(ax[1], '(b)', x=0.85, y=0.15)
    ax[2].axhline(0, color='k', lw=0.6)
    ax[2].plot(Q, R['MFDFA']['tau_q'] - th, 'v-', color=C_DFA, mfc='none', label='MFDFA')
    ax[2].plot(Q, R['MFDMA']['tau_q'] - th, 'o-', color=C_DMA, mfc='none', label='MFDMA')
    for t in TAUS:
        ax[2].plot(Q, R[t]['tau_q'] - th, '-', color=TCOL[t], label=rf'$\lambda={t}$')
    ax[2].set(xlabel='$q$', ylabel=r'$\Delta\tau(q)$')
    ax[2].legend(frameon=False, fontsize=5.3, ncol=2, loc='lower left')
    lab(ax[2], '(c)', x=0.85)
    ax[3].plot(a_th, f_th, 'k-', lw=1.6, label='theory')
    ax[3].plot(R['MFDFA']['alpha'], R['MFDFA']['f_alpha'], 'v', color=C_DFA, mfc='none',
               label='MFDFA')
    ax[3].plot(R['MFDMA']['alpha'], R['MFDMA']['f_alpha'], 'o', color=C_DMA, mfc='none',
               label='MFDMA')
    ax[3].plot(R[0.25]['alpha'], R[0.25]['f_alpha'], 's', color=TCOL[0.25], ms=2.5,
               label=r'$\lambda=0.25$')
    ax[3].set(xlabel=r'$\alpha$', ylabel=r'$f(\alpha)$')
    lab(ax[3], '(d)', x=0.85)
    fig.tight_layout(w_pad=0.4)
    save(fig, 'fig03_binomial1d')


# ------------------------------------------------------------------ Fig 4
def fig4():
    R = cached('binomial_p03_k16', run_binomial)
    th = binomial_tau_theory(Q, 0.3)
    hth = np.where(np.abs(Q) > 1e-9, (th + 1) / np.where(Q == 0, 1, Q),
                   -(np.log(0.3) + np.log(0.7)) / (2 * np.log(2)))
    def mono():
        out = {}
        for H in (0.2, 0.35, 0.5, 0.65, 0.8, 0.9):
            hs = {'MFDFA': [], 'MFDMA0.5': [], 'MFDMA': []}
            for t in TAUS: hs[t] = []
            for sd in range(3):
                x = fgn(2 ** 16, H, seed=9100 + sd)
                hs['MFDFA'].append(mfdfa(x, Q, scales=SD)['hq'])
                hs['MFDMA'].append(mfdma(x, Q, scales=SM, theta=0.0)['hq'])
                hs['MFDMA0.5'].append(mfdma(x, Q, scales=SM, theta=0.5)['hq'])
                for t, r in mfqdma_multi(x, Q, scales=SM, theta=0.0,
                                         taus=TAUS).items():
                    hs[t].append(r['hq'])
            out[H] = {k: np.array(v) for k, v in hs.items()}
        return out
    M = cached('fgn_mono', mono)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.0))
    ax[0].plot(Q, hth, 'k-', lw=1.6, label='theory')
    for t in TAUS:
        ax[0].plot(Q, R[t]['hq'], '-', color=TCOL[t], label=rf'$\lambda={t}$')
    ax[0].plot(Q, R['MFDMA']['hq'], 'o', color=C_DMA, mfc='none', label='MFDMA')
    ax[0].set(xlabel='$q$', ylabel=r'$h(q,\lambda)$')
    ax[0].legend(frameon=False, fontsize=5.6, loc='upper right')
    lab(ax[0], '(a)', y=0.12)
    Hs = sorted(M)
    q2 = int(np.argmin(np.abs(Q - 2)))
    ax[1].plot([0.15, 0.95], [0.15, 0.95], 'k-', lw=0.7)
    for k, c, mk, lb in [('MFDFA', C_DFA, 'v', 'MFDFA'), ('MFDMA', C_DMA, 'o', r'MFDMA $\theta=0$')]:
        ax[1].errorbar(Hs, [M[H][k][:, q2].mean() for H in Hs],
                       [M[H][k][:, q2].std() for H in Hs], fmt=mk, color=c,
                       mfc='none', capsize=1.5, label=lb)
    for t in (0.1, 0.5, 0.9):
        ax[1].errorbar(np.array(Hs) + 0.008 * (t - 0.5) * 4,
                       [M[H][t][:, q2].mean() for H in Hs],
                       [M[H][t][:, q2].std() for H in Hs], fmt='s', ms=2.4,
                       color=TCOL[t], capsize=1.5, label=rf'MFQDMA $\lambda={t}$')
    ax[1].set(xlabel='$H$', ylabel='$h(2)$')
    ax[1].legend(frameon=False, fontsize=5.6, loc='upper left')
    lab(ax[1], '(b)', x=0.86, y=0.12)
    for k, c, mk, lb in [('MFDFA', C_DFA, 'v', 'MFDFA'), ('MFDMA', C_DMA, 'o', 'MFDMA'),
                         ('MFDMA0.5', '#8fa0e0', 'D', r'MFDMA $\theta=0.5$')]:
        ax[2].plot(Hs, [np.mean(np.ptp(M[H][k], axis=1)) for H in Hs], mk + '-',
                   color=c, mfc='none', label=lb)
    for t in TAUS:
        ax[2].plot(Hs, [np.mean(np.ptp(M[H][t], axis=1)) for H in Hs], 's-',
                   ms=2.4, color=TCOL[t], label=rf'$\lambda={t}$')
    ax[2].set(xlabel='$H$', ylabel=r'$\max_q h-\min_q h$')
    ax[2].legend(frameon=False, fontsize=5.3, ncol=2, loc='upper left')
    lab(ax[2], '(c)', x=0.86)
    fig.tight_layout(w_pad=0.5)
    save(fig, 'fig04_tauinv_mono')


# ------------------------------------------------------------------ Fig 5
def fig5():
    d = json.load(open(f'{RES}/advantage_all.json'))['contamination']
    import pandas as pd
    D = pd.DataFrame(d)
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.95))
    names = {'outlier': r'(a) $\pm20\sigma$ outliers', 'student_t': r'(b) Student-$t_2$',
             'levy': '(c) Lévy ' + r'$\alpha_L=1.2$'}
    for a, kind in zip(ax[:3], ('outlier', 'student_t', 'levy')):
        S = D[D.kind == kind]
        for m, c, mk, lb in [('MFDFA', C_DFA, 'v', 'MFDFA'), ('MFDMA', C_DMA, 'o', 'MFDMA')]:
            g = S[S.method == m].groupby('frac').rmse.mean()
            a.plot(100 * g.index, g.values, mk + '-', color=c, mfc='none', label=lb)
        for t in TAUS:
            g = S[S.method == f'MFQDMA_tau{t}'].groupby('frac').rmse.mean()
            a.plot(100 * g.index, g.values, 's-', ms=2.3, color=TCOL[t], label=rf'$\lambda={t}$')
        a.set(yscale='log', xlabel=r'$\varepsilon$ (%)', title=names[kind])
    ax[0].set_ylabel(r'RMSE of $\tau(q)$')
    ax[0].legend(frameon=False, fontsize=5.0, ncol=2, loc='lower right')
    # (d) RMSE ratio MFQDMA / MFDMA, paired by contamination replicate
    for kind, ls in (('outlier', '-'), ('student_t', '--'), ('levy', ':')):
        S = D[(D.kind == kind) & (D.frac > 0)]
        base = S[S.method == 'MFDMA'].set_index(['frac', 'rep']).rmse
        for t in (0.1, 0.25, 0.9):
            v = S[S.method == f'MFQDMA_tau{t}'].set_index(['frac', 'rep']).rmse
            r = (v / base).groupby('frac').mean()
            ax[3].plot(100 * r.index, r.values, 's' + ls, ms=2.3, color=TCOL[t],
                       label=(rf'$\lambda={t}$' if kind == 'outlier' else None))
    ax[3].axhline(1, color=C_DMA, lw=0.8)
    ax[3].plot([], [], 'k-', label='outlier'); ax[3].plot([], [], 'k--', label='Student')
    ax[3].plot([], [], 'k:', label='Lévy')
    ax[3].set(xlabel=r'$\varepsilon$ (%)', ylabel='RMSE / RMSE$_{\\rm MFDMA}$', yscale='log', ylim=(0.5, 25),
              title='(d) relative to MFDMA')
    ax[3].legend(frameon=False, fontsize=5.0, ncol=2, loc='upper right')
    fig.tight_layout(w_pad=0.4)
    save(fig, 'fig05_contamination')


# ------------------------------------------------------------------ Fig 6
def fig6():
    import pandas as pd
    d = json.load(open(f'{RES}/advantage_all.json'))
    L = pd.DataFrame(d['levy']); M = pd.DataFrame(d['monotonic'])
    fig, ax = plt.subplots(1, 2, figsize=(W1 * 1.25, 1.75))
    for m, c, mk, lb in [('MFDFA', C_DFA, 'v', 'MFDFA'), ('MFDMA', C_DMA, 'o', 'MFDMA')] + \
            [(f'MFQDMA_tau{t}', TCOL[t], 's', rf'$\lambda={t}$') for t in TAUS]:
        g = L[L.method == m].groupby('alpha').width.mean()
        ax[0].plot(g.index, g.values, mk + '-', color=c, mfc='none' if mk != 's' else c,
                   ms=2.5, label=lb)
    ax[0].set(xlabel=r'stable index $\alpha_L$', ylabel=r'spurious $\max_q h-\min_q h$')
    ax[0].legend(frameon=False, fontsize=5, ncol=1)
    lab(ax[0], '(a)', x=0.85)
    tr = ['sqrt', 'cuberoot', 'log', 'gaussianize', 'cube']
    meths = ['MFDFA', 'MFDMA'] + [f'MFQDMA_tau{t}' for t in TAUS]
    cols = [C_DFA, C_DMA] + [TCOL[t] for t in TAUS]
    w = 0.11
    for j, (m, c) in enumerate(zip(meths, cols)):
        v = [M[(M.method == m) & (M['transform'] == t)].mean_dev.mean() for t in tr]
        ax[1].bar(np.arange(len(tr)) + (j - 3) * w, v, w, color=c)
    ax[1].set_xticks(range(len(tr)))
    ax[1].set_xticklabels(['$\\sqrt{\\cdot}$', '$\\sqrt[3]{\\cdot}$', 'log', 'Gauss.', '$(\\cdot)^3$'])
    ax[1].set(ylabel=r'mean $|h_g(q)-h(q)|$')
    lab(ax[1], '(b)')
    fig.tight_layout(w_pad=0.5)
    save(fig, 'fig06_negative')


# ------------------------------------------------------------------ Fig 7
P2D = (0.05, 0.15, 0.25, 0.55)


def fig7():
    def run():
        Z = cascade_2d(P2D, 9)
        s = np.unique(np.logspace(np.log10(6), np.log10(64), 8).astype(int))
        out = {'MFDMA': mfdma_2d(Z, Q, scales=s, theta=0.0, tau=None, demean=False)}
        for t in (0.25, 0.5):
            out[t] = mfdma_2d(Z, Q, scales=s, theta=0.0, tau=t, demean=False)
        return out
    R = cached('cascade2d_A_k9', run)
    Z = cascade_2d(P2D, 9)
    qq = np.linspace(-4, 4, 161)
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.95))
    im = ax[0].imshow(Z / Z.mean(), norm=LogNorm(vmin=1e-3, vmax=30), cmap='magma',
                      interpolation='nearest')
    ax[0].set(xticks=[], yticks=[])
    plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
    ax[0].set_title(r'(a) $\mu/\langle\mu\rangle$, $512^2$')
    r = R[0.25]
    for qi, qv in enumerate(Q):
        if qv in (-4, -2, 0, 2, 4):
            F = r['Fq'][qi]; k = np.isfinite(F)
            c = plt.cm.viridis((qv + 4) / 8.5)
            sh = 10 ** (0.8 * qv / 2)
            ax[1].loglog(r['scales'][k], F[k] * sh, 'o', color=c, ms=2.5, label=f'$q={int(qv)}$')
            ax[1].loglog(r['scales'][k], np.exp(np.polyval(np.polyfit(
                np.log(r['scales'][k]), np.log(F[k]), 1), np.log(r['scales'][k]))) * sh,
                '-', color=c, lw=0.7)
    ax[1].set(xlabel='$n$', ylabel='$F_q(n)$ (shifted)', title=r'(b) 2D-MFQDMA, $\lambda=0.25$')
    logticks(ax[1], [8, 16, 32, 64])
    ax[1].legend(frameon=False, fontsize=5.3, loc='lower right')
    th = cascade2d_tau(Q, P2D)
    ax[2].axhline(0, color='k', lw=0.6)
    ax[2].plot(Q, R['MFDMA']['tau_q'] - th, 'o-', color=C_DMA, mfc='none', label='2D-MFDMA')
    for t in (0.25, 0.5):
        ax[2].plot(Q, R[t]['tau_q'] - th, 's-', ms=2.5, color=TCOL[t],
                   label=rf'2D-MFQDMA $\lambda={t}$')
    ax[2].set(xlabel='$q$', ylabel=r'$\Delta\tau(q)$', title=r'(c) $\tau(q)-\tau_{\rm th}(q)$')
    ax[2].legend(frameon=False, fontsize=5.5)
    ax[3].plot(cascade2d_alpha(qq, P2D), cascade2d_falpha(qq, P2D), 'k-', lw=1.6, label='theory')
    ax[3].plot(R['MFDMA']['alpha'], R['MFDMA']['f_alpha'], 'o', color=C_DMA, mfc='none',
               label='2D-MFDMA')
    for t in (0.25, 0.5):
        ax[3].plot(R[t]['alpha'], R[t]['f_alpha'], 's', ms=2.5, color=TCOL[t],
                   label=rf'$\lambda={t}$')
    ax[3].set(xlabel=r'$\alpha$', ylabel=r'$f(\alpha)$', title=r'(d) spectrum')
    ax[3].legend(frameon=False, fontsize=5.5, loc='lower center')
    fig.tight_layout(w_pad=0.4)
    save(fig, 'fig07_cascade2d')
    return R


# ------------------------------------------------------------------ Fig 8
def fig8():
    import pandas as pd
    d = json.load(open(f'{RES}/analysis_2d_all.json'))
    A = pd.DataFrame(d['A']); A = A[A.k == 8]
    C = pd.DataFrame(d['C'])
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.05),
                           gridspec_kw=dict(width_ratios=[1.05, 1, 1]))
    rows = ['mean', 0.25, 0.5, 0.75]
    M = np.zeros((4, 3))
    for j, th in enumerate((0.0, 0.5, 1.0)):
        M[0, j] = A[A.method == f'MFDMA2D_th{th}'].rmse.mean()
        for i, t in enumerate((0.25, 0.5, 0.75)):
            M[i + 1, j] = A[A.method == f'MFQDMA2D_th{th}_tau{t}'].rmse.mean()
    im = ax[0].imshow(M, cmap='viridis_r', aspect='auto')
    for i in range(4):
        for j in range(3):
            ax[0].text(j, i, f'{M[i, j]:.3f}', ha='center', va='center', fontsize=6.5,
                       color='w' if M[i, j] > M.mean() else 'k')
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels([r'$\theta=0$', r'$\theta=0.5$', r'$\theta=1$'])
    ax[0].set_yticks(range(4)); ax[0].set_yticklabels(['mean', r'$\lambda=0.25$', r'$\lambda=0.5$', r'$\lambda=0.75$'])
    ax[0].set_title(r'(a) RMSE of $\tau(q)$, $256^2$ cascades')
    for a, kind, tt in zip(ax[1:], ('outlier', 'student_t'),
                           (r'(b) $\pm20\sigma$ outlier pixels', r'(c) Student-$t_2$ pixels')):
        S = C[C.kind == kind]
        g = S[S.method == 'MFDMA2D_th0.0'].groupby('frac').rmse.mean()
        a.plot(100 * g.index, g.values, 'o-', color=C_DMA, mfc='none', label='2D-MFDMA')
        for t in (0.25, 0.5, 0.75):
            g = S[S.method == f'MFQDMA2D_th0.0_tau{t}'].groupby('frac').rmse.mean()
            a.plot(100 * g.index, g.values, 's-', ms=2.5, color=TCOL[t], label=rf'$\lambda={t}$')
        a.set(xlabel=r'$\varepsilon$ (%)', yscale='log', title=tt)
    ax[1].set_ylabel(r'RMSE of $\tau(q)$')
    ax[1].legend(frameon=False, fontsize=5.6)
    fig.tight_layout(w_pad=0.5)
    save(fig, 'fig08_2d_theta_contam')


# ------------------------------------------------------------------ Fig 9
def fig9():
    import pandas as pd
    B = pd.DataFrame(json.load(open(f'{RES}/analysis_2d_all.json'))['B'])
    Mo = json.load(open(f'{RES}/paper_2d_mono.json'))
    D = pd.DataFrame(Mo['rows'])
    q = np.array(Mo['q'])
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.0))
    meth = [('MFDMA2D_th0', 'MFDMA2D_th0.0', C_DMA, 'o', '2D-MFDMA')] + \
           [(f'MFQDMA2D_th0_tau{t}', f'MFQDMA2D_th0.0_tau{t}', TCOL[t], 's',
             rf'$\lambda={t}$') for t in (0.25, 0.5, 0.75)]
    ax[0].plot([0.15, 0.95], [0.15, 0.95], 'k-', lw=0.7, label='$h=H$')
    for mi, mr, c, mk, lb in meth:
        g = B[B.method == mr].groupby('H').h2.mean()
        ax[0].plot(g.index, g.values, mk + '--', color=c, mfc='none', ms=2.5)
        g = D[D.method == mi].groupby('H').h2
        ax[0].errorbar(g.mean().index, g.mean(), g.std(), fmt=mk + '-', color=c,
                       ms=2.5, capsize=1.2, label=lb)
    ax[0].text(0.55, 1.75, 'raw surface $X$ (dashed)', fontsize=6.5, ha='center')
    ax[0].text(0.6, 0.25, r'increments $\Delta_1\Delta_2X$', fontsize=6.5)
    ax[0].set(xlabel='$H$', ylabel='$h(2)$', ylim=(0, 2.3))
    ax[0].legend(frameon=False, fontsize=5.4, loc='center left')
    lab(ax[0], '(a)', y=0.97)
    for H, ls in ((0.3, '-'), (0.7, '--')):
        for mi, mr, c, mk, lb in meth:
            hq = np.array(D[(D.method == mi) & (D.H == H)].hq.tolist()).mean(0)
            ax[1].plot(q, hq, ls, color=c)
        ax[1].axhline(H, color='k', lw=0.6, ls=ls)
    ax[1].set(xlabel='$q$', ylabel='$h(q)$', title=r'increments, $H=0.3$ (—), $0.7$ (- -)')
    lab(ax[1], '(b)')
    r = D[(D.method == 'MFQDMA2D_th0_tau0.25') & (D.H == 0.5) & (D.seed == 0)].iloc[0]
    F = np.array(r['Fq']); s = np.array(r['scales'])
    for qi, qv in enumerate(q):
        if qv in (-4, -2, 0, 2, 4):
            c = plt.cm.viridis((qv + 4) / 8.5); sh = 10 ** (0.5 * qv / 2)
            ax[2].loglog(s, F[qi] * sh, 'o', color=c, ms=2.5, label=f'$q={int(qv)}$')
            ax[2].loglog(s, np.exp(np.polyval(np.polyfit(np.log(s), np.log(F[qi]), 1),
                                              np.log(s))) * sh, '-', color=c, lw=0.7)
    logticks(ax[2], [6, 12, 24])
    ax[2].set(xlabel='$n$', ylabel='$F_q(n)$ (shifted)',
              title=r'2D-MFQDMA $\lambda=0.25$, $H=0.5$')
    ax[2].legend(frameon=False, fontsize=5.3, loc='lower right')
    lab(ax[2], '(c)')
    fig.tight_layout(w_pad=0.5)
    save(fig, 'fig09_2d_monofractal')


# ------------------------------------------------------------------ Fig 10
def fig10():
    from .analysis_2d import load_real_fields, sc2d
    F = load_real_fields()['dem_terrain']
    G = np.diff(np.diff(F, axis=0), axis=1)
    s = sc2d(min(G.shape), lo=6, hi=max(12, min(G.shape) // 8), n=8)
    def run():
        rng = np.random.RandomState(5)
        Gs = rng.permutation(G.ravel()).reshape(G.shape)
        Fg = np.fft.fft2(G - G.mean())
        ph = np.angle(np.fft.fft2(rng.randn(*G.shape)))     # Hermitian phases
        Gp = np.real(np.fft.ifft2(np.abs(Fg) * np.exp(1j * ph)))
        out = {}
        for cond, Z in (('orig', G), ('shuf', Gs), ('phase', Gp)):
            out[cond] = {'MFDMA': mfdma_2d(Z, Q, scales=s, theta=0.0, tau=None, demean=True)}
            for t in (0.25, 0.5, 0.75):
                out[cond][t] = mfdma_2d(Z, Q, scales=s, theta=0.0, tau=t, demean=True)
        return out
    R = cached('dem_surrogates', run)
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.95),
                           gridspec_kw=dict(width_ratios=[1.05, 1.05, 1, 1]))
    ax[0].imshow(F, cmap='gray', interpolation='nearest')
    ax[0].set(xticks=[], yticks=[]); lab(ax[0], r'(a) DEM $X$')
    lim = np.percentile(np.abs(G), 99)
    ax[1].imshow(G, cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
    ax[1].set(xticks=[], yticks=[]); lab(ax[1], r'(b) $\Delta_1\Delta_2X$')
    r = R['orig'][0.25]
    for qi, qv in enumerate(Q):
        if qv in (-4, -2, 0, 2, 4):
            c = plt.cm.viridis((qv + 4) / 8.5); sh = 10 ** (0.6 * qv / 2)
            Fq = r['Fq'][qi]
            ax[2].loglog(s, Fq * sh, 'o', color=c, ms=2.4, label=f'$q={int(qv)}$')
            ax[2].loglog(s, np.exp(np.polyval(np.polyfit(np.log(s), np.log(Fq), 1),
                                              np.log(s))) * sh, '-', color=c, lw=0.7)
    logticks(ax[2], [6, 12, 24, 42])
    ax[2].set(xlabel='$n$', ylabel='$F_q(n)$ (shifted)')
    ax[2].legend(frameon=False, fontsize=5, loc='upper left')
    lab(ax[2], r'(c) $\lambda=0.25$')
    for cond, ls, mfc in (('orig', '-', None), ('shuf', ':', 'none'), ('phase', '--', 'none')):
        for k, c, mk in (('MFDMA', C_DMA, 'o'), (0.25, TCOL[0.25], 's')):
            rr = R[cond][k]
            ax[3].plot(rr['alpha'], rr['f_alpha'], mk + ls, color=c, ms=2.0,
                       mfc=c if mfc is None else 'none', lw=0.8)
    ax[3].plot([], [], 'o-', color=C_DMA, label='2D-MFDMA')
    ax[3].plot([], [], 's-', color=TCOL[0.25], label=r'$\lambda=0.25$')
    ax[3].plot([], [], 'k-', label='original'); ax[3].plot([], [], 'k:', label='shuffled')
    ax[3].plot([], [], 'k--', label='phase-rand.')
    ax[3].set(xlabel=r'$\alpha$', ylabel=r'$f(\alpha)$')
    ax[3].legend(frameon=False, fontsize=5, loc='lower center')
    lab(ax[3], '(d)')
    fig.tight_layout(w_pad=0.5)
    save(fig, 'fig10_dem')
    for cond in R:
        for k in R[cond]:
            rr = R[cond][k]; a = rr['alpha'][np.isfinite(rr['alpha'])]
            print('DEM', cond, k, 'h2=%.3f dalpha=%.3f R2=%.4f' % (
                rr['hq'][np.argmin(np.abs(Q - 2))], a.max() - a.min(), np.nanmean(rr['r2'])))


# ------------------------------------------------------------------ Fig 11
def fig11():
    D = json.load(open(f'{RES}/real_eeg.json'))
    recs = D['records']; SETS = ['Z', 'O', 'N', 'F', 'S']
    meths = [('MFDFA', C_DFA, 'MFDFA'), ('MFDMA_th0', C_DMA, 'MFDMA')] + \
            [(f'MFQDMA_tau{t}', TCOL[t], rf'$\lambda={t}$') for t in TAUS]
    def W(r, c, m):
        return r['original'][m]['width'] if c == 'original' else \
            float(np.mean([s[m]['width'] for s in r[c]]))
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.0))
    w = 0.11
    for j, (m, c, lb) in enumerate(meths):
        v = [np.mean([W(r, 'original', m) for r in recs if r['set'] == s]) for s in SETS]
        ax[0].bar(np.arange(5) + (j - 3) * w, v, w, color=c, label=lb)
    ax[0].set_xticks(range(5)); ax[0].set_xticklabels(SETS)
    ax[0].set(ylabel=r'$\Delta\alpha$ (originals)', xlabel='Bonn set')
    ax[0].legend(frameon=False, fontsize=5, ncol=2, loc='upper left')
    lab(ax[0], '(a)', x=0.9)
    for j, (m, c, lb) in enumerate(meths):
        for cond, mk in (('shuffled', 'v'), ('phase_rand', '^')):
            v = []
            for s in SETS:
                rs = [r for r in recs if r['set'] == s]
                v.append(100 * np.mean([W(r, cond, m) for r in rs]) /
                         np.mean([W(r, 'original', m) for r in rs]))
            ax[1].plot(np.arange(5) + (j - 3) * 0.08, v, mk, color=c, ms=3,
                       mfc=c if cond == 'phase_rand' else 'none')
    ax[1].axhline(100, color='k', lw=0.6)
    ax[1].plot([], [], 'kv', mfc='none', label='shuffled'); ax[1].plot([], [], 'k^', label='phase-rand.')
    ax[1].set_xticks(range(5)); ax[1].set_xticklabels(SETS)
    ax[1].set(ylabel=r'$\Delta\alpha_{\rm surr}/\Delta\alpha$ (%)', xlabel='Bonn set', ylim=(0, 135))
    ax[1].legend(frameon=False, fontsize=6, loc='center right')
    lab(ax[1], '(b)')
    for j, (m, c, lb) in enumerate(meths):
        v = [np.mean([r['original'][m]['r2'] for r in recs if r['set'] == s]) for s in SETS]
        ax[2].plot(range(5), v, 'o-' if 'QDMA' in m else 'v--', color=c, ms=2.5, label=lb)
    ax[2].set_xticks(range(5)); ax[2].set_xticklabels(SETS)
    ax[2].set(ylabel=r'mean fit $R^2$', xlabel='Bonn set')
    lab(ax[2], '(c)')
    fig.tight_layout(w_pad=0.5)
    save(fig, 'fig11_eeg_surrogates')



# ------------------------------------------------------------------ EEG main
def _eeg():
    z = np.load(RES / 'eeg_main.npz')
    Qe = z['Q']; TA = [float(t) for t in z['TAUS']]; SETS = ['Z', 'O', 'N', 'F', 'S']
    H = {s: {t: z[f'H_{s}_{t}'] for t in TA} for s in SETS}
    HM = {s: z[f'HM_{s}'] for s in SETS}
    return Qe, TA, SETS, H, HM


ECOL = {'Z': '#1b9e77', 'O': '#66a61e', 'N': '#7570b3', 'F': '#d95f02', 'S': '#e7298a'}


def fig12():
    Qe, TA, SETS, H, HM = _eeg()
    LBL = {'Z': 'Z healthy, open', 'O': 'O healthy, closed', 'N': 'N interictal, contra.',
           'F': 'F interictal, focus', 'S': 'S ictal'}
    M = {s: np.array([[np.nanmedian(H[s][t][:, qi]) for qi in range(len(Qe))] for t in TA])
         for s in SETS}
    vmin = min(m.min() for m in M.values()); vmax = max(m.max() for m in M.values())
    fig, ax = plt.subplots(1, 5, figsize=(W2, 1.75), sharey=True,
                           gridspec_kw=dict(wspace=0.08))
    yb = np.arange(len(TA) + 1) - 0.5
    for a, s in zip(ax, SETS):
        im = a.pcolormesh(Qe, np.arange(len(TA)), M[s], cmap='RdYlBu_r', shading='auto',
                          vmin=vmin, vmax=vmax)
        a.set_title(f'({"abcde"[SETS.index(s)]}) set {s}', fontsize=8, loc='left')
        a.set_xlabel('$q$')
    ax[0].set_yticks(range(len(TA))); ax[0].set_yticklabels([str(t) for t in TA])
    ax[0].set_ylabel(r'$\lambda$')
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label(r'median $h(q,\lambda)$', fontsize=7); cb.ax.tick_params(labelsize=6)
    save(fig, 'fig12_eeg_surfaces')


def fig13():
    from .stats_tools import cliffs_delta
    from scipy.stats import spearmanr
    Qe, TA, SETS, H, HM = _eeg()
    def Wq(s, qq):
        qi = int(np.argmin(np.abs(Qe - qq)))
        Mx = np.column_stack([H[s][t][:, qi] for t in TA])
        return np.nanmax(Mx, 1) - np.nanmin(Mx, 1)
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.95),
                           gridspec_kw=dict(width_ratios=[1, 1, 1.15, 1]))
    data = [Wq(s, -5.0) for s in SETS]
    bp = ax[0].boxplot(data, tick_labels=SETS, patch_artist=True, widths=0.6, showfliers=False)
    for p, s in zip(bp['boxes'], SETS):
        p.set_facecolor(ECOL[s]); p.set_alpha(0.75); p.set_linewidth(0.6)
    for m in bp['medians']: m.set_color('k')
    ax[0].set(ylabel=r'$W(q{=}-5)$', xlabel='Bonn set')
    lab(ax[0], '(a)')
    for s in SETS:
        ax[1].plot(Qe[::2], [np.nanmedian(Wq(s, qq)) for qq in Qe[::2]], 'o-', ms=2.2,
                   color=ECOL[s], label=s)
    ax[1].set(xlabel='$q$', ylabel=r'median $W(q)$')
    ax[1].legend(frameon=False, ncol=2, fontsize=5.6)
    lab(ax[1], '(b)')
    D = np.array([[cliffs_delta(H['F'][t][:, qi], H['N'][t][:, qi]) for qi in range(len(Qe))]
                  for t in TA])
    im = ax[2].pcolormesh(Qe, np.arange(len(TA)), np.abs(D), cmap='magma', shading='auto')
    ax[2].set_yticks(range(len(TA))); ax[2].set_yticklabels([str(t) for t in TA])
    ax[2].set(xlabel='$q$', ylabel=r'$\lambda$')
    cb = plt.colorbar(im, ax=ax[2], fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=6)
    ax[2].set_title(r"(c) F vs N, $|\delta_{\rm Cliff}|$", loc='left', fontsize=8)
    tasks = ['F/N', 'S/F', 'Z/S', 'Z/O']
    A = json.load(open(RES / 'eeg_auc.json'))      # experiments_paperI.run_eeg_classification()
    tk = ['F vs N', 'S vs F', 'Z vs S', 'Z vs O']
    base = [A[k]['base'] for k in tk]
    best1 = [max(A[k]['per_lambda'].values()) for k in tk]
    x = np.arange(4); w = 0.38
    ax[3].bar(x - w / 2, base, w, color=C_DMA, label='MFDMA $h(q)$')
    ax[3].bar(x + w / 2, best1, w, color=TCOL[0.25], label=r'MFQDMA $h(q,\lambda^*)$')
    ax[3].set_xticks(x); ax[3].set_xticklabels(tasks)
    ax[3].set(ylim=(0.6, 1.05), ylabel='cross-validated AUC')
    ax[3].legend(frameon=False, fontsize=5.4, loc='upper left')
    lab(ax[3], '(d)')
    fig.tight_layout(w_pad=1.6)
    save(fig, 'fig13_eeg_tau_resolved')
    allv = np.concatenate(data)
    sev = {'Z': 0, 'O': 0, 'N': 1, 'F': 2, 'S': 3}
    alls = np.concatenate([np.full(len(d), sev[s]) for d, s in zip(data, SETS)])
    print('medians W', [round(float(np.median(d)), 4) for d in data],
          'spearman', spearmanr(allv, alls))




# ================================================================== FINANCE
FIN = RES / 'finance.json'
FSER = ['SP500', 'DJ', 'NASDAQ', 'FTSE', 'DAX', 'CAC', 'SMI', 'EURSTOXX', 'NIKKEI',
        'HSI', 'SSEC', 'CSI', 'EUR_USD', 'GBP_USD', 'JPY_USD', 'GOLD', 'OIL_Brent', 'BTC']
FSHORT = {'SP500': 'SPX', 'DJ': 'DJI', 'NASDAQ': 'NDX', 'FTSE': 'FTSE', 'DAX': 'DAX',
          'CAC': 'CAC', 'SMI': 'SMI', 'EURSTOXX': 'SX5E', 'NIKKEI': 'N225', 'HSI': 'HSI',
          'SSEC': 'SSEC', 'CSI': 'CSI300', 'EUR_USD': 'EUR', 'GBP_USD': 'GBP',
          'JPY_USD': 'JPY', 'GOLD': 'Gold', 'OIL_Brent': 'Brent', 'BTC': 'BTC'}
FMETH = [('MFDFA', C_DFA, 'v', 'MFDFA'), ('MFDMA', C_DMA, 'o', 'MFDMA')] + \
        [(f'MFQDMA_{t}', TCOL[t], 's', rf'$\lambda={t}$') for t in TAUS]
QF = np.arange(-4, 4.5, 0.5)


def _fin():
    return json.load(open(FIN))


def fig14():
    """S&P 500 and SSEC: returns, F_q(n), h(q), f(alpha)."""
    from . import finance as fz
    F = _fin()['AB']
    fig, ax = plt.subplots(1, 4, figsize=(W2, 1.95))
    d, r = fz.returns('SP500')
    t = d.astype('datetime64[D]').astype(float) / 365.25 + 1970
    ax[0].plot(t, r, lw=0.3, color='k')
    i = int(np.argmin(r)); ax[0].annotate('19 Oct 1987', xy=(t[i], r[i]), xytext=(t[i] - 30, r[i] + 0.03),
                                          fontsize=6, arrowprops=dict(arrowstyle='->', lw=0.5))
    ax[0].set(xlabel='year', ylabel=r'$r(t)=\Delta\ln P$')
    lab(ax[0], '(a) S&P 500')
    o = F['SP500']['orig']['MFQDMA_0.25']
    s = np.array(o['scales']); Fq = np.array(o['Fq'])
    for qi, qv in enumerate(QF):
        if qv in (-4, -2, 0, 2, 4):
            c = plt.cm.viridis((qv + 4) / 8.5); sh = 10 ** (0.6 * qv / 2)
            ax[1].loglog(s, Fq[qi] * sh, 'o', color=c, ms=2.3, label=f'$q={int(qv)}$')
            ax[1].loglog(s, np.exp(np.polyval(np.polyfit(np.log(s), np.log(Fq[qi]), 1),
                                              np.log(s))) * sh, '-', color=c, lw=0.7)
    ax[1].set(xlabel='$n$ (days)', ylabel='$F_q(n)$ (shifted)')
    ax[1].legend(frameon=False, fontsize=5, loc='upper left')
    lab(ax[1], r'(b) $\lambda=0.25$')
    for m, c, mk, lb in FMETH:
        if m in ('MFQDMA_0.75', 'MFQDMA_0.9'):
            continue
        ax[2].plot(QF, F['SP500']['orig'][m]['hq'], mk + '-', color=c, ms=2.3,
                   mfc='none' if mk != 's' else c, label=lb)
    ax[2].axhline(0.5, color='k', lw=0.5, ls=':')
    ax[2].set(xlabel='$q$', ylabel='$h(q)$')
    ax[2].legend(frameon=False, fontsize=5.2)
    lab(ax[2], '(c)')
    for nm, ls in (('SP500', '-'), ('SSEC', '--'), ('BTC', ':')):
        for m, c, mk, lb in FMETH:
            if m not in ('MFDMA', 'MFQDMA_0.25'):
                continue
            o = F[nm]['orig'][m]
            ax[3].plot(o['alpha'], o['f_alpha'], ls, color=c, lw=1.0)
    for nm, ls in (('SP500', '-'), ('SSEC', '--'), ('BTC', ':')):
        ax[3].plot([], [], 'k' + ls, label=fz.LABEL[nm])
    ax[3].plot([], [], '-', color=C_DMA, label='MFDMA'); ax[3].plot([], [], '-', color=TCOL[0.25], label=r'$\lambda=0.25$')
    ax[3].set(xlabel=r'$\alpha$', ylabel=r'$f(\alpha)$')
    ax[3].legend(frameon=False, fontsize=4.8, loc='lower left')
    lab(ax[3], '(d)')
    fig.tight_layout(w_pad=0.4)
    save(fig, 'fig14_fin_spectra')


def fig15():
    """Cross-market spectral width and surrogate decomposition."""
    F = _fin()['AB']
    fig, ax = plt.subplots(2, 1, figsize=(W2, 3.3), sharex=True)
    x = np.arange(len(FSER)); w = 0.11
    for j, (m, c, mk, lb) in enumerate(FMETH):
        ax[0].bar(x + (j - 3) * w, [F[s]['orig'][m]['dalpha'] for s in FSER], w, color=c, label=lb)
    ax[0].set(ylabel=r'$\Delta\alpha$')
    ax[0].legend(frameon=False, ncol=7, fontsize=5.6, loc='upper left')
    lab(ax[0], '(a) original returns')
    for j, (m, c, mk, lb) in enumerate(FMETH):
        o = np.array([F[s]['orig'][m]['dalpha'] for s in FSER])
        sh = np.array([F[s]['surr'][m]['shuf_dalpha'] for s in FSER])
        ph = np.array([F[s]['surr'][m]['phase_dalpha'] for s in FSER])
        ax[1].plot(x + (j - 3) * w, 100 * sh / o, 'v', color=c, ms=3, mfc='none')
        ax[1].plot(x + (j - 3) * w, 100 * ph / o, '^', color=c, ms=3)
    ax[1].axhline(100, color='k', lw=0.5)
    ax[1].plot([], [], 'kv', mfc='none', label='shuffled'); ax[1].plot([], [], 'k^', label='phase-randomized')
    ax[1].set(ylabel=r'$\Delta\alpha_{\rm surr}/\Delta\alpha$ (%)', yscale='log')
    ax[1].legend(frameon=False, fontsize=6, loc='upper left', ncol=2)
    ax[1].set_xticks(x); ax[1].set_xticklabels([FSHORT[s] for s in FSER], rotation=45, fontsize=6.5)
    lab(ax[1], '(b) surrogates (mean of 5)')
    fig.tight_layout(h_pad=0.4)
    save(fig, 'fig15_fin_markets')


def fig16():
    """Influence of genuine extreme days and of injected data errors."""
    import pandas as pd
    J = _fin()
    C = J['C']
    fig, ax = plt.subplots(1, 4, figsize=(W2, 2.0))
    ks = [1, 5, 10, 20, 50]
    for m, c, mk, lb in FMETH:
        if m in ('MFQDMA_0.75',):
            continue
        med = [np.median([C[s]['rec'][str(k)][m]['dev_h'] for s in FSER]) for k in ks]
        ax[0].plot(ks, med, mk + '-', color=c, ms=2.5, mfc='none' if mk != 's' else c, label=lb)
    ax[0].set(xscale='log', yscale='log', xlabel='$k$ extreme days removed',
              ylabel=r'median $D_h$')
    ax[0].legend(frameon=False, fontsize=5)
    lab(ax[0], '(a) genuine extremes')
    rows = []
    for s in FSER:
        for rr in J['D'][s]['rows']:
            rr = dict(rr); rr['series'] = s; rows.append(rr)
    D = pd.DataFrame(rows)
    for a, kind, tt, xl in zip(ax[1:], ('badtick', 'outlier', 'noise'),
                               ('(b) bad ticks', '(c) return outliers', '(d) microstructure noise'),
                               (r'$\varepsilon$ (%)', r'$\varepsilon$ (%)', r'$\eta$')):
        S = D[D.kind == kind]
        for m, c, mk, lb in FMETH:
            if m in ('MFQDMA_0.75',):
                continue
            g = S[S.method == m].groupby(['series', 'level']).dev_h.mean().groupby('level').median()
            xv = g.index * (100 if kind != 'noise' else 1)
            a.plot(xv, g.values, mk + '-', color=c, ms=2.5, mfc='none' if mk != 's' else c)
        a.set(xscale='log', yscale='log', xlabel=xl)
        lab(a, tt)
    ax[1].set_ylabel(r'median $D_h$')
    fig.tight_layout(w_pad=0.3)
    save(fig, 'fig16_fin_robustness')


def fig17():
    """DJIA constituents (paired) and rolling S&P 500 window."""
    import pandas as pd
    J = _fin()
    rows = []
    for s, v in J['E'].items():
        for rr in v['rows']:
            rr = dict(rr); rr['stock'] = s; rows.append(rr)
    E = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.05), gridspec_kw=dict(width_ratios=[1.1, 1, 1.5]))
    S = E[(E.kind == 'badtick') & (E.level == 0.005)].groupby(['stock', 'method']).dev_h.mean().unstack()
    meths = [m for m, *_ in FMETH]
    bp = ax[0].boxplot([S[m].values for m in meths], patch_artist=True, widths=0.6, showfliers=False)
    for p, (m, c, mk, lb) in zip(bp['boxes'], FMETH):
        p.set_facecolor(c); p.set_alpha(0.8); p.set_linewidth(0.5)
    for mm in bp['medians']: mm.set_color('k')
    ax[0].set_xticks(range(1, len(meths) + 1))
    ax[0].set_xticklabels(['DFA', 'DMA', '.1', '.25', '.5', '.75', '.9'], fontsize=5.5, rotation=60)
    ax[0].set(ylabel='$D_h$', yscale='log')
    lab(ax[0], r'(a) bad ticks, $\varepsilon=0.5\%$')
    S2 = E[(E.kind == 'noise') & (E.level == 0.5)].groupby(['stock', 'method']).dev_h.mean().unstack()
    bp = ax[1].boxplot([S2[m].values for m in meths], patch_artist=True, widths=0.6, showfliers=False)
    for p, (m, c, mk, lb) in zip(bp['boxes'], FMETH):
        p.set_facecolor(c); p.set_alpha(0.8); p.set_linewidth(0.5)
    for mm in bp['medians']: mm.set_color('k')
    ax[1].set_xticks(range(1, len(meths) + 1))
    ax[1].set_xticklabels(['DFA', 'DMA', '.1', '.25', '.5', '.75', '.9'], fontsize=5.5, rotation=60)
    ax[1].set(yscale='log')
    lab(ax[1], r'(b) noise, $\eta=0.5$')
    Fr = J['F']
    t = np.array([np.datetime64(x).astype('datetime64[D]').astype(float) / 365.25 + 1970 for x in Fr['t']])
    for m, c, lb in (('MFDFA', C_DFA, 'MFDFA'), ('MFDMA', C_DMA, 'MFDMA'),
                     ('MFQDMA_0.1', TCOL[0.1], r'MFQDMA $\lambda=0.1$')):
        dd = np.array(Fr['dirty'][m]) - np.array(Fr['clean'][m])
        ax[2].plot(t, dd, '-', color=c, lw=0.9, label=lb + f' (mean $|\\cdot|$={np.mean(np.abs(dd)):.3f})')
    ax[2].axhline(0, color='k', lw=0.5)
    ax[2].set(xlabel='end of 10-yr window', ylabel=r'$\Delta\alpha_{\rm ticks}-\Delta\alpha_{\rm clean}$')
    ax[2].legend(frameon=False, fontsize=5.2, loc='upper left')
    lab(ax[2], '(c) S&P 500, rolling')
    fig.tight_layout(w_pad=0.4)
    save(fig, 'fig17_fin_stocks_rolling')





# ------------------------------------------------------------ manuscript order
# (figure number, output file name, internal name, function). Figures are
# written to figures_out/<output file name>.pdf/.png.
FIGURES = [
    (1, 'Fig01_schematic', 'fig01_schematic', fig1),
    (2, 'Fig02_binomial1d', 'fig03_binomial1d', fig3),
    (3, 'Fig03_boxsum_degeneracy', 'fig02_boxsum_degeneracy', fig2),
    (4, 'Fig04_lambda_invariance_monofractal', 'fig04_tauinv_mono', fig4),
    (5, 'Fig05_contamination', 'fig05_contamination', fig5),
    (6, 'Fig06_negative_results', 'fig06_negative', fig6),
    (7, 'Fig07_cascade2d', 'fig07_cascade2d', fig7),
    (8, 'Fig08_2d_theta_contamination', 'fig08_2d_theta_contam', fig8),
    (9, 'Fig09_2d_monofractal', 'fig09_2d_monofractal', fig9),
    (10, 'Fig10_finance_spectra', 'fig14_fin_spectra', fig14),
    (11, 'Fig11_finance_markets', 'fig15_fin_markets', fig15),
    (12, 'Fig12_finance_robustness', 'fig16_fin_robustness', fig16),
    (13, 'Fig13_finance_stocks_rolling', 'fig17_fin_stocks_rolling', fig17),
    (14, 'Fig14_eeg_surrogates', 'fig11_eeg_surrogates', fig11),
    (15, 'Fig15_eeg_lambda_surfaces', 'fig12_eeg_surfaces', fig12),
    (16, 'Fig16_eeg_lambda_resolved', 'fig13_eeg_tau_resolved', fig13),
    (17, 'Fig17_terrain_dem', 'fig10_dem', fig10),
]
for _num, _out, _internal, _fn in FIGURES:
    FIGNAME[_internal] = _out


def make_figure(number):
    """Draw manuscript figure `number` (1-17) and save it to figures_out/."""
    for num, out, _, fn in FIGURES:
        if num == number:
            t0 = time.time()
            fn()
            print(f'Fig. {num} ({out}) done in {time.time() - t0:.0f} s', flush=True)
            return out
    raise ValueError(f'no figure {number}')
