"""Shared plotting style and helpers (PRE column widths, colours)."""
import json
import numpy as np
import matplotlib.pyplot as plt
from . import paths

RES = paths.RESULTS
OUT = paths.FIGURES
CACHE = paths.CACHE

plt.rcParams.update({
    'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8,
    'axes.labelsize': 8.5, 'axes.titlesize': 8.5, 'legend.fontsize': 6.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'lines.linewidth': 1.1,
    'lines.markersize': 3.2, 'axes.linewidth': 0.6, 'figure.dpi': 150,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True})
W1, W2 = 3.4, 7.0                       # PRE single / double column width (in)
Q = np.arange(-4, 4.5, 0.5)
C_DFA, C_DMA = '#7f7f7f', '#3b4cc0'
C_MID, C_PEA = '#2a9d8f', '#000000'
TAUS = [0.1, 0.25, 0.5, 0.75, 0.9]
TCOL = {t: plt.cm.YlOrRd(0.35 + 0.6 * i / 4) for i, t in enumerate(TAUS)}
FIGNAME = {}                            # internal name -> manuscript file name


def lab(ax, s, x=None, y=None):
    t = ax.get_title()
    ax.set_title(s + (' ' + t if t else ''), loc='left', fontsize=8.5)


def logticks(ax, ticks):
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())


def save(fig, name):
    """Save as PDF + PNG under figures_out/ and show inline in notebooks."""
    name = FIGNAME.get(name, name)
    fig.savefig(OUT / f'{name}.pdf'); fig.savefig(OUT / f'{name}.png')
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import display
            display(fig)
    except Exception:
        pass
    plt.close(fig)
    print('wrote', OUT / f'{name}.pdf', flush=True)


def sc(lo, hi, n=14):
    return np.unique(np.logspace(np.log10(lo), np.log10(hi), n).astype(int))


SM, SD = sc(16, 2048), sc(64, 8192)


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def cached(name, fn):
    """Cache an expensive dict-valued computation in cache/<name>.npy."""
    p = CACHE / f'{name}.npy'
    if p.exists():
        return np.load(p, allow_pickle=True).item()
    r = fn(); np.save(p, r, allow_pickle=True); return r


def J(name):
    with open(RES / f'{name}.json') as f:
        return json.load(f)
