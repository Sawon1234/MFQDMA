"""Compare regenerated figures with the reference figures of the manuscript."""
import numpy as np
import pandas as pd
from PIL import Image

from . import paths


def compare_figures(names=None):
    """Mean absolute pixel difference (0-255 scale) and the fraction of pixels
    that differ by more than 32 grey levels, regenerated vs reference PNG.
    'identical': pixel-identical; 'close': fewer than 1% of the pixels differ
    noticeably (typically anti-aliasing from another Matplotlib version)."""
    ref_dir, out_dir = paths.REPO / 'figures', paths.FIGURES
    names = names or sorted(p.stem for p in ref_dir.glob('*.png'))
    rows = []
    for nm in names:
        r, o = ref_dir / f'{nm}.png', out_dir / f'{nm}.png'
        if not o.exists():
            rows.append(dict(figure=nm, status='not generated')); continue
        a = np.asarray(Image.open(r).convert('L'), float)
        b = Image.open(o).convert('L')
        if b.size != (a.shape[1], a.shape[0]):
            b = b.resize((a.shape[1], a.shape[0]))
        b = np.asarray(b, float)
        d = np.abs(a - b)
        rows.append(dict(figure=nm, mean_abs_diff=round(float(d.mean()), 3),
                         frac_pixels_changed=round(float((d > 32).mean()), 5),
                         status='identical' if d.max() == 0 else
                         ('close' if (d > 32).mean() < 1e-2 else 'DIFFERS')))
    return pd.DataFrame(rows).set_index('figure')
