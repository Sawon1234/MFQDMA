"""Non-parametric statistics for the MFQDMA EEG analysis.
statsmodels is unavailable in this sandbox, so Benjamini-Hochberg FDR and
Cliff's delta are implemented directly (both are short and exact)."""
import numpy as np
from scipy.stats import mannwhitneyu


def cliffs_delta(a, b):
    """Cliff's delta = P(a>b) - P(a<b), in [-1,1].
    Computed via the Mann-Whitney U identity: delta = 2U/(n1 n2) - 1."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    u = mannwhitneyu(a, b, alternative='two-sided').statistic
    return 2.0 * u / (len(a) * len(b)) - 1.0


def cliff_magnitude(d):
    """Romano et al. (2006) thresholds."""
    a = abs(d)
    if not np.isfinite(a): return 'na'
    if a < 0.147: return 'negligible'
    if a < 0.330: return 'small'
    if a < 0.474: return 'medium'
    return 'large'


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg. Returns (reject, qvals)."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    rej = np.zeros(len(p), bool)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return rej, q
    ps = p[idx]
    order = np.argsort(ps)
    m = len(ps)
    ranked = ps[order]
    qv = ranked * m / (np.arange(m) + 1)
    qv = np.minimum.accumulate(qv[::-1])[::-1]   # enforce monotonicity
    qv = np.minimum(qv, 1.0)
    q[idx[order]] = qv
    rej[idx[order]] = qv <= alpha
    return rej, q


def compare(a, b):
    """Mann-Whitney p, Cliff's delta, and medians for two 1-D samples."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return dict(p=np.nan, delta=np.nan, med_a=np.nan, med_b=np.nan)
    p = mannwhitneyu(a, b, alternative='two-sided').pvalue
    return dict(p=p, delta=cliffs_delta(a, b),
                med_a=float(np.median(a)), med_b=float(np.median(b)))


def cv_auc(X, y, n_splits=10, n_repeats=5, seed=0):
    """Repeated stratified CV AUC with a standardised logistic model.
    Returns (mean_auc, sd_auc)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.impute import SimpleImputer

    X = np.asarray(X, float); y = np.asarray(y, int)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    aucs = []
    for tr, te in cv.split(X, y):
        pipe = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=1.0))
        pipe.fit(X[tr], y[tr])
        s = pipe.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], s))
    return float(np.mean(aucs)), float(np.std(aucs))
