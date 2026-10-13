"""Pairwise causal-direction scores, in NumPy/SciPy/scikit-learn.

These replace the four functions pyspi used from the Causal Discovery Toolbox
(`cdt.causality.pairwise`): the ANM independence score, the conditional
distribution similarity statistic, the RECI regression-error score, and IGCI.
Nothing else in cdt was ever used, and cdt eagerly imports its Torch-backed
models at package load, so a ~2 GB dependency and a multi-second import were
being paid for four functions that need none of it.

Each function below is a transcription of the corresponding cdt 0.6
implementation, kept close enough that the numbers are bit-identical on
continuous data (see `tests/test_pairwise_causal.py`, which checks against
values frozen from cdt 0.6 before the dependency was dropped). Where cdt's code
was itself a transcription of the original author's, the original attribution is
kept below.

Provenance and licence
----------------------
Derived from the Causal Discovery Toolbox, Copyright (c) 2018 Diviyan
Kalainathan, MIT licence (a copy of which is retained at
`pyspi/lib/LICENSE-cdt.txt`).

Algorithms:

* HSIC with a Gamma approximation -- Gretton, A., Fukumizu, K., Teo, C.,
  Song, L., Schölkopf, B. & Smola, A. (2007). A kernel statistical test of
  independence. *NIPS*.
* ANM -- Hoyer, P., Janzing, D., Mooij, J., Peters, J. & Schölkopf, B. (2009).
  Nonlinear causal discovery with additive noise models. *NIPS*.
* CDS -- Fonollosa, J. A. R. (2016). Conditional distribution variability
  measures for causality detection. (cdt's implementation is Fonollosa's.)
* RECI -- Blöbaum, P., Janzing, D., Washio, T., Shimizu, S. & Schölkopf, B.
  (2018). Cause-effect inference by comparing regression errors. *AISTATS*.
* IGCI -- Daniušis, P., Janzing, D., Mooij, J., Zscheischler, J., Steudel, B.,
  Zhang, K. & Schölkopf, B. (2010). Inferring deterministic causal relations.
  *UAI*.
"""

from collections import Counter

import numpy as np
import pandas as pd
from scipy.special import psi
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, minmax_scale

__all__ = ["normalized_hsic", "cds_score", "reci_score", "igci_score"]


# ---------------------------------------------------------------------------
# HSIC (drives the additive-noise-model score)
# ---------------------------------------------------------------------------

def _rbf_dot(X, deg):
    """Gaussian kernel matrix; bandwidth from the median pairwise distance."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, np.newaxis]
    m = X.shape[0]
    G = np.sum(X * X, axis=1)[:, np.newaxis]
    Q = np.tile(G, (1, m))
    H = Q + Q.T - 2.0 * np.dot(X, X.T)
    if deg == -1:
        dists = (H - np.tril(H)).flatten()
        deg = np.sqrt(0.5 * np.median(dists[dists > 0]))
    return np.exp(-H / 2.0 / (deg ** 2))


def _fast_hsic_test_gamma(X, Y, sig=(-1, -1), maxpnt=200):
    """Biased HSIC statistic, on at most `maxpnt` evenly-spaced samples.

    The subsampling is part of the estimator as pyspi has always used it, not an
    optimisation: it fixes the statistic's scale, so removing it would change
    every `anm` value.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    m = X.shape[0]
    if m > maxpnt:
        indx = np.floor(np.r_[0:m:float(m - 1) / (maxpnt - 1)]).astype(int)
        Xm = X[indx].astype(float)
        Ym = Y[indx].astype(float)
        m = Xm.shape[0]
    else:
        Xm = X.astype(float)
        Ym = Y.astype(float)

    H = np.eye(m) - 1.0 / m * np.ones((m, m))
    Kc = np.dot(H, np.dot(_rbf_dot(Xm, sig[0]), H))
    Lc = np.dot(H, np.dot(_rbf_dot(Ym, sig[1]), H))

    stat = (1.0 / m) * (Kc.T * Lc).sum()
    return 0 if not np.isfinite(stat) else stat


def normalized_hsic(x, y):
    """HSIC between standardised `x` and `y`. Lower means more independent."""
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    return _fast_hsic_test_gamma(x, y)


# ---------------------------------------------------------------------------
# CDS -- conditional distribution similarity (Fonollosa 2016)
# ---------------------------------------------------------------------------

def _count_unique(x):
    return len(np.unique(x)) if isinstance(x, np.ndarray) else len(set(x))


def _discretized_values(x, ffactor, maxdev):
    if _count_unique(x) > (2 * ffactor * maxdev + 1):
        return range(-ffactor * maxdev, ffactor * maxdev + 1)
    return sorted(set(x))


def _discretized_sequence(x, ffactor, maxdev, norm=True):
    if not norm or _count_unique(x) > len(_discretized_values(x, ffactor, maxdev)):
        if norm:
            x = (x - np.mean(x)) / np.std(x)
            xf = x[abs(x) < maxdev]
            x = (x - np.mean(xf)) / np.std(xf)
        x = np.round(x * ffactor)
        vmax = ffactor * maxdev
        x[x > vmax] = vmax
        x[x < -vmax] = -vmax
    return x


def cds_score(x_te, y_te, ffactor=2, maxdev=3, minc=12):
    """Std. of the rescaled y values after binning in x; lower favours x -> y."""
    if isinstance(x_te, np.ndarray):
        x_te = pd.Series(np.asarray(x_te).reshape(-1))
        y_te = pd.Series(np.asarray(y_te).reshape(-1))

    xd = _discretized_sequence(x_te, ffactor, maxdev)
    yd = _discretized_sequence(y_te, ffactor, maxdev)
    cx, cy = Counter(xd), Counter(yd)
    yrange = sorted(cy.keys())
    ny = len(yrange)
    py = np.array([cy[i] for i in yrange], dtype=float)
    py = py / py.sum()

    pyx = []
    for a in cx:
        if cx[a] <= minc:
            continue
        yx = y_te[xd == a]
        if _count_unique(y_te) > len(_discretized_values(y_te, ffactor, maxdev)):
            yx = (yx - np.mean(yx)) / np.std(y_te)
            yx = _discretized_sequence(yx, ffactor, maxdev, norm=False)
            cyx = Counter(yx.astype(int))
            pyxa = np.array(
                [cyx[i] for i in _discretized_values(y_te, ffactor, maxdev)],
                dtype=float)
        else:
            # Discrete y: align the conditional histogram to the marginal by
            # the shift that maximises their cross-correlation, so a pure
            # location shift between bins is not read as a shape difference.
            cyx = Counter(yx)
            pyxa = [cyx[i] for i in yrange]
            padded = np.array([0] * (ny - 1) + pyxa + [0] * (ny - 1), dtype=float)
            xcorr = [sum(py * padded[i:i + ny]) for i in range(2 * ny - 1)]
            imax = xcorr.index(max(xcorr))
            pyxa = np.array([0] * (2 * ny - 2 - imax) + pyxa + [0] * imax,
                            dtype=float)
        pyx.append(pyxa / pyxa.sum())

    if not pyx:
        return 0
    pyx = np.array(pyx)
    return float(np.std(pyx - pyx.mean(axis=0)))


# ---------------------------------------------------------------------------
# RECI -- regression error causal inference (Bloebaum et al. 2018)
# ---------------------------------------------------------------------------

def reci_score(x, y, degree=3):
    """Mean squared error of a monomial fit of y on x, both min-max scaled.

    The first two polynomial columns (the intercept and the linear term) are
    zeroed, which is cdt's implementation of the paper's monomial regressor: the
    fit is over x**2..x**degree only.
    """
    x = np.reshape(minmax_scale(np.asarray(x, dtype=float).reshape(-1, 1)), (-1, 1))
    y = np.reshape(minmax_scale(np.asarray(y, dtype=float).reshape(-1, 1)), (-1, 1))
    poly_x = PolynomialFeatures(degree=degree).fit_transform(x)
    poly_x[:, 1] = 0
    poly_x[:, 2] = 0
    y_predict = LinearRegression().fit(poly_x, y).predict(poly_x)
    return mean_squared_error(y_predict, y)


# ---------------------------------------------------------------------------
# IGCI -- information-geometric causal inference (Daniusis et al. 2010)
# ---------------------------------------------------------------------------

def _eval_entropy(x):
    """Kozachenko-Leonenko 1-D entropy from consecutive order-statistic gaps."""
    hx = 0.0
    sx = sorted(x)
    for i, j in zip(sx[:-1], sx[1:]):
        delta = j - i
        if bool(delta):
            hx += np.log(np.abs(delta))
    return hx / (len(x) - 1) + psi(len(x)) - psi(1)


def _integral_approx_estimator(x, y):
    a = b = 0.0
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    idx, idy = np.argsort(x), np.argsort(y)
    for x1, x2, y1, y2 in zip(x[idx][:-1], x[idx][1:], y[idx][:-1], y[idx][1:]):
        if x1 != x2 and y1 != y2:
            a += np.log(np.abs((y2 - y1) / (x2 - x1)))
    for x1, x2, y1, y2 in zip(x[idy][:-1], x[idy][1:], y[idy][:-1], y[idy][1:]):
        if x1 != x2 and y1 != y2:
            b += np.log(np.abs((x2 - x1) / (y2 - y1)))
    return (a - b) / len(x)


def igci_score(x, y, ref_measure="gaussian", estimator="entropy"):
    """Entropy (or integral-approximation) asymmetry; > 0 favours x -> y."""
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if ref_measure == "gaussian":
        scale = lambda v: (v - v.mean()) / v.std()
    elif ref_measure == "uniform":
        scale = lambda v: (v - v.min()) / (v.max() - v.min())
    elif ref_measure == "None":
        scale = lambda v: v
    else:
        raise ValueError(f"Unknown ref_measure {ref_measure!r}.")

    a, b = scale(x), scale(y)
    if estimator == "entropy":
        return _eval_entropy(a) - _eval_entropy(b)
    if estimator == "integral":
        return _integral_approx_estimator(a, b)
    raise ValueError(f"Unknown estimator {estimator!r}.")
