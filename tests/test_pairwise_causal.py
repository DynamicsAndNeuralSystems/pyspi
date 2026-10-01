"""The four pairwise causal scores, after cdt and torch were removed.

pyspi used exactly four functions from `cdt.causality.pairwise`: the ANM
independence score, the conditional distribution similarity statistic, the RECI
regression-error score and IGCI. They are now transcribed in
`pyspi/lib/pairwise_causal.py`.

Three kinds of evidence here, deliberately separated:

* **Independent formulas.** Each score recomputed from its definition with
  different machinery -- explicit double sums for HSIC, `lstsq` for RECI, a
  direct order-statistic sum for the IGCI entropy. These would catch a
  transcription error that the differential fixture below cannot, because they
  do not descend from cdt's code at all.
* **Invariants and known directions.** Scale/translation behaviour, reverse-pair
  antisymmetry, determinism, and synthetic pairs whose causal direction is known
  by construction.
* **A frozen cdt 0.6 fixture**, `tests/data/fixtures/cdt_pairwise_reference.npz`,
  recorded from cdt before the dependency was dropped. Regression evidence, not
  ground truth: it pins that removing cdt changed nothing, and it is the reason
  no baseline moved.
"""
import os

import numpy as np
import pytest

from pyspi.lib.pairwise_causal import (cds_score, igci_score, normalized_hsic,
                                       reci_score)

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "fixtures", "cdt_pairwise_reference.npz")


@pytest.fixture(scope="module")
def cdt_reference():
    """[(name, x, y, {score: value_from_cdt_0_6})]"""
    with np.load(FIXTURE, allow_pickle=False) as archive:
        names = [str(n) for n in archive["names"]]
        lengths = archive["lengths"]
        xs = np.split(archive["x"], np.cumsum(lengths)[:-1])
        ys = np.split(archive["y"], np.cumsum(lengths)[:-1])
        expected = {k: archive[k] for k in ("hsic", "cds", "reci", "igci")}
    return [(names[i], xs[i], ys[i], {k: v[i] for k, v in expected.items()})
            for i in range(len(names))]


# ---------------------------------------------------------------------------
# Differential regression against the dependency that was removed
# ---------------------------------------------------------------------------

def test_local_scores_reproduce_cdt_0_6_exactly(cdt_reference):
    """Bit-identical, so no bundled SPI value changed when cdt was dropped.

    Five fixtures spanning the input shapes the estimators behave differently
    on: a nonlinear additive-noise pair, a random walk, a quantised source (CDS
    takes its discrete branch there), an independent pair, and a short record.
    """
    for name, x, y, expected in cdt_reference:
        X, Y = x.reshape(-1, 1), y.reshape(-1, 1)
        for key, got in (("hsic", normalized_hsic(X, Y)),
                         ("cds", cds_score(X, Y)),
                         ("reci", reci_score(X, Y)),
                         ("igci", igci_score(X, Y))):
            assert float(got) == pytest.approx(float(expected[key]), rel=1e-12,
                                               abs=1e-15), f"{name}/{key}"


# ---------------------------------------------------------------------------
# Independent formulas
# ---------------------------------------------------------------------------

def test_hsic_matches_an_explicit_double_sum():
    """HSIC = (1/m) * trace(Kc Lc) with Kc = HKH, written out elementwise.

    Small enough that the maxpnt=200 subsampling does not engage, so the two
    computations see the same points.
    """
    rng = np.random.default_rng(0)
    n = 120
    x = rng.standard_normal((n, 1))
    y = np.tanh(x) + 0.4 * rng.standard_normal((n, 1))

    xs = (x - x.mean()) / x.std()
    ys = (y - y.mean()) / y.std()

    def kernel(v):
        sq = (v - v.T) ** 2
        lower = (sq - np.tril(sq)).flatten()
        width = np.sqrt(0.5 * np.median(lower[lower > 0]))
        return np.exp(-sq / (2.0 * width ** 2))

    K, L = kernel(xs), kernel(ys)
    H = np.eye(n) - np.ones((n, n)) / n
    expected = np.trace(H @ K @ H @ (H @ L @ H)) / n

    assert normalized_hsic(x, y) == pytest.approx(expected, rel=1e-10)


def test_reci_matches_an_explicit_least_squares_fit():
    """Min-max scale both, fit y on [1, 0, 0, x^3..x^d], take the MSE.

    The two zeroed columns are cdt's rendering of the paper's monomial
    regressor: only the cubic-and-above terms carry any signal, alongside the
    intercept that `LinearRegression` fits separately.
    """
    rng = np.random.default_rng(1)
    n = 300
    x = rng.standard_normal(n)
    y = x ** 3 + 0.3 * rng.standard_normal(n)

    scale = lambda v: (v - v.min()) / (v.max() - v.min())
    xs, ys = scale(x), scale(y)
    design = np.column_stack([np.ones(n), np.zeros(n), np.zeros(n), xs ** 3])
    beta, *_ = np.linalg.lstsq(design, ys, rcond=None)
    expected = float(np.mean((design @ beta - ys) ** 2))

    assert reci_score(x, y, degree=3) == pytest.approx(expected, rel=1e-9)


def test_igci_entropy_matches_a_direct_order_statistic_sum():
    """h(x) = mean(log(gap)) + psi(n) - psi(1) over consecutive sorted values."""
    from scipy.special import psi

    rng = np.random.default_rng(2)
    n = 400
    x = rng.standard_normal(n)
    y = np.exp(x)

    def entropy(v):
        gaps = np.diff(np.sort(v))
        gaps = gaps[gaps != 0]
        return np.sum(np.log(np.abs(gaps))) / (len(v) - 1) + psi(len(v)) - psi(1)

    standard = lambda v: (v - v.mean()) / v.std()
    expected = entropy(standard(x)) - entropy(standard(y))
    assert igci_score(x, y) == pytest.approx(expected, rel=1e-12)


def test_cds_prefers_the_cause_as_the_conditioning_variable_where_it_can():
    """CDS bins on its first argument, and a lower score favours that direction.

    Asserted on a saturating additive-noise pair, where it works: 8 of 8 seeds.
    It is *not* asserted universally, because CDS does not hold universally --
    it is one heuristic feature of the Jarfo model rather than a consistent
    estimator, and on a cubic pair `y = x**3 + noise` the same comparison goes
    the wrong way in 8 of 8 seeds. Pinning a universal direction here would
    encode a property the statistic does not have.
    """
    rng = np.random.default_rng(6)
    for _ in range(4):
        x = rng.standard_normal(600)
        y = np.tanh(2 * x) + 0.3 * rng.standard_normal(600)
        X, Y = x.reshape(-1, 1), y.reshape(-1, 1)
        assert cds_score(X, Y) < cds_score(Y, X)


# ---------------------------------------------------------------------------
# Invariants, known directions, determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score", [normalized_hsic, cds_score, reci_score,
                                   igci_score])
@pytest.mark.parametrize("a,b", [(3.0, 5.0), (0.25, -2.0)])
def test_scores_are_invariant_to_positive_affine_rescaling(score, a, b):
    """Each standardises or min-max scales its inputs first, so an affine change
    of units must not move the score.

    RECI and CDS scale to a fixed range, so they are invariant to `a` of either
    sign; HSIC and IGCI standardise, which is invariant only up to the sign of
    the scale. Only positive `a` is asserted, which is what all four share.
    """
    rng = np.random.default_rng(4)
    n = 300
    x = rng.standard_normal(n)
    y = np.tanh(x) + 0.3 * rng.standard_normal(n)
    scaled_a, scaled_b = abs(a) * x + b, abs(a) * y - b

    base = float(score(x.reshape(-1, 1), y.reshape(-1, 1)))
    moved = float(score(scaled_a.reshape(-1, 1), scaled_b.reshape(-1, 1)))
    assert moved == pytest.approx(base, rel=1e-6, abs=1e-9)


def test_igci_is_antisymmetric_in_its_arguments():
    """It is a difference of two entropies, so reversing the pair negates it."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal(400)
    y = np.exp(x) + 0.1 * rng.standard_normal(400)
    assert igci_score(x, y) == pytest.approx(-igci_score(y, x), rel=1e-12)


def test_anm_and_reci_prefer_the_true_direction_on_a_nonlinear_pair():
    """y = f(x) + noise with f nonlinear and the noise independent of x.

    The additive-noise model is identifiable here, so the residual of a fit in
    the true direction is independent of the cause (low HSIC) while the reverse
    fit leaves a dependent residual. RECI's regression error is likewise smaller
    in the true direction under the paper's scaling assumptions.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor

    rng = np.random.default_rng(6)
    n = 300
    x = rng.uniform(-2.5, 2.5, n)
    y = x ** 3 + 0.5 * rng.standard_normal(n)
    X, Y = x.reshape(-1, 1), y.reshape(-1, 1)

    def anm(cause, effect):
        gp = GaussianProcessRegressor(random_state=42).fit(cause, effect)
        return normalized_hsic(gp.predict(cause).reshape(-1, 1) - effect, cause)

    assert anm(X, Y) < anm(Y, X)
    assert reci_score(X, Y) < reci_score(Y, X)


@pytest.mark.parametrize("score", [normalized_hsic, cds_score, reci_score,
                                   igci_score])
def test_scores_are_deterministic(score):
    rng = np.random.default_rng(7)
    x = rng.standard_normal((250, 1))
    y = (np.tanh(x) + 0.3 * rng.standard_normal((250, 1)))
    assert float(score(x, y)) == float(score(x, y))


def test_cdt_and_torch_are_not_imported_by_pyspi():
    """The point of the exercise: neither is a dependency any more.

    `cdt` eagerly imports its Torch-backed models at package load, so importing
    it pulled in torch -- roughly 2 GB installed -- for four functions that
    need neither.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, pyspi.calculator, pyspi.statistics.causal;"
         "leaked = sorted(m for m in sys.modules if m.split('.')[0] in "
         "{'cdt', 'torch'});"
         "print(leaked)"],
        capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]", result.stdout
