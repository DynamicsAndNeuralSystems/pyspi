"""Regression test against JIDT 1.6.1 reference values.

pyspi 2.x computed its information-theoretic measures with JIDT (Lizier, 2014;
https://github.com/jlizier/jidt, GPL-3.0-or-later). pyspi 3 reimplements the
KSG, kernel, Gaussian and symbolic estimators in NumPy, with no JIDT jar. This
module checks that the NumPy estimators reproduce JIDT: the constants below
were produced by JIDT itself and are asserted against pyspi's estimators. **No
Java is needed to run this test.**

Provenance of the constants
---------------------------
Generated with JIDT 1.6.1 (``infodynamics.jar`` as shipped by pyspi 2.0.2) on
``tests/data/fixtures/jidt_ar_pair_T500.csv`` (the first 500 samples of a
coupled AR(1) pair, x -> y; column 0 = x, column 1 = y), by
``tests/tools/jidt_reference/JidtReference.java``; see the README in that
directory to regenerate. Values are pasted at full double precision. KSG uses
k=4, algorithm 1, ``ADD_NOISE=false`` (JIDT's default dither is stochastic) and
JIDT's default ``NORMALISE=true``, which pyspi mirrors by standardising
continuous inputs. The data are tie-free, so pyspi's tie-breaking jitter never
engages.

Units: JIDT's kernel and symbolic estimators report bits, its Gaussian and KSG
estimators nats; pyspi reports nats throughout. The constants are stored in
JIDT's native units and converted (x ln 2) on comparison.

Tolerances (achieved maximum |difference| on this fixture in brackets)
----------------------------------------------------------------------
* KSG MI and TE, including Theiler windows, DCE and auto-embedding: 1e-12
  (2e-15).
* Gaussian MI/TE: 1e-8 (5.5e-9). pyspi ridge-regularises the covariance at
  1e-8 relative to each variable's variance, the deterministic analogue of
  JIDT's stochastic ``NOISE_LEVEL_TO_ADD``; this is the whole difference.
* Kernel MI and TE (k_history 1 and 2), symbolic TE: 1e-12 after the ln 2
  conversion (3e-16 / 2e-16 / 2e-15). pyspi's kernel SPIs exclude the point
  itself, which is JIDT's ``DYN_CORR_EXCL = 1``; the calculators are also
  checked at JIDT's default of 0.
* Kernel MI, time-lagged MI and TE (k_history=2) with JIDT's dynamic
  correlation exclusion ``DYN_CORR_EXCL`` of 1, 5 and 20 (pyspi windows none,
  4 and 19) at kernel widths 0.5 and 0.1: 1e-12 after the ln 2 conversion
  (6e-16). Width 0.1 leaves samples with an empty joint box, which JIDT
  counts as 0.
"""

import os

import numpy as np
import pytest

from pyspi.data import Data
from pyspi.statistics.infotheory import (
    KernelMICalculator,
    KernelTECalculator,
    MutualInfo,
    TimeLaggedMutualInfo,
    TransferEntropy,
)

LN2 = np.log(2.0)

FIXTURE = os.path.join(os.path.dirname(__file__), "data", "fixtures",
                       "jidt_ar_pair_T500.csv")

# ---------------------------------------------------------------------------
# JIDT 1.6.1 outputs (see module docstring). Units as JIDT reports them.
# ---------------------------------------------------------------------------
JIDT = {
    # Gaussian, nats
    "mi_gaussian": 0.037544112813742175,
    "tlmi_gaussian": 0.21998110375622046,
    "te_gaussian_k1": 0.18475489153224076,
    "te_gaussian_k3": 0.18542418742024203,
    "te_gaussian_k2t2_l3t2": 0.1863993998423251,
    "te_gaussian_auto": 0.18463443229211318,  # MAX_CORR_AIS, search k<=10, tau<=4 (JIDT chose k=6 ktau=1 l=6 ltau=4)
    # KSG algorithm 1, k=4, nats
    "mi_ksg_w0": 0.04517809185828536,
    "mi_ksg_w5": 0.06604489802304947,
    "mi_ksg_w20": 0.12737994398423336,
    "te_ksg_k1": 0.12066600409154526,
    "te_ksg_k3": 0.1392619704896605,
    "te_ksg_k2t2_l3t2": 0.1563034393305236,
    "te_ksg_k1_dce10": 0.12622057707308,
    "te_ksg_dce10_auto": 0.15009111284231103,  # MAX_CORR_AIS + DCE 10 (JIDT chose k=1 ktau=1 l=4 ltau=1)
    # Kernel, width 0.5, bits
    "mi_kernel_bits": 0.09538573546518486,
    "te_kernel_bits_k1": 0.3111627487933465,
    "te_kernel_bits_k2": 0.44662763106247877,
    "te_kernel_bits_k1_W0.5_dce1": 0.19569119734024015,
    # Kernel with DYN_CORR_EXCL W (excludes |t - i| < W), bits. TLMI is kernel MI
    # of (x[:-1], y[1:]); TE uses k_history=2.
    "mi_kernel_bits_W0.5_dce1": 0.027092073335829674,
    "tlmi_kernel_bits_W0.5_dce1": 0.2589376908163566,
    "te_kernel_bits_k2_W0.5_dce1": 0.21215642217217362,
    "mi_kernel_bits_W0.5_dce5": 0.026124908431031595,
    "tlmi_kernel_bits_W0.5_dce5": 0.26645938746450387,
    "te_kernel_bits_k2_W0.5_dce5": 0.207569405965963,
    "mi_kernel_bits_W0.5_dce20": 0.021395123882834433,
    "tlmi_kernel_bits_W0.5_dce20": 0.2701562276388299,
    "te_kernel_bits_k2_W0.5_dce20": 0.18611451617066044,
    "mi_kernel_bits_W0.1_dce1": 0.13867092504379724,
    "tlmi_kernel_bits_W0.1_dce1": 0.26022039954607196,
    "te_kernel_bits_k2_W0.1_dce1": 0.014397439761932352,
    "mi_kernel_bits_W0.1_dce5": 0.15107106122165262,
    "tlmi_kernel_bits_W0.1_dce5": 0.26851678590353345,
    "te_kernel_bits_k2_W0.1_dce5": 0.014397439761932352,
    "mi_kernel_bits_W0.1_dce20": 0.17181020492147228,
    "tlmi_kernel_bits_W0.1_dce20": 0.2606747073955967,
    "te_kernel_bits_k2_W0.1_dce20": 0.014397439761932352,
    # Symbolic, bits
    "te_symbolic_k2": 0.045972510764248756,
    "te_symbolic_k3": 0.1852520627825706,
}

KSG_TOL = 1e-12
GAUSSIAN_TOL = 1e-8   # the Gaussian ridge, see module docstring
EXACT_TOL = 1e-12     # kernel/symbolic after the ln 2 conversion


@pytest.fixture(scope="module")
def data():
    arr = np.loadtxt(FIXTURE, delimiter=",")
    assert arr.shape == (500, 2)
    # zscore=False: the estimators do their own standardisation, as JIDT does.
    return Data(arr.T, dim_order="ps", zscore=False)


def _mi(est, data, **kw):
    return MutualInfo(estimator=est, **kw).bivariate(data, i=0, j=1)


def _te(est, data, **kw):
    return TransferEntropy(estimator=est, **kw).bivariate(data, i=0, j=1)


def _close(value, ref, tol, name):
    assert abs(value - ref) <= tol, (
        f"{name}: pyspi {value!r} vs JIDT 1.6.1 {ref!r} (|diff| {abs(value - ref):.3e} > {tol:g})")


# -- Gaussian ---------------------------------------------------------------

def test_gaussian_mi(data):
    _close(_mi("gaussian", data), JIDT["mi_gaussian"], GAUSSIAN_TOL, "mi_gaussian")


def test_gaussian_tlmi(data):
    v = TimeLaggedMutualInfo(estimator="gaussian").bivariate(data, i=0, j=1)
    _close(v, JIDT["tlmi_gaussian"], GAUSSIAN_TOL, "tlmi_gaussian")


@pytest.mark.parametrize("name, kw", [
    ("te_gaussian_k1", {}),
    ("te_gaussian_k3", dict(k_history=3)),
    ("te_gaussian_k2t2_l3t2", dict(k_history=2, k_tau=2, l_history=3, l_tau=2)),
    ("te_gaussian_auto", dict(auto_embed_method="MAX_CORR_AIS")),
])
def test_gaussian_te(data, name, kw):
    _close(_te("gaussian", data, **kw), JIDT[name], GAUSSIAN_TOL, name)


# -- KSG --------------------------------------------------------------------

@pytest.mark.parametrize("w", [0, 5, 20])
def test_ksg_mi_theiler_window(data, w):
    """Neighbour search and counts match JIDT exactly; the psi(N) term does not.

    JIDT 1.6.1 keeps psi(N) with a Theiler window although the window shrinks
    every point's neighbour pool (its source marks this as an open TODO), which
    adds ~ln(N / (N - 2w)): its own MI on this fixture climbs 0.045 -> 0.066 ->
    0.127 across w = 0, 5, 20. pyspi uses the mean psi(m_i + 1) of the reduced
    pools. The difference is that one closed-form term, so adding it back must
    recover JIDT's value to round-off: everything else is still pinned.
    """
    from scipy.special import digamma

    # w=0 is pyspi's "no exclusion" (dyn_corr_excl=None).
    v = _mi("kraskov", data, dyn_corr_excl=(None if w == 0 else w))
    n = data.to_numpy().shape[1]
    i = np.arange(n)
    pool = n - 1 - np.minimum(i, w) - np.minimum(n - 1 - i, w)
    jidt_psi_n_offset = digamma(n) - np.mean(digamma(pool + 1))
    _close(v + jidt_psi_n_offset, JIDT[f"mi_ksg_w{w}"], KSG_TOL, f"mi_ksg_w{w}")
    if w:
        # pyspi's value does not drift with the window.
        assert abs(v - JIDT["mi_ksg_w0"]) < 0.002


@pytest.mark.parametrize("name, kw", [
    ("te_ksg_k1", {}),
    ("te_ksg_k3", dict(k_history=3)),
    ("te_ksg_k2t2_l3t2", dict(k_history=2, k_tau=2, l_history=3, l_tau=2)),
    ("te_ksg_k1_dce10", dict(dyn_corr_excl=10)),
    ("te_ksg_dce10_auto", dict(dyn_corr_excl=10, auto_embed_method="MAX_CORR_AIS")),
])
def test_ksg_te(data, name, kw):
    _close(_te("kraskov", data, **kw), JIDT[name], KSG_TOL, name)


# -- Kernel (JIDT reports bits; pyspi nats) ----------------------------------

@pytest.mark.parametrize("name, kw", [
    ("mi_kernel_bits", {}),
    ("te_kernel_bits_k1", dict(k_history=1)),
    ("te_kernel_bits_k2", dict(k_history=2)),
])
def test_kernel_calculators_match_jidt_default(data, name, kw):
    """The calculators with JIDT's default DYN_CORR_EXCL of 0 (the point itself
    counted); the SPIs always exclude it."""
    x, y = data.to_numpy(squeeze=True)
    if name.startswith("mi"):
        calc = KernelMICalculator()
        calc.initialise(1, 1)
    else:
        calc = KernelTECalculator()
        calc.setProperty("k_HISTORY", str(kw["k_history"]))
        calc.initialise()
    calc.setProperty("KERNEL_WIDTH", "0.5")
    calc.setObservations(x, y)
    _close(calc.computeAverageLocalOfObservations() / LN2, JIDT[name], EXACT_TOL, name)


@pytest.mark.parametrize("width", [0.5, 0.1])
@pytest.mark.parametrize("W", [1, 5, 20])
@pytest.mark.parametrize("name, cls, kw", [
    ("mi_kernel_bits", MutualInfo, {}),
    ("tlmi_kernel_bits", TimeLaggedMutualInfo, {}),
    ("te_kernel_bits_k2", TransferEntropy, dict(k_history=2)),
])
def test_kernel_dynamic_correlation_exclusion(data, name, cls, kw, W, width):
    """JIDT's kernel DYN_CORR_EXCL W: |t - i| < W excluded, per-sample pool size.

    pyspi's window w excludes |t - i| <= w, as for KSG, so it is JIDT's
    W = w + 1; W = 1 (the point itself only) is no window.
    """
    spi = cls(estimator="kernel", kernel_width=width,
              dyn_corr_excl=(W - 1 or None), **kw)
    v = spi.bivariate(data, i=0, j=1) / LN2
    key = f"{name}_W{width}_dce{W}"
    _close(v, JIDT[key], EXACT_TOL, key)


def test_kernel_te_k1(data):
    v = _te("kernel", data, kernel_width=0.5) / LN2
    _close(v, JIDT["te_kernel_bits_k1_W0.5_dce1"], EXACT_TOL, "te_kernel_k1")


# -- Symbolic (bits in JIDT) -------------------------------------------------

@pytest.mark.parametrize("k", [2, 3])
def test_symbolic_te(data, k):
    v = _te("symbolic", data, k_history=k) / LN2
    _close(v, JIDT[f"te_symbolic_k{k}"], EXACT_TOL, f"te_symbolic_k{k}")
