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
* Kernel MI, kernel TE at k_history=2, symbolic TE: 1e-12 after the ln 2
  conversion (3e-16 / 2e-16 / 2e-15).
* Kernel TE at k_history=1 is **not** reproduced exactly. pyspi normalises
  each embedding column separately, whereas JIDT normalises by the standard
  deviation of the whole series, so the kernel widths differ slightly. On this
  fixture the difference is 1.1e-4 bits; across the harness's longer runs it
  reached 1.9e-4 bits. The test pins that known discrepancy (bounded above by
  2e-4 bits, and required to be nonzero so that the day it is fixed the
  exclusion gets noticed) rather than claiming parity.
"""

import os

import numpy as np
import pytest

from pyspi.data import Data
from pyspi.statistics.infotheory import (
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
    # Symbolic, bits
    "te_symbolic_k2": 0.045972510764248756,
    "te_symbolic_k3": 0.1852520627825706,
}

KSG_TOL = 1e-12
GAUSSIAN_TOL = 1e-8   # the Gaussian ridge, see module docstring
EXACT_TOL = 1e-12     # kernel/symbolic after the ln 2 conversion
KERNEL_TE_K1_BOUND = 2e-4   # bits; known normalisation discrepancy, <= 1.9e-4 observed


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

def test_kernel_mi(data):
    v = _mi("kernel", data, kernel_width=0.5) / LN2
    _close(v, JIDT["mi_kernel_bits"], EXACT_TOL, "mi_kernel")


def test_kernel_te_k2_matches(data):
    v = _te("kernel", data, kernel_width=0.5, k_history=2) / LN2
    _close(v, JIDT["te_kernel_bits_k2"], EXACT_TOL, "te_kernel_k2")


def test_kernel_te_k1_known_discrepancy(data):
    """Documented non-parity: per-column vs whole-series normalisation."""
    v = _te("kernel", data, kernel_width=0.5) / LN2
    diff = abs(v - JIDT["te_kernel_bits_k1"])
    assert 1e-8 < diff <= KERNEL_TE_K1_BOUND, (
        f"kernel TE (k_history=1) now differs from JIDT by {diff:.3e} bits; the known "
        f"normalisation discrepancy is ~1e-4 and <= {KERNEL_TE_K1_BOUND:g}. If it has been "
        f"fixed, turn this into an exact comparison.")


# -- Symbolic (bits in JIDT) -------------------------------------------------

@pytest.mark.parametrize("k", [2, 3])
def test_symbolic_te(data, k):
    v = _te("symbolic", data, k_history=k) / LN2
    _close(v, JIDT[f"te_symbolic_k{k}"], EXACT_TOL, f"te_symbolic_k{k}")
