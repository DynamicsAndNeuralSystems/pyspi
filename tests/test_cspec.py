"""Conditional spectral measures ``cpdc_*``, ``cgpdc_*``, ``cddtf_*``, ``pcohmag_*``.

Checked against ``A(f)``, ``H(f)``, ``S(f)`` and ``S(f)^{-1}`` written out
with loops, against the normalisations that define PDC and the full-frequency
DTF, on a chain, on an oscillatory coupling, and for sharing ``cgc_*``'s fit.
"""
import warnings

import numpy as np
import pytest
import yaml

from pyspi import _parallel
from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.statistics import spectral, var
from pyspi.statistics.var import (
    ConditionalDirectDirectedTransferFunction as CDDTF,
    ConditionalGeneralizedPartialDirectedCoherence as CGPDC,
    ConditionalGrangerCausality as CGC,
    ConditionalPartialDirectedCoherence as CPDC,
    PartialCoherence as PCOH,
    fit_var,
    frequency_grid,
    full_frequency_dtf,
    var_frequency_response,
)

CLASSES = {"pdc": CPDC, "gpdc": CGPDC, "ddtf": CDDTF, "pcoh": PCOH}
STEMS = {"pdc": "cpdc", "gpdc": "cgpdc", "ddtf": "cddtf", "pcoh": "pcohmag"}
BANDS = {"fmin-0_fmax-0-5": dict(), "fmin-0_fmax-0-25": dict(fmin=0, fmax=0.25),
         "fmin-0-25_fmax-0-5": dict(fmin=0.25, fmax=0.5)}
BUNDLED = {f"{STEMS[m]}_p-bic_mean_{band}": (CLASSES[m], dict(order="bic", **kw))
           for m in CLASSES for band, kw in BANDS.items()}
FAST = {"cpdc_p-bic_mean_fmin-0_fmax-0-5", "pcohmag_p-bic_mean_fmin-0_fmax-0-5"}


def _simulate(coefs, T, seed, burn=500):
    """VAR(len(coefs)) with unit-variance Gaussian innovations; coefs[k-1][j, i]."""
    coefs = [np.asarray(A, dtype=float) for A in coefs]
    M = coefs[0].shape[0]
    rng = np.random.default_rng(seed)
    x = np.zeros((M, T + burn))
    for t in range(len(coefs), T + burn):
        x[:, t] = sum(A @ x[:, t - k] for k, A in enumerate(coefs, start=1))
        x[:, t] += rng.standard_normal(M)
    return x[:, burn:]


def _data(x):
    return Data(data=np.asarray(x), dim_order="ps")


def _offdiag(M):
    return ~np.eye(M, dtype=bool)


# A stable VAR(2) on four processes with unequal innovation variances (so gPDC
# differs from PDC) and correlated innovations (so partial coherence is not
# read off the coefficients alone).
_A1 = np.array([[0.4, 0.2, 0.0, 0.0], [0.0, 0.3, -0.3, 0.0],
                [0.2, 0.0, 0.3, 0.1], [0.0, 0.0, 0.25, 0.2]])
_A2 = np.array([[-0.2, 0.0, 0.0, 0.1], [0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.2, 0.0], [0.0, 0.15, 0.0, 0.0]])


def _fixture(T=600, seed=0):
    x = _simulate([_A1, _A2], T, seed)
    mix = np.array([[1.0, 0, 0, 0], [0.3, 2.0, 0, 0], [0, 0, 0.5, 0],
                    [0, 0.4, 0, 1.5]])
    return mix @ x


# The chain x0 -> x1 -> x2: no direct link 0 -> 2.
_CHAIN = np.array([[0.5, 0.0, 0.0], [0.6, 0.5, 0.0], [0.0, 0.6, 0.5]])


@pytest.fixture(scope="module")
def chain():
    return _simulate([_CHAIN], 4000, seed=0)


# --------------------------------------------------------------------------
# 1. Independent reference: A(f), H(f), S(f) and S(f)^{-1} written out
# --------------------------------------------------------------------------

def _reference_spectra(coef, sigma, freq):
    """The four measures by loops, each ``[f, source, target]``.

    ``A(f)`` element by element, ``H`` and ``S^{-1}`` by explicit inversion
    of ``A(f)`` and of ``S = H Sigma H^H`` (the module instead forms
    ``S^{-1} = A^H Sigma^{-1} A``), normalisations as sums over loops.
    """
    p, M, _ = coef.shape
    F = len(freq)
    pdc, gpdc, pcoh = (np.zeros((F, M, M)) for _ in range(3))
    H_all = np.zeros((F, M, M), dtype=complex)
    for n, f in enumerate(freq):
        A = np.zeros((M, M), dtype=complex)
        for tgt in range(M):
            for src in range(M):
                a = 1.0 if tgt == src else 0.0
                for k in range(1, p + 1):
                    a -= coef[k - 1, tgt, src] * np.exp(-2j * np.pi * f * k)
                A[tgt, src] = a
        H = np.linalg.inv(A)
        S = H @ sigma @ H.conj().T
        Q = np.linalg.inv(S)
        H_all[n] = H
        for src in range(M):
            out = sum(abs(A[k, src]) ** 2 for k in range(M))
            gout = sum(abs(A[k, src]) ** 2 / sigma[k, k] for k in range(M))
            for tgt in range(M):
                pdc[n, src, tgt] = abs(A[tgt, src]) ** 2 / out
                gpdc[n, src, tgt] = abs(A[tgt, src]) ** 2 / sigma[tgt, tgt] / gout
                pcoh[n, src, tgt] = abs(Q[tgt, src]) ** 2 / (
                    Q[tgt, tgt].real * Q[src, src].real)
    ffdtf = np.zeros((F, M, M))
    for tgt in range(M):
        inflow = sum(abs(H_all[n, tgt, k]) ** 2 for n in range(F) for k in range(M))
        for src in range(M):
            ffdtf[:, src, tgt] = np.abs(H_all[:, tgt, src]) / np.sqrt(inflow)
    return {"pdc": pdc, "gpdc": gpdc, "pcoh": pcoh,
            "ddtf": ffdtf * np.sqrt(pcoh), "ffdtf": ffdtf}


def _reference_for(data, order, nfreq):
    fit = data.var[CPDC(order=order)._cache_subkey]
    return _reference_spectra(fit["coef"], fit["sigma"], frequency_grid(nfreq))


def _reference_for_fresh(data, nfreq):
    CPDC(order=2).multivariate(data)            # make sure the fit is cached
    return _reference_for(data, 2, nfreq)


@pytest.mark.parametrize("order", [2, "bic"])
@pytest.mark.parametrize("measure", list(CLASSES))
def test_spectra_match_explicit_loops(measure, order):
    data = _data(_fixture())
    _, got = CLASSES[measure](order=order).spectrum(data)
    want = _reference_for(data, order, 256)[measure]
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-13)
    # The fit those references read is fit_var's on the z-scored series.
    fit = data.var[(order, None if order == 2 else 10)]
    np.testing.assert_array_equal(
        fit["coef"], fit_var(data.to_numpy(squeeze=True), fit["order"])["coef"])


@pytest.mark.parametrize("measure", list(CLASSES))
def test_single_frequency_bands_through_multivariate_and_bivariate(measure):
    """A band holding one grid frequency reads the spectrum at that frequency."""
    data = _data(_fixture())
    nfreq = 11                                  # 0, 0.05, ..., 0.5
    want = _reference_for_fresh(data, nfreq)[measure]
    off = _offdiag(4)
    for f in (0.0, 0.1, 0.25, 0.4, 0.5):
        n = int(round(f / 0.05))
        spi = CLASSES[measure](order=2, fmin=f, fmax=f, nfreq=nfreq)
        assert spi._band.tolist() == [n]
        got = spi.multivariate(data)
        np.testing.assert_allclose(got[off], want[n][off], rtol=1e-10, atol=1e-13)
        assert np.isnan(np.diag(got)).all()
        for i, j in [(0, 1), (1, 0), (2, 3), (3, 0)]:
            assert spi.bivariate(data, i=i, j=j) == pytest.approx(
                want[n, i, j], rel=1e-10, abs=1e-13)


@pytest.mark.parametrize("statistic", ["mean", "max"])
@pytest.mark.parametrize("band", list(BANDS))
@pytest.mark.parametrize("measure", list(CLASSES))
def test_band_statistics_match_the_reference(measure, band, statistic):
    data = _data(_fixture())
    spi = CLASSES[measure](order="bic", statistic=statistic, **BANDS[band])
    assert spi.identifier == f"{STEMS[measure]}_p-bic_{statistic}_{band}"
    got = spi.multivariate(data)
    freq = frequency_grid(256)
    lo, hi = BANDS[band].get("fmin", 0), BANDS[band].get("fmax", 0.5)
    inside = (freq >= lo - 1e-12) & (freq <= hi + 1e-12)
    assert inside.sum() == (256 if band == "fmin-0_fmax-0-5" else 128)
    reduce = np.mean if statistic == "mean" else np.max
    want = reduce(_reference_for(data, "bic", 256)[measure][inside], axis=0)
    off = _offdiag(4)
    np.testing.assert_allclose(got[off], want[off], rtol=1e-10, atol=1e-13)
    for i, j in [(0, 2), (2, 0), (1, 3)]:
        assert spi.bivariate(data, i=i, j=j) == pytest.approx(want[i, j], rel=1e-10)
    # A fresh dataset through bivariate alone gives the same value.
    fresh = _data(_fixture())
    assert spi.bivariate(fresh, i=3, j=1) == pytest.approx(got[3, 1], rel=1e-12)


def test_two_processes_pdc_is_dtf_and_three_are_not(chain):
    """On a pair PDC and DTF coincide identically (why pdcoh == dtf); not on M = 3."""
    def dtf(data):
        fit = data.var[("bic", 10)]
        H = np.linalg.inv(var_frequency_response(fit["coef"], frequency_grid()))
        power = np.abs(H) ** 2
        return np.swapaxes(power / power.sum(axis=-1, keepdims=True), -1, -2)

    # Off the diagonal: |A_ji|^2 / (|A_ji|^2 + |A_ii|^2) both, as H = adj(A) / det A.
    pair = _data(chain[:2])
    _, pdc = CPDC().spectrum(pair)
    off = _offdiag(2)
    np.testing.assert_allclose(pdc[:, off], dtf(pair)[:, off], rtol=1e-10, atol=1e-14)

    full = _data(chain)
    _, pdc = CPDC().spectrum(full)
    assert np.abs(pdc - dtf(full))[:, 0, 2].max() > 0.1


# --------------------------------------------------------------------------
# 2. Defining normalisations and symmetries
# --------------------------------------------------------------------------

def test_pdc_and_ffdtf_normalisations():
    data = _data(_fixture())
    for measure in ("pdc", "gpdc"):
        _, spec = CLASSES[measure]().spectrum(data)
        # sum_i |pi_ij(f)|^2 = 1 over targets i for each source j, every f:
        # row = source here, so the sum runs along each row.
        np.testing.assert_allclose(spec.sum(axis=2), 1.0, rtol=1e-12)
        assert np.all((spec >= 0) & (spec <= 1))

    fit = data.var[("bic", 10)]
    H = np.linalg.inv(var_frequency_response(fit["coef"], frequency_grid()))
    eta = full_frequency_dtf(H)                         # [f, target, source]
    # sum_f sum_j eta_ij(f)^2 = 1 for each target i, over sources j.
    np.testing.assert_allclose((eta ** 2).sum(axis=(0, 2)), 1.0, rtol=1e-12)
    np.testing.assert_allclose(
        eta, np.swapaxes(_reference_for(data, "bic", 256)["ffdtf"], 1, 2),
        rtol=1e-10, atol=1e-14)

    _, pcoh = PCOH().spectrum(data)
    _, ddtf = CDDTF().spectrum(data)
    np.testing.assert_allclose(ddtf, np.swapaxes(eta, 1, 2) * np.sqrt(pcoh),
                               rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.diagonal(pcoh, axis1=1, axis2=2), 1.0, rtol=1e-12)
    np.testing.assert_array_equal(pcoh, np.swapaxes(pcoh, 1, 2))
    assert np.all((pcoh >= 0) & (pcoh <= 1 + 1e-12))


@pytest.mark.parametrize("statistic", ["mean", "max"])
def test_partial_coherence_is_exactly_symmetric(statistic):
    A = PCOH(statistic=statistic).multivariate(_data(_fixture(seed=3)))
    off = _offdiag(4)
    np.testing.assert_array_equal(A[off], A.T[off])
    assert np.ptp(A[off]) > 0


def test_gpdc_is_invariant_to_rescaling_one_process_and_pdc_is_not():
    """Baccalá et al. (2007)'s point, without the default z-scoring."""
    x = _fixture()
    y = x.copy()
    y[2] *= 10.0
    def raw(z):
        return Data(data=z, dim_order="ps", zscore=False)

    off = _offdiag(4)
    g_x, g_y = CGPDC(order=2).multivariate(raw(x)), CGPDC(order=2).multivariate(raw(y))
    np.testing.assert_allclose(g_y[off], g_x[off], rtol=1e-8)
    p_x, p_y = CPDC(order=2).multivariate(raw(x)), CPDC(order=2).multivariate(raw(y))
    assert np.abs(p_y[off] - p_x[off]).max() > 0.05


# --------------------------------------------------------------------------
# 3. Known truth: a chain x0 -> x1 -> x2
# --------------------------------------------------------------------------

def test_chain_conditional_measures_drop_the_indirect_link(chain):
    data = _data(chain)
    cpdc, cgpdc, cddtf, pcoh = (cls().multivariate(data)
                                for cls in (CPDC, CGPDC, CDDTF, PCOH))
    assert data.var[("bic", 10)]["order"] == 1

    # The bivariate spectral SPIs at the same cell (full band, mean).
    ident = "_multitaper_mean_fs-1_fmin-0_fmax-0-5"
    pdcoh = spectral.PartialDirectedCoherence()
    dtf = spectral.DirectedTransferFunction()
    ddtf = spectral.DirectDirectedTransferFunction()
    cohmag = spectral.CoherenceMagnitude()
    for spi, stem in [(pdcoh, "pdcoh"), (dtf, "dtf"), (ddtf, "ddtf"),
                      (cohmag, "cohmag")]:
        assert spi.identifier == stem + ident
    pdcoh, dtf, ddtf = (s.multivariate(data) for s in (pdcoh, dtf, ddtf))
    with warnings.catch_warnings():
        # cohmag's NaN diagonal: "Mean of empty slice".
        warnings.simplefilter("ignore", RuntimeWarning)
        cohmag = cohmag.multivariate(data)

    # PDC: the direct links are clearly positive, the indirect one is not.
    for direct in (cpdc, cgpdc):
        assert direct[0, 1] > 0.15 and direct[1, 2] > 0.15
        assert direct[0, 2] < 1e-3
        assert direct[0, 2] < 0.005 * min(direct[0, 1], direct[1, 2])
    # ...where the pairwise PDC and DTF (identical on a pair) are not.
    np.testing.assert_allclose(pdcoh[_offdiag(3)], dtf[_offdiag(3)], rtol=1e-6)
    assert pdcoh[0, 2] > 0.1 and dtf[0, 2] > 0.5 * dtf[0, 1]

    # dDTF: the indirect path keeps a full-frequency DTF but loses its
    # partial coherence. Ratios, because ddtf_* normalises over the
    # backend's T-dependent grid and cddtf over 256 frequencies.
    assert cddtf[0, 1] > 0.01 and cddtf[1, 2] > 0.01
    assert cddtf[0, 2] < 0.1 * cddtf[0, 1]
    assert ddtf[0, 2] > 0.5 * ddtf[0, 1]

    # Partial coherence of 0 and 2 given 1 vanishes; ordinary coherence not.
    assert pcoh[0, 1] > 0.1 and pcoh[1, 2] > 0.1
    assert pcoh[0, 2] < 0.005
    assert cohmag[0, 2] > 0.2

    # And the directed measures are directed: nothing flows back up the chain.
    for direct in (cpdc, cgpdc):
        assert max(direct[1, 0], direct[2, 1], direct[2, 0]) < 1e-3


# --------------------------------------------------------------------------
# 4. Frequency specificity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("f0", [0.1, 0.4])
def test_coupling_from_a_resonant_source_peaks_in_its_band(f0):
    """Source 0 is an AR(2) oscillator at f0 (r = 0.95) driving 1 at lag 1.

    PDC is ``|A_10|^2 / (|A_00|^2 + |A_10|^2)``, and the source's own
    ``|A_00(f)|`` dips at its resonance, so the direct influence is
    concentrated in the half band holding f0.
    """
    r = 0.95
    A1 = np.array([[2 * r * np.cos(2 * np.pi * f0), 0.0], [0.5, 0.3]])
    A2 = np.array([[-r * r, 0.0], [0.0, 0.0]])
    data = _data(_simulate([A1, A2], 4000, seed=1))
    low, high = BANDS["fmin-0_fmax-0-25"], BANDS["fmin-0-25_fmax-0-5"]
    inside, outside = (low, high) if f0 < 0.25 else (high, low)
    for cls in (CPDC, CGPDC, CDDTF, PCOH):
        on = cls(**inside).multivariate(data)
        off = cls(**outside).multivariate(data)
        assert on[0, 1] > 4 * off[0, 1], cls.__name__
    # The peak itself sits at the resonance.
    freq, spec = CPDC().spectrum(data)
    assert abs(freq[np.argmax(spec[:, 0, 1])] - f0) < 0.02
    assert CPDC(**inside).multivariate(data)[1, 0] < 0.01


# --------------------------------------------------------------------------
# 5. Sharing cgc_*'s fit
# --------------------------------------------------------------------------

def test_shares_the_cgc_fit_and_never_refits(monkeypatch):
    data = _data(_fixture())
    CGC(order="bic").multivariate(data)
    entry = data.var[("bic", 10)]

    def refit(*args, **kwargs):
        raise AssertionError("the VAR was fitted again")

    monkeypatch.setattr(var, "fit_var", refit)
    monkeypatch.setattr(var, "select_order_bic", refit)
    for cls in CLASSES.values():
        for kw in BANDS.values():
            cls(**kw).multivariate(data)
        cls().bivariate(data, i=0, j=1)
    assert list(data.var) == [("bic", 10)] and data.var[("bic", 10)] is entry
    # One spectrum per measure, kept on the shared fit and reused by every band.
    assert set(entry["spectra"]) == {(m, 256) for m in CLASSES}
    assert CPDC(fmax=0.25).spectrum(data)[1] is entry["spectra"][("pdc", 256)]

    # Same scheduling bucket as cgc_* of the same order, not of another.
    bucket = _parallel.cache_bucket
    for cls in CLASSES.values():
        assert bucket(cls()) == bucket(CGC(order="bic")) == ("var", "bic", 10)
        assert bucket(cls(order=2)) == bucket(CGC(order=2))
        assert bucket(cls(order=2)) != bucket(cls())


def test_results_are_not_the_cache_and_the_cache_follows_the_data():
    x, y = _fixture(seed=1), _fixture(seed=2)
    data = _data(x)
    spi = CPDC()
    first = spi.multivariate(data)
    first[0, 1] = 1e6
    assert spi.multivariate(data)[0, 1] != 1e6
    _, spec = spi.spectrum(data)
    assert not spec.flags.writeable

    data.set_data(y, dim_order="ps")
    for cls in CLASSES.values():
        np.testing.assert_array_equal(cls().multivariate(data),
                                      cls().multivariate(_data(y)))


def test_singular_innovations_are_refused_where_they_matter():
    """A duplicated process: Sigma is singular, so S^{-1} does not exist."""
    x = _fixture()
    data = _data(np.vstack([x, x[:1]]))
    for cls in (PCOH, CDDTF):
        with pytest.raises(ValueError, match="singular"):
            cls(order=1).multivariate(data)
    # PDC needs only A(f), which the minimum-norm fit still defines.
    assert np.isfinite(CPDC(order=1).multivariate(data)[_offdiag(5)]).all()


# --------------------------------------------------------------------------
# Construction, identifiers and labels
# --------------------------------------------------------------------------

def test_identifiers_and_labels():
    assert CPDC().identifier == "cpdc_p-bic_mean_fmin-0_fmax-0-5"
    assert CGPDC(order=1, statistic="max").identifier == \
        "cgpdc_p-1_max_fmin-0_fmax-0-5"
    assert CDDTF(fmin=0.25, fmax=0.5).identifier == \
        "cddtf_p-bic_mean_fmin-0-25_fmax-0-5"
    assert PCOH(max_order=4, fmax=0.25, nfreq=64).identifier == \
        "pcohmag_p-bic_pmax-4_mean_fmin-0_fmax-0-25_nf-64"
    assert PCOH(fmax=np.float64(0.25)).identifier == PCOH(fmax=0.25).identifier
    for cls in (CPDC, CGPDC, CDDTF):
        assert "directed" in cls().labels and "undirected" not in cls().labels
        assert not cls().issigned()
    assert "undirected" in PCOH().labels and not PCOH().issigned()


@pytest.mark.parametrize("kwargs, exc, match", [
    (dict(order="aic"), ValueError, "order"),
    (dict(order=0), ValueError, "order"),
    (dict(order=2, max_order=5), ValueError, "only used"),
    (dict(statistic="median"), ValueError, "statistic"),
    (dict(fmin=0.3, fmax=0.2), ValueError, "above"),
    (dict(fmax=0.6), ValueError, "outside"),
    (dict(fmin=-0.1), ValueError, "outside"),
    (dict(fmin="0"), TypeError, "fmin"),
    (dict(nfreq=1), ValueError, "nfreq"),
    (dict(fmin=0.1, fmax=0.12, nfreq=3), ValueError, "no frequency"),
])
def test_constructor_rejects_bad_arguments(kwargs, exc, match):
    for cls in CLASSES.values():
        with pytest.raises(exc, match=match):
            cls(**kwargs)


# --------------------------------------------------------------------------
# 6. Calculator round trip and bundled configs
# --------------------------------------------------------------------------

def test_calculator_round_trip(tmp_path):
    cfg = {".statistics.var": {
        cls.__name__: {
            "labels": ["linear", "unsigned", "multivariate", "frequency-dependent"],
            "configs": [{"order": "bic", **kw} for kw in BANDS.values()]}
        for cls in CLASSES.values()}}
    cfg[".statistics.var"]["ConditionalGrangerCausality"] = {
        "labels": ["directed"], "configs": [{"order": "bic"}]}
    path = tmp_path / "cspec.yaml"
    path.write_text(yaml.safe_dump(cfg))
    x = _fixture(T=500, seed=5)
    calc = Calculator(dataset=_data(x), config=str(path))
    calc.compute()

    assert calc.errors == {}
    assert set(calc.spis) == set(BUNDLED) | {"cgc_p-bic"}
    off = _offdiag(4)
    for ident, (cls, kwargs) in BUNDLED.items():
        got = calc.table[ident].to_numpy(dtype=float)
        want = cls(**kwargs).multivariate(_data(x))
        np.testing.assert_allclose(got[off], want[off], rtol=1e-12, atol=1e-15)
    assert "undirected" in calc.spis["pcohmag_p-bic_mean_fmin-0_fmax-0-5"].labels
    assert "directed" in calc.spis["cddtf_p-bic_mean_fmin-0_fmax-0-5"].labels


def test_bundled_configs_construct():
    full = Calculator(config="full").spis
    ours = {k for k in full if k.split("_")[0] in STEMS.values()}
    assert ours == set(BUNDLED)
    for ident, (cls, _) in BUNDLED.items():
        assert type(full[ident]) is cls
        assert "multivariate" in full[ident].labels
    fast = Calculator(config="fast").spis
    assert {k for k in fast if k.split("_")[0] in STEMS.values()} == FAST
