"""Declared structural traits must match observed behaviour.

Symmetry is three-valued, not two. A binary directed/undirected vocabulary has
no word for measures satisfying ``A[i,j] == -A[j,i]`` -- PLI, wPLI, PSI, and
CCM's "diff" statistic -- and labelling them ``undirected`` (which implies
symmetry) misdescribes them for any downstream filtering or grouping.

Resolved here:

* ``Cointegration`` declared ``Undirected`` while ``aeg`` computes
  ``stattools.coint(z[i], z[j])``, which is not symmetric in its arguments
  (measured: ~0.8 mean absolute difference between orientations, up to ~1.6).
  The cache then wrote the one computed value to both ``(i, j)`` and
  ``(j, i)``, so which orientation you got depended on visit order. ``aeg`` is
  now ``directed`` and reports what it computes; ``johansen``, which is
  symmetric to ~3e-14, keeps the alias.
* Wavelet ``PhaseSlopeIndex`` filled its upper triangle from the lower one
  *without negating*, inverting the lead/lag sign for half of every matrix.
* ``hhg`` was declared directed but is exactly symmetric; ``ce``, ``dcorrx``
  and ``mgcx`` were labelled undirected in configs but are directed.

One open finding remains, marked ``xfail(strict=True)`` with its reasoning in
the marker. It is recorded rather than silently patched because it needs a
scientific decision, not a code change.

The audit reads the committed baseline matrices, so it costs no computation.
"""
import os

import numpy as np
import pytest

from pyspi.calculator import load_spis_from_yaml, resolve_config
from pyspi.data import Data
from pyspi.statistics.misc import Cointegration

BASELINE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "baselines", "var1_M3_T100.npz"
)


def _offdiag(a):
    m = a.shape[0]
    return a[~np.eye(m, dtype=bool)]


# --------------------------------------------------------------------------
# AEG semantics
# --------------------------------------------------------------------------

def test_aeg_value_is_independent_of_process_order():
    """Permuting the input processes must not change a pair's AEG value."""
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((3, 200)).cumsum(axis=1)  # integrated series

    spi = Cointegration(method="aeg", statistic="tstat")

    forward = spi.multivariate(Data(data=arr, dim_order="ps", zscore=False))

    perm = [2, 1, 0]
    inv = np.argsort(perm)
    permuted = Cointegration(method="aeg", statistic="tstat").multivariate(
        Data(data=arr[perm], dim_order="ps", zscore=False)
    )
    restored = permuted[np.ix_(inv, inv)]

    assert np.allclose(_offdiag(forward), _offdiag(restored), equal_nan=True), (
        "AEG changed under a permutation of the processes: the cached value "
        "depends on which orientation was computed first."
    )


def test_aeg_declared_symmetry_matches_the_statistic():
    """If AEG is labelled undirected, the underlying statistic must be symmetric."""
    from statsmodels.tsa import stattools

    rng = np.random.default_rng(1)
    x = rng.standard_normal(200).cumsum()
    y = rng.standard_normal(200).cumsum()

    fwd = stattools.coint(x, y, autolag="aic", maxlag=10, trend="c")[0]
    rev = stattools.coint(y, x, autolag="aic", maxlag=10, trend="c")[0]

    spi = Cointegration(method="aeg")
    declared_undirected = "undirected" in getattr(spi, "labels", [])

    if declared_undirected:
        assert np.isclose(fwd, rev), (
            f"Cointegration(method='aeg') is labelled undirected, but the AEG "
            f"t-statistic is asymmetric: coint(x,y)={fwd:.6g} vs coint(y,x)={rev:.6g}."
        )


# --------------------------------------------------------------------------
# Declared labels vs observed matrices
# --------------------------------------------------------------------------

def _classify(mat):
    """Classify an MxM matrix as symmetric / antisymmetric / asymmetric.

    Antisymmetry (``A[i,j] == -A[j,i]``) is a genuine third category, not a
    broken form of either other one: phase-based measures such as PLI, wPLI and
    PSI carry a sign that encodes lead/lag. The label vocabulary currently has
    no word for it, which is why they show up as "undirected but asymmetric".
    """
    finite = np.isfinite(mat)
    pairwise = finite & finite.T & ~np.eye(mat.shape[0], dtype=bool)
    if not pairwise.any():
        return "undetermined"
    a, at = mat[pairwise], mat.T[pairwise]
    if np.allclose(a, at, rtol=1e-9, atol=1e-12):
        # A constant (e.g. identically zero) matrix is vacuously symmetric;
        # calling it "symmetric" would mask a degenerate estimator.
        return "degenerate" if np.ptp(a) == 0 else "symmetric"
    if np.allclose(a, -at, rtol=1e-9, atol=1e-12):
        return "antisymmetric"
    return "asymmetric"


def _label_symmetry_audit():
    """Return {identifier: (declared, observed)} disagreements from baselines."""
    z = np.load(BASELINE, allow_pickle=False)
    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)

    disagreements = {}
    for ident, spi in spis.items():
        if ident not in z.files:
            continue
        mat = np.asarray(z[ident], dtype=float)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            continue

        observed = _classify(mat)
        if observed in ("undetermined", "degenerate"):
            continue  # covered by the degeneracy tests, not by this one

        # Only undirected -> asymmetric is a contradiction. A *directed*
        # measure may legitimately produce a symmetric matrix on a particular
        # dataset: lmfit_* and gpfit_DotProduct are symmetric on z-scored data
        # because a linear model's R^2 is, yet with zscore=False their measured
        # asymmetry is ~22.7-23.3. Flagging that direction produced false
        # positives, not findings.
        labels = set(getattr(spi, "labels", []) or [])
        if "undirected" in labels and observed == "asymmetric":
            disagreements[ident] = ("undirected", observed)
    return disagreements


def test_declared_symmetry_matches_observed_matrices():
    bad = _label_symmetry_audit()
    assert not bad, "Declared/observed symmetry disagreements:\n" + "\n".join(
        f"  {k}: declared {v[0]}, observed {v[1]}" for k, v in sorted(bad.items())
    )


def test_antisymmetric_measures_are_labelled_as_such():
    """PLI/wPLI/PSI encode lead-lag in their sign; 'undirected' misdescribes them."""
    z = np.load(BASELINE, allow_pickle=False)
    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)

    mislabelled = []
    for ident, spi in spis.items():
        if ident not in z.files:
            continue
        if _classify(np.asarray(z[ident], dtype=float)) != "antisymmetric":
            continue
        labels = set(getattr(spi, "labels", []) or [])
        if "antisymmetric" not in labels:
            mislabelled.append(f"{ident} (labelled: {sorted(labels & {'directed', 'undirected'})})")

    assert not mislabelled, (
        "Antisymmetric SPIs carry no 'antisymmetric' label:\n  "
        + "\n  ".join(sorted(mislabelled))
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Fixture/low-data finding, not a proven universal defect. "
        "dspli_*_max and dswpli_*_max variants return a "
        "constant matrix on var1_M3_T100 (M=3, T=100), so on that fixture "
        "they carry no pairwise information. Whether it holds at larger M or "
        "T has not been established, and none should be removed from "
        "the shipped set on this evidence alone."
    ),
)
def test_no_bundled_spi_returns_a_constant_matrix():
    """A constant matrix carries no pairwise information."""
    z = np.load(BASELINE, allow_pickle=False)
    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)

    degenerate = [
        ident for ident in spis
        if ident in z.files
        and _classify(np.asarray(z[ident], dtype=float)) == "degenerate"
    ]
    assert not degenerate, (
        "SPIs returning a constant matrix on var1_M3_T100: " + ", ".join(sorted(degenerate))
    )


def test_conditional_entropy_label_matches_implementation():
    """Under the default z-scoring the Gaussian form is symmetric, so the
    bundled variants are labelled undirected; the kernel and kozachenko forms
    remain asymmetric and that is checked by the symmetry audit above."""
    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)
    ce = {k: v for k, v in spis.items() if k.startswith("ce_")}
    assert ce, "Precondition: full config must contain ConditionalEntropy variants."

    contradictory = [
        k for k, v in ce.items()
        if {"directed", "undirected"} <= set(getattr(v, "labels", []) or [])
    ]
    assert not contradictory, (
        f"ConditionalEntropy variants carry both labels: {sorted(contradictory)}"
    )


# --------------------------------------------------------------------------
# Wavelet PSI band statistics
# --------------------------------------------------------------------------

def test_circular_nanmean_refuses_unresolved_resultants_and_preserves_wraparound():
    from pyspi.statistics.spectral import _circular_nanmean

    # Exact cancellation leaves only sin(pi)'s floating-point residue. Tiling
    # it exercises the count-scaled error bound rather than one special pair.
    assert np.isnan(_circular_nanmean(np.array([0.0, np.pi]), axis=0))
    cancellation = np.tile([0.0, np.pi], 64)
    assert np.isnan(_circular_nanmean(cancellation, axis=0))
    assert np.isnan(_circular_nanmean(np.array([np.nan, np.nan]), axis=0))

    # +/-pi denote the same circular location and therefore cannot supply a
    # unique signed orientation in an ordinary-float antisymmetric matrix.
    assert np.isnan(_circular_nanmean(np.array([np.pi, np.pi]), axis=0))
    assert np.isnan(_circular_nanmean(np.array([-np.pi, -np.pi]), axis=0))

    wrapped = np.array([np.pi - 0.1, -np.pi + 0.1, np.pi - 0.05])
    expected = np.angle(np.mean(np.exp(1j * wrapped)))
    got = _circular_nanmean(wrapped, axis=0)
    assert np.isfinite(got)
    assert got == pytest.approx(expected)


def test_wavelet_coherence_phase_uses_a_circular_mean_and_negates_orientation():
    from pyspi.statistics.wavelet import CoherencePhase

    phase = np.zeros((3, 3, 3))
    phase[1, 0] = [np.pi - 0.1, -np.pi + 0.1, np.pi - 0.05]
    phase[2, 0] = [0.1, 0.3, 0.5]
    phase[2, 1] = [-0.4, -0.2, 0.6]
    lower = np.exp(1j * phase)
    lower[np.triu_indices(3, 1)] = 0
    freq = np.array([0.1, 0.2, 0.3])

    spi = CoherencePhase(statistic="mean", fmin=0, fmax=0.5)
    spi._get_cache = lambda data: (lower, np.arange(freq.size))
    got = spi.multivariate(Data(data=np.ones((3, 8)), dim_order="ps",
                                zscore=False))

    for i, j in ((1, 0), (2, 0), (2, 1)):
        expected = np.angle(np.mean(np.exp(1j * phase[i, j])))
        assert got[i, j] == pytest.approx(expected)
        assert got[j, i] == pytest.approx(-expected)
    assert set(spi.labels) & {"directed", "undirected", "antisymmetric",
                              "asymmetric"} == {"antisymmetric"}
    assert "signed" in spi.labels and spi.issigned()


def test_wavelet_coherence_phase_is_process_permutation_covariant():
    from pyspi.statistics.wavelet import CoherencePhase

    full = np.zeros((3, 3, 3))
    full[1, 0] = [-0.7, 0.2, 0.4]
    full[2, 0] = [0.1, 0.3, 0.5]
    full[2, 1] = [-0.4, -0.2, 0.6]
    full = full - full.transpose(1, 0, 2)
    data = Data(data=np.ones((3, 8)), dim_order="ps", zscore=False)

    def calculate(order):
        phase = full[np.ix_(order, order, np.arange(3))]
        lower = np.exp(1j * phase)
        lower[np.triu_indices(3, 1)] = 0
        spi = CoherencePhase(statistic="mean", fmin=0, fmax=0.5)
        spi._get_cache = lambda unused: (lower, np.arange(3))
        return spi.multivariate(data)

    order = [2, 0, 1]
    base = calculate([0, 1, 2])
    moved = calculate(order)
    assert np.allclose(base[np.ix_(order, order)], moved, equal_nan=True)


def test_wavelet_coherence_phase_refuses_exact_antiphase():
    from pyspi.statistics.wavelet import CoherencePhase

    phase = np.zeros((2, 2, 3))
    phase[1, 0] = np.pi
    lower = np.exp(1j * phase)
    lower[np.triu_indices(2, 1)] = 0
    spi = CoherencePhase(statistic="mean", fmin=0, fmax=0.5)
    spi._get_cache = lambda unused: (lower, np.arange(3))
    got = spi.multivariate(Data(data=np.ones((2, 8)), dim_order="ps",
                                zscore=False))
    assert np.isnan(got[0, 1]) and np.isnan(got[1, 0])


def test_spectral_coherence_phase_uses_a_circular_mean_and_is_permutation_covariant():
    from pyspi.statistics.spectral import CoherencePhase

    full = np.zeros((3, 3, 3))
    full[:, 1, 0] = [np.pi - 0.1, -np.pi + 0.1, np.pi - 0.05]
    full[:, 2, 0] = [0.1, 0.3, 0.5]
    full[:, 2, 1] = [-0.4, -0.2, 0.6]
    full = full - full.transpose(0, 2, 1)
    freq = np.array([0.1, 0.2, 0.3])
    data = Data(data=np.ones((3, 8)), dim_order="ps", zscore=False)

    def calculate(order):
        phase = full[:, order][:, :, order]
        spi = CoherencePhase(statistic="mean", fmin=0, fmax=0.5)
        spi._get_cache = lambda unused: (phase[None, ...], freq)
        return spi.multivariate(data)

    base = calculate([0, 1, 2])
    spi = CoherencePhase(statistic="mean", fmin=0, fmax=0.5)
    assert "signed" in spi.labels and "antisymmetric" in spi.labels
    assert "unsigned" not in spi.labels and "undirected" not in spi.labels
    assert spi.issigned()
    expected = np.angle(np.mean(np.exp(1j * full[:, 1, 0])))
    assert base[1, 0] == pytest.approx(expected)
    assert base[0, 1] == pytest.approx(-expected)
    assert np.allclose(base, -base.T, equal_nan=True)

    order = [2, 0, 1]
    moved = calculate(order)
    assert np.allclose(base[np.ix_(order, order)], moved, equal_nan=True)


def test_spectral_coherence_phase_var_fixture_is_exactly_antisymmetric_and_covariant():
    from pyspi.statistics.spectral import CoherencePhase

    raw = np.load(os.path.join(
        os.path.dirname(BASELINE), "..", "fixtures", "var1_M3_T100.npy"
    ))

    def calculate(values):
        data = Data(data=values, dim_order="sp", zscore=False)
        return CoherencePhase(
            statistic="mean", fs=1, fmin=0, fmax=0.5
        ).multivariate(data)

    base = calculate(raw)
    assert np.nanmax(np.abs(base + base.T)) == 0.0
    order = [2, 0, 1]
    moved = calculate(raw[:, order])
    assert np.allclose(base[np.ix_(order, order)], moved, equal_nan=True)


def test_spectral_coherence_phase_actual_backend_refuses_exact_antiphase():
    from pyspi.statistics.spectral import CoherencePhase

    x = np.random.default_rng(0).standard_normal(256)

    def calculate(values):
        return CoherencePhase(
            statistic="mean", fs=1, fmin=0, fmax=0.5
        ).multivariate(Data(data=values, dim_order="ps", zscore=False))

    base = calculate(np.vstack([x, -x]))
    moved = calculate(np.vstack([-x, x]))
    assert np.isnan(base[0, 1]) and np.isnan(base[1, 0])
    assert np.allclose(base[::-1, ::-1], moved, equal_nan=True)


def test_coherence_phase_refuses_branch_dependent_maximum():
    from pyspi.statistics.spectral import CoherencePhase as SpectralPhase
    from pyspi.statistics.wavelet import CoherencePhase as WaveletPhase

    for cls in (SpectralPhase, WaveletPhase):
        with pytest.raises(ValueError, match="branch-cut-independent ordinary maximum"):
            cls(statistic="max")

@pytest.mark.parametrize("statistic", ["mean", "max"])
def test_wavelet_psi_is_permutation_invariant(statistic):
    """Both band statistics must survive a permutation of the processes.

    mne_connectivity returns a lower-triangular tensor and the upper triangle
    is filled by negating. That fill must happen *before* the band statistic:
    negating after reduction is only valid for a statistic commuting with
    negation. mean commutes, max does not --
    ``max_f(-v) = -min_f(v) != -max_f(v)`` -- so reducing first made the max
    variants permutation-dependent by up to 11.5.
    """
    from pyspi.statistics.wavelet import PhaseSlopeIndex

    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(400))
    arr = np.vstack([x, np.roll(x, 4) + 0.1 * rng.standard_normal(400),
                     rng.standard_normal(400)])
    perm = [2, 0, 1]
    inv = np.argsort(perm)

    fwd = PhaseSlopeIndex(statistic=statistic).multivariate(
        Data(data=arr, dim_order="ps", zscore=True))
    permuted = PhaseSlopeIndex(statistic=statistic).multivariate(
        Data(data=arr[perm], dim_order="ps", zscore=True))
    restored = permuted[np.ix_(inv, inv)]

    off = ~np.eye(3, dtype=bool)
    assert np.allclose(fwd[off], restored[off], equal_nan=True), (
        f"psi_wavelet statistic={statistic} is not permutation-invariant; "
        f"max|d|={np.nanmax(np.abs(fwd[off] - restored[off])):.6g}"
    )


def test_antisymmetric_spis_report_themselves_as_signed():
    """`issigned()` drives a transform, so it cannot disagree with the values.

    `Calculator._rmmin` subtracts the minimum from every SPI that reports
    unsigned. On an antisymmetric matrix that shifts A[i,j] and A[j,i] by the
    same amount, destroying the antisymmetry that carries the lead/lag;
    `set_group` separately correlates unsigned SPIs through `abs()`, folding
    lead onto lag. The shipped configs declared `unsigned` for `phase`, `pli`,
    `wpli`, `psi` (both the multitaper and wavelet families), `gd` and
    `ccm_*_diff`.
    """
    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)
    bad = [
        ident for ident, spi in spis.items()
        if ({"antisymmetric", "asymmetric"} & set(spi.labels)) and not spi.issigned()
    ]
    assert not bad, "antisymmetric SPIs reporting unsigned:\n  " + "\n  ".join(sorted(bad))


def test_no_spi_declares_both_signed_and_unsigned():
    """One authority. `_merge_spi_labels` resolves the label from `issigned()`."""
    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)
    bad = [i for i, s in spis.items() if {"signed", "unsigned"} <= set(s.labels)]
    assert not bad, "contradictory signedness labels:\n  " + "\n  ".join(sorted(bad))


def test_every_bundled_spi_declares_a_signedness():
    """`_rmmin` calls `issigned()` unguarded, so a missing one is an exception.

    `CrossPairwiseDistance` subclassed only `Undirected`, which supplies no
    `issigned`, so `Calculator._rmmin()` raised AttributeError on any config
    containing it.
    """
    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)
    missing = [i for i, s in spis.items() if not hasattr(s, "issigned")]
    assert not missing, "SPIs with no issigned():\n  " + "\n  ".join(sorted(missing))


# --------------------------------------------------------------------------
# Structural traits are authoritative over config-declared directedness
# --------------------------------------------------------------------------

_GROUP_DELAY_YAML = """.statistics.spectral:
  GroupDelay:
    labels: [directed, linear, unsigned, bivariate, M01]
    configs:
      - {fmin: 0, fmax: 0.5, statistic: delay}
      - {fmin: 0, fmax: 0.5, statistic: slope}
      - {fmin: 0, fmax: 0.5, statistic: rvalue}
"""

_EXPECTED_GD_TRAITS = {
    "delay": ({"antisymmetric", "signed"}, True),
    "slope": ({"antisymmetric", "signed"}, True),
    "rvalue": ({"undirected", "unsigned"}, False),
}
_TRAITS = {"directed", "undirected", "antisymmetric", "asymmetric",
           "signed", "unsigned"}


@pytest.mark.parametrize("statistic", ["delay", "slope", "rvalue"])
def test_group_delay_traits_survive_direct_construction(statistic):
    from pyspi.statistics.spectral import GroupDelay

    spi = GroupDelay(statistic=statistic, fmin=0, fmax=0.5)
    expected, signed = _EXPECTED_GD_TRAITS[statistic]
    assert set(spi.labels) & _TRAITS == expected
    assert spi.issigned() is signed


def test_group_delay_traits_survive_a_yaml_family_label(tmp_path):
    """A family-level `directed` must not displace the SPI's own trait.

    `gd_*_rvalue` is symmetric by construction -- it stores |r| -- and declares
    itself undirected, but the family label put `directed` back alongside it, so
    `filter_spis(["directed"])` and `filter_spis(["undirected"])` both returned
    it. `delay` and `slope` are antisymmetric, which displaces both.
    """
    from pyspi.calculator import load_spis_from_yaml

    config = tmp_path / "gd.yaml"
    config.write_text(_GROUP_DELAY_YAML)
    for identifier, spi in load_spis_from_yaml(str(config), quiet=True).items():
        statistic = identifier.split("_")[2]
        expected, signed = _EXPECTED_GD_TRAITS[statistic]
        assert set(spi.labels) & _TRAITS == expected, identifier
        assert spi.issigned() is signed, identifier


def test_yaml_cannot_add_a_competing_structural_trait(tmp_path):
    """The instance's one structural trait displaces all YAML competitors."""
    from pyspi.calculator import load_spis_from_yaml

    config = tmp_path / "conflicting_traits.yaml"
    config.write_text(
        ".statistics.infotheory:\n"
        "  MutualInfo:\n"
        "    configs:\n"
        "      - {estimator: gaussian, labels: [directed, antisymmetric, asymmetric]}\n"
        "  TransferEntropy:\n"
        "    configs:\n"
        "      - {estimator: gaussian, labels: [undirected, antisymmetric, asymmetric]}\n"
        ".statistics.spectral:\n"
        "  CoherencePhase:\n"
        "    configs:\n"
        "      - {statistic: mean, fmin: 0, fmax: 0.5, labels: [directed, undirected, asymmetric]}\n"
    )
    expected = {
        "mi_gaussian": "undirected",
        "gc_gaussian_k-1_kt-1_l-1_lt-1": "directed",
        "phase_multitaper_mean_fs-1_fmin-0_fmax-0-5": "antisymmetric",
    }
    spis = load_spis_from_yaml(str(config), quiet=True)
    assert set(spis) == set(expected)
    for identifier, trait in expected.items():
        assert set(spis[identifier].labels) & set(_TRAITS) == {
            trait,
            "signed" if trait in {"antisymmetric", "asymmetric"} else "unsigned",
        }


def test_no_spi_in_any_bundled_config_declares_two_directedness_traits():
    from pyspi.calculator import bundled_configs, load_spis_from_yaml, resolve_config

    for name in bundled_configs():
        for identifier, spi in load_spis_from_yaml(resolve_config(name),
                                                   quiet=True).items():
            labels = set(spi.labels)
            structural = labels & {
                "directed", "undirected", "antisymmetric", "asymmetric"
            }
            assert len(structural) == 1, f"{name}/{identifier}: {structural}"


# --------------------------------------------------------------------------
# Causal-statistic metadata
# --------------------------------------------------------------------------

def test_igci_is_named_and_labelled_for_what_it_computes():
    """It is Information-Geometric Causal *Inference*, and its score is signed.

    The score is a difference of two entropies, hence exactly antisymmetric.
    Reporting it unsigned was not cosmetic: `Calculator._rmmin` shifts every
    column it believes unsigned by that column's minimum, which on an
    antisymmetric matrix moves both orientations equally and destroys the sign,
    and `set_group` folds the directions together through `abs()`.
    """
    from pyspi.data import Data
    from pyspi.statistics.causal import InformationGeometricCausalInference

    spi = InformationGeometricCausalInference()
    assert "causal inference" in spi.name.lower()
    assert "conditional independence" not in spi.name.lower()
    assert set(spi.labels) & _TRAITS == {"antisymmetric", "signed"}
    assert spi.issigned()

    rng = np.random.default_rng(0)
    x = rng.standard_normal(300)
    y = np.exp(x) + 0.1 * rng.standard_normal(300)
    table = spi.multivariate(Data(data=np.vstack([x, y]), dim_order="ps"))
    assert table[0, 1] == pytest.approx(-table[1, 0], rel=1e-12)


def test_the_old_igci_name_still_works_and_warns():
    """Compatibility alias, so existing configs and scripts keep running."""
    import pyspi.statistics.causal as causal

    with pytest.warns(DeprecationWarning, match="Causal"):
        old = causal.InformationGeometricConditionalIndependence()
    assert isinstance(old, causal.InformationGeometricCausalInference)
    assert old.identifier == causal.InformationGeometricCausalInference().identifier


def test_additive_noise_model_is_not_labelled_linear():
    """It fits a Gaussian process and tests independence with an RBF HSIC."""
    from pyspi.statistics.causal import AdditiveNoiseModel

    labels = set(AdditiveNoiseModel().labels)
    assert "nonlinear" in labels and "linear" not in labels


def test_igci_stays_out_of_the_bundled_configs():
    """A metadata correction, not a claim that the heuristic is reliable."""
    from pyspi.calculator import bundled_configs, load_spis_from_yaml, resolve_config

    for name in bundled_configs():
        assert "igci" not in load_spis_from_yaml(resolve_config(name), quiet=True)
