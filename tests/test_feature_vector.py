"""``feature_vector`` keeps one value per independent pairwise feature.

A results table stores every SPI as a full ``M x M`` matrix. For an
``undirected`` SPI the lower triangle is a copy of the upper one, and for an
``antisymmetric`` SPI (PLI, wPLI, PSI, group delay, CCM diff, ...) it is the
negation, so flattening the whole matrix counts every pair twice. The feature
vector keeps ``i < j`` for those two traits and both orientations for
``directed`` and ``asymmetric`` SPIs, and drops self-pairs unless asked.
"""
import numpy as np
import pandas as pd
import pytest

import pyspi
from pyspi import Calculator
from pyspi.base import DIRECTEDNESS_TRAITS, Directed, Undirected
from pyspi.calculator import feature_vector, load_spis_from_yaml, resolve_config
from pyspi.statistics.basic import Covariance
from pyspi.statistics.infotheory import ConditionalEntropy
from pyspi.statistics.spectral import PhaseLagIndex


# --------------------------------------------------------------------------
# The function on a hand-built table
# --------------------------------------------------------------------------

PROCS = ["a", "b", "c"]


def _table(matrices):
    """A ``Calculator.table``-shaped frame from ``{spi: (3, 3) array}``."""
    spis = list(matrices)
    columns = pd.MultiIndex.from_product([spis, PROCS], names=["spi", "process"])
    table = pd.DataFrame(
        np.concatenate([matrices[k] for k in spis], axis=1),
        columns=columns, index=PROCS,
    )
    table.columns.name = "process"
    return table


U = np.array([[np.nan, 1.0, 2.0], [1.0, np.nan, 3.0], [2.0, 3.0, np.nan]])
A = np.array([[np.nan, 1.0, 2.0], [-1.0, np.nan, 3.0], [-2.0, -3.0, np.nan]])
D = np.array([[np.nan, 1.0, 2.0], [4.0, np.nan, 3.0], [5.0, 6.0, np.nan]])
TRAITS = {"u": "undirected", "a": "antisymmetric", "d": "directed", "s": "asymmetric"}


def test_one_triangle_for_undirected_and_antisymmetric_both_for_the_rest():
    v = feature_vector(_table({"u": U, "a": A, "d": D, "s": D}), TRAITS)

    assert list(v.index.names) == ["spi", "source", "target"]
    assert v.name == "value"
    # Upper triangle in row-major order, i.e. (a, b), (a, c), (b, c).
    assert v["u"].to_dict() == {("a", "b"): 1.0, ("a", "c"): 2.0, ("b", "c"): 3.0}
    # The kept antisymmetric entries are the positive orientation (row leads).
    assert v["a"].to_dict() == {("a", "b"): 1.0, ("a", "c"): 2.0, ("b", "c"): 3.0}
    # Directed and asymmetric keep every off-diagonal entry.
    assert v["d"].to_dict() == {
        ("a", "b"): 1.0, ("a", "c"): 2.0, ("b", "a"): 4.0,
        ("b", "c"): 3.0, ("c", "a"): 5.0, ("c", "b"): 6.0,
    }
    assert v["s"].equals(v["d"])
    # SPI blocks come in table column order.
    assert list(v.index.get_level_values("spi").unique()) == ["u", "a", "d", "s"]
    assert len(v) == 3 + 3 + 6 + 6


def test_self_pairs_are_dropped_by_default_and_kept_on_request():
    table = _table({"u": U, "d": D})
    assert not any(s == t for _, s, t in feature_vector(table, TRAITS).index)

    with_self = feature_vector(table, TRAITS, include_self=True)
    assert len(with_self) == (3 + 3) + (6 + 3)
    for spi in ("u", "d"):
        for p in PROCS:
            assert np.isnan(with_self[(spi, p, p)])
    # Self-pairs sit in row-major order with the rest, not appended.
    assert list(with_self["u"].index) == [
        ("a", "a"), ("a", "b"), ("a", "c"), ("b", "b"), ("b", "c"), ("c", "c")
    ]


def test_nans_are_kept_so_the_length_is_fixed():
    table = _table({"u": np.full((3, 3), np.nan), "d": D})
    v = feature_vector(table, TRAITS)
    assert len(v) == 3 + 6
    assert v["u"].isna().all()


def test_unknown_spi_falls_back_to_both_orientations():
    """A missing trait must not drop data: it is treated as directed."""
    v = feature_vector(_table({"x": U}), {})
    assert len(v) == 6


def test_unknown_trait_raises():
    with pytest.raises(ValueError, match="Unknown directedness"):
        feature_vector(_table({"u": U}), {"u": "symmetric"})


def test_table_without_run_spec_needs_an_explicit_mapping():
    with pytest.raises(ValueError, match="directedness="):
        feature_vector(_table({"u": U}))


def test_table_with_unresolvable_config_raises_rather_than_guessing(tmp_path):
    table = _table({"u": U})
    table.attrs["run_spec"] = {"config": str(tmp_path / "gone.yaml")}
    with pytest.raises(ValueError, match="Could not rebuild"):
        feature_vector(table)


def test_mismatched_process_names_are_refused():
    table = _table({"u": U})
    table.index = ["a", "b", "z"]
    with pytest.raises(ValueError, match="different processes"):
        feature_vector(table, TRAITS)


# --------------------------------------------------------------------------
# Trait accessor
# --------------------------------------------------------------------------

def test_directedness_of_bare_instances_falls_back_on_the_base_class():
    assert Covariance().directedness() == "undirected"
    assert ConditionalEntropy().directedness() == "directed"
    # Set in __init__ from the statistic, before any config is involved.
    assert PhaseLagIndex(statistic="mean").directedness() == "antisymmetric"
    assert PhaseLagIndex(statistic="max").directedness() == "asymmetric"
    assert Directed().directedness() == "directed"
    assert Undirected().directedness() == "undirected"


def test_every_configured_spi_has_exactly_one_trait_and_reports_it():
    spis = load_spis_from_yaml(resolve_config("full"), quiet=True)
    seen = set()
    for key, spi in spis.items():
        declared = [t for t in DIRECTEDNESS_TRAITS if t in spi.labels]
        assert len(declared) == 1, (key, spi.labels)
        assert spi.directedness() == declared[0], key
        seen.add(declared[0])
    assert {"undirected", "antisymmetric", "directed"} <= seen


# --------------------------------------------------------------------------
# Through a Calculator
# --------------------------------------------------------------------------

CONFIG = """
.statistics.basic:
  Covariance:
    labels: [undirected, linear, signed, multivariate, contemporaneous, M14]
    dependencies:
    configs:
      - estimator: EmpiricalCovariance
.statistics.spectral:
  PhaseLagIndex:
    labels: [undirected, linear, unsigned, bivariate, frequency-dependent, M05]
    dependencies:
    configs:
      - fs: 1
        fmin: 0
        fmax: 0.5
        statistic: mean
      - fs: 1
        fmin: 0
        fmax: 0.5
        statistic: max
.statistics.infotheory:
  ConditionalEntropy:
    labels: [directed, nonlinear, unsigned, bivariate, contemporaneous, M04]
    dependencies:
    configs:
      - estimator: gaussian
"""


@pytest.fixture(scope="module")
def computed(tmp_path_factory):
    configfile = tmp_path_factory.mktemp("fv") / "fv.yaml"
    configfile.write_text(CONFIG, encoding="utf-8")
    rng = np.random.default_rng(0)
    calc = Calculator(
        dataset=rng.standard_normal((4, 300)), config=str(configfile), verbose=False
    )
    calc.compute()
    assert not calc.errors
    return calc


def test_calculator_feature_vector_counts_follow_the_traits(computed):
    M = 4
    v = computed.feature_vector()
    sizes = v.groupby(level="spi", sort=False).size().to_dict()
    assert sizes == {
        "cov_EmpiricalCovariance": M * (M - 1) // 2,                 # undirected
        "pli_multitaper_mean_fs-1_fmin-0_fmax-0-5": M * (M - 1) // 2,  # antisymmetric
        "pli_multitaper_max_fs-1_fmin-0_fmax-0-5": M * (M - 1),       # asymmetric
        "ce_gaussian": M * (M - 1),                                   # directed
    }
    assert len(computed.feature_vector(include_self=True)) == len(v) + M * computed.n_spis


def test_calculator_feature_vector_loses_nothing(computed):
    """Each matrix is recovered from its vector plus the declared symmetry."""
    procs = list(computed.dataset.procnames)
    pos = {p: i for i, p in enumerate(procs)}
    M = len(procs)
    v = computed.feature_vector()
    for key, spi in computed.spis.items():
        rebuilt = np.full((M, M), np.nan)
        for (s, t), val in v[key].items():
            rebuilt[pos[s], pos[t]] = val
            if spi.directedness() == "undirected":
                rebuilt[pos[t], pos[s]] = val
            elif spi.directedness() == "antisymmetric":
                rebuilt[pos[t], pos[s]] = -val
        original = computed.table[key].to_numpy(dtype=float)
        off = ~np.eye(M, dtype=bool)
        np.testing.assert_allclose(rebuilt[off], original[off], err_msg=key)


def test_saved_table_gives_the_same_vector(computed, tmp_path):
    """``load_table`` keeps the config in ``attrs``; the traits are rebuilt from it."""
    path = computed.save(tmp_path / "fv.npz")
    loaded = pyspi.load_table(path)
    direct = computed.feature_vector()
    via_file = pyspi.feature_vector(loaded)
    assert via_file.index.equals(direct.index)
    np.testing.assert_array_equal(via_file.to_numpy(), direct.to_numpy())
