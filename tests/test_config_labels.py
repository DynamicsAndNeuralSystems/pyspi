"""Label consistency across the bundled configs.

The same SPI must carry the same labels whichever config it is loaded from,
no SPI may carry a contradictory pair, every label must belong to the fixed
vocabulary of CONTRIBUTING.md (which catches typos such as ``sigend``, a
stray ``directed,`` or a compound such as ``linear/nonlinear``), and every
SPI carries exactly one label from each axis of that vocabulary. One axis
says what the value is: ``strength``, the kind of an SPI that declares no
other, or ``timing`` or ``profile``, set by the class on fixed sets of bundled
SPIs. ``signed`` is a flag, carried exactly when ``issigned()`` is true;
``unsigned`` is not a label, and one in a config is dropped. ``inverted`` is a
flag set by the class, only on a ``strength`` SPI that is not ``signed``.
The labels a config writes for an SPI must also agree with the SPI itself:
its sign with ``issigned()``, its directedness with the class trait, its
linearity with the class where the class declares one, and each SPI carries
exactly one module label.
"""
import collections
import copy
import importlib
import re

import pytest

from pyspi.base import DIRECTEDNESS_TRAITS, VALUE_KINDS, Undirected
from pyspi.calculator import (
    _expand_lagged_correlation_configs,
    _is_module_label,
    _merge_spi_labels,
    _split_config_params,
    bundled_configs,
    load_config_yaml,
    load_spis_from_yaml,
    resolve_config,
)

CONTRADICTORY = (
    ("linear", "nonlinear"),
    ("directed", "undirected"),
)
# The label vocabulary of CONTRIBUTING.md: each SPI carries exactly one
# label from each axis, and the flags ``signed`` and ``inverted`` where they
# apply.
LABEL_AXES = {
    "directedness": {"directed", "undirected", "antisymmetric"},
    "linearity": {"linear", "nonlinear"},
    "arity": {"bivariate", "multivariate"},
    "time": {"contemporaneous", "time-dependent", "frequency-dependent"},
    "module": {f"M{n:02d}" for n in range(1, 15)} | {"MXX"},
    "value": set(VALUE_KINDS),
}
# Descriptive tags a class may carry in addition, none required.
DESCRIPTIVE_TAGS = {
    "basic", "causal", "distance", "embedding", "infotheory", "lagged", "misc",
    "normal", "rank", "spectral", "symbolic", "temporal", "unordered",
    "wavelet",
}
FLAGS = {"signed", "inverted"}
VOCABULARY = set().union(*LABEL_AXES.values()) | DESCRIPTIVE_TAGS | FLAGS
# The SPIs of `full` whose value is an offset (timing) or describes how the
# dependence is distributed (profile).
TIMING_IN_FULL = {
    "lagcorr_lag_L-None", "lagcorr_lag_L-10", "lccm_E-None_k-5_tp",
    "phase_multitaper_mean_fs-1_fmin-0_fmax-0-5",
    "phase_multitaper_mean_fs-1_fmin-0_fmax-0-25",
    "phase_multitaper_mean_fs-1_fmin-0-25_fmax-0-5",
}
PROFILE_IN_FULL = {
    "edge_excessvar", "edge_topshare_f-0-05", "edge_acf1",
    "tailasym_q-0-95", "tailasym_q-0-9",
}
# One-ended SPIs whose scale runs the other way: lower means stronger.
INVERTED_IN_FULL = {
    # distances and dissimilarities
    "dtw", "dtw_constraint-itakura", "dtw_constraint-sakoe-chiba_radius-auto",
    "softdtw", "softdtw_constraint-itakura", "softdtw_constraint-sakoe-chiba",
    "pdist_cityblock", "pdist_chebyshev", "pdist_canberra", "pdist_braycurtis",
    "xpdist_euclidean_tau-1_min_rmse", "xpdist_euclidean_tau-1_mean_rmse",
    # entropies
    "je_kozachenko", "je_kernel_W-0.5", "ce_kozachenko", "ce_kernel_W-0.5",
    "cce_gaussian_n-5", "cce_kozachenko_n-5", "xme_gaussian_k1",
    "xme_kozachenko_k1", "xme_kernel_W-0.5_k1", "xme_gaussian_k10",
    "xme_kozachenko_k10",
    # Engle-Granger t-statistic: more negative is stronger evidence
    "coint_aeg_tstat_trend-c_autolag-aic_maxlag-10",
    "coint_aeg_tstat_trend-ct_autolag-aic_maxlag-10",
    "coint_aeg_tstat_trend-ct_autolag-bic_maxlag-10",
    # causal scores and prediction errors: lower is stronger evidence
    "anm", "cds", "reci", "gpfit_DotProduct", "gpfit_RBF",
}


@pytest.fixture(scope="module")
def labels_by_config():
    out = {}
    for name in bundled_configs():
        spis = load_spis_from_yaml(resolve_config(name), quiet=True)
        out[name] = {ident: tuple(sorted(spi.labels)) for ident, spi in spis.items()}
    return out


def test_all_bundled_configs_are_checked(labels_by_config):
    assert len(labels_by_config) == 8


def test_same_spi_has_same_labels_in_every_config(labels_by_config):
    seen = collections.defaultdict(dict)
    for config, spis in labels_by_config.items():
        for ident, labels in spis.items():
            seen[ident][config] = labels
    inconsistent = {
        ident: by_config for ident, by_config in seen.items()
        if len(set(by_config.values())) > 1
    }
    assert not inconsistent, "SPIs with config-dependent labels:\n" + "\n".join(
        f"  {ident}: {by_config}" for ident, by_config in sorted(inconsistent.items())
    )


def test_no_contradictory_label_pairs(labels_by_config):
    bad = [
        f"{config}/{ident}: {pair}"
        for config, spis in labels_by_config.items()
        for ident, labels in spis.items()
        for pair in CONTRADICTORY
        if all(label in labels for label in pair)
    ]
    assert not bad, "\n".join(bad)


def test_labels_come_from_the_documented_vocabulary(labels_by_config):
    unknown = sorted(
        {
            f"{config}/{ident}: {label!r}"
            for config, spis in labels_by_config.items()
            for ident, labels in spis.items()
            for label in labels
            if label not in VOCABULARY
        }
    )
    assert not unknown, "\n".join(unknown)


def test_every_spi_has_one_label_per_axis(labels_by_config):
    bad = [
        f"{config}/{ident}: {axis} {sorted(set(labels) & axis_labels)}"
        for config, spis in labels_by_config.items()
        for ident, labels in spis.items()
        for axis, axis_labels in LABEL_AXES.items()
        if len(set(labels) & axis_labels) != 1
    ]
    assert not bad, "\n".join(bad)


def test_the_vocabulary_matches_contributing():
    """Each axis label and module code above is named in CONTRIBUTING.md."""
    import pathlib

    text = (pathlib.Path(__file__).parents[1] / "CONTRIBUTING.md").read_text()
    named = set(re.findall(r"`([A-Za-z0-9-]+)`", text))
    for axis, axis_labels in LABEL_AXES.items():
        if axis != "module":
            assert axis_labels <= named, (axis, sorted(axis_labels - named))
    assert {"M01", "M14", "MXX"} | FLAGS <= named
    assert DESCRIPTIVE_TAGS <= named, sorted(DESCRIPTIVE_TAGS - named)


def test_merge_rejects_unresolvable_contradictions():
    from pyspi.calculator import _merge_spi_labels

    class Bare:
        labels = []

    with pytest.raises(ValueError, match="contradictory"):
        _merge_spi_labels(Bare(), config_labels=["linear", "nonlinear"])


def test_class_trait_wins_over_config_linearity():
    from pyspi.calculator import _merge_spi_labels

    class Nonlinear:
        labels = ["nonlinear"]

    spi = Nonlinear()
    _merge_spi_labels(spi, config_labels=["linear"])
    assert spi.labels == ["nonlinear", "strength"]


def _config_entries(name):
    """
    Each SPI of a bundled config with the labels the config writes for it.

    The written labels are the family's followed by the entry's, with the
    family's module label dropped when the entry sets its own, as
    ``_merge_spi_labels`` does.

    Parameters
    ----------
    name : str
        Name of a bundled config.

    Returns
    -------
    list of tuple
        ``(written labels, SPI as constructed, SPI with merged labels)`` per
        SPI, in config order.
    """
    out = []
    for module_name, module_spis in load_config_yaml(resolve_config(name)).items():
        module = importlib.import_module(module_name, "pyspi")
        for fcn, entry in module_spis.items():
            family_labels = list(entry.get("labels") or [])
            configs = entry.get("configs")
            if fcn == "LaggedCorrelation" and configs is not None:
                configs = _expand_lagged_correlation_configs(configs)
            for params in configs if configs is not None else [{}]:
                params, config_labels = _split_config_params(params)
                config_labels = list(config_labels or [])
                written = family_labels
                if any(_is_module_label(label) for label in config_labels):
                    written = [l for l in written if not _is_module_label(l)]
                spi = getattr(module, fcn)(**params)
                merged = copy.copy(spi)
                _merge_spi_labels(merged, family_labels, config_labels)
                out.append((written + config_labels, spi, merged))
    return out


@pytest.fixture(scope="module")
def entries_by_config():
    return {name: _config_entries(name) for name in bundled_configs()}


def _class_trait(spi):
    """
    The directedness the SPI settles before any config label is merged.

    Parameters
    ----------
    spi : object
        An SPI as constructed.

    Returns
    -------
    str or None
        The trait among the SPI's own labels, else ``'undirected'`` for an
        :class:`~pyspi.base.Undirected` subclass, else None (the config
        decides).
    """
    own = [t for t in DIRECTEDNESS_TRAITS if t in spi.labels]
    if own:
        return own[0]
    return "undirected" if isinstance(spi, Undirected) else None


def test_config_sign_label_agrees_with_issigned(entries_by_config):
    bad = []
    for config, entries in entries_by_config.items():
        for written, spi, merged in entries:
            expected = ["signed"] if spi.issigned() else []
            declared = [l for l in written if l in ("signed", "unsigned")]
            own = [l for l in spi.labels if l in ("signed", "unsigned")]
            if declared != expected or own != expected \
                    or [l for l in merged.labels if l == "signed"] != expected:
                bad.append(f"{config}/{merged.identifier}: config {declared}, "
                           f"class labels {own}, issigned() -> {spi.issigned()}")
    assert not bad, "\n".join(bad)


def test_config_directedness_agrees_with_the_class(entries_by_config):
    bad = []
    for config, entries in entries_by_config.items():
        for written, spi, merged in entries:
            declared = [l for l in written if l in DIRECTEDNESS_TRAITS]
            trait = _class_trait(spi)
            if len(declared) != 1 or trait not in (None, declared[0]) \
                    or merged.directedness() != declared[0]:
                bad.append(f"{config}/{merged.identifier}: config {declared}, "
                           f"class {trait}, directedness() -> "
                           f"{merged.directedness()}")
    assert not bad, "\n".join(bad)


def test_config_linearity_agrees_with_the_class(entries_by_config):
    bad = []
    for config, entries in entries_by_config.items():
        for written, spi, merged in entries:
            own = [l for l in spi.labels if l in ("linear", "nonlinear")]
            declared = [l for l in written if l in ("linear", "nonlinear")]
            if len(declared) > 1 or (own and declared and declared != own):
                bad.append(f"{config}/{merged.identifier}: config {declared}, "
                           f"class {own}")
    assert not bad, "\n".join(bad)


def test_every_spi_has_one_module_label(entries_by_config):
    bad = [
        f"{config}/{merged.identifier}: "
        f"{[l for l in merged.labels if _is_module_label(l)]}"
        for config, entries in entries_by_config.items()
        for _, _, merged in entries
        if sum(_is_module_label(l) for l in merged.labels) != 1
    ]
    assert not bad, "\n".join(bad)


def test_unsigned_is_not_a_label(labels_by_config):
    bad = [f"{config}/{ident}" for config, spis in labels_by_config.items()
           for ident, labels in spis.items() if "unsigned" in labels]
    assert not bad, "\n".join(bad)


def test_an_unsigned_label_in_a_config_is_dropped(tmp_path):
    """Configs written for pyspi 2.x label SPIs ``unsigned``; they still load."""
    config = tmp_path / "old_labels.yaml"
    config.write_text(
        ".statistics.basic:\n"
        "  SpearmanR:\n"
        "    labels: [undirected, nonlinear, unsigned, bivariate, M01]\n"
        "    configs:\n"
        "      - squared: True\n"
        "      - squared: False\n"
        "        labels: [unsigned]\n"
    )
    spis = load_spis_from_yaml(str(config), quiet=True)
    assert set(spis) == {"spearmanr-sq", "spearmanr"}
    assert "unsigned" not in spis["spearmanr-sq"].labels
    assert "signed" not in spis["spearmanr-sq"].labels
    # issigned() decides: Spearman's rho is signed whatever the config says.
    assert "unsigned" not in spis["spearmanr"].labels
    assert "signed" in spis["spearmanr"].labels


def test_filter_spis_explains_that_unsigned_is_not_a_label(tmp_path, monkeypatch):
    from pyspi.utils import filter_spis

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="'unsigned' is not a label.*filter by 'strength'"):
        filter_spis(["unsigned"])
    with pytest.raises(ValueError, match="'unsigned' is not a label"):
        filter_spis(["linear", "unsigned"])


@pytest.mark.parametrize("flag,expected", [("timing", TIMING_IN_FULL),
                                           ("profile", PROFILE_IN_FULL),
                                           ("inverted", INVERTED_IN_FULL)])
def test_timing_profile_and_inverted_sets_of_full(labels_by_config, flag, expected):
    full = labels_by_config["full"]
    assert {ident for ident, labels in full.items() if flag in labels} == expected
    # The same SPIs carry the flag in every config that includes them.
    for config, spis in labels_by_config.items():
        assert {i for i, labels in spis.items() if flag in labels} \
            == expected & set(spis), config


def test_every_other_spi_of_full_is_a_strength(labels_by_config):
    full = labels_by_config["full"]
    strength = {ident for ident, labels in full.items() if "strength" in labels}
    assert strength == set(full) - TIMING_IN_FULL - PROFILE_IN_FULL
    signed = {ident for ident in strength if "signed" in full[ident]}
    assert signed and not signed & INVERTED_IN_FULL
    assert INVERTED_IN_FULL <= strength


def _merged(own, config_labels=None):
    class Stub:
        labels = list(own)

    spi = Stub()
    _merge_spi_labels(spi, config_labels=config_labels)
    return spi.labels


def test_an_spi_that_declares_no_kind_is_a_strength():
    assert _merged(["linear"]) == ["linear", "strength"]
    assert _merged(["linear"], ["strength"]) == ["linear", "strength"]


@pytest.mark.parametrize("kind", ["timing", "profile"])
def test_a_declared_kind_replaces_strength(kind):
    assert set(_merged([kind])) & set(VALUE_KINDS) == {kind}
    # A config may give the kind of a class that declares none.
    assert set(_merged(["linear"], [kind])) & set(VALUE_KINDS) == {kind}
    # The class's own kind overrides the config's, as its trait does.
    assert set(_merged([kind], ["strength"])) & set(VALUE_KINDS) == {kind}


def test_two_kinds_are_refused():
    with pytest.raises(ValueError, match="by its class with more than one of "
                                         "'strength', 'timing', 'profile'"):
        _merged(["timing", "profile"])
    with pytest.raises(ValueError, match="by its config with more than one of"):
        _merged(["linear"], ["timing", "profile"])
    with pytest.raises(ValueError, match="by its config with more than one of"):
        _merged(["linear"], ["strength", "timing"])


def test_merging_twice_gives_the_same_labels():
    class Stub:
        labels = ["linear"]

    spi = Stub()
    _merge_spi_labels(spi, config_labels=["timing"])
    once = list(spi.labels)
    _merge_spi_labels(spi, config_labels=["timing"])
    assert spi.labels == once and "timing" in once and "strength" not in once


def test_unbundled_variants_get_their_kind_when_merged():
    from pyspi.statistics.distance import Barycenter
    from pyspi.statistics.spectral import GroupDelay

    def kind(spi):
        _merge_spi_labels(spi)
        (found,) = set(spi.labels) & set(VALUE_KINDS)
        return found

    assert kind(GroupDelay(statistic="delay")) == "timing"
    assert kind(GroupDelay(statistic="rvalue")) == "strength"
    assert kind(Barycenter(statistic="max_time")) == "timing"
    assert kind(Barycenter(mode="dtw", statistic="mean")) == "strength"


def test_inverted_goes_only_with_an_unsigned_strength():
    from pyspi.base import Signed

    assert _merged(["inverted"]) == ["inverted", "strength"]
    with pytest.raises(ValueError, match="'inverted' and 'timing'"):
        _merged(["inverted", "timing"])
    with pytest.raises(ValueError, match="'inverted' and 'signed' and 'timing'"):
        _merge_spi_labels(type("SignedTiming", (Signed,),
                               {"labels": ["inverted", "timing"]})())
    with pytest.raises(ValueError, match="'inverted' and 'profile'"):
        _merged(["inverted"], ["profile"])

    class SignedStub(Signed):
        labels = ["inverted"]

    with pytest.raises(ValueError, match="'inverted' and 'signed'"):
        _merge_spi_labels(SignedStub())


def test_filter_spis_selects_by_kind(tmp_path, monkeypatch):
    from pyspi.utils import filter_spis

    monkeypatch.chdir(tmp_path)
    filter_spis(["timing"], output_name="timing")
    kept = load_spis_from_yaml(str(tmp_path / "timing.yaml"), quiet=True)
    assert set(kept) == TIMING_IN_FULL
    filter_spis(["strength", "signed"], output_name="signed_strength")
    kept = load_spis_from_yaml(str(tmp_path / "signed_strength.yaml"), quiet=True)
    full = load_spis_from_yaml(resolve_config("full"), quiet=True)
    assert set(kept) == {ident for ident, spi in full.items()
                         if spi.issigned() and "strength" in spi.labels}
    assert kept and not set(kept) & (TIMING_IN_FULL | PROFILE_IN_FULL)


def test_config_value_flags_agree_with_the_class(entries_by_config):
    bad = []
    for config, entries in entries_by_config.items():
        for written, spi, merged in entries:
            for flag in ("timing", "profile", "inverted"):
                if (flag in written) != (flag in spi.labels):
                    bad.append(f"{config}/{merged.identifier}: {flag}")
    assert not bad, "\n".join(bad)


def test_unbundled_timing_statistics():
    from pyspi.statistics.distance import Barycenter
    from pyspi.statistics.events import TailDependenceAsymmetry
    from pyspi.statistics.spectral import GroupDelay

    def flags(spi):
        return set(spi.labels) & {"timing", "profile"}

    assert flags(GroupDelay(statistic="delay")) == {"timing"}
    assert flags(GroupDelay(statistic="slope")) == {"timing"}
    assert flags(GroupDelay(statistic="rvalue")) == set()
    assert flags(Barycenter(statistic="max_time")) == {"timing"}
    for statistic in ("mean", "max", "corr"):
        assert flags(Barycenter(mode="dtw", statistic=statistic)) == set()
    assert flags(TailDependenceAsymmetry(statistic="upper")) == set()
    assert flags(TailDependenceAsymmetry(statistic="lower")) == set()


def test_inverted_is_only_on_one_ended_spis(labels_by_config):
    bad = [f"{config}/{ident}" for config, spis in labels_by_config.items()
           for ident, labels in spis.items()
           if "inverted" in labels and {"signed", "timing", "profile"} & set(labels)]
    assert not bad, bad


def test_unbundled_inverted_statistics():
    from pyspi.statistics.distance import GromovWassersteinTau, LongestCommonSubsequence
    from pyspi.statistics.misc import Cointegration, LinearModel

    assert "inverted" in GromovWassersteinTau().labels
    assert "inverted" in LinearModel("Ridge").labels
    assert "inverted" not in LongestCommonSubsequence().labels
    for statistic in ("trace_stat", "max_eig_stat"):
        assert "inverted" not in Cointegration(method="johansen",
                                               statistic=statistic).labels


def test_no_label_is_listed_twice(entries_by_config):
    """In a config's own label lists, and after merging with the class."""
    bad = []
    for name in bundled_configs():
        for module_spis in load_config_yaml(resolve_config(name)).values():
            for fcn, entry in module_spis.items():
                lists = [list(entry.get("labels") or [])] + [
                    list((params or {}).get("labels") or [])
                    for params in entry.get("configs") or []]
                bad += [f"{name}/{fcn}: {l}" for l in lists if len(l) != len(set(l))]
    for config, entries in entries_by_config.items():
        bad += [f"{config}/{merged.identifier}: {merged.labels}"
                for _, _, merged in entries
                if len(merged.labels) != len(set(merged.labels))]
    assert not bad, "\n".join(bad)
