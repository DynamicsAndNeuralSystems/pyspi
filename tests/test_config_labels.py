"""Label consistency across the bundled configs.

The same SPI must carry the same labels whichever config it is loaded from,
no SPI may carry a contradictory pair, every label must belong to the fixed
vocabulary of CONTRIBUTING.md (which catches typos such as ``sigend``, a
stray ``directed,`` or a compound such as ``linear/nonlinear``), and every
SPI carries exactly one label from each axis of that vocabulary.
``signed`` is a flag, carried exactly when ``issigned()`` is true; ``unsigned``
is not a label, and one in a config is dropped. ``timing`` and ``profile`` are
flags set by the class, on a fixed set of bundled SPIs.
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

from pyspi.base import DIRECTEDNESS_TRAITS, Undirected
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
# label from each axis, and the flags ``signed``, ``timing`` and ``profile``
# where they apply.
LABEL_AXES = {
    "directedness": {"directed", "undirected", "antisymmetric"},
    "linearity": {"linear", "nonlinear"},
    "arity": {"bivariate", "multivariate"},
    "time": {"contemporaneous", "time-dependent", "frequency-dependent"},
    "module": {f"M{n:02d}" for n in range(1, 15)} | {"MXX"},
}
# Descriptive tags a class may carry in addition, none required.
DESCRIPTIVE_TAGS = {
    "basic", "causal", "distance", "embedding", "infotheory", "lagged", "misc",
    "normal", "rank", "spectral", "symbolic", "temporal", "unordered",
    "wavelet",
}
FLAGS = {"signed", "timing", "profile"}
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
    assert spi.labels == ["nonlinear"]


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
    with pytest.raises(ValueError, match="'unsigned' is not a label.*without 'signed'"):
        filter_spis(["unsigned"])
    with pytest.raises(ValueError, match="'unsigned' is not a label"):
        filter_spis(["linear", "unsigned"])


@pytest.mark.parametrize("flag,expected", [("timing", TIMING_IN_FULL),
                                           ("profile", PROFILE_IN_FULL)])
def test_value_flags_in_full(labels_by_config, flag, expected):
    full = labels_by_config["full"]
    assert {ident for ident, labels in full.items() if flag in labels} == expected
    # The same SPIs carry the flag in every config that includes them.
    for config, spis in labels_by_config.items():
        assert {i for i, labels in spis.items() if flag in labels} \
            == expected & set(spis), config


def test_no_spi_is_both_timing_and_profile(labels_by_config):
    bad = [f"{config}/{ident}" for config, spis in labels_by_config.items()
           for ident, labels in spis.items() if {"timing", "profile"} <= set(labels)]
    assert not bad, bad


def test_config_value_flags_agree_with_the_class(entries_by_config):
    bad = []
    for config, entries in entries_by_config.items():
        for written, spi, merged in entries:
            for flag in ("timing", "profile"):
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
    assert flags(GroupDelay(statistic="slope")) == set()
    assert flags(GroupDelay(statistic="rvalue")) == set()
    assert flags(Barycenter(statistic="max_time")) == {"timing"}
    for statistic in ("mean", "max", "corr"):
        assert flags(Barycenter(mode="dtw", statistic=statistic)) == set()
    assert flags(TailDependenceAsymmetry(statistic="upper")) == set()
    assert flags(TailDependenceAsymmetry(statistic="lower")) == set()
