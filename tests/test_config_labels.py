"""Label consistency across the bundled configs.

The same SPI must carry the same labels whichever config it is loaded from,
no SPI may carry a contradictory pair, and every label must belong to the
vocabulary used by ``full.yaml`` and the SPI classes (which catches typos
such as ``unisgned`` or a stray ``directed,``).
"""
import collections
import re

import pytest

from pyspi.calculator import bundled_configs, load_spis_from_yaml, resolve_config

CONTRADICTORY = (
    ("linear", "nonlinear"),
    ("directed", "undirected"),
    ("signed", "unsigned"),
)
MODULE_LABEL = re.compile(r"^M(\d\d|XX)$")


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


def test_labels_come_from_the_known_vocabulary(labels_by_config):
    vocabulary = {
        label for labels in labels_by_config["full"].values() for label in labels
    }
    unknown = sorted(
        {
            f"{config}/{ident}: {label!r}"
            for config, spis in labels_by_config.items()
            for ident, labels in spis.items()
            for label in labels
            if label not in vocabulary and not MODULE_LABEL.match(label)
        }
    )
    assert not unknown, "\n".join(unknown)
    assert all(label == label.strip() and "," not in label for label in vocabulary)


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
