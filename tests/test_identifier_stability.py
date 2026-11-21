"""SPI identifiers: equal parameters give equal identifiers, and bundled ones are fixed.

An identifier names a table column, a checkpoint file and a drift baseline,
so a numpy scalar must name the same SPI as the equal Python number, an
integral float the same SPI as the integer, and no bundled identifier may
change. ``tests/data/bundled_identifiers.json`` lists every bundled
identifier, config by config, in config order.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from pyspi.calculator import bundled_configs, load_spis_from_yaml, resolve_config
from pyspi.statistics import (
    basic,
    causal,
    crossfreq,
    distance,
    events,
    infotheory,
    misc,
    spectral,
    var,
    wavelet,
)
from pyspi.utils import fmt_param

FROZEN = Path(__file__).parent / "data" / "bundled_identifiers.json"


@pytest.mark.parametrize("value, expected", [
    (1, "1"),
    (1.0, "1"),
    (np.int64(1), "1"),
    (np.float64(1.0), "1"),
    (np.float32(1.0), "1"),
    (0.25, "0.25"),
    (np.float64(0.25), "0.25"),
    (np.float32(0.25), "0.25"),
    (1e-5, "1e-05"),
    (np.float64(1e-5), "1e-05"),
    (-0.0, "0"),
    (True, "True"),
    (np.True_, "True"),
    (None, "None"),
    ("bic", "bic"),
    (float("nan"), "nan"),
    (float("inf"), "inf"),
])
def test_fmt_param_formats_equal_values_alike(value, expected):
    assert fmt_param(value) == expected


def test_fmt_param_keeps_unequal_values_apart():
    assert fmt_param(0.1) != fmt_param(np.float32(0.1))
    assert fmt_param(1.5) != fmt_param(15)


# (class, Python-number kwargs, equal kwargs spelled with numpy scalars or
# integral floats), one or more per statistics module.
EQUIVALENT_SPELLINGS = [
    (basic.ChatterjeeXi, dict(tau=1), dict(tau=np.int64(1))),
    (basic.LaggedCorrelation, dict(tau=2), dict(tau=np.int32(2))),
    (causal.NonlinearInterdependence, dict(m=3, k=4), dict(m=np.int64(3), k=np.int64(4))),
    (crossfreq.PhaseAmplitudeCoupling,
     dict(phase_band=(0.01, 0.05), amp_band=(0.15, 0.4)),
     dict(phase_band=(np.float64(0.01), np.float64(0.05)),
          amp_band=np.array([0.15, 0.4]))),
    (distance.CrossDistanceCorrelation, dict(max_lag=10), dict(max_lag=10.0)),
    (distance.CrossDistanceCorrelation, dict(max_lag=10), dict(max_lag=np.int64(10))),
    (distance.DynamicTimeWarping,
     dict(global_constraint="sakoe_chiba", sakoe_chiba_ratio=0.1),
     dict(global_constraint="sakoe_chiba", sakoe_chiba_ratio=np.float64(0.1))),
    (events.EventSynchronisation, dict(quantile=0.9), dict(quantile=np.float64(0.9))),
    (events.TailDependenceAsymmetry, dict(quantile=0.95), dict(quantile=np.float64(0.95))),
    (infotheory.MutualInfo, dict(estimator="kernel", kernel_width=0.25),
     dict(estimator="kernel", kernel_width=np.float64(0.25))),
    (infotheory.MutualInfo, dict(estimator="kernel", kernel_width=1),
     dict(estimator="kernel", kernel_width=1.0)),
    (infotheory.IntegratedInformation, dict(normalization=0), dict(normalization=0.0)),
    (misc.Cointegration, dict(method="aeg", maxlag=10), dict(method="aeg", maxlag=np.int64(10))),
    (misc.InterDependenceScore, dict(bandwidth=0.25), dict(bandwidth=np.float64(0.25))),
    (misc.EdgeTimeSeries, dict(statistic="topshare", frac=0.05),
     dict(statistic="topshare", frac=np.float64(0.05))),
    (spectral.CoherenceMagnitude, dict(fmax=0.25), dict(fmax=np.float64(0.25))),
    (spectral.CoherenceMagnitude, dict(fs=1, fmin=0, fmax=0.5),
     dict(fs=1.0, fmin=0.0, fmax=np.float64(0.5))),
    (spectral.CoherenceMagnitude, dict(fs=4, fmax=1), dict(fs=4.0, fmax=1.0)),
    (spectral.CoherenceMagnitude, dict(time_halfbandwidth_product=4),
     dict(time_halfbandwidth_product=4.0)),
    (spectral.SpectralGrangerCausality, dict(fs=1, fmax=0.25),
     dict(fs=np.float64(1.0), fmax=np.float64(0.25))),
    (spectral.SpectralGrangerCausality, dict(method="parametric", order=2),
     dict(method="parametric", order=np.int64(2))),
    (var.ConditionalPartialDirectedCoherence, dict(fmin=0, fmax=0.25),
     dict(fmin=np.int64(0), fmax=np.float64(0.25))),
    (var.ConditionalGrangerCausality, dict(order=2), dict(order=np.int64(2))),
    (wavelet.PhaseSlopeIndex, dict(fmax=0.25), dict(fmax=np.float64(0.25))),
    (wavelet.WaveletCoherence, dict(fs=2, fmax=1, n_cycles=8),
     dict(fs=2.0, fmax=np.float64(1.0), n_cycles=np.int64(8))),
]


@pytest.mark.parametrize(
    "cls, plain, spelled", EQUIVALENT_SPELLINGS,
    ids=[f"{c.__module__.rsplit('.', 1)[-1]}.{c.__name__}-{n}"
         for n, (c, _p, _s) in enumerate(EQUIVALENT_SPELLINGS)],
)
def test_equal_parameters_give_equal_identifiers(cls, plain, spelled):
    a = cls(**plain).identifier
    b = cls(**spelled).identifier
    assert a == b
    # Identifiers name files, so no numpy repr or parentheses may leak in.
    assert "np" not in b.split("_")[0] and "(" not in b and " " not in b


def test_integral_float_reads_as_the_integer():
    assert (spectral.CoherenceMagnitude(fs=1.0).identifier
            == "cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5")
    assert (spectral.CoherenceMagnitude(fmax=np.float64(0.25)).identifier
            == "cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25")


@pytest.fixture(scope="module")
def bundled_identifiers():
    return {
        name: list(load_spis_from_yaml(resolve_config(name), quiet=True))
        for name in bundled_configs()
    }


def test_bundled_identifiers_match_the_frozen_list(bundled_identifiers):
    frozen = json.loads(FROZEN.read_text())
    assert sorted(frozen) == sorted(bundled_identifiers)
    for name, identifiers in bundled_identifiers.items():
        added = sorted(set(identifiers) - set(frozen[name]))
        removed = sorted(set(frozen[name]) - set(identifiers))
        assert not added and not removed, (
            f"{name}: identifiers added {added}, removed {removed}. Drift "
            f"baselines are keyed by identifier; update the frozen list only "
            f"for a deliberate change to the bundled set.")
        assert identifiers == frozen[name], f"{name}: order changed"


def test_close_but_unequal_parameters_give_different_identifiers():
    a = misc.EdgeTimeSeries(statistic="topshare", frac=0.1234567).identifier
    b = misc.EdgeTimeSeries(statistic="topshare", frac=0.12345671).identifier
    assert a != b
