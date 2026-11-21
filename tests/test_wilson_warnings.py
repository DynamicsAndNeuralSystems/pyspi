"""Every SPI read off a Wilson factorisation warns when the factorisation did not converge.

The directed spectral SPIs (``dcoh``, ``pdcoh``, ...) share one factorisation
per ordered pair, and the nonparametric spectral GC SPIs one per dataset, all
cached on the ``Data``. Each SPI that reads a factorisation which reached
Wilson's iteration cap must warn for itself, whichever SPI computed it, and
the Calculator attributes the warning to that SPI however it is run.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

import pyspi
from pyspi.calculator import Calculator
from pyspi.data import Data
from pyspi.statistics import spectral

FIXTURES = Path(__file__).parent / "data" / "fixtures"
CAP = "Wilson's spectral factorisation did not converge"


def _unconverged_data():
    """Seven Kuramoto oscillators: four ordered pairs reach Wilson's iteration cap."""
    return Data(data=str(FIXTURES / "kuramoto_M7_T100.npy"), dim_order="sp")


def _converged_data():
    return Data(np.random.default_rng(0).standard_normal((4, 200)), dim_order="ps")


def _wilson_config(tmp_path):
    """The ``DirectedCoherence`` and nonparametric ``SpectralGrangerCausality`` entries of ``full``."""
    full = yaml.safe_load(
        (Path(pyspi.__file__).parent / "configs" / "full.yaml").read_text())
    classes = full[".statistics.spectral"]
    sgc = dict(classes["SpectralGrangerCausality"])
    sgc["configs"] = [c for c in sgc["configs"] if c.get("method") == "nonparametric"]
    config = tmp_path / "wilson.yaml"
    config.write_text(yaml.safe_dump({".statistics.spectral": {
        "DirectedCoherence": classes["DirectedCoherence"],
        "SpectralGrangerCausality": sgc}}))
    return config


def _relayed(caught):
    """``{identifier: [message, ...]}`` for the non-convergence warnings a Calculator relayed."""
    found = {}
    for w in caught:
        message = str(w.message)
        if CAP in message:
            spi = message.split('SPI "')[1].split('"')[0]
            found.setdefault(spi, []).append(message)
    return found


def _direct(caught):
    """The non-convergence warnings caught from calling an SPI directly."""
    return [str(w.message) for w in caught if CAP in str(w.message)]


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_every_wilson_spi_of_full_warns_when_the_factorisation_does_not_converge(
        tmp_path, n_jobs):
    calc = Calculator(dataset=_unconverged_data(), config=str(_wilson_config(tmp_path)))
    expected = set(calc.spis)
    assert len(expected) == 12
    assert {s.split("_")[0] for s in expected} == {"dcoh", "sgc"}
    assert all("parametric_" not in s or "nonparametric_" in s for s in expected)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calc.compute(n_jobs=n_jobs, progress=False,
                     mp_context="spawn" if n_jobs > 1 else None)

    found = _relayed(caught)
    assert set(found) == expected
    # One warning per SPI, which the Calculator relays under its identifier.
    for spi, messages in found.items():
        assert len(messages) == 1, messages
        assert messages[0].startswith(
            f'SPI "{spi}": RuntimeWarning: Wilson\'s spectral factorisation')
    # The values are kept: non-convergence is reported, not an error.
    assert not calc.errors


def test_the_warning_names_the_ordered_pairs_that_did_not_converge():
    data = _unconverged_data()
    spis = [spectral.DirectedCoherence(), spectral.DirectedCoherence(statistic="max"),
            spectral.PartialDirectedCoherence(), spectral.DirectedTransferFunction()]
    for spi in spis:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spi.multivariate(data)
        found = _direct(caught)
        assert len(found) == 1, (spi.identifier, found)
        assert found[0].startswith(
            "Wilson's spectral factorisation did not converge for 4 of 42 ordered "
            "pairs: (0, 6), (4, 6), (6, 0), (6, 4).")


def test_spectral_gc_counts_the_pairs_that_did_not_converge():
    data = _unconverged_data()
    for statistic in ("mean", "max"):
        spi = spectral.SpectralGrangerCausality(statistic=statistic)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            spi.multivariate(data)
        found = _direct(caught)
        assert len(found) == 1, (spi.identifier, found)
        assert "did not converge for 2 of 21 pairs." in found[0]


def test_a_single_pair_warns_from_bivariate():
    data = _unconverged_data()
    spi = spectral.DirectedCoherence()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spi.bivariate(data, i=6, j=0)
        spi.bivariate(data, i=1, j=0)
    found = _direct(caught)
    assert len(found) == 1
    assert "for 1 of 1 ordered pairs: (6, 0)." in found[0]


def test_a_converged_factorisation_gives_no_warning(tmp_path):
    calc = Calculator(dataset=_converged_data(), config=str(_wilson_config(tmp_path)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calc.compute(progress=False)
    assert not _relayed(caught)
    assert not [w for w in caught if "spectral_connectivity" in str(w.message)]
