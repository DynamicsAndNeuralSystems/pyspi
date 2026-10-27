"""A checkpoint must identify the run that produced it.

These began as red tests and are now green. Resume used to validate only the
SPI identifier and an ``(M, M)`` shape, so any other run of the same width
silently inherited the previous run's numbers -- valid-looking results with no
warning and no trace. Checkpoints are now bound to ``Calculator.run_digest``
via a ``run.json`` manifest.

Also covered: identifier collisions. Identifiers were formatted with ``.3g``/
``.4g``, so distinct parameterisations could render to the same string, and
dictionary insertion overwrote the loser before any duplicate check ran.
"""
import numpy as np
import pytest

from pyspi.calculator import Calculator
from pyspi.data import Data

CONFIG = "fabfour"


def _data(seed, m=3, t=80, name=None):
    rng = np.random.default_rng(seed)
    return Data(data=rng.standard_normal((m, t)), dim_order="ps",
                zscore=False, name=name)


def _run(dataset, cp_dir, config=CONFIG, **kw):
    calc = Calculator(dataset=dataset, config=config, verbose=False)
    calc.compute(checkpoint_dir=cp_dir, progress=False, **kw)
    return calc


def _assert_foreign_directory_refused(dataset, cp_dir, config=CONFIG):
    """A checkpoint directory owned by another run must be refused, not reused.

    Deleting its contents to make room would be destructive, and if interrupted
    partway would relabel whatever survived as the new run.
    """
    with pytest.raises(ValueError, match="different run"):
        _run(dataset, cp_dir, config=config)


def _first_spi_values(calc):
    key = sorted(calc.spis)[0]
    return key, np.asarray(calc.table[key].to_numpy(dtype=float))


# --------------------------------------------------------------------------
# Checkpoint identity
# --------------------------------------------------------------------------

def test_checkpoint_rejects_a_different_dataset(tmp_path):
    """A different dataset of the same width must not reuse checkpoints."""
    _run(_data(seed=1), tmp_path)
    _assert_foreign_directory_refused(_data(seed=2), tmp_path)


def test_checkpoint_rejects_a_different_config(tmp_path):
    dataset = _data(seed=3)
    _run(dataset, tmp_path, config="fabfour")
    _assert_foreign_directory_refused(dataset, tmp_path, config="fast")


def test_checkpoint_rejects_permuted_processes(tmp_path):
    rng = np.random.default_rng(7)
    arr = rng.standard_normal((3, 80))

    _run(Data(data=arr, dim_order="ps", zscore=False, procnames=["a", "b", "c"]),
         tmp_path)

    perm = [2, 0, 1]
    _assert_foreign_directory_refused(
        Data(data=arr[perm], dim_order="ps", zscore=False,
             procnames=["c", "a", "b"]),
        tmp_path,
    )


def test_failed_checkpoints_are_retried_by_default(tmp_path):
    """An SPI that failed previously must be recomputed, not resumed as NaN."""
    dataset = _data(seed=4)
    calc = Calculator(dataset=dataset, config=CONFIG, verbose=False)
    key = sorted(calc.spis)[0]

    # Simulate a prior run of *this* calculator in which `key` failed, manifest
    # included -- otherwise the directory reads as foreign and is refused.
    tmp_path.mkdir(parents=True, exist_ok=True)
    from pyspi import _parallel
    _parallel.write_manifest(tmp_path, calc.run_digest, calc.run_spec)
    np.save(tmp_path / f"{key}.npy", np.full((3, 3), np.nan))
    (tmp_path / f"{key}.error").write_text("RuntimeError: simulated prior failure")

    calc.compute(checkpoint_dir=tmp_path, progress=False)
    got = np.asarray(calc.table[key].to_numpy(dtype=float))

    assert np.isfinite(got[~np.eye(3, dtype=bool)]).any(), (
        f"'{key}' was resumed from a failed checkpoint instead of being retried."
    )


# --------------------------------------------------------------------------
# Identifier collisions
# --------------------------------------------------------------------------

def test_duplicate_identifiers_are_rejected_at_insertion():
    """Two SPIs with the same identifier must raise, not silently overwrite."""
    from pyspi.calculator import load_spis_from_yaml
    import tempfile, textwrap, os

    yaml_text = textwrap.dedent("""
        .statistics.basic:
          Covariance:
            labels: [undirected]
            configs:
              - estimator: EmpiricalCovariance
              - estimator: EmpiricalCovariance
        """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(yaml_text)
        path = fh.name
    try:
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            load_spis_from_yaml(path)
    finally:
        os.unlink(path)


def test_identifier_does_not_collide_under_float_rounding():
    """Parameterisations that differ numerically must differ in identifier."""
    from pyspi.statistics.spectral import CoherenceMagnitude

    a = CoherenceMagnitude(fmin=0.123456, fmax=0.5)
    b = CoherenceMagnitude(fmin=0.123499, fmax=0.5)

    assert a.identifier != b.identifier, (
        f"Distinct fmin values collide after .3g rounding: {a.identifier!r}."
    )


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def test_npz_round_trips_without_pickle(tmp_path):
    """Saved tables must load with allow_pickle=False and match exactly."""
    from pyspi.calculator import load_table

    calc = Calculator(dataset=_data(seed=11), config=CONFIG, verbose=False)
    calc.compute(progress=False)
    out = calc.save(tmp_path / "t.npz")

    # The load path must not need pickle; assert it directly as well as via
    # load_table, so a future regression to dtype=object is caught here.
    with np.load(out, allow_pickle=False) as f:
        assert set(f.files) >= {"values", "spis", "processes", "schema"}
        assert f["spis"].dtype.kind == "U", "SPI names stored as pickled objects"
        assert f["processes"].dtype.kind == "U"

    reloaded = load_table(out)
    for key in calc.spis:
        np.testing.assert_allclose(
            reloaded[key].to_numpy(dtype=float),
            calc.table[key].to_numpy(dtype=float),
            equal_nan=True,
        )


def test_npz_records_its_provenance(tmp_path):
    from pyspi.calculator import load_table  # noqa: F401  (import parity)

    calc = Calculator(dataset=_data(seed=12), config=CONFIG, verbose=False)
    calc.compute(progress=False)
    out = calc.save(tmp_path / "t.npz")

    import json
    with np.load(out, allow_pickle=False) as f:
        spec = json.loads(str(f["run_spec"]))
        assert spec["config"] == CONFIG
        assert spec["n_processes"] == 3
        assert str(f["run_digest"]) == calc.run_digest


def test_malformed_npz_is_rejected(tmp_path):
    from pyspi.calculator import load_table

    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, values=np.zeros((2, 3, 3)),
                        spis=np.array(["a"], dtype="U"),
                        processes=np.array(["p0", "p1", "p2"], dtype="U"))
    with pytest.raises(ValueError, match="malformed"):
        load_table(bad)


def test_non_pyspi_npz_is_rejected(tmp_path):
    from pyspi.calculator import load_table

    bad = tmp_path / "other.npz"
    np.savez_compressed(bad, something_else=np.zeros(3))
    with pytest.raises(ValueError, match="not a pyspi results table"):
        load_table(bad)


def test_invalid_checkpoint_is_retried_not_kept(tmp_path):
    """A non-finite checkpoint must be recomputed, not resumed.

    The validation used to run *after* the retry decision, so an inf-filled
    matrix was marked failed and then kept.
    """
    from pyspi import _parallel

    dataset = _data(seed=21)
    calc = Calculator(dataset=dataset, config=CONFIG, verbose=False)
    key = sorted(calc.spis)[0]

    tmp_path.mkdir(parents=True, exist_ok=True)
    _parallel.write_manifest(tmp_path, calc.run_digest, calc.run_spec)
    np.save(tmp_path / f"{key}.npy", np.full((3, 3), np.inf))

    calc.compute(checkpoint_dir=tmp_path, progress=False)
    got = np.asarray(calc.table[key].to_numpy(dtype=float))
    assert np.isfinite(got[~np.eye(3, dtype=bool)]).any(), (
        f"'{key}' kept an infinite checkpoint instead of recomputing it."
    )


def test_checkpoint_rejects_a_different_computation_version(tmp_path):
    """Checkpoints must not outlive the algorithm that produced them."""
    from pyspi import _parallel

    dataset = _data(seed=22)
    calc = Calculator(dataset=dataset, config=CONFIG, verbose=False)
    tmp_path.mkdir(parents=True, exist_ok=True)
    _parallel.write_manifest(tmp_path, calc.run_digest, calc.run_spec)

    manifest = tmp_path / _parallel.MANIFEST_NAME
    import json
    payload = json.loads(manifest.read_text())
    payload["computation"] = "0.0.0-ancient"
    manifest.write_text(json.dumps(payload))
    np.save(tmp_path / f"{sorted(calc.spis)[0]}.npy", np.zeros((3, 3)))

    with pytest.raises(ValueError, match="different run"):
        calc.compute(checkpoint_dir=tmp_path, progress=False)
