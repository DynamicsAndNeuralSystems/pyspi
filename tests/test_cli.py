import numpy as np
import pytest

from pyspi import __main__ as cli


@pytest.fixture
def data_file(tmp_path):
    path = tmp_path / "ts.npy"
    np.save(path, np.random.default_rng(0).standard_normal((3, 60)))
    return path


def _boom(*a, **k):
    raise AssertionError("computation started before arguments were validated")


def test_bad_output_suffix_fails_before_computing(monkeypatch, data_file, capsys):
    monkeypatch.setattr(cli, "Calculator", _boom)
    with pytest.raises(SystemExit) as exc:
        cli.main(["compute", "--data", str(data_file), "--output", "out.parquet"])
    assert exc.value.code == 2
    assert "unsupported output suffix" in capsys.readouterr().err


def test_missing_output_directory_fails_before_computing(monkeypatch, data_file, tmp_path, capsys):
    monkeypatch.setattr(cli, "Calculator", _boom)
    with pytest.raises(SystemExit) as exc:
        cli.main(["compute", "--data", str(data_file),
                  "--output", str(tmp_path / "missing" / "out.npz")])
    assert exc.value.code == 2
    assert "does not exist" in capsys.readouterr().err


def test_zero_n_jobs_fails_before_computing(monkeypatch, data_file, capsys):
    monkeypatch.setattr(cli, "Calculator", _boom)
    with pytest.raises(SystemExit) as exc:
        cli.main(["compute", "--data", str(data_file), "--n-jobs", "0"])
    assert exc.value.code == 2
    assert "n_jobs" in capsys.readouterr().err


def test_cli_honours_pyspi_n_jobs(monkeypatch, data_file):
    seen = {}
    real = cli.Calculator.compute

    def spy(self, n_jobs=None, **kw):
        seen["n_jobs"] = n_jobs
        return real(self, n_jobs=1, **kw)

    monkeypatch.setattr(cli.Calculator, "compute", spy)
    monkeypatch.setenv("PYSPI_N_JOBS", "3")
    assert cli.main(["compute", "--data", str(data_file), "--config", "fabfour",
                     "--quiet"]) == 0
    assert seen["n_jobs"] is None  # left to the library, which reads PYSPI_N_JOBS


def test_dim_order_sp_reads_one_column_per_process(tmp_path):
    import pyspi

    X = np.random.default_rng(0).standard_normal((3, 60))
    np.save(tmp_path / "ps.npy", X)
    np.save(tmp_path / "sp.npy", X.T)
    cli.main(["compute", "--data", str(tmp_path / "ps.npy"), "--config", "fabfour",
              "--output", str(tmp_path / "ps.npz"), "--quiet"])
    cli.main(["compute", "--data", str(tmp_path / "sp.npy"), "--config", "fabfour",
              "--dim-order", "sp", "--output", str(tmp_path / "sp.npz"), "--quiet"])
    assert pyspi.load_table(tmp_path / "sp.npz").equals(
        pyspi.load_table(tmp_path / "ps.npz"))


def _orientation_warnings(record):
    return [w for w in record if issubclass(w.category, UserWarning)
            and "processes of" in str(w.message)]


@pytest.mark.parametrize("flag, warns", [([], True), (["--dim-order", "ps"], False),
                                         (["--dim-order", "sp"], False)])
def test_dim_order_flag_states_the_layout(tmp_path, flag, warns):
    import warnings

    np.save(tmp_path / "wide.npy", np.random.default_rng(0).standard_normal((30, 20)))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        cli.main(["compute", "--data", str(tmp_path / "wide.npy"), "--config", "fabfour",
                  "--output", str(tmp_path / "out.npz"), "--quiet", "--allow-partial", *flag])
    found = _orientation_warnings(record)
    assert bool(found) == warns
    if warns:
        assert "dim_order='sp'" in str(found[0].message)
        assert "dim_order='ps'" in str(found[0].message)
