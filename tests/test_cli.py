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
