"""A library import / construction must not alter the host program's I/O or logging."""
import logging
import subprocess
import sys

import numpy as np
import pytest

from pyspi import Calculator, _parallel


def test_import_leaves_std_streams_and_env_alone():
    code = (
        "import sys, os\n"
        "out, err = sys.stdout, sys.stderr\n"
        "omp = os.environ.get('OMP_NUM_THREADS')\n"
        "import pyspi\n"
        "assert sys.stdout is out and sys.stderr is err, 'std stream replaced'\n"
        "assert os.environ.get('OMP_NUM_THREADS') == omp, 'environment mutated'\n"
    )
    env = {k: v for k, v in __import__("os").environ.items() if k != "OMP_NUM_THREADS"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_verbose_false_adds_no_handlers_and_keeps_level():
    root, pyspi_logger = logging.getLogger(), logging.getLogger("pyspi")
    before = (list(root.handlers), list(pyspi_logger.handlers),
              pyspi_logger.level, pyspi_logger.propagate, root.level)
    Calculator(config="fabfour", verbose=False)
    after = (list(root.handlers), list(pyspi_logger.handlers),
             pyspi_logger.level, pyspi_logger.propagate, root.level)
    assert before == after


def test_verbose_true_attaches_one_handler_once_and_propagates():
    pyspi_logger = logging.getLogger("pyspi")
    saved = (list(pyspi_logger.handlers), pyspi_logger.level)
    try:
        for h in list(pyspi_logger.handlers):
            if not isinstance(h, logging.NullHandler):
                pyspi_logger.removeHandler(h)
        for _ in range(3):
            Calculator(config="fabfour", verbose=True)
        streams = [h for h in pyspi_logger.handlers
                   if not isinstance(h, logging.NullHandler)]
        assert len(streams) == 1
        assert pyspi_logger.propagate is True
    finally:
        pyspi_logger.handlers[:] = saved[0]
        pyspi_logger.setLevel(saved[1])


def test_records_reach_caplog_with_verbose_false(caplog):
    with caplog.at_level(logging.INFO, logger="pyspi"):
        Calculator(config="fabfour", verbose=False)
    assert any("initialised" in r.getMessage() for r in caplog.records)


def test_resolve_n_jobs(monkeypatch):
    monkeypatch.setattr(_parallel, "available_cores", lambda: 8)
    assert _parallel.resolve_n_jobs(3) == 3
    assert _parallel.resolve_n_jobs(-1) == 8
    assert _parallel.resolve_n_jobs(-2) == 7
    assert _parallel.resolve_n_jobs(-100) == 1
    with pytest.raises(ValueError, match="must not be 0"):
        _parallel.resolve_n_jobs(0)
    with pytest.raises(TypeError):
        _parallel.resolve_n_jobs(1.5)
    with pytest.raises(TypeError):
        _parallel.resolve_n_jobs(True)


def test_resolve_n_jobs_env(monkeypatch):
    monkeypatch.setenv("PYSPI_N_JOBS", "4")
    assert _parallel.resolve_n_jobs(None) == 4
    monkeypatch.setenv("PYSPI_N_JOBS", "four")
    with pytest.raises(ValueError, match="PYSPI_N_JOBS"):
        _parallel.resolve_n_jobs(None)
    monkeypatch.delenv("PYSPI_N_JOBS")
    assert _parallel.resolve_n_jobs(None) == 1


def test_compute_rejects_zero_n_jobs():
    calc = Calculator(np.random.default_rng(0).standard_normal((3, 50)),
                      config="fabfour", verbose=False)
    with pytest.raises(ValueError, match="n_jobs"):
        calc.compute(n_jobs=0)


def test_compute_n_jobs_minus_one_uses_available_cores(monkeypatch):
    seen = {}
    monkeypatch.setattr(_parallel, "available_cores", lambda: 1)
    calc = Calculator(np.random.default_rng(0).standard_normal((3, 50)),
                      config="fabfour", verbose=False)
    monkeypatch.setattr(_parallel, "guard_oversubscription",
                        lambda n: seen.setdefault("n", n))
    calc.compute(n_jobs=-1, progress=False)
    assert seen["n"] == 1 and not calc.errors
