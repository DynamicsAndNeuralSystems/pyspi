"""Parallel runs under spawn need a main module the workers can re-import.

Each program runs in its own interpreter, with the spawn start method forced
so the tests behave the same on Linux, where the default is fork.
"""
import multiprocessing as mp
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from pyspi.calculator import Calculator

MESSAGE = "pyspi could not start the worker processes"

BODY = """\
calc = Calculator(dataset=np.random.default_rng(0).standard_normal((3, 100)),
                  config="fabfour", verbose=False)
calc.compute(n_jobs=2, mp_context="spawn", progress=False)
print("errors:", len(calc.errors), "finite values:", int(np.isfinite(calc.table.to_numpy()).sum()))
"""

HEADER = "import numpy as np\nfrom pyspi import Calculator\n"

UNGUARDED = HEADER + BODY
GUARDED = HEADER + 'if __name__ == "__main__":\n' + textwrap.indent(BODY, "    ")


def _run(args, tmp_path, **kwargs):
    return subprocess.run(
        [sys.executable, *args], cwd=tmp_path, capture_output=True, text=True,
        timeout=180, **kwargs,
    )


def test_unguarded_script_fails_with_pyspis_message(tmp_path):
    script = tmp_path / "unguarded.py"
    script.write_text(UNGUARDED)
    out = _run([str(script)], tmp_path)
    assert out.returncode != 0
    assert MESSAGE in out.stderr
    assert '__name__ == "__main__"' in out.stderr and "n_jobs=1" in out.stderr


def test_script_from_standard_input_fails_with_pyspis_message(tmp_path):
    out = _run(["-"], tmp_path, input=GUARDED)
    assert out.returncode != 0
    assert MESSAGE in out.stderr
    assert "standard input" in out.stderr


def test_guarded_script_runs(tmp_path):
    script = tmp_path / "guarded.py"
    script.write_text(GUARDED)
    out = _run([str(script)], tmp_path)
    assert out.returncode == 0, out.stderr
    assert MESSAGE not in out.stderr
    # Four SPIs, each finite on the six ordered pairs of three processes.
    assert "errors: 0 finite values: 24" in out.stdout


def test_compute_refuses_to_start_workers_while_its_process_bootstraps(monkeypatch):
    calc = Calculator(dataset=np.random.default_rng(0).standard_normal((3, 50)),
                      config="fabfour", verbose=False)
    monkeypatch.setattr(mp.current_process(), "_inheriting", True, raising=False)
    with pytest.raises(RuntimeError, match=MESSAGE):
        calc.compute(n_jobs=2, mp_context="spawn", progress=False)
    # Serial runs start no process and are unaffected.
    calc.compute(n_jobs=1, progress=False)
    assert not calc.errors
