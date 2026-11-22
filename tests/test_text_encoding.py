"""Text files pyspi reads and writes are UTF-8, whatever the locale encoding.

Each test runs a short script in a subprocess whose locale encoding is ASCII
(``LC_ALL=C`` with UTF-8 mode and locale coercion turned off), so a file
opened in the default encoding fails to read or write a non-ASCII character.
The subprocess also runs with ``-X warn_default_encoding`` and turns an
``EncodingWarning`` raised from pyspi into an error, so a file opened without
an explicit encoding is caught even on a platform where ``LC_ALL=C`` still
gives UTF-8.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

TESTS_DIR = Path(__file__).parent
REPO_ROOT = TESTS_DIR.parent

# A non-ASCII label and comment: 'é' and 'Ý' (whose UTF-8 encoding contains
# byte 0x9d, which cp1252 does not define).
LABEL = "linéaire-Ý"

PRELUDE = """
import locale
import sys
import warnings

warnings.filterwarnings("error", category=EncodingWarning, module=r"pyspi(\\.|$)")
print("locale encoding:", locale.getencoding(), file=sys.stderr)
"""


def _run(script, cwd):
    """Run ``script`` under an ASCII locale encoding; fail with its output if it fails."""
    env = dict(os.environ)
    for name in ("LANG", "LANGUAGE", "LC_CTYPE", "PYTHONWARNINGS"):
        env.pop(name, None)
    existing = env.get("PYTHONPATH", "")
    env.update(
        LC_ALL="C",
        PYTHONUTF8="0",
        PYTHONCOERCECLOCALE="0",
        PYTHONIOENCODING="utf-8",
        PYTHONPATH=os.pathsep.join(
            p for p in (str(REPO_ROOT), str(TESTS_DIR), existing) if p
        ),
    )
    proc = subprocess.run(
        [sys.executable, "-X", "warn_default_encoding", "-c",
         textwrap.dedent(PRELUDE) + textwrap.dedent(script)],
        cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
        timeout=300,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return proc.stdout


def _write_config(path):
    path.write_text(
        "# Covariance, étiquette non ASCII\n"
        ".statistics.basic:\n"
        "  Covariance:\n"
        f"    labels: [undirected, linear, signed, {LABEL}]\n"
        "    configs:\n"
        "      - estimator: EmpiricalCovariance\n",
        encoding="utf-8",
    )


def test_config_with_non_ascii_label_loads_intact(tmp_path):
    _write_config(tmp_path / "custom.yaml")
    out = _run(f"""
        import numpy as np
        from pyspi import Calculator

        data = np.random.default_rng(0).standard_normal((3, 100))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            calc = Calculator(dataset=data, config="custom.yaml")
        assert {LABEL!r} in calc.spis["cov_EmpiricalCovariance"].labels
        print("ok")
    """, cwd=tmp_path)
    assert out.splitlines()[-1] == "ok"


def test_filter_spis_round_trips_a_non_ascii_label(tmp_path):
    _write_config(tmp_path / "custom.yaml")
    out = _run(f"""
        import yaml
        from pyspi.utils import filter_spis

        filter_spis([{LABEL!r}], output_name="filtered", configfile="custom.yaml")
        with open("filtered.yaml", encoding="utf-8") as f:
            written = yaml.safe_load(f)
        assert {LABEL!r} in written[".statistics.basic"]["Covariance"]["labels"]
        print("ok")
    """, cwd=tmp_path)
    assert out.splitlines()[-1] == "ok"


def test_checkpoint_with_a_non_ascii_error_message_resumes_intact(tmp_path):
    message = "échec délibéré Ý"
    (tmp_path / "failing.yaml").write_text(
        "failing_spis:\n"
        "  AlwaysRaises:\n"
        "    labels: [test]\n"
        "    configs:\n"
        f"      - message: {message}\n",
        encoding="utf-8",
    )
    out = _run(f"""
        import numpy as np
        from pyspi import Calculator

        data = np.random.default_rng(0).standard_normal((3, 100))

        def run(**kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                calc = Calculator(dataset=data, config="failing.yaml")
                calc.compute(n_jobs=1, progress=False, checkpoint_dir="cp", **kwargs)
            return calc

        first = run()
        assert {message!r} in first.errors["always_raises"], first.errors
        resumed = run(resume=True, retry_failed=False)
        assert resumed.errors["always_raises"] == first.errors["always_raises"]
        print("ok")
    """, cwd=tmp_path)
    assert out.splitlines()[-1] == "ok"
