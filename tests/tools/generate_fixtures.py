"""Generate the frozen synthetic datasets used by ``tests/test_baseline_drift.py``.

Three qualitatively different generating processes, at three different process
counts, so the SPI set is exercised across a range of ``M`` rather than at a
single width:

  * ``var1_M3_T100``     -- linear VAR(1), stable, sparse coupling (M=3)
  * ``cml_M5_T100``      -- coupled map lattice, chaotic logistic map (M=5)
  * ``kuramoto_M7_T100`` -- coupled phase oscillators, sin(phase) observed (M=7)

Conventions
-----------
Arrays are saved as ``(observations, processes)`` — i.e. ``dim_order='sp'`` —
matching the ``.npy`` files bundled in ``pyspi/data/``. The generators build
``(processes, observations)`` internally and transpose on write.

``T = 100`` throughout. That is short enough to keep the 322-SPI drift suite
fast (SPI cost is at worst quadratic in ``M`` and roughly linear-to-quadratic in
``T``) while still leaving enough samples for the embedding- and
spectrum-based estimators, which need a few dozen effective observations after
lagging. It also matches the ``T`` of the fixtures these replace, so baseline
run times are directly comparable.

These are test fixtures, not shipped data: they live under ``tests/data/fixtures/``
and are deliberately *not* reachable via ``pyspi.data.load_dataset``.

Usage
-----
    python tests/tools/generate_fixtures.py            # write all three
    python tests/tools/generate_fixtures.py -f var1_M3_T100
    python tests/tools/generate_fixtures.py --out /tmp/fixtures
"""
import argparse
import os

import numpy as np

# <repo>/tests/tools/this_file.py -> <repo>/tests/data/fixtures
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.dirname(_HERE), "data", "fixtures")


def generate_var1(M=3, T=100, seed=0):
    """Stable VAR(1) with sparse off-diagonal coupling and Gaussian noise."""
    rng = np.random.default_rng(seed)

    A = rng.uniform(-0.4, 0.4, size=(M, M))
    A *= rng.random((M, M)) < 0.5          # sparsify the coupling
    A[np.diag_indices(M)] = 0.5            # keep every process autocorrelated
    # Rescale below unit spectral radius so the process is stationary.
    A *= 0.85 / np.max(np.abs(np.linalg.eigvals(A)))

    Y = np.zeros((M, T))
    Y[:, 0] = rng.standard_normal(M)
    for t in range(1, T):
        Y[:, t] = A @ Y[:, t - 1] + 0.5 * rng.standard_normal(M)
    return Y


def generate_cml(M=5, T=100, coupling=0.2, r=4.0, burn_in=500, seed=0):
    """Diffusively coupled logistic maps on a ring (fully chaotic at r=4)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 0.9, size=M)

    Y = np.zeros((M, T))
    for t in range(burn_in + T):
        fx = r * x * (1.0 - x)
        x = (1.0 - coupling) * fx + 0.5 * coupling * (np.roll(fx, 1) + np.roll(fx, -1))
        if t >= burn_in:
            Y[:, t - burn_in] = x
    return Y


def generate_kuramoto(M=7, T=100, dt=0.2, K=0.8, omega_spread=2.0, seed=0):
    """Kuramoto phase oscillators with uniform all-to-all coupling.

    ``K`` and the natural-frequency spread are set for *partial* synchronisation.
    Strong coupling drives the lattice to a single locked phase, at which point
    every process is a copy of every other and most pairwise statistics become
    degenerate (that was the failure mode of the fixture this replaces).
    """
    rng = np.random.default_rng(seed)
    omega = rng.uniform(-omega_spread, omega_spread, size=M)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=M)

    Y = np.zeros((M, T))
    for t in range(T):
        dtheta = omega + (K / M) * np.sin(theta[None, :] - theta[:, None]).sum(axis=1)
        theta = theta + dt * dtheta
        Y[:, t] = np.sin(theta)
    return Y


# name -> zero-argument generator returning a (processes, observations) array.
FIXTURES = {
    "var1_M3_T100": lambda: generate_var1(M=3, T=100, seed=0),
    "cml_M5_T100": lambda: generate_cml(M=5, T=100, seed=0),
    "kuramoto_M7_T100": lambda: generate_kuramoto(M=7, T=100, seed=0),
}


def check(name, Y):
    """Fail loudly on a degenerate fixture; return a one-line summary."""
    assert np.all(np.isfinite(Y)), f"{name}: non-finite entries"
    sd = Y.std(axis=1)
    assert np.all(sd > 1e-3), f"{name}: near-constant process(es), std={sd}"

    # No two processes may be (anti-)identical up to scale, or every pairwise
    # statistic between them degenerates.
    corr = np.corrcoef(Y)
    off = np.abs(corr[~np.eye(len(Y), dtype=bool)])
    assert off.max() < 0.999, f"{name}: duplicated process(es), max |corr|={off.max():.4f}"

    return (f"{name:<18} shape={Y.T.shape} (obs, proc)  "
            f"range=[{Y.min():+.3f}, {Y.max():+.3f}]  "
            f"std=[{sd.min():.3f}, {sd.max():.3f}]  max|corr|={off.max():.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-f", "--fixture", choices=tuple(FIXTURES) + ("all",), default="all",
        help="Fixture to regenerate (default: all).",
    )
    parser.add_argument(
        "-o", "--out", default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT}).",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    names = tuple(FIXTURES) if args.fixture == "all" else (args.fixture,)
    for name in names:
        Y = FIXTURES[name]()
        summary = check(name, Y)
        np.save(os.path.join(args.out, f"{name}.npy"), Y.T)  # -> (obs, processes)
        print(summary)


if __name__ == "__main__":
    main()
