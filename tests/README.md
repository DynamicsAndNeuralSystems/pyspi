# pyspi test suite

## Running

```bash
pytest                      # fast suite (default: -m 'not slow'), ~2 min
pytest -m slow              # baseline drift suite only, ~3.5 min
pytest -m ''                # everything
```

`addopts = "-m 'not slow'"` is set in `pyproject.toml`, so the slow marker is
opt-in. Nothing outside `tests/tools/` performs work at import time: a missing or
corrupt baseline can never break collection of unrelated tests.

## What each file covers

| File | Covers |
| --- | --- |
| `test_infotheory_analytic.py` | **Closed-form correctness** of the pure-NumPy info-theory estimators: bivariate-Gaussian MI vs `-0.5*ln(1-rho^2)` for all four estimators, differential entropy of `N(0, sigma^2)`, Gaussian TE on an analytic AR(1), independence collapsing to ~0, and the entropy identities. The only place a `return np.zeros(...)` stub would be caught. |
| `test_calculator.py` | `Calculator` / `Data` API: `config=` resolution, `zscore=`, labels, grouping, dataset loading. |
| `test_utils.py` | Utility helpers. |
| `test_smoke.py` | Cheap shape / finiteness / sign checks across SPI families. Sanity, not correctness. |
| `test_parallel.py` | `Calculator.compute()` parallel path, checkpointing, per-SPI failure isolation. |
| `test_phi_native.py` | Native (non-JIDT) integrated-information implementation. |
| `test_baseline_drift.py` | `slow`. Every SPI on three frozen fixtures (M=3/5/7) vs a stored baseline. |
| `test_state_integrity.py` | `Data` ownership, read-only exposure, cache invalidation on mutation, builder path, process-name lifecycle, `dim_order` validation. |
| `test_cache_keys.py` | Parameterised statistic caches must key on every parameter that reaches the identifier. |
| `test_run_identity.py` | Checkpoints must identify the run that produced them; identifier collisions must be rejected at insertion. |
| `test_execution_parity.py` | Serial and parallel must agree on *failure* semantics, not only on numbers. Uses `failing_spis.py` + `parity_failure_config.yaml`. |
| `test_estimator_contracts.py` | An SPI must compute the estimator it advertises, or refuse. Symbolic/KSG preconditions. |
| `test_structural_traits.py` | Declared symmetry labels vs observed baseline matrices; AEG process-order dependence. |

### Contract tests

These files assert behaviour the package must have.
None carries an `xfail` marker.

Two rules when working on these:

1. **Do not relax an assertion to make one pass.** Fix the code, or mark the
   test `xfail(strict=True)` with the reason.
2. **Check the failure reason, not just the xfail count.**
   A test can fail for reasons unrelated to the defect under test -- a vacuous comparison, a wrong keyword, a config name passed where a path was wanted.
   An xfail proves nothing until you have seen the message.
   Run with `--runxfail` to see it.

One trap worth naming: `parse_bivariate`'s signature is `(self, data, data2=None, i=None, j=None)`, so `spi.bivariate(data, 0, 1)` binds `data2=0, i=1`.
It is rejected with a message naming the signature.
Always pass `i=`/`j=` by keyword.

Log base: every information-theoretic estimator reports **nats**.
JIDT reports bits for its box-kernel and discrete estimators, so divide `kernel` and `symbolic` values by ln 2 to compare with JIDT.

## Baselines

`tests/data/baselines/{var1_M3_T100,cml_M5_T100,kuramoto_M7_T100}.npz` — one
`MxM` matrix per SPI identifier, plus `__dataset__` / `__config__` / `__seed__`
provenance entries. Regenerate with:

```bash
python tests/tools/generate_benchmark_tables.py                 # all three
python tests/tools/generate_benchmark_tables.py -d cml_M5_T100
```

The frozen `.npy` fixtures themselves live in `tests/data/fixtures/` and come
from `tests/tools/generate_fixtures.py`. They are **test inputs, not shipped
data**: they are deliberately outside `pyspi/data/`, so they are not in the
wheel and not reachable via `pyspi.data.load_dataset` (which exposes only
the three demo datasets `forex`, `cml`, `standard_normal`). Three generating
processes at three widths — VAR(1) at `M=3`, coupled map lattice at `M=5`,
Kuramoto at `M=7`, all `T=100` — so the SPI set is exercised across a range of
`M` rather than at a single width.

These baselines are generated from **the current code**, not from upstream pyspi 2.0.1.
pyspi 3 reimplements every information-theoretic estimator, so upstream values are the wrong oracle for exactly the code that most needs one.
The baselines are therefore a *forward-looking change detector*: they tell you that something moved, not that it was right before.
Independent correctness lives in `test_infotheory_analytic.py`.

A single seeded pass is stored per dataset — not a mean over trials. The
datasets are frozen fixtures and nearly every SPI is a deterministic function of
them, so an exact oracle is more useful than an average that no individual run
reproduces.

## Drift is enforced

`test_baseline_drift.py` **fails** on:

1. the baseline SPI set differing from the current `Calculator`'s SPI set, so a
   newly-broken or renamed SPI cannot escape by having no baseline;
2. a matrix shape change;
3. a change in the **NaN pattern** — an SPI going from finite to all-NaN (or
   back) is a categorical regression, not drift;
4. a baseline with no finite off-diagonal value at all, which cannot detect a
   regression however closely the current run reproduces it;
5. any numerical difference outside the SPI's tolerance.
   A session-end summary table lists these failures.

Two further gates run per fixture: no SPI may raise (`calc.errors` must be
empty) and none may produce an entirely non-finite column. The single
documented exception lives in `KNOWN_UNESTIMABLE`, shared with the baseline
generator so the two cannot disagree, and the suite fails if a listed exception
starts succeeding.

Tolerances are `1e-9` relative / `1e-12` absolute for every SPI, including `causal` and `misc`, whose optimisers and permutation tests reproduce bit-exactly.
Per-SPI exceptions go in `LOOSE_SPIS`; `tools/measure_reproducibility.py` checks reproducibility.

## Open findings

None.
