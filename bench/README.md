# pyspi benchmark suite

Reproducible timing for `Calculator.compute()`, and the cost model behind the
shipped `pyspi/configs/benchmarked_p{80,90,95,99}.yaml` subsets.

## Install

```bash
pip install -e '.[bench]'
```

The `bench` extra adds `psutil` (per-cell RSS), `matplotlib`/`seaborn`/`plotly`
(analysis plots) and `nbformat` — none of which are runtime dependencies of
pyspi itself. Run everything from the repo root so `bench.*` is importable.

## Run

```bash
# Per-SPI walltime sweep over an M x T grid (n_jobs=1) — feeds the cost model
python -m bench.bench_compute --preset scaling --config full

# Parallel speedup curve at M=16, T=800 (n_jobs = 1,2,4,8,16)
python -m bench.bench_compute --preset parallel --config benchmarked_p90

# Custom grid
python -m bench.bench_compute --m 8,16,32 --t 200,800 --n-jobs 1,4,8 --config fast
```

`--config` takes a bundled config name (`full`, `fast`, `sonnet`, `fabfour`,
`benchmarked_p80/p90/p95/p99`) or a path to your own YAML — it is passed
straight to `pyspi.calculator.resolve_config`, so the two forms behave exactly
as they do for `Calculator(config=...)`.

## Presets

| preset      | grid                                          | purpose |
|-------------|-----------------------------------------------|---------|
| `headline`  | (M=10,T=500), (M=20,T=1000), n_jobs=1         | quick reference points |
| `scaling`   | M={4,8,16,32} x T={200,400,800,1600}, n_jobs=1| per-SPI walltime for cutting `benchmarked_p*.yaml` |
| `parallel`  | M=16, T=800, n_jobs={1,2,4,8,16}              | parallel speedup curve |
| `amortized` | M={8,16}, T=800, n_jobs=1                     | per-SPI walltime, minimal grid |

## Output

One JSON file per cell, written atomically to `bench/results/cells/` (the
default `--output-dir`) as `<label>_M<M>_T<T>_n<n_jobs>.json`. `--resume` skips
cells whose JSON exists with `repeats >= --repeats`. Each file is
self-contained, with an `environment` block (pyspi git sha, dependency versions
+ fingerprint, platform) pinning results to an exact environment. Per-cell
fields:

- `cell_wall_seconds {mean, std, values}`, `n_spis`, `n_spis_failed`, `failed_spis: [...]`
- `rss_mb_end`, `rss_mb_delta` (per-cell, via `psutil.Process().memory_info().rss`)
- `spi_seconds: {identifier: {mean, std, values, category, labels}}`
  - `category` is one of `basic | distance | causal | infotheory | spectral | wavelet | misc`
  - `labels` is the SPI's merged label list (includes `Mxx` size tags + stat-type tags)

`--repeats` defaults to **2**. The 22-cell reference campaign used to derive the
tracked summaries was measured with `--repeats 1` — at M=64, T=3200 a single
repeat of the full config is already a multi-day job — so its
`cell_wall_seconds.std` is 0 by construction, and cross-cell consistency (see
`report.md`) stands in for a within-cell error bar. Raw per-cell JSON is
machine-specific and intentionally untracked; retain it locally or archive it
externally if the campaign may need to be reanalysed.

## Cut a benchmarked config

`cut_config.py` turns a single per-cell JSON into `pyspi/configs/benchmarked_p<N>.yaml`.

```bash
python -m bench.bench_compute --preset amortized --config full
python -m bench.cut_config --bench-json bench/results/cells/<file>.json --keep 90
```

Cost model (`--mode`):

- `amortized` (default) — SPIs sharing a `_cache_namespace` (Covariance/Precision,
  the multitaper spectral pairs, Cointegration, Barycenter, CCM, ...) split the
  group's total cost evenly: `cost = sum(group walltimes) / group size`. This is
  the true per-variant budget impact — the shared computation is built once.
- `raw` — each SPI's own measured walltime. Written to
  `benchmarked_p<N>_raw.yaml` so it never overwrites the shipped amortized cut.

Output goes to `pyspi/configs/benchmarked_p<N>.yaml`; by default dropped SPIs
are commented out (not deleted) so the YAML carries the full provenance of the
cut. Pass `--no-preserve-dropped` to delete instead.

**`benchmarked_p90.yaml` carries a hand edit**: the two `te_kraskov_..._DCE_k-{1,2}`
variants were added back after the cut for methodological reasons. Re-running
`cut_config` overwrites it. The rationale lives in the config's own header
(`pyspi/configs/benchmarked_p90.yaml`, lines 6-16) — read it before regenerating.

## Analyse

```bash
# Cross-cell analysis (anchor stability, scaling fits, cumulative cost).
# With no arguments this analyses every local cell against the full config.
python -m bench.analyse_cells

# Explicit form
python -m bench.analyse_cells \
    --results-glob 'bench/results/cells/physics_config_M*_T*_n1.json' \
    --config full --percentiles 80,90,95,99 \
    --output-dir bench/results/analysis

# Predict cell wall time at a target (M, T) from the scaling fits
```

`analyse_cells` writes into `bench/results/analysis/`. Only the small
human-readable summaries are committed — `report.md`, `scaling.csv`,
`cell_summary.csv` (and `dropped_spi_comparison.md`). The bulk artefacts
(`long_costs.csv`, `jaccard_p*.csv`, `plot_*.png`)
are gitignored and regenerate in seconds; the Jaccard matrices are also
read `long_costs.csv`, so run `analyse_cells` once before either.

## Cluster (PBS)

One generic PBS Pro script, `bench/run_benchmark.pbs`, plus a worked site
example. To run it anywhere:

1. Clone the repo on the cluster and create a venv with pyspi installed
   editable (default location `<repo>/.venv`, override with `-v VENV=/path`;
   set `VENV=` empty if python already comes from a module or conda).
2. Edit the two `#PBS -l` lines at the top of `run_benchmark.pbs` to size
   walltime/cpus/mem for the largest `(M,T)` cell you plan to run. They are
   the only site-specific values in the file.
3. Submit from the repo root, passing your account/queue/storage/mail on the
   qsub command line — `#PBS` directives are never shell-expanded, so nothing
   site-specific can come from an env var there.

```bash
# one (M,T) cell per array task, n_jobs=1 (the config-cutting regime):
M=32 T=200,400,800,1600,3200 CONFIG=full REPEATS=1 \
  qsub -J 1-5 -v M,T,CONFIG,REPEATS bench/run_benchmark.pbs

# a single cell, on a named project/queue, with mail and a longer walltime:
M=64 T=3200 CONFIG=full \
  qsub -P myproj -q normal -l storage=scratch/myproj -l walltime=168:00:00 \
       -m bea -M you@example.org -v M,T,CONFIG bench/run_benchmark.pbs

# a bundled preset, walked sequentially in one job:
qsub -v PRESET=parallel,CONFIG=benchmarked_p90 bench/run_benchmark.pbs
```

`qsub -v` splits on commas, so comma-valued vars (`M`, `T`) must be exported in
the shell and passed by name, as above. Other env vars: `NJOBS`, `PRESET`,
`CONFIG`, `REPEATS`, `LABEL`, `PYSPI_DIR`, `VENV`, `MODULES` (space-separated
modules to load) — all documented in the script header. Every run is
`--resume`, so resubmitting skips cells that already have a JSON.

**`bench/physics/run_bench.pbs`** is a worked example to copy and adapt: the
USYD Physics queue configuration that produced the committed
`physics_config_M*_T*_n1.json` cells (full config, `repeats=1`, `n_jobs=1`,
M={4,8,16,32,64} x T={200,400,800,1600,3200}, one cell per array task). It
pins those defaults and delegates to `run_benchmark.pbs`.

## Notes

- **n_jobs and amortized configs**: per-SPI cost is invariant to `n_jobs` under
  the cache-aware scheduler (each cache group runs sequentially within one
  worker). Use `n_jobs=1` for config-cutting measurements; `n_jobs` only
  changes makespan, which the `parallel` preset measures.
- **Start method**: `Calculator.compute()` defaults to `fork` on Linux,
  `spawn` on macOS/Windows. fork is ~2x faster (workers inherit imported
  state via copy-on-write). Override with `--mp-context`.
- This suite measures **inner** parallelism (SPIs within one dataset). Outer
  parallelism (many datasets) belongs to the job scheduler — e.g. a PBS array.
- **JIDT parity harness (removed)**: `bench/jidt_parity/` validated the pure-NumPy
  information-theory estimators against JIDT 1.6.1 before the Java dependency was
  dropped. Gaussian/kernel MI and symbolic TE matched to machine precision
  (~1e-16), and Kozachenko entropy to ~1e-5. The KSG comparison used different
  normalisation and tie policies and is historical evidence rather than a parity
  oracle. The harness required `jpype` and the removed `infodynamics.jar`, so it
  is not part of 3.0.0.
