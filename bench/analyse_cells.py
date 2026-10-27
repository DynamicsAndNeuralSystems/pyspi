#!/usr/bin/env python
"""Analyse a directory of per-cell bench JSONs to choose the anchor + percentile
recipe for the pyspi/configs/benchmarked_p<N>.yaml cuts.

Writes into --output-dir. Committed (small, human-readable):
  cell_summary.csv      (M, T, n_spis, n_failed, cell_wall_s, sum_amortized_s)
  scaling.csv           per-SPI fit  log t = a + p*log M + q*log T  (amortized)
  report.md             human-readable summary with elbow / drift / recommendation
Gitignored (bulk / derived, regenerate in seconds):
  long_costs.csv        (M, T, identifier, raw_s, amortized_s, cache_namespace)
  jaccard_p{N}.csv      kept-set Jaccard between cells at percentile N (also in report.md)
  plot_cumulative.png   cumulative amortized cost vs kept-fraction, one line per cell
  plot_kept_drift_p{N}.png  binary heatmap: SPI x cell (1 = kept @ percentile, 0 = dropped)

Usage:
    python -m bench.analyse_cells                      # defaults: all committed cells
    python -m bench.analyse_cells \
        --results-glob 'bench/results/cells/physics_config_M*_T*_n1.json' \
        --config full \
        --percentiles 80,90,95 \
        --output-dir bench/results/analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bench._config_walk import cache_bucket, walk_spis  # noqa: E402
from pyspi.calculator import bundled_configs, resolve_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_GLOB = "bench/results/cells/*.json"


def cache_namespace_map(configfile) -> dict[str, tuple | None]:
    """Return {identifier: bucket_key} for every SPI in the config.

    The bucket key is ``(namespace, *cache_subkey_values)`` if the class
    declares ``_cache_namespace``, else ``None``.
    """
    return {ident: cache_bucket(spi)
            for _, _, _, ident, spi in walk_spis(configfile)}


def amortized(raw: dict[str, float], ns_map: dict[str, str | None]) -> dict[str, float]:
    groups: dict[str | None, list[str]] = defaultdict(list)
    for ident in raw:
        groups[ns_map.get(ident)].append(ident)
    cost: dict[str, float] = {}
    for ns, ids in groups.items():
        if ns is None:
            for i in ids:
                cost[i] = raw[i]
        else:
            share = sum(raw[i] for i in ids) / len(ids)
            for i in ids:
                cost[i] = share
    return cost


def load_cells(paths: list[Path]) -> list[dict]:
    cells = []
    for p in sorted(paths):
        d = json.loads(p.read_text())
        if "spi_seconds" not in d or "error" in d:
            print(f"[skip] {p.name}: error or empty", file=sys.stderr)
            continue
        d["__path__"] = str(p)
        cells.append(d)
    return cells


def long_df(cells: list[dict], ns_map: dict[str, tuple | None]) -> pd.DataFrame:
    rows = []
    for c in cells:
        raw = {k: v["mean"] for k, v in c["spi_seconds"].items()}
        amo = amortized(raw, ns_map)
        for ident, r in raw.items():
            bucket = ns_map.get(ident)
            rows.append({
                "M": c["M"], "T": c["T"],
                "identifier": ident,
                "raw_s": r,
                "amortized_s": amo[ident],
                "cache_namespace": bucket[0] if bucket else None,
                "cache_subkey": str(bucket[1:]) if bucket and len(bucket) > 1 else None,
            })
    return pd.DataFrame(rows)


def cell_summary(cells: list[dict], df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in cells:
        sub = df[(df["M"] == c["M"]) & (df["T"] == c["T"])]
        rows.append({
            "M": c["M"], "T": c["T"],
            "n_spis": c["n_spis"],
            "n_failed": c["n_spis_failed"],
            "cell_wall_s": c["cell_wall_seconds"]["mean"],
            "sum_amortized_s": float(sub.amortized_s.sum()),
            "median_amortized_s": float(sub.amortized_s.median()),
            "max_amortized_s": float(sub.amortized_s.max()),
        })
    return pd.DataFrame(rows).sort_values(["M", "T"]).reset_index(drop=True)


def kept_set(df_cell: pd.DataFrame, pct: int) -> set[str]:
    """The fastest pct% of SPIs by amortized cost at this cell."""
    n_keep = round(pct / 100 * len(df_cell))
    return set(df_cell.nsmallest(n_keep, "amortized_s").identifier)


def jaccard_matrix(df: pd.DataFrame, pct: int) -> pd.DataFrame:
    cells = df[["M", "T"]].drop_duplicates().sort_values(["M", "T"]).itertuples(index=False)
    cells = list(cells)
    labels = [f"M{m}T{t}" for m, t in cells]
    keeps = [kept_set(df[(df["M"] == m) & (df["T"] == t)], pct) for m, t in cells]
    n = len(cells)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = keeps[i], keeps[j]
            J[i, j] = len(a & b) / len(a | b) if (a | b) else 1.0
    return pd.DataFrame(J, index=labels, columns=labels)


def fit_scaling(df: pd.DataFrame, min_seconds: float = 0.01) -> pd.DataFrame:
    """Fit log t = a + p log M + q log T per SPI on amortized cost.

    Returns rows with intercept, p_M, q_T, residual std on log scale, n_points.
    Skips SPIs whose amortized cost is below min_seconds in *all* cells (noise floor).
    """
    rows = []
    for ident, sub in df.groupby("identifier"):
        if (sub.amortized_s < min_seconds).all():
            continue
        sub = sub[sub.amortized_s > 0]
        if len(sub) < 4:
            continue
        X = np.column_stack([np.ones(len(sub)),
                             np.log(sub["M"].values), np.log(sub["T"].values)])
        y = np.log(sub.amortized_s.values)
        # least squares
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        rows.append({
            "identifier": ident,
            "intercept": float(beta[0]),
            "p_M": float(beta[1]),
            "q_T": float(beta[2]),
            "log_resid_std": float(resid.std(ddof=0)),
            "n_points": int(len(sub)),
            "amortized_at_largest": float(sub.sort_values(["M", "T"]).amortized_s.iloc[-1]),
        })
    return pd.DataFrame(rows).sort_values("amortized_at_largest", ascending=False).reset_index(drop=True)


def plot_cumulative(df: pd.DataFrame, percentiles: list[int], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for (M, T), sub in df.groupby(["M", "T"]):
        costs = np.sort(sub.amortized_s.values)
        frac = np.arange(1, len(costs) + 1) / len(costs)
        ax.plot(frac * 100, np.cumsum(costs) / costs.sum(), label=f"M={M} T={T}",
                lw=1.2, alpha=0.75)
    for p in percentiles:
        ax.axvline(p, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("Kept-fraction (%)")
    ax.set_ylabel("Cumulative amortized cost / total")
    ax.set_title("Cumulative amortized cost vs kept-fraction (elbow ≈ cut point)")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_kept_drift(df: pd.DataFrame, pct: int, out: Path) -> None:
    cells = df[["M", "T"]].drop_duplicates().sort_values(["M", "T"]).itertuples(index=False)
    cells = list(cells)
    keeps = [kept_set(df[(df["M"] == m) & (df["T"] == t)], pct) for m, t in cells]
    # only show SPIs that are NOT kept in at least one cell (i.e. drift exists)
    union_drop = set.union(*[set(df.identifier.unique()) - k for k in keeps])
    if not union_drop:
        return
    sorted_drop = sorted(union_drop)
    mat = np.array([[int(ident in k) for k in keeps] for ident in sorted_drop])
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(cells)), max(4, 0.18 * len(sorted_drop))))
    ax.imshow(mat, aspect="auto", cmap="Greys_r", interpolation="nearest")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"M{m}T{t}" for m, t in cells], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(sorted_drop)))
    ax.set_yticklabels(sorted_drop, fontsize=6)
    ax.set_title(f"Kept-set drift @ p{pct}  (white = kept, black = dropped)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _md_table(df: pd.DataFrame, floatfmt: str = ".3f", index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table without tabulate."""
    work = df.copy()
    if index:
        work = work.reset_index().rename(columns={work.index.name or "index": "idx"})
    cols = [str(c) for c in work.columns]
    body = []
    for _, r in work.iterrows():
        row = []
        for v in r.values:
            if isinstance(v, float):
                row.append(format(v, floatfmt))
            else:
                row.append(str(v))
        body.append(row)
    widths = [max(len(cols[i]), *(len(b[i]) for b in body)) if body else len(cols[i])
              for i in range(len(cols))]
    sep = "|".join("-" * (w + 2) for w in widths)
    head = "|".join(f" {cols[i]:<{widths[i]}} " for i in range(len(cols)))
    rows = ["|".join(f" {body[r][i]:<{widths[i]}} " for i in range(len(cols)))
            for r in range(len(body))]
    return "\n".join([f"|{head}|", f"|{sep}|"] + [f"|{r}|" for r in rows])


def write_report(out: Path, summary: pd.DataFrame, jac: dict[int, pd.DataFrame],
                 scaling: pd.DataFrame, df: pd.DataFrame,
                 percentiles: list[int]) -> None:
    lines = ["# Bench-cell analysis", ""]
    lines.append(f"Cells analysed: {len(summary)}.  Per cell: {int(summary.n_spis.iloc[0])} SPIs.")
    lines.append("")
    lines.append("## Cell summary (sorted by M, T)")
    lines.append(_md_table(summary, ".2f"))
    lines.append("")

    for p in percentiles:
        J = jac[p]
        offdiag = J.values[~np.eye(len(J), dtype=bool)]
        lines.append(f"## Jaccard kept-set similarity @ p{p}")
        lines.append(f"Off-diagonal: mean={offdiag.mean():.3f}  min={offdiag.min():.3f}  "
                     f"max={offdiag.max():.3f}")
        lines.append("")
        lines.append(_md_table(J.round(3), ".3f", index=True))
        lines.append("")

    # Elbow heuristic: where does cumulative cost cross 90% of total?
    lines.append("## Where does cumulative cost cross 50%, 80%, 90%, 95% of total?")
    rows = []
    for (M, T), sub in df.groupby(["M", "T"]):
        costs = np.sort(sub.amortized_s.values)
        frac_cost = np.cumsum(costs) / costs.sum()
        frac_kept = np.arange(1, len(costs) + 1) / len(costs)
        row = {"M": M, "T": T}
        for target in [0.50, 0.80, 0.90, 0.95]:
            i = int(np.searchsorted(frac_cost, target))
            row[f"keep_for_{int(target*100)}pct_cost"] = round(frac_kept[i] * 100, 1)
        rows.append(row)
    lines.append(_md_table(pd.DataFrame(rows), ".2f"))
    lines.append("")

    lines.append("## Top 25 SPIs by amortized cost at the largest cell")
    lines.append(_md_table(scaling.head(25).round(3), ".3f"))
    lines.append("")

    out.write_text("\n".join(lines))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-glob", default=DEFAULT_RESULTS_GLOB,
                   help=f"Glob for per-cell bench JSONs, relative to repo root "
                        f"(default: {DEFAULT_RESULTS_GLOB}).")
    p.add_argument("--config", default="full",
                   help=f"Source config for the cache-namespace map and SPI list: "
                        f"a bundled name ({'/'.join(bundled_configs())}) or a path "
                        f"(default: full).")
    p.add_argument("--percentiles", default="80,90,95",
                   help="Comma-separated percentiles to evaluate kept-set Jaccard at.")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "bench" / "results" / "analysis")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    percentiles = [int(x) for x in args.percentiles.split(",") if x.strip()]

    paths = sorted(Path(REPO_ROOT).glob(args.results_glob))
    if not paths:
        raise SystemExit(f"no JSONs match {args.results_glob}")
    print(f"[analyse] {len(paths)} cells matched", file=sys.stderr)

    ns_map = cache_namespace_map(resolve_config(args.config))
    print(f"[analyse] cache-namespace map: {len(ns_map)} SPIs", file=sys.stderr)

    cells = load_cells(paths)
    df = long_df(cells, ns_map)
    summary = cell_summary(cells, df)

    df.to_csv(args.output_dir / "long_costs.csv", index=False)
    summary.to_csv(args.output_dir / "cell_summary.csv", index=False)

    jac: dict[int, pd.DataFrame] = {}
    for p in percentiles:
        J = jaccard_matrix(df, p)
        J.to_csv(args.output_dir / f"jaccard_p{p}.csv")
        jac[p] = J

    sf = fit_scaling(df)
    sf.to_csv(args.output_dir / "scaling.csv", index=False)

    plot_cumulative(df, percentiles, args.output_dir / "plot_cumulative.png")
    for p in percentiles:
        plot_kept_drift(df, p, args.output_dir / f"plot_kept_drift_p{p}.png")

    write_report(args.output_dir / "report.md", summary, jac, sf, df, percentiles)

    print(f"[analyse] wrote artefacts -> {args.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
