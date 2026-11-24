#!/usr/bin/env python
"""Cut a benchmarked SPI subset config from a bench_compute.py per-cell JSON.

Reads ONE per-cell JSON (``<label>_M<M>_T<T>_n<n>.json``), ranks the SPIs it
timed by cost, and emits a ``pyspi/configs/benchmarked_p<keep>.yaml`` containing
only the fastest ``--keep`` percent. SPIs of the source config with no timing in
the JSON are left out. (M, T, n_jobs) are read from the JSON itself, not parsed
from the filename.

Cost model — two modes:
  raw        each SPI's own measured wall time at this (M, T).
  amortized  (default) SPIs sharing a within-class cache (``_cache_namespace``
             — Covariance/Precision, multitaper spectral pairs, CCM,
             Cointegration, Barycenter, ...) split the group's total cost
             evenly:
                 cost(spi) = sum(group wall times) / (group size)
             The shared computation is built once and reused, so blaming its
             full cost to one variant overcounts. Ungrouped SPIs use raw time.

The shipped 3.0.0 sets are this script's cut of the costs in
``bench/results/analysis/tiers_3_0.csv``, followed by 11 hand placements that
the script does not apply. Re-apply them after a re-cut, and keep each config's
header note on them (see ``bench/README.md``):

    in benchmarked_p90 and up:  te_kraskov_NN-4_DCE-AUTO_k-2_kt-1_l-1_lt-1, cds
                                (above the cutoff), anm, hsic (kept although
                                their cost grows about as T^2)
    in benchmarked_p95 and up:  softdtw_constraint-itakura
    in benchmarked_p99 only:    softdtw, bary_dtw_corr, lcss, hsic_biased,
                                lccm_E-None_k-5_skill, lccm_E-None_k-5_tp

Usage:
    python -m bench.cut_config --bench-json bench/results/cells/pbs_config_M16_T800_n1.json --keep 90
    python -m bench.cut_config --bench-json <path> --keep 80 --mode raw
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from bench._config_walk import cache_bucket, walk_spis
from pyspi.calculator import CONFIG_DIR, bundled_configs, resolve_config


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench-json", type=Path, required=True,
                   help="Per-cell timing JSON from bench_compute.py.")
    p.add_argument("--config", default="full",
                   help=f"Source config to cut from: a bundled name "
                        f"({'/'.join(bundled_configs())}) or a path (default: full).")
    p.add_argument("--keep", type=int, default=90,
                   help="Percent of SPIs to keep, fastest-first (default: 90).")
    p.add_argument("--mode", choices=["amortized", "raw"], default="amortized",
                   help="Cost model (default: amortized).")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output config path (default: pyspi/configs/benchmarked_p<keep>.yaml, "
                        "suffixed _raw under --mode raw).")
    p.add_argument("--preserve-dropped", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Keep dropped variants/classes in the output as commented-out "
                        "blocks (default: delete them; the dropped SPIs are listed on "
                        "stderr either way).")
    return p.parse_args(argv)


def _cache_buckets(records: list) -> dict[tuple, list[str]]:
    """Return {(namespace, *subkey): [identifiers]} for all cache-grouped SPIs."""
    buckets: dict = defaultdict(list)
    for _, _, _, identifier, spi in records:
        bucket = cache_bucket(spi)
        if bucket is None:
            continue
        buckets[bucket].append(identifier)
    return buckets


def snap_to_cache_buckets(records: list, kept: set[str]) -> tuple[set[str], list[str]]:
    """Promote partially-kept cache buckets to fully kept.

    Once the cache for a bucket is built (because ANY member is kept), the
    remaining members are essentially free at compute time — they're just
    cheap transforms of the same cached value. Dropping a strict subset of a
    bucket therefore pays the full cache cost for fewer SPIs, which is
    strictly worse than keeping the whole bucket.

    Returns ``(snapped_kept, promoted_ids)``. The kept count may exceed the
    original percentile target by the size of the promoted partial buckets.
    """
    snapped = set(kept)
    promoted: list[str] = []
    for ids in _cache_buckets(records).values():
        in_kept = [i for i in ids if i in snapped]
        if not in_kept or len(in_kept) == len(ids):
            continue  # whole bucket kept or whole bucket dropped — nothing to do
        for i in ids:
            if i not in snapped:
                snapped.add(i)
                promoted.append(i)
    return snapped, sorted(promoted)


def amortized_costs(records: list, raw: dict[str, float]) -> dict[str, float]:
    """Amortize cost within each shared-cache bucket.

    A bucket is identified by ``(_cache_namespace, *_cache_subkey)``. The
    subkey (a per-instance tuple, default ``()``) lets a class declare which
    constructor params split its namespace into independent caches — e.g.
    Barycenter caches per ``mode``, so its variants amortize per mode rather
    than across all 16 of them.
    """
    groups: dict = defaultdict(list)
    for _, _, _, identifier, spi in records:
        groups[cache_bucket(spi)].append(identifier)
    cost: dict[str, float] = {}
    for key, ids in groups.items():
        if key is None:
            for i in ids:
                cost[i] = raw[i]
        else:
            share = sum(raw[i] for i in ids) / len(ids)
            for i in ids:
                cost[i] = share
    return cost


def _dump_variant(params: dict | None) -> list[str]:
    """Render one config variant (a param dict, or None for no-args) as YAML lines.

    Returns a list of lines like ``["- estimator: kraskov", "  prop_k: 4"]``. For
    None (no-args SPI), returns ``["- {}"]`` — but that case shouldn't reach here
    in normal use (it's handled at the class level via ``configs: null``).
    """
    if params is None or params == {}:
        return ["- {}"]
    text = yaml.dump([params], sort_keys=False, default_flow_style=False, indent=2).rstrip()
    return text.splitlines()


def _render_class_block(class_name: str, src_entry: dict,
                        kept: list, dropped: list) -> str:
    """Render a class block: labels, dependencies, configs (kept), then commented dropped.

    The class header is at the LEFT MARGIN (the caller indents under the module).
    Returns a string with NO trailing newline.

    Special cases:
      - single no-args SPI (configs: null in source):  kept=[None] -> configs: null
        kept; kept=[] -> the whole class is fully dropped (caller comments it out).
      - some kept, some dropped: kept variants emitted normally, dropped appended as
        ``  # - estimator: ...`` comments under the configs: list.
    """
    lines = [f"{class_name}:"]
    labels = src_entry.get("labels")
    if labels is not None:
        lines.append("  labels:")
        for lab in labels:
            lines.append(f"    - {lab}")
    if "dependencies" in src_entry:
        deps = src_entry["dependencies"]
        if deps is None:
            lines.append("  dependencies:")
        else:
            lines.append("  dependencies:")
            for d in deps:
                lines.append(f"    - {d}")

    # configs handling
    if kept == [None]:
        lines.append("  configs:")
    else:
        lines.append("  configs:")
        for p in kept:
            if p is None:
                continue
            for vl in _dump_variant(p):
                lines.append("  " + vl)
        for p in dropped:
            if p is None:
                # commented no-args means the class is fully dropped — handled by caller
                lines.append("  # (no-args variant dropped)")
                continue
            for vl in _dump_variant(p):
                lines.append("  # " + vl)
    return "\n".join(lines)


def emit_config(source_path: Path, records: list, kept_ids: set[str], header: str,
                preserve_dropped: bool = False) -> str:
    """Emit the cut YAML. If preserve_dropped, dropped variants/classes are kept
    as commented blocks instead of being removed."""
    source = yaml.safe_load(source_path.read_text())

    # Group records by (module, class), preserving source order.
    grouped: dict[tuple[str, str], list[tuple]] = {}
    order: list[tuple[str, str]] = []
    for module, cls, params, ident, _ in records:
        key = (module, cls)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((params, ident))

    out_lines: list[str] = [header.rstrip()]
    current_module: str | None = None

    for (module, cls) in order:
        variants = grouped[(module, cls)]
        kept = [p for p, i in variants if i in kept_ids]
        dropped = [p for p, i in variants if i not in kept_ids]

        if not preserve_dropped and not kept:
            continue  # drop the class entirely

        if module != current_module:
            if current_module is not None:
                out_lines.append("")
            out_lines.append(f"{module}:")
            current_module = module

        src_entry = source[module][cls]
        if kept or not preserve_dropped:
            # Some variants kept: render normally with kept + commented dropped.
            block = _render_class_block(cls, src_entry, kept, dropped if preserve_dropped else [])
            out_lines.append("\n".join("  " + l for l in block.splitlines()))
        else:
            # All variants dropped: render the full class block and comment every line.
            block = _render_class_block(cls, src_entry, dropped, [])
            out_lines.append("\n".join("  # " + l for l in block.splitlines()))
        out_lines.append("")  # blank line between class blocks

    return "\n".join(out_lines) + "\n"


def main(argv=None) -> int:
    args = parse_args(argv)
    cell = json.loads(args.bench_json.read_text())
    if "spi_seconds" not in cell:
        raise SystemExit(
            f"{args.bench_json} is not a per-cell bench JSON "
            "(no 'spi_seconds' at top level).")
    if "error" in cell:
        raise SystemExit(f"{args.bench_json} has an error entry: {cell['error']}")
    raw_cell = {spi: v["mean"] for spi, v in cell["spi_seconds"].items()}
    M, T, n_jobs = cell.get("M"), cell.get("T"), cell.get("n_jobs")

    source_config = Path(resolve_config(args.config))
    records = list(walk_spis(source_config))
    ids = [r[3] for r in records]
    missing = sorted(i for i in ids if i not in raw_cell)
    if missing:
        print(f"[cut] WARNING: {len(missing)} SPI(s) in config but not in bench JSON "
              f"— left out: {', '.join(missing[:5])}"
              f"{' ...' if len(missing) > 5 else ''}", file=sys.stderr)
    timed = [r for r in records if r[3] in raw_cell]
    raw = {r[3]: raw_cell[r[3]] for r in timed}

    cost = raw if args.mode == "raw" else amortized_costs(timed, raw)
    ranked = sorted(raw, key=lambda i: cost[i])
    n_target = round(args.keep / 100 * len(ranked))
    kept = set(ranked[:n_target])
    kept_max = cost[ranked[n_target - 1]] if n_target else 0.0
    drop_min = cost[ranked[n_target]] if n_target < len(ranked) else float("inf")

    # Cache-aware snap: promote partial buckets to fully-kept. The kept count
    # may exceed n_target by the size of the promoted partial buckets.
    if args.mode == "amortized":
        kept, promoted = snap_to_cache_buckets(timed, kept)
    else:
        promoted = []
    dropped = [i for i in ranked if i not in kept]

    output = args.output or (
        CONFIG_DIR / f"benchmarked_p{args.keep}{'_raw' if args.mode == 'raw' else ''}.yaml")

    cost_note = ("amortized cost (SPIs that share a cache split its cost)"
                 if args.mode == "amortized" else "measured cost")
    cores = "on one core" if n_jobs == 1 else f"with n_jobs = {n_jobs}"
    whole = ", with cache groups kept whole" if args.mode == "amortized" else ""
    src = source_config.stem
    ranked_from = (f"the {len(ids)} SPIs in `{src}`" if not missing
                   else f"the {len(ranked)} SPIs of `{src}` that the benchmark timed")
    header = (
        f"# {Path(output).stem}: about the cheapest {args.keep}% of {ranked_from}, ranked by\n"
        f"# {cost_note} at M = {M}, T = {T}\n"
        f"# {cores}{whole}.\n"
        f"# Current set: {len(kept)} SPIs (of the {len(ids)} in `{src}`).\n"
    )
    if missing:
        header += f"# The {len(missing)} SPIs of `{src}` that the benchmark did not time are left out.\n"
    header += "#\n"
    text = emit_config(source_config, records, kept, header,
                       preserve_dropped=args.preserve_dropped)

    Path(output).write_text(text)
    if promoted:
        print(f"[cut] {len(promoted)} SPI(s) kept to complete partially kept cache groups",
              file=sys.stderr)
    print(f"[cut] cutoff: fastest kept <= {kept_max:.3f}s; slowest dropped >= {drop_min:.3f}s",
          file=sys.stderr)
    for i in reversed(dropped):
        print(f"[cut] dropped {cost[i]:9.3f}s  {i}", file=sys.stderr)
    print(f"[cut] {len(kept)}/{len(ranked)} SPIs kept ({args.mode}, M={M} T={T}) -> {output}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
