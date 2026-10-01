#!/usr/bin/env python
"""Cut a benchmarked SPI subset config from a bench_compute.py per-cell JSON.

Reads ONE per-cell JSON (``<label>_M<M>_T<T>_n<n>.json``), ranks SPIs by cost,
and emits a ``pyspi/configs/benchmarked_p<keep>.yaml`` containing only the
fastest ``--keep`` percent. (M, T, n_jobs) are read from the JSON itself, not
parsed from the filename.

Cost model — two modes:
  raw        each SPI's own measured wall time at this (M, T).
  amortized  (default) SPIs sharing a within-class cache (``_cache_namespace``
             — Covariance/Precision, multitaper spectral pairs, CCM,
             Cointegration, Barycenter, ...) split the group's total cost
             evenly:
                 cost(spi) = sum(group wall times) / (group size)
             The shared computation is built once and reused, so blaming its
             full cost to one variant overcounts. Ungrouped SPIs use raw time.

Usage:
    python -m bench.cut_config --bench-json bench/results/cells/physics_config_M16_T800_n1.json --keep 90
    python -m bench.cut_config --bench-json <path> --keep 80 --mode raw
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from bench._config_walk import cache_bucket, walk_spis
from pyspi.calculator import CONFIG_DIR, bundled_configs, resolve_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def _rel(path: Path) -> str:
    """Render a path relative to the repo root when possible.

    Generated config headers are committed, so they must not carry the
    absolute path of whichever checkout produced them.
    """
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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
    p.add_argument("--no-preserve-dropped", action="store_true",
                   help="Delete dropped variants/classes from the output instead of "
                        "commenting them out (default: preserve as comments).")
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
                preserve_dropped: bool = True) -> str:
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
              f"— kept unconditionally (cost 0): {', '.join(missing[:5])}"
              f"{' ...' if len(missing) > 5 else ''}", file=sys.stderr)
    raw = {i: raw_cell.get(i, 0.0) for i in ids}

    cost = raw if args.mode == "raw" else amortized_costs(records, raw)
    ranked = sorted(ids, key=lambda i: cost[i])
    n_target = round(args.keep / 100 * len(ranked))
    kept = set(ranked[:n_target])
    kept_max = cost[ranked[n_target - 1]] if n_target else 0.0
    drop_min = cost[ranked[n_target]] if n_target < len(ranked) else float("inf")

    # Cache-aware snap: promote partial buckets to fully-kept. The kept count
    # may exceed n_target by the size of the promoted partial buckets.
    if args.mode == "amortized":
        kept, promoted = snap_to_cache_buckets(records, kept)
    else:
        promoted = []
    dropped = [i for i in ranked if i not in kept]

    output = args.output or (
        CONFIG_DIR / f"benchmarked_p{args.keep}{'_raw' if args.mode == 'raw' else ''}.yaml")

    env = cell.get("environment") or {}
    sha = env.get("pyspi_git_sha") or "?"
    snap_note = (f"; +{len(promoted)} snapped from partial cache buckets"
                 if promoted else "")
    header = (
        f"# {Path(output).name}\n"
        f"# Generated by bench/cut_config.py on {datetime.now():%Y-%m-%d}.\n"
        f"# Source config : {_rel(source_config)}\n"
        f"# Bench JSON    : {args.bench_json.name} (pyspi {sha[:12]}, "
        f"M={M} T={T} n_jobs={n_jobs})\n"
        f"# Cost model    : {args.mode}\n"
        f"# Keep {args.keep}% : kept {len(kept)} / {len(ranked)} SPIs (target {n_target}{snap_note}), "
        f"dropped {len(dropped)}.\n"
        f"# Cutoff        : fastest kept <= {kept_max:.3f}s ; slowest dropped >= "
        f"{drop_min:.3f}s.\n#\n"
    )
    text = emit_config(source_config, records, kept, header,
                       preserve_dropped=not args.no_preserve_dropped)
    if dropped:
        text += "\n# --- DROPPED (slowest %d, %s cost) ---\n" % (len(dropped), args.mode)
        text += "".join(f"#   {cost[i]:9.3f}s  {i}\n" for i in reversed(dropped))

    Path(output).write_text(text)
    print(f"[cut] {len(kept)}/{len(ranked)} SPIs kept ({args.mode}, M={M} T={T}) -> {output}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
