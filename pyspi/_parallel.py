"""Parallel SPI execution backend for Calculator.compute().

Design:
- Dataset bytes are placed in a single multiprocessing.shared_memory block.
  Workers attach read-only; no per-worker pickling of the array.
- SPIs are bucketed by their ``_cache_namespace`` class attribute. Each bucket
  becomes one task assigned to a single worker, so a cache populated lazily on
  the Data object (e.g. data.spectral_bv, data.covariance) is reused across
  every variant in that bucket. SPIs without a tag run as single-SPI tasks.
- Per-SPI failures yield a NaN matrix; the rest of the run continues.
- If ``checkpoint_dir`` is set, each finished SPI is atomically written to
  ``<dir>/<identifier>.npy`` (and ``<identifier>.error`` on failure). A
  subsequent run with ``resume=True`` will load these and skip the SPIs.
- Progress is per-SPI, not per-bucket: workers post a lightweight event to a
  shared queue after each SPI so the tqdm bar advances one tick per SPI and
  a stuck SPI is visible (the bar stalls). Results themselves still travel
  back via the futures, which also surfaces a hard worker crash.
"""

from __future__ import annotations

import multiprocessing as mp
import multiprocessing.shared_memory as shm
import concurrent.futures as cf
import os
import queue as _queue
import sys
import time
import warnings
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from ._logging import get_logger

logger = get_logger("pyspi.parallel")


def default_mp_context() -> str:
    """Best start method for the current platform.

    fork on Linux: workers inherit the parent's already-imported modules and
    instantiated SPIs via copy-on-write, so worker startup is near-instant.
    Measured ~2x faster end-to-end than spawn for a 252-SPI run.

    spawn elsewhere: fork is unsafe on macOS (Accelerate/CoreFoundation after
    init) and absent on Windows; spawn re-imports per worker but is correct.
    """
    return "fork" if sys.platform.startswith("linux") else "spawn"

# Worker-local state populated by _worker_init. Module globals are safe here
# because each worker process has its own independent copy.
_WORKER_STATE: dict = {}


def _attach_data(shm_name, shape, dtype_str, procnames, name):
    """Build a Data object that views an existing shared-memory block.

    The shared array is the parent's already-normalised/detrended ``_dataset._data``,
    so we bypass Data.__init__ to avoid re-applying those transforms.
    """
    from pyspi.data import Data

    shared = shm.SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shared.buf)
    data = Data._from_prepared_array(arr, procnames=procnames, name=name)
    return data, shared


_BLAS_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
)


def _pin_blas_env() -> None:
    """Force BLAS/threading env vars to 1 in the parent, before workers spawn.

    threadpool_limits (see _pin_worker_thread_pools) pins OpenBLAS/MKL/OpenMP
    at runtime; these env vars additionally cover numba and — for spawn
    workers, which import numpy before _worker_init runs — make BLAS start
    single-threaded from process start. Caveat: macOS Accelerate only partly
    honours these (its vDSP/FFT path threads independently of any documented
    env var), so on macOS n_jobs>1 can still oversubscribe FFT-heavy SPIs;
    Linux (OpenBLAS/MKL) is fully covered.
    """
    for var in _BLAS_ENV_VARS:
        os.environ[var] = "1"


def available_cores() -> int:
    """Cores this process may actually use.

    ``sched_getaffinity`` honours cgroup/cpuset pinning, so under PBS or Slurm
    this returns the cores the scheduler actually granted -- not the machine's
    physical core count. That distinction is the whole point of the check in
    :func:`guard_oversubscription`.
    """
    try:
        return len(os.sched_getaffinity(0))  # Linux
    except AttributeError:
        return os.cpu_count() or 1


def _requested_threads() -> int:
    """Largest thread count any BLAS/OpenMP backend has been told to use."""
    counts = [1]
    for var in _BLAS_ENV_VARS:
        try:
            counts.append(int(os.environ.get(var, "1") or 1))
        except ValueError:
            pass
    return max(counts)


def guard_oversubscription(n_jobs: int) -> None:
    """Warn -- or intervene -- when threads x processes exceeds the cores we hold.

    The dangerous case is dataset-level parallelism on a cluster: many
    single-core pyspi processes, each inheriting a site-wide
    ``OMP_NUM_THREADS=8``, so a 48-core node runs 384 threads and thrashes.
    ``pyspi/__init__`` only *defaults* the variable to 1, so an inherited value
    survives by design -- a user who sets it deliberately should keep it.

    When the scheduler granted exactly one core, more than one thread is never
    right, so that case is pinned outright. Anything else only warns, since
    pyspi cannot see how many sibling processes the scheduler started.
    """
    threads = _requested_threads()
    cores = available_cores()
    requested = n_jobs * threads

    # pyEDM (ConvergentCrossMapping) self-parallelises over *processes*, so no
    # BLAS thread count would cover it. It no longer needs covering here: its
    # pools are unconditionally off at the call site, because pyEDM 2.5 starts
    # them with forkserver/spawn and so re-imports the caller's __main__ (see
    # statistics/causal.py).

    if requested <= cores:
        return

    if cores == 1 and threads > 1:
        _pin_blas_env()
        try:
            from threadpoolctl import threadpool_limits
            global _THREADPOOL_LIMITER
            _THREADPOOL_LIMITER = threadpool_limits(limits=1)
        except ImportError:
            pass
        logger.warning(
            "Only 1 core is available to this process but the BLAS thread count "
            "is %d; pinned it to 1. This is the usual symptom of a scheduler "
            "array job inheriting a site-wide OMP_NUM_THREADS -- set "
            "OMP_NUM_THREADS=1 in your job script to silence this.",
            threads,
        )
        return

    logger.warning(
        "Oversubscription: n_jobs=%d x %d BLAS thread(s) = %d workers for %d "
        "available core(s). If you are running one dataset per process, set "
        "OMP_NUM_THREADS=1; if you meant to parallelise within this dataset, "
        "lower n_jobs.",
        n_jobs, threads, requested, cores,
    )


_THREADPOOL_LIMITER = None  # module-global so the limiter is never GC'd


def _pin_worker_thread_pools():
    """Pin every nested thread/process pool to 1 so process workers don't oversubscribe.

    n_jobs workers each running a library that itself spawns cpu_count() threads
    = quadratic blow-up. Pinning BLAS alone is not enough. The pools:
      - BLAS + OpenMP (numpy/scipy/sklearn): threadpool_limits, all user APIs.
    That is now the whole list. cdt used to be here (it autoset SETTINGS.NJOBS
    to cpu_count() at import) and torch with it, but both are gone: the four
    pairwise causal scores are in pyspi/lib/pairwise_causal.py, and torch never
    drove any pyspi computation -- InterDependenceScore is NumPy, and torch
    entered only because cdt eagerly imports its Torch-backed models.

    pyEDM (drives ConvergentCrossMapping) is process-based, not thread-based, so
    it can't be pinned here — it is instead switched off unconditionally at the
    call site in statistics/causal.py, for correctness rather than scheduling.
    """
    global _THREADPOOL_LIMITER
    try:
        from threadpoolctl import threadpool_limits
        _THREADPOOL_LIMITER = threadpool_limits(limits=1)  # blas + openmp
    except ImportError:
        pass


def _worker_init(shm_name, shape, dtype_str, procnames, ds_name, configfile, progress_q,
                 config_bytes=None):
    """ProcessPoolExecutor initializer. Runs once per worker.

    Re-instantiates SPIs from the configfile (some SPI classes use closures in
    ``__init__`` that aren't picklable, so we can't ship instances across the
    process boundary).
    """
    data, shared = _attach_data(shm_name, shape, dtype_str, procnames, ds_name)

    # Direct call to the shared loader — no throwaway Calculator instantiation,
    # no stdout suppression needed.
    from pyspi.calculator import load_spis_from_yaml
    # Load from the parent's snapshot, not the path: the file on disk may have
    # changed since the parent instantiated its SPIs, which would bind results
    # to parameters that did not produce them.
    if config_bytes is not None:
        import tempfile, os as _os
        fd, tmp = tempfile.mkstemp(suffix=".yaml")
        try:
            with _os.fdopen(fd, "wb") as fh:
                fh.write(config_bytes)
            spis = load_spis_from_yaml(tmp)
        finally:
            _os.unlink(tmp)
    else:
        spis = load_spis_from_yaml(configfile)

    # Pin nested thread pools AFTER the SPI modules import: a library that
    # sizes its pool at import time has to be pinned once it exists.
    _pin_worker_thread_pools()

    _WORKER_STATE["data"] = data
    _WORKER_STATE["shm"] = shared
    _WORKER_STATE["spis"] = spis
    _WORKER_STATE["progress_q"] = progress_q


def _run_task(spi_keys, checkpoint_dir):
    """Compute a bucket of SPIs sequentially in this worker.

    Posts ``(key, failed)`` to the progress queue after each SPI, and returns
    the list of ``(key, matrix, error_str_or_None, warnings, elapsed)`` tuples.
    """
    data = _WORKER_STATE["data"]
    spis = _WORKER_STATE["spis"]
    progress_q = _WORKER_STATE["progress_q"]
    M = data.n_processes
    out = []
    for key in spi_keys:
        S, err, warns, elapsed = run_spi(spis[key], data, key, M)
        write_checkpoint(checkpoint_dir, key, S, err)
        # Warnings travel back to the parent rather than being emitted (and
        # lost) here in the worker process.
        out.append((key, S, err, warns, elapsed))
        if progress_q is not None:
            progress_q.put((key, err is not None))
    return out


def run_spi(spi, data, key, M):
    """Compute one SPI, validate it, and capture failures and warnings.

    The single execution primitive shared by the serial and parallel paths.
    Previously each path had its own copy: only the parallel one validated the
    returned shape, and only the parallel one suppressed warnings, so the two
    modes disagreed about what counted as a failure and about what the caller
    got to see.

    Returns ``(S, err, warns, elapsed)`` where ``err`` is ``None`` on success
    and ``warns`` is a list of formatted warning strings raised during the
    computation (returned rather than emitted, so the parallel path can
    re-emit them in the parent process).

    A result is a failure if it raises, has the wrong shape, contains an
    infinity, or is entirely NaN off the diagonal. The last case previously
    passed silently, which is how three group-delay SPIs shipped as all-NaN
    columns with no warning attached.
    """
    t0 = time.perf_counter()
    err = None
    warns: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            S = spi.multivariate(data)
        warns = [f"{w.category.__name__}: {w.message}" for w in caught]

        S = np.array(S, dtype=float, copy=True)
        if S.shape != (M, M):
            raise ValueError(f"SPI returned shape {S.shape}, expected ({M},{M})")
        np.fill_diagonal(S, np.nan)

        offdiag = S[~np.eye(M, dtype=bool)]
        if offdiag.size:
            if np.isinf(offdiag).any():
                raise ValueError("SPI returned infinite value(s)")
            if not np.isfinite(offdiag).any():
                raise ValueError("SPI returned no finite off-diagonal values")
    except Exception as e:
        S = np.full((M, M), np.nan)
        err = f"{type(e).__name__}: {e}"

    return S, err, warns, time.perf_counter() - t0


def write_checkpoint(checkpoint_dir, key, S, err):
    """Persist one SPI result plus its error sidecar, atomically."""
    if checkpoint_dir is None:
        return
    d = Path(checkpoint_dir)
    _atomic_npy_write(d / f"{key}.npy", S)
    err_path = d / f"{key}.error"
    if err is not None:
        err_path.write_text(err)
    elif err_path.exists():
        err_path.unlink()


def _atomic_npy_write(path: Path, arr: np.ndarray) -> None:
    """Atomic write: numpy.save then os.replace. POSIX rename is atomic.

    Note: ``np.save(path, arr)`` auto-appends ``.npy`` if absent — that
    rewrites our ``.npy.tmp`` to ``.npy.tmp.npy`` and breaks the rename.
    Passing a file handle bypasses that behaviour.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def cache_bucket(spi):
    """``(namespace, *_cache_subkey)`` -- the key that actually shares a cache.

    The one definition of "these SPIs share work", used by the scheduler here,
    by ``calculator.warn_partial_cache_buckets`` and by ``bench/cut_config.py``.
    ``None`` for an SPI that caches nothing.

    Namespace alone is too coarse. ``_cache_subkey`` splits a namespace into
    independent caches -- ``Barycenter`` caches per mode, so ``bary_dtw`` and
    ``bary_softdtw`` share nothing, and the multitaper spectral SPIs cache per
    class and sampling frequency. On the ``full`` config the namespaces divide
    as: spectral_mv 84 SPIs across 16 independent caches, covariance 32 across
    8, spectral_bv 30 across 5, barycenter 16 across 4, coint 11 across 7,
    ccm 9 across 3.
    """
    ns = getattr(type(spi), "_cache_namespace", None)
    if ns is None:
        return None
    return (ns, *tuple(getattr(spi, "_cache_subkey", ())))


# Amortized cost per SPI at the M=16, T=800 anchor cell
# (bench/results/analysis/report.md). Used only to decide which task a worker
# picks up first, so a stale or missing entry costs some makespan and nothing
# else -- it cannot change a computed value. Anything unlisted is treated as
# cheap.
_NAMESPACE_COST = {"ccm": 292.7, "barycenter": 11.1, "spectral_bv": 1.6,
                   "spectral_mv": 0.3, "coint": 0.3, "covariance": 0.3}


def build_tasks(spi_keys, spis) -> list[list[str]]:
    """Bucket SPI keys by the cache they actually share.

    SPIs in one bucket form a multi-SPI task, so the cached intermediate is
    built once on the worker's Data and reused; SPIs that cache nothing become
    single-SPI tasks.

    Bucketing by ``_cache_namespace`` alone -- as this did -- serialises SPIs
    that share no cache at all. On ``full`` it produced one 84-member
    ``spectral_mv`` task covering 16 independent caches, and the longest task
    is what bounds the makespan: no amount of parallelism could split it.
    Bucketing on ``cache_bucket`` gives 43 shareable groups whose largest has
    24 members.

    Ordering is by estimated cost (members times the namespace's measured
    amortized cost) rather than member count, so a 3-member ``ccm`` bucket
    starts before a 24-member ``covariance`` one. This affects scheduling only;
    every task is computed identically whichever order it runs in.
    """
    cacheless: list[list[str]] = []
    grouped: dict[tuple, list[str]] = defaultdict(list)
    for key in spi_keys:
        bucket = cache_bucket(spis[key])
        if bucket is None:
            cacheless.append([key])
        else:
            grouped[bucket].append(key)

    def cost(item):
        bucket, keys = item
        return len(keys) * _NAMESPACE_COST.get(bucket[0], 0.1)

    grouped_tasks = [keys for _, keys in
                     sorted(grouped.items(), key=cost, reverse=True)]
    return grouped_tasks + cacheless


MANIFEST_NAME = "run.json"
SCHEMA_VERSION = 1
# Bumped when a change alters computed values, so checkpoints cannot outlive the
# algorithm that produced them, and so `Calculator.run_digest` separates results
# that differ only by implementation. `<release>.r<revision>`: the revision is a
# counter within a release, not a version of anything -- only equality is ever
# tested.
#   r2 -- the KSG per-occurrence dither, full MAX_CORR_AIS, the group-delay
#         sample scaling and |r|, and the cross-correlation threshold returning
#         zero when nothing clears it.
#   r3 -- the KSG dither key is taken from the normalised column rounded to 12
#         decimals rather than its exact bytes. On tie-free data that changes
#         nothing (only the integer neighbour counts enter the estimate), but on
#         tied or quantised data a different dither separates the ties
#         differently, so values there can move.
#   r4 -- KSG uses the rounded coordinate in its geometry, canonicalises
#         reflections, and assigns numerical key collisions by content rather
#         than coordinate order. Tied/quantised and tolerance-boundary values
#         can move.
#   r5 -- KSG refuses tied coordinates and no longer rounds or dithers valid
#         continuous inputs. This removes arbitrary order-dependent tie
#         breaking and the artificial 12-decimal conditioning boundary.
#   r6 -- KSG strict marginal counts use the immediately preceding
#         representable radius instead of shrinking epsilon by 1e-10.
#   r7 -- coherence phase uses a circular mean and refuses zero-resultant and
#         antipodal locations that have no unique signed float orientation.
COMPUTATION_VERSION = "3.0.0.r7"


def read_manifest(checkpoint_dir: Path):
    """Return the manifest dict for a checkpoint directory, or None."""
    path = Path(checkpoint_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write_manifest(checkpoint_dir: Path, digest: str, spec: dict) -> None:
    """Record which run owns this checkpoint directory."""
    path = Path(checkpoint_dir) / MANIFEST_NAME
    payload = {"schema": SCHEMA_VERSION, "computation": COMPUTATION_VERSION,
               "digest": digest, "spec": spec}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


def checkpoint_owner_matches(checkpoint_dir: Path, digest: str):
    """Return (matches, reason). A directory with no manifest is unowned.

    An unowned directory is treated as a mismatch rather than a match: it was
    written by a version that did not record provenance, and there is no way to
    tell whether it belongs to this run.
    """
    manifest = read_manifest(checkpoint_dir)
    if manifest is None:
        # Nothing written yet is fine; a populated directory without a manifest
        # is not.
        existing = any(Path(checkpoint_dir).glob("*.npy"))
        if not existing:
            return True, None
        return False, "checkpoint directory has results but no run manifest"
    if manifest.get("schema") != SCHEMA_VERSION:
        return False, (
            f"manifest schema {manifest.get('schema')!r} != {SCHEMA_VERSION}"
        )
    if manifest.get("computation") != COMPUTATION_VERSION:
        return False, (f"checkpoint computed by pyspi {manifest.get('computation')!r}, "
                       f"not {COMPUTATION_VERSION!r}")
    if manifest.get("digest") != digest:
        return False, "checkpoint was written by a different run"
    return True, None


def load_checkpoints(checkpoint_dir: Path, spi_keys, M: int, retry_failed: bool = True):
    """Return (done_results, remaining_keys).

    done_results: dict[key] -> (matrix, error_or_None, warns, 0.0).
    A key is considered done if ``<key>.npy`` exists and has shape (M, M).

    A checkpoint carrying an ``.error`` sidecar records a *failed* SPI. By
    default those are retried rather than resumed: a failure is usually caused
    by something transient or since-fixed, and silently inheriting a NaN column
    from a previous run is the outcome resume is least likely to be wanted for.
    Pass ``retry_failed=False`` to resume them as-is.
    """
    done: dict = {}
    remaining: list = []
    for key in spi_keys:
        npy = checkpoint_dir / f"{key}.npy"
        if not npy.exists():
            remaining.append(key)
            continue
        try:
            arr = np.load(npy)
        except Exception:
            remaining.append(key)
            continue
        if arr.shape != (M, M):
            remaining.append(key)
            continue
        err_path = checkpoint_dir / f"{key}.error"
        err = err_path.read_text() if err_path.exists() else None
        # Validate *before* the retry decision, so an invalid matrix is retried
        # rather than being marked failed and then kept.
        off = arr[~np.eye(M, dtype=bool)] if M > 1 else arr.ravel()
        if off.size and (np.isinf(off).any() or not np.isfinite(off).any()):
            err = err or "ValueError: checkpoint contains no finite values"
        if err is not None and retry_failed:
            remaining.append(key)
            continue
        done[key] = (arr, err, [], 0.0)
    return done, remaining


def run_parallel(
    spis: dict,
    dataset,
    spi_keys: list[str],
    n_jobs: int,
    mp_context: str,
    checkpoint_dir: Optional[Path],
    progress: bool,
    configfile: str,
    config_bytes: bytes | None = None,
) -> dict:
    """Execute ``spi_keys`` across ``n_jobs`` workers; return dict[key] -> (S, err, elapsed)."""
    from tqdm import tqdm

    # Pin BLAS env before any worker spawns — workers inherit single-threaded
    # BLAS from process start (the only lever for macOS Accelerate).
    _pin_blas_env()

    arr = np.ascontiguousarray(dataset._data)
    M = arr.shape[0]
    tasks = build_tasks(spi_keys, spis)
    cp_str = str(checkpoint_dir) if checkpoint_dir is not None else None

    # Create resources inside the try so a failure constructing either one
    # (e.g. mp.Manager() raising) still runs the cleanup in finally.
    shared = None
    manager = None
    try:
        shared = shm.SharedMemory(create=True, size=arr.nbytes)
        manager = mp.Manager()
        progress_q = manager.Queue()
        shared_view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shared.buf)
        shared_view[:] = arr

        ctx = mp.get_context(mp_context)
        results: dict = {}
        pbar = tqdm(total=len(spi_keys), desc="SPIs", disable=not progress)
        with cf.ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(
                shared.name, arr.shape, str(arr.dtype),
                list(dataset.procnames), getattr(dataset, "_name", None),
                configfile, progress_q, config_bytes,
            ),
        ) as ex:
            future_to_task = {ex.submit(_run_task, task, cp_str): task for task in tasks}
            pending = set(future_to_task)
            while pending:
                # Per-SPI progress ticks (cosmetic; bar stalls on a stuck SPI).
                while True:
                    try:
                        key, _failed = progress_q.get_nowait()
                        pbar.update(1)
                        pbar.set_postfix_str(key[:32])
                    except _queue.Empty:
                        break
                # Harvest finished futures (source of truth for results).
                done = {f for f in pending if f.done()}
                for fut in done:
                    task = future_to_task[fut]
                    try:
                        for key, S, err, warns, elapsed in fut.result():
                            results[key] = (S, err, warns, elapsed)
                    except Exception as exc:  # worker process died (segfault/OOM)
                        for key in task:
                            results.setdefault(
                                key,
                                (np.full((M, M), np.nan), f"worker died: {exc}", [], 0.0),
                            )
                pending -= done
                if pending:
                    time.sleep(0.05)
        # Drain any progress events that arrived after the last poll, then
        # hard-sync the bar (a dead worker can leave it a few ticks short).
        while True:
            try:
                progress_q.get_nowait()
                pbar.update(1)
            except _queue.Empty:
                break
        pbar.n = len(results)
        pbar.refresh()
        pbar.close()
        return results
    finally:
        if manager is not None:
            manager.shutdown()
        if shared is not None:
            shared.close()
            try:
                shared.unlink()
            except FileNotFoundError:
                pass
