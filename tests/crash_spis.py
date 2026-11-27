"""Two SPIs in one cache bucket, for tests/test_parallel.py.

A config names this module, so workers build these SPIs themselves.
`Finisher` runs first in the bucket and returns a value; `Crasher` then
kills its worker process, as a segfault or the OOM killer would.
"""

import os

import numpy as np

from pyspi.base import Undirected, Unsigned, parse_multivariate


class Finisher(Undirected, Unsigned):
    name = "Finishes before its bucket-mate kills the worker"
    identifier = "crash_test_finisher"
    labels = ["undirected", "unordered", "linear"]
    _cache_namespace = "crash_test"

    @parse_multivariate
    def multivariate(self, data):
        return np.abs(np.corrcoef(data.to_numpy(squeeze=True)))


class Crasher(Undirected, Unsigned):
    name = "Kills its worker process"
    identifier = "crash_test_crasher"
    labels = ["undirected", "unordered", "linear"]
    _cache_namespace = "crash_test"

    @parse_multivariate
    def multivariate(self, data):
        os._exit(1)
