# Contributing to pyspi

Thank you for helping improve _pyspi_.
This page sets out the conventions that keep the code and its documentation consistent.
They follow those of its sister package, [_pyhctsa_](https://github.com/DynamicsAndNeuralSystems/pyhctsa), so that the two read and behave alike.

For a step-by-step guide to adding a new SPI, see [Incorporating new SPIs](https://time-series-features.gitbook.io/pyspi/development/development/incorporating-new-spis).
For bug reports and the code of conduct, see [Contributing to pyspi](https://time-series-features.gitbook.io/pyspi/development/development/contributing-to-pyspi).

## Setting up

```bash
git clone https://github.com/DynamicsAndNeuralSystems/pyspi.git
cd pyspi
uv sync
uv run pytest -q                                          # fast suite
uv run pytest -m slow -q tests/test_baseline_drift.py     # drift suite: values against frozen baselines
```

Run both suites before opening a pull request.

## Code style

Names follow [PEP 8](https://peps.python.org/pep-0008/): `snake_case` for variables, functions and modules, `PascalCase` for classes, `ALL_CAPITALS` for constants.

Imports come in three groups, each sorted alphabetically: the standard library first, then third-party packages, then pyspi's own modules.

```python
import inspect
from functools import partial

import numpy as np
from scipy import stats

from pyspi.base import Directed, Unsigned, parse_bivariate
from pyspi.utils import require_int
```

Where an import must come first for a reason (an environment variable that has to be set before a library is imported, say), leave it in place with a one-line comment saying why.

## Docstrings

A docstring describes what the code computes now.
Docstrings use the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) layout, as in _pyhctsa_.

### SPI classes

Each SPI class has, in this order:

1. A summary line, on the line after the opening quotes, saying what the SPI is, with `[1]_` where a reference applies.
2. Prose covering:
   - what is computed, with the formula inline in double backticks;
   - the choices made (estimators, preprocessing, where pyspi departs from the published method or from an upstream library);
   - its properties;
   - its direction and sign: row = source, column = target;
   - caveats such as ties and the inputs that give NaN.
3. `References`, with one `.. [1]` entry per source: authors, title in quotes, journal, volume(issue), pages and year.
   Check each entry against the publisher's page or Crossref.
   No DOIs or URLs.
4. `Parameters`, one entry per constructor argument: `name : type, optional`, a description with the allowed values, and `Default: …`.
   Check each name, type and default against `__init__`.

SPI classes have no `Returns` section, since the prose gives the output's orientation.

```python
class ChatterjeeXi(Directed, Unsigned):
    """
    Chatterjee's rank correlation xi [1]_.

    For a source ``X`` and a target ``Y``, sort the pairs by ``X`` and let
    ``r_k`` be the rank of the k-th target in that order. Then
    ``xi_n(X, Y) = 1 - 3 sum_k |r_{k+1} - r_k| / (n^2 - 1)`` (no ties).

    It uses ranks only and tends to 0 if and only if ``Y`` is independent of
    ``X``, so it also detects non-monotone dependence such as ``Y = X^2``.

    Asymmetric and unsigned: ``A[i, j] = xi(X = x_i, Y = x_j)``, row = source.
    The value is NaN when the target is constant or a sample is not finite.

    References
    ----------
    .. [1] S. Chatterjee, "A new coefficient of correlation", Journal of the
           American Statistical Association 116(536), 2009-2022 (2021).

    Parameters
    ----------
    tau : int, optional
        Lag in samples, at least 0. Default: 0.
    """
```

(The class in `pyspi/statistics/basic.py` is the full version.)

### Functions and methods

Public functions and methods have a summary line and prose, then `Parameters` and `Returns` sections.
Base classes and private helpers need only a short plain docstring.

### What to leave out

- **History:** how the code reached its current form ("previously", "no longer", "changed to", "an earlier version").
  That belongs in commit messages and the CHANGELOG.
- **Development narratives:** simulation settings, scores or timings measured on particular datasets, "we tested X and found Y".
  A brief general reason is fine.
- **Long discussions of what else a measure responds to:** one plain sentence for a known confound is enough.
- **Pointers to test files or benchmark scripts.**
- **Display maths:** no `.. math::` blocks, and `:math:` only sparingly, since `help()` shows them as raw text.

## Comments

A comment is one or two lines, written only where the reason for the code isn't obvious from the code itself.
A note on where pyspi differs from an upstream implementation is worth keeping.

## How an SPI behaves

- **Orientation:** the output is an `M x M` matrix with row = source and column = target.
  The class's base (`Directed` or `Undirected`, `Signed` or `Unsigned`) and its config labels must agree with what it returns; `Signed` classes carry the `signed` label.
- **Input it can't handle:** a constant series, too few samples or a fit that doesn't exist gives NaN, not an exception.
  Raise an exception only for invalid arguments.
- **Randomness:** any random draw uses a fixed seed, so every value is reproducible.
- **Identifiers:** an SPI's identifier names its table column and its checkpoint files, so every configuration must have a unique one.
  Parameters are appended as `_<name>-<value>`, formatted with `pyspi.utils.fmt_param` and with dots written as dashes (`_nw-2-5`).
  A parameter added to an existing class is appended only when it differs from its default, so existing identifiers don't change.

## Configs

The bundled configs live in `pyspi/configs/`.
Each class entry carries its labels: directedness (`directed`, `undirected` or `antisymmetric`), `linear` or `nonlinear`, `bivariate` or `multivariate`, `contemporaneous`, `time-dependent` or `frequency-dependent`, and one module code (`M01` to `M14`).
An SPI not assigned to a module carries the code `MXX` instead.
Each config entry adds the labels that depend on its parameters.
Merged with the class's own labels, every SPI ends up with exactly one label from each of these groups, spelled as here.
Four optional flags say what an SPI's value is.
By default, with no flag, the value is a one-ended strength of dependence, higher meaning stronger, whatever values it takes: it answers "how much".
`timing` marks a value that is a time or phase offset, such as a lag in samples or a phase in radians, not a strength: it answers "when".
`profile` marks a value that describes how the dependence is distributed, over time or over the range of values, not how strong it is.
`signed` marks a value for which zero has a special meaning (no relationship, or no lead) and the sign says which of two opposite cases holds, so that its absolute value is the size of the effect, as for a covariance; it combines with the default, with `timing` (a lag whose sign says which process leads) and with `profile`.
`inverted` marks a one-ended strength whose scale runs the other way, so that lower means stronger dependence or stronger evidence, as for a distance, an entropy or the Engle-Granger t-statistic, whose values can be negative without the SPI being `signed`; it never combines with `signed`, `timing` or `profile`.
An SPI carries at most one of `timing` and `profile`, and `unsigned` is not a label.
`timing` says what the value is; `time-dependent` and the descriptive tag `lagged` say how the SPI is computed.
A class may also carry descriptive tags: `basic`, `causal`, `distance`, `embedding`, `infotheory`, `lagged`, `misc`, `normal`, `rank`, `spectral`, `symbolic`, `temporal`, `unordered` and `wavelet`.
Comments in a config are one or two lines per class.

## Tests and values

- **New SPI in a bundled config:** add drift baselines for it as a family file, using `tests/tools/generate_benchmark_tables.py`, and check that no other baseline array changes.
- **Change to a computed value:** say so in the pull request.
  Regenerate exactly the affected baseline arrays.
  Increment the revision of `COMPUTATION_VERSION` in `pyspi/_parallel.py`, with a one-line entry in the log above it.
- **Test names** describe what they test (`test_spectral_phase_reference.py`), not the work that prompted them.

## CHANGELOG

Each user-facing change gets a short entry in `CHANGELOG.md`, written relative to the last release.
In Markdown files, start each sentence on a new line, so that diffs show one changed sentence per line.

## Public text

The repository, its issues and its pull requests are public.
Code, comments, docstrings, commit messages and pull request descriptions describe what the change is and why, in terms any reader can follow.
Leave out local file paths, machine or cluster names, user names, job IDs, and labels for private experiments.
