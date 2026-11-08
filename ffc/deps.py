"""Locate the reference code this package builds on and put it on ``sys.path``.

functional_flow_climate reuses, instead of re-implementing:

* ``pgcfm`` from physical_guide_flowmatching (pg-flow): the KAI 67-channel ERA5 contract, the
  primitive-equation residuals on the sphere, the ClimODE-style tendency core (the climate-ODE
  baseline), the WeatherBench2 scoring wrappers and the resumable training loop.
* ``neuralop`` from neuraloperator: the (spherical) Fourier neural operator layers.

Both are git submodules under ``third_party/`` pinned to the commit the experiments ran with.
A sibling checkout (``../physical_guide_flowmatching``, ``../neuraloperator``) or the environment
variables ``PGFLOW_ROOT`` / ``NEURALOP_ROOT`` are accepted too, and an installed ``neuralop``
package is the last resort.
"""

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

_CANDIDATES = {
    "pgcfm": ("PGFLOW_ROOT", "third_party/pgflow", "../physical_guide_flowmatching"),
    "neuralop": ("NEURALOP_ROOT", "third_party/neuraloperator", "../neuraloperator"),
}


def _root(name: str):
    env, sub, sibling = _CANDIDATES[name]
    for cand in (os.environ.get(env), os.path.join(ROOT, sub), os.path.join(ROOT, sibling)):
        if cand and os.path.isdir(os.path.join(cand, name)):
            return os.path.abspath(cand)
    return None


def pgflow_root() -> str:
    r = _root("pgcfm")
    if r is None:
        raise ImportError("pgcfm not found: run `git submodule update --init third_party/pgflow` "
                          "or set PGFLOW_ROOT to a physical_guide_flowmatching checkout")
    return r


def setup() -> None:
    """Idempotent: add the reference checkouts to ``sys.path`` (before any installed copy)."""
    for name in _CANDIDATES:
        r = _root(name)
        if r is not None and r not in sys.path:
            sys.path.insert(0, r)
    importlib.invalidate_caches()


setup()
