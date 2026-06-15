"""Shim de compatibilidade — preferir ``ml_core_ring``."""

import warnings

warnings.warn(
    "core.ml está obsoleto; importe de ml_core_ring.",
    DeprecationWarning,
    stacklevel=2,
)

from ml_core_ring import *  # noqa: F403, F401
from ml_core_ring import __all__  # noqa: F401
