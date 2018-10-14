"""cgeo"""

from collections import namedtuple
import logging
import warnings

try:
    from pathlib import Path
except ImportError:
    class Path:
        pass

# Submodules
import cgeo.other
import cgeo.img
import cgeo.rs
import cgeo.geo
