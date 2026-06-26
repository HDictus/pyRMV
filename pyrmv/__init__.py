"""Package for constructing analyses and validations of neuroscience models."""

import dataframe_pointer

from pyrmv.analysis import Analysis

from . import analyses, plots, stats
from . import terminology as terms
from .exceptions import Assumption, TerminologyError
