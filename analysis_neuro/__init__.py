"""Package for constructing analyses and validations of neuroscience models."""

import dataframe_pointer

from .analysis import Analysis
from . import analyses, stats, plots
from . import terminology as terms
from .exceptions import Assumption, TerminologyError
