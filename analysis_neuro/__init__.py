"""Package for constructing analyses and validations of neuroscience models."""

import dataframe_pointer

from . import analyses, mtypes, plots
from . import terminology as terms
from .analysis import Analysis
from .exceptions import Assumption, TerminologyError
