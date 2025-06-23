"""Package for constructing analyses and validations of neuroscience models."""

import dataframe_pointer

from analysis_neuro.analysis import Analysis

from . import analyses, plots, stats
from . import terminology as terms
from .exceptions import Assumption, TerminologyError
