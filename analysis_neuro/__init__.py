"""Package for constructing analyses and validations of neuroscience models."""
from . import plots
from . import terminology as terms
from .exceptions import TerminologyError, Assumption
from .measurement_utils import DATA_TERMS, measure, extract_parameters
from .analysis import Analysis
