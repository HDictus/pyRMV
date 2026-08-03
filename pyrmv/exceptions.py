"""Errors and warnings."""


class TerminologyError(ValueError):
    """Error in use of terminology."""


class Assumption(UserWarning):
    """A warning representing an assumption made during the analysis process."""
