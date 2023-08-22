"""Analysis class for composition of analyses and validations."""
import inspect
from collections.abc import Callable

import pandas as pd
from lazy import lazy

from . import terminology as terms
from .measurement_utils import (
    validate_measurement,
    validate_observations,
    measure,
    extract_parameters
)



def _join_columns(dataframe):
    """Combine the columns of dataframe into a single series."""
    series_name = ", ".join(dataframe.columns)
    values = [" ".join(str(v) for v in row) for row in dataframe.values]
    return pd.Series(values, name=series_name)


def _check_callable(obj, args):
    """Check that obj is a callable which takes args."""
    if obj is None:
        # no object was actually provided to Analysis constructor
        # ignore it
        return True
    if not isinstance(obj, Callable):
        return False
    params = inspect.signature(obj).parameters
    for param in params.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    for arg in args:
        if arg not in params:
            return False
    return True


class Analysis:
    """An object for defining analyses.

    Analysis is NOT a base class, instead it uses callables provided as
    arguments to compose a specific *instance* of an analysis.
    this analysis can then be run on models.

    These components, provided at initialization are as follows:

    measurement: a string describing the measurement to analyze.
       should have a corresponding entry in analysis_neuro.terminology.measurements
    observations: parameters or parameterized experimental data to use
       for the measurements.
    stats (optional): a callable for statistical tests accepting :
        (measurement_data, dependent, independent, compare)
        and returning a dict of {<hypothesis description>: dataframe}
        where the dataframes contain the test statistics for the hypotheses
        tested
    plotter (optional): a callable for plotting in one of two forms:
        (x: pd.Series, y: pd.Series, hue: pd.Series) -> figure
        (several plots from the seaborn package satisfy this)
        OR:
        (data: pd.DataFrame, dependent: str, independent: list[str], compare: str)
    verdict (optional): a callable for rendering verdicts on hypotheses
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        measurement,
        observations,
        stats=None,
        plotter=None,
        verdict=None,
        doc=None,
    ):
        """Initialize an Analysis from various components."""
        validate_measurement(measurement)
        self.measurement = measurement


        validate_observations(observations)
        self.observations = observations

        if not (
            _check_callable(plotter, ["x", "y", "hue"])
            or _check_callable(plotter, ["data", "dependent", "independent", "compare"])
        ):
            raise ValueError(
                "plotter must be a callable of the form:\n"
                "(x, y, hue) -> matplotlib.pyplot.Axis OR\n",
                "(data, dependent, independent, compare) -> "
                "matplotlib.pyplot.Figure",
            )

        self.plotter = plotter

        if not _check_callable(stats, ["data", "dependent", "independent", "compare"]):
            raise ValueError(
                "stats must be a callable of the form:\n"
                "(data, dependent, independent, compare) -> {hypothesis: pd.DataFrame}"
                "where hypothesis is a string describing the hypothesis tested"
            )

        self.stats = stats
        if not _check_callable(verdict, ["stats"]):
            raise ValueError(
                "verdict must be a callable of the form:\n"
                "(stats: dict[str: DataFrame]) -> dict[str: str]"
                "Where stats is a dict with hypotheses as the keys and "
                "dataframes of test statistics as the values, and the output"
                "is a dict of hypotheses and the associated verdict "
                "( e.g. 'Pass', 'Fail')"
            )
        self.verdict = verdict
        self.doc = doc

    def measure(self, model):
        """Measure the required measurements on model.

        Model must have the method required to measure self.measurement
        see analysis_neuro.terminology.measurements
        """
        measured = measure(
            model, measurement=self.measurement, parameters=self.parameters
        )
        # the model's label should be included to distinguish it from other
        # models and experimental data
        measured[terms.DATASET] = model.label
        return measured

    @lazy
    def parameters(self):
        """Determine validation parameters from provided observations."""
        return extract_parameters(self.observations, self.measurement)

    @lazy
    def varying_parameters(self):
        """Parameters which are not constants."""
        return [
            col
            for col in self.parameters.columns
            if len(self.parameters[col].unique()) > 1
        ]

    def statistical_tests(self, measurements):
        """Run the statistical tests for this analysis on some data."""
        if self.stats is None:
            return "No statistical tests performed"
        return self.stats(
            data=measurements,
            dependent=self.measurement,
            independent=self.varying_parameters,
            compare=terms.DATASET,
        )

    def __call__(self, *models):
        """Run this analysis instance on a model."""
        to_concat = [self.measure(model) for model in models]

        # if observations represents experimental values, we want to
        # include those in the dataframe
        if self.measurement in self.observations:
            to_concat = [self.observations] + to_concat

        measurements = pd.concat(to_concat)
        stats = self.statistical_tests(measurements)
        verdict = "No verdict rendered" if self.verdict is None else self.verdict(stats)
        report = {
            "Introduction": self.doc,
            "measurements": measurements,
            "stats": stats,
            "verdict": verdict,
        }
        if self.plotter is not None:
            if _check_callable(self.plotter, ["x", "y", "hue"]):
                dependent = measurements[self.measurement]
                independent = _join_columns(measurements[self.varying_parameters])
                compare = measurements[terms.DATASET]
                figure = self.plotter(
                    x=independent, y=dependent, hue=compare
                ).get_figure()
            else:
                figure = self.plotter(
                    data=measurements,
                    dependent=self.measurement,
                    independent=self.varying_parameters,
                    compare=terms.DATASET,
                )
            report["figures"] = figure

        return report

    def with_fields(self, **fields):
        """Duplicate this analysis, overwriting some fields."""
        current_fields = {
            'measurement': self.measurement,
            'observations': self.observations,
            'plotter': self.plotter,
            'stats': self.stats,
            'verdict': self.verdict,
            'doc': self.doc,
        }
        current_fields.update(fields)
        return self.__class__(**current_fields)
