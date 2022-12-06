"""Package for constructing analyses and validations of neuroscience models."""
import inspect
from collections.abc import Callable
from pathlib import Path
import pandas as pd
from . import terminology as terms

DATA_TERMS = [terms.DATASET, terms.CITATION, terms.NOTES]


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
            print(arg, params)
            return False
    return True


class Analysis:
    """An object for defining analyses.

    Analysis is NOT a base class, instead it uses callables provided as
    arguments to compose a specific *instance* of an analysis.
    this analysis can then be run on models.

    These components, provided at initialization are as follows:

    measurement: a string describing the measurement to analyze.
       should have a corresponding entry in analysis_neuro.measurements
    observations: parameters or parameterized experimental data to use
       for the measurements.
    stats (optional): a callable for statistical tests accepting :
        (measurement_data, dependent, independent, compare)
        and returning a dict of {<hypothesis description>: dataframe}
        where the dataframe contains the test statistic for the hypotheses
        tested
    plotter (optional): a callable for plotting following the seaborn convention
         accepting x, y, and hue and returning a figure
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
        self.measurement = measurement
        exclude_from_parameters = [measurement] + DATA_TERMS

        if not isinstance(observations, pd.DataFrame):
            thispath = Path(__file__).parent
            examplepath = thispath / "analyses" / "data" / "schuz_neuron_density_1989.csv"

            raise ValueError(
                "observations must be a pandas.DataFrame of the form:\n"
                f"|parameter1|parameter2|...|measured_quantity|{terms.DATASET}|\n"
                "|value     | value    |...|measured value   | <some name>   |\n"
                "|...       |...       |...|...              |               |\n"
                "namely, the columns represent the measurement and its parameterization\n"
                f"see {str(examplepath)} for example data")

        self.parameters = observations[
            [col for col in observations if col not in exclude_from_parameters]
        ]
        self.observations = observations

        if not _check_callable(plotter, ['x', 'y', 'hue']):
            raise ValueError(
                "plotter must be a callable of the form:\n"
                "(x, y, hue) -> matplotlib.pyplot.Figure")

        self.plotter = plotter

        if not _check_callable(stats, ['data', 'dependent', 'independent', 'compare']):
            raise ValueError(
                "stats must be a callable of the form:\n"
                "(data, dependent, independent, compare) -> {hypothesis: pd.DataFrame}"
                "where hypothesis is a string describing the hypothesis tested")

        self.stats = stats

        self.verdict = verdict
        self.doc = doc

    def measure(self, model):
        """Measure the required measurements on model."""
        method = terms.measurements[self.measurement]["method name"]
        measured = getattr(model, method)(self.parameters)
        # the model's label should be included to distinguish it from other
        # models and experimental data
        measured[terms.DATASET] = model.label
        return measured

    @property
    def varying_parameters(self):
        """Parameters which are not constants."""
        return [
            col
            for col in self.parameters.columns
            if len(self.parameters[col].unique()) > 1
        ]

    def __call__(self, *models):
        """Run this analysis instance on a model."""
        to_concat = [self.measure(model) for model in models]
        # if observations represents experimental values, we want to
        # include those in the dataframe
        if self.measurement in self.observations:
            to_concat = [self.observations] + to_concat
        measurements = pd.concat(to_concat)
        report = {
            "Introduction": self.doc,
            "measurements": measurements,
            "stats": "TODO: not yet implemented",
            "verdict": "TODO: not yet implemented",
        }
        if self.plotter is not None:
            # we will need to expand on this behavior as we try to support more
            # maybe by inspecting the plotter's signature?
            # otherwise, we may need to define a different plotting interface
            # this will mean you can't pass seaborn functions directly,
            # instead we will need to wrap them
            dependent = measurements[self.measurement]
            independent = _join_columns(measurements[self.varying_parameters])
            compare = measurements[terms.DATASET]
            axis = self.plotter(x=independent, y=dependent, hue=compare)
            report["figures"] = axis.get_figure()

        return report

    def with_fields(self, **fields):
        """Duplicate this analysis, overwriting some fields."""
        current_fields = dict(
            measurement=self.measurement,
            observations=self.observations,
            plotter=self.plotter,
            stats=self.stats,
            verdict=self.verdict,
            doc=self.doc,
        )
        for key, value in fields.items():
            current_fields[key] = value
        return self.__class__(**current_fields)
