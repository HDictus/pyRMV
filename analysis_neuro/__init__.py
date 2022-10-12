"""Package for constructing analyses and validations of neuroscience models."""
import pandas as pd
from . import terminology as terms

_measurements = {
    terms.CELL_DENSITY: {"method name": "cell_density"},
    terms.CELL_COUNT: {"method name": "cell_count"},
}


def _join_columns(dataframe):
    """Combine the columns of dataframe into a single series."""
    series_name = ", ".join(dataframe.columns)
    values = [" ".join(str(v) for v in row) for row in dataframe.values]
    return pd.Series(values, name=series_name)


class Analysis:
    """
    An object for defining analyses.

    Analysis is NOT a base class, instead it uses callables provided as
    arguments to compose a specific *instance* of an analysis.
    this analysis can then be run on models.

    These components, provided at initialization are as follows:

    measurement: a string describing the measurement to analyze.
       should have a corresponding entry in analysis_neuro.measurements
    observations: parameters or parameterized experimental data to use
       for the measurements.
    stats (optional): a callable for statistical tests accepting :
        (measurement_data, dependent_var, independent_vars, compared_vars)
        and returning a dict of {<hypothesis description>: dataframe}
        where the dataframe contains the test statistic for the hypotheses
        tested
    plotter (optional): a callable for plotting accepting:
         (measurement_data, dependent_var, independent_vars, compared_vars)
         and returning a dict of {'caption': 'figure'}
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
        exclude_from_parameters = [
            measurement,
            terms.DATASET,
            terms.CITATION,
            terms.NOTES,
        ]
        self.parameters = observations[
            [col for col in observations if col not in exclude_from_parameters]
        ]
        self.observations = observations
        self.plotter = plotter
        self.stats = stats
        self.verdict = verdict
        self.doc = doc

    def measure(self, model):
        """Measure the required measurements on model."""
        method = _measurements[self.measurement]["method name"]
        measured = getattr(model, method)(self.parameters)
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

    def statistical_tests(self, measurements):
        """
        Run the statistical tests for this analysis on some data
        """
        if self.stats is None:
            return "No statistical tests performed"
        return self.stats(measurements,
                          dependent=self.measurement,
                          independent=self.varying_parameters,
                          compare=terms.DATASET)
    
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
            "stats": self.statistical_tests(measurements),
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
