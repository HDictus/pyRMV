"""Tools for extracting measurements from models."""
import warnings
from pathlib import Path

import pandas as pd

from .exceptions import TerminologyError
from functools import partial
import logging
from . import terminology as terms
from . import measurements


DATA_TERMS = [terms.DATASET, terms.CITATION, terms.NOTES, terms.CELL_ID, terms.TRIAL_ID]


def validate_measurement(measurement, measurements_library=measurements.measurements):
    """Check that <measurement> is a valid measurement."""
    if measurement not in measurements_library:
        raise TerminologyError(
            f"Provided measurement '{measurement}' is not defined in "
            "the measurements library (default analysis_neuro.measurements.measurements")


def validate_measured(measured_data, measurement, parameters):
    """Check that measured data are of the right form.

    Run this on the output of a model's measurement method to
    verify that it conforms to the required structure. raises
    a ValueError if not.

    Arguments:
        measured_data: the measurement extracted from a model
        measurement: Term or string, the quantity measured,
        parameters: a dataframe of parameter values for which
            the measurement was conducted
    """
    def _exception(msg):
        fake_example_measurement = pd.DataFrame(
            [
                {**row, **{measurement: "<some value>"}}
                for __, row in parameters.iterrows()
                for _ in range(4)
            ]
        )

        raise ValueError(
            f"The measurement method {measurements.measurements[measurement]['method name']} must return"
            " a pandas DataFrame containing the parameters and measurements.\n"
            f"e.g. \n: {fake_example_measurement}\n\n"
            f"Recieved instead:\n{measured_data}\n\n"
            f"The problem with that is that {msg}."
        )

    if not isinstance(measured_data, pd.DataFrame):
        _exception("it is not a DataFrame")

    if measurement not in measured_data.columns:
        _exception("it does not contain a column for the measurement")

    if any(c not in measured_data for c in parameters.columns):
        _exception(
            "it does not include all the parameters. "
            "It can be hard to tell measurements apart without them"
        )


def validate_observations(observations):
    """Check that experimental observations are reported in the required format."""
    if not isinstance(observations, pd.DataFrame):
        thispath = Path(__file__).parent
        examplepath = thispath / "analyses" / "data" / "schuz_neuron_density_1989.csv"

        raise ValueError(
            f"invalid observations {observations}\n\n"
            "observations must be a pandas.DataFrame of the form:\n"
            f"|parameter1|parameter2|...|measured_quantity|{terms.DATASET}|\n"
            "|value     | value    |...|measured value   | <some name>   |\n"
            "|...       |...       |...|...              |               |\n"
            "namely, the columns represent the measurement and its parameterization\n"
            f"see {str(examplepath)} for example data"
        )
    # pylint: disable=protected-access
    for column in observations.columns:
        if not isinstance(column, str):
            raise ValueError("column headers must be strings")
        if column not in terms._ALLTERMS:
            # check if it is a prefix-postfix combo
            is_valid_prefixed = False
            for term in terms._ALLTERMS:
                if term.endswith(" "):
                    # if it is prefixed and the postfix term exists
                    if column.startswith(term) and column[len(term):] in terms._ALLTERMS:
                        is_valid_prefixed = True
                        break
            if not is_valid_prefixed:
                warnings.warn(
                    f"Column header '{column}' is not defined in analysis_neuro.terminology"
                )


def _measurement_method(model, measurement, measurements_library):
    method_name = measurements_library[measurement]['method name']
    if hasattr(model, method_name):
        return getattr(model, method_name)

    for key, value in measurements_library[measurement].items():
        if key == 'method name':
            continue
        if all(_measurement_method(model, other, measurements_library) for other in key):
            logging.debug(f"Using method {value} to measure {measurement} from {model} using {key}")
            return partial(value, model, measurements_library=measurements_library)
    return None


# TODO: maybe at this point measurements_library should be a class of objects
# which has a measure method and can be subclassed / initialized for specifics?
def measure(model, measurement, parameters, measurements_library=measurements.measurements):
    """Measure the quantity measurement from a model.

    Arguments:
        model: a class implementing a method measuring measurement
        measurement: string representing a measurement type.
            should be one of the terms represented in
            analysis_neuro.measurements
        parameters: a DataFrame describing the parameters of the
            measurements to make. Use terminology from
            analysis_neuro.terminology to ensure consistency.
    """
    validate_measurement(measurement, measurements_library)
    validate_observations(parameters)
    measurement_method = _measurement_method(model, measurement, measurements_library)
    if measurement_method is None:
        raise TypeError(
            f"The model does not have the functionality needed to measure {measurement}.")
    measured = measurement_method(parameters)
    validate_measured(measured, measurement, parameters)
    return measured


def extract_parameters(observations, measurement=None):
    """Identify the measurement parameters from experimental observations.

    Excludes the column of measured values and any parameters describing
    the dataset (citation, notes, sample size), leaving only the parameters
    of the measurement in question. Only each unique combination of parameters
    is retained, collapsing multiple measurements.

    Arguments:
       observations: a dataframe of experimental measurements. Each column
           corresponds to a variable, each row to an observation.
       measurement: a string indicating the column which corresponds to the
           measured quantity. will be excluded from the parameters.
    """
    exclude_from_parameters = DATA_TERMS + [terms.STD + measurement, terms.SAMPLE_SIZE]
    if measurement is not None:
        exclude_from_parameters += [measurement]
    paramcols = [col for col in observations if col not in exclude_from_parameters]

    def _multicolumn_unique(dframe, cols):
        """Return the unique combinations of cols in dframe."""
        return dframe[cols].groupby(cols).sum().reset_index()

    return _multicolumn_unique(observations, paramcols)
