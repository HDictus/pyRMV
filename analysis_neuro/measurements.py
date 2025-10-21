"""Methods for measuring properties.

All methods can be overwritten by a model. Those implemented here serve as defaults
That allow the computation of one measurement type from another.

For example, a method here may implement calculating density based on mass and volume.

Then, any model which has measurement methods for both mass and volume can have
orientation selectivity implemented for it.

The dict measurements contains measurable properties as keys and a description
of the requirements to measure the property as values.
The values can specify the name of a method a model has to implement to measure
that property, and it can specify a method to measure the property on the basis
of other properties.

"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import analysis_neuro.terminology as terms
from analysis_neuro.exceptions import TerminologyError

DATA_TERMS = [terms.DATASET, terms.CITATION, terms.NOTES, terms.CELL_ID, terms.TRIAL_ID]


def validate_measurement(measurement):
    """Check that <measurement> is a valid measurement."""
    if isinstance(measurement, list):
        for m in measurement:
            validate_measurement(m)
        return
    try:
        measurement = terms.ALL_TERMS[measurement]
    except KeyError as exc:
        raise TerminologyError(f"{measurement} is not a defined Term.") from exc
    if measurement.measurement_method is None:
        raise TerminologyError(
            f"{measurement} is not a measurable property, "
            "as it does not have an associated measurement_method."
        )


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
            f"The measurement method {measurement.measurement_method}"
            " must return a pandas DataFrame containing the parameters and measurements.\n"
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
        if column not in terms.ALL_TERMS:
            # check if it is a prefix-postfix combo
            is_valid_prefixed = False
            for term in terms.ALL_TERMS:
                if term.endswith(" "):
                    # if it is prefixed and the postfix term exists
                    if (
                        column.startswith(term)
                        and column[len(term):] in terms.ALL_TERMS
                    ):
                        is_valid_prefixed = True
                        break
            if not is_valid_prefixed:
                warnings.warn(
                    f"Column header '{column}' is not defined in analysis_neuro.terminology"
                )


def _measurement_method(model, measurement):
    measurement = terms.ALL_TERMS[measurement]
    method_name = measurement.measurement_method
    if hasattr(model, method_name):
        return getattr(model, method_name)
    raise TypeError(
        f"The model does not have the functionality needed to measure {measurement}."
        f" It should define a method named '{method_name}"
    )


def measure(model, measurement, parameters):
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
    if isinstance(measurement, list):
        first = measure(model, measurement[0], parameters)
        for m in measurement[1:]:
            first[m] = measure(model, m, parameters)[m]
        return first

    validate_measurement(measurement)
    validate_observations(parameters)
    measurement_method = _measurement_method(model, measurement)

    measured = measurement_method(parameters)
    validate_measured(measured, measurement, parameters)
    measured[terms.DATASET] = model.label
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
    exclude_from_parameters = DATA_TERMS + [
        terms.STD + measurement,
        terms.SAMPLE_SIZE,
        terms.MEAN + measurement,
    ]
    if measurement is not None:
        exclude_from_parameters += [measurement]
    paramcols = [col for col in observations if col not in exclude_from_parameters]

    def _multicolumn_unique(dframe, cols):
        """Return the unique combinations of cols in dframe."""
        return dframe[cols].groupby(cols, dropna=False).sum().reset_index()

    return _multicolumn_unique(observations, paramcols)


def _calculate_osi(dataframe):
    rates = dataframe[terms.FIRING_RATE]
    orientations = dataframe[terms.STIM_ORIENTATION]
    return np.abs(
        np.sum(rates * np.exp(2 * 1j * np.deg2rad(orientations))) / np.sum(rates)
    )


def orientation_selectivity(model, parameters, response_measurement=terms.FIRING_RATE):
    """Measure orientation selectivity on the basis of some response property (e.g. Firing rate).

    Arguments:
       model: an object which can measure the response_measurement
       parameters: a dataframe of measurement parameters, e.g. stimuli, cell populations
       response_measurement: (default: FIRING_RATE) the response property from which
           to calculate orientation selectivity.
    """
    # we loop through the parameters at the moment.
    # there may be a more efficient way to do this with batch processing
    # but I haven't come up with it
    out = []

    for _, row in tqdm(parameters.iterrows(), total=len(parameters)):
        stimuli_shown = row[terms.STIMULUS].df
        columns_both = [
            c
            for c in parameters.columns
            if c in stimuli_shown and row[c] not in ["optimal"]
        ]
        if len(columns_both) > 0:
            stimuli_shown = (
                stimuli_shown.set_index(columns_both)
                .loc[row[columns_both]]
                .reset_index()
            )
        other_parameters = [c for c in parameters.columns if row[c] not in ["optimal"]]
        stimuli_shown = stimuli_shown.assign(**row[other_parameters])

        response = measure(model, response_measurement, stimuli_shown)
        if len(response) == 0 or not np.any(~np.isnan(response[response_measurement])):
            continue

        # if temporal frequency is set to optimal, we select a different
        # temporal frequency for each cell. Specifically, the one to which
        # it responds most strongly
        tf_optimal = (
            terms.TEMPORAL_FREQUENCY in parameters.columns
            and row[terms.TEMPORAL_FREQUENCY] == "optimal"
        )
        if tf_optimal:
            conditionwise_rates = (
                response.groupby(
                    [
                        c
                        for c in response
                        if c not in (response_measurement, terms.TRIAL_ID)
                    ]
                )[response_measurement]
                .mean()
                .reset_index()
            )
            optimal_tf = (
                conditionwise_rates.set_index(terms.TEMPORAL_FREQUENCY)
                .groupby(terms.CELL_ID)[response_measurement]
                .idxmax()
            )
            response = (
                response.set_index([terms.CELL_ID, terms.TEMPORAL_FREQUENCY])
                .loc[zip(optimal_tf.index, optimal_tf.values)]
                .reset_index()
            )

        response["scaled"] = response[response_measurement] * np.exp(
            2 * 1j * np.deg2rad(response[terms.STIM_ORIENTATION])
        )
        grouped_by_cell = response.groupby(terms.CELL_ID)
        selectivity = np.abs(
            grouped_by_cell["scaled"].sum()
            / grouped_by_cell[response_measurement].sum()
        )
        selectivity.name = terms.ORIENTATION_SELECTIVITY
        out.append(selectivity.reset_index().assign(**row))

    return pd.concat(out, axis=0)
