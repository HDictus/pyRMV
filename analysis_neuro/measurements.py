"""Methods for measuring properties.

All methods can be overwritten by a model. Those implemented here serve as defaults
That allow the computation of one measurement type from another.

For example, a method here may implement calculating density based on mass and volume.
"""
import warnings
import logging
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
from analysis_neuro import terms, TerminologyError


def _calculate_osi(dataframe):
    rates = dataframe[terms.FIRING_RATE]
    orientations = dataframe[terms.STIM_ORIENTATION]
    return np.abs(np.sum(rates * np.exp(2 * 1j * np.deg2rad(orientations))) / np.sum(rates))


def osi_firing_rate(model, parameters, measurements_library):
    """Measure orientation selectivity using firing rates."""
    # we loop through the parameters at the moment.
    # there may be a more efficient way to do this with batch processing
    # but I haven't come up with it
    out = []
    for _, row in parameters.iterrows():
        stimuli_shown = row[terms.STIMULUS].df
        columns_both = [
            c for c in parameters.columns if c in stimuli_shown
            and row[c] not in ['optimal']
        ]
        if len(columns_both) > 0:
            stimuli_shown = stimuli_shown.set_index(columns_both).loc[
                row[columns_both]].reset_index()
        other_parameters = [c for c in parameters.columns if row[c] not in ['optimal']]
        stimuli_shown = stimuli_shown.assign(**row[other_parameters])

        firing_rate = measure(
            model, terms.FIRING_RATE,
            stimuli_shown, measurements_library
        )
        # if temporal frequency is set to optimal, we select a different
        # temporal frequency for each cell. Specifically, the one to which
        # it responds most strongly
        tf_optimal = (
            terms.TEMPORAL_FREQUENCY in parameters.columns
            and row[terms.TEMPORAL_FREQUENCY] == 'optimal'
        )
        if tf_optimal:
            conditionwise_rates = firing_rate.groupby(
                [c for c in firing_rate if c != terms.FIRING_RATE])[
                    terms.FIRING_RATE].mean().reset_index()
            optimal_tf = conditionwise_rates.set_index(
                terms.TEMPORAL_FREQUENCY).groupby(
                terms.CELL_ID)[terms.FIRING_RATE].idxmax()
            firing_rate = firing_rate.set_index([
                terms.CELL_ID, terms.TEMPORAL_FREQUENCY]).loc[
                    zip(optimal_tf.index, optimal_tf.values)
            ].reset_index()
        selectivity = firing_rate.groupby(
            terms.CELL_ID).apply(_calculate_osi)\
            .rename(terms.ORIENTATION_SELECTIVITY).reset_index()\
            .assign(**row)

        out.append(selectivity)

    return pd.concat(out, axis=0)


measurements = {
    terms.CELL_DENSITY: {"method name": "cell_density"},
    terms.CELL_COUNT: {"method name": "cell_count"},
    terms.CONNECTION_PROBABILITY: {"method name": "connection_probability"},
    terms.SYNAPSES_PER_CONNECTION: {'method name': 'synapses_per_connection'},
    terms.NUM_SYNAPSES: {'method name': 'num_synapses'},
    terms.INTERSOMATIC_DISTANCE: {"method name": "intersomatic_distance"},
    terms.ORIENTATION_SELECTIVITY: {
        "method name": "orientation_selectivity",
        (terms.FIRING_RATE,): osi_firing_rate},
    terms.FIRING_RATE: {"method name": "firing_rate"},
}

DATA_TERMS = [terms.DATASET, terms.CITATION, terms.NOTES, terms.CELL_ID, terms.TRIAL_ID]


# pylint: disable=dangerous-default-value
def validate_measurement(measurement, measurements_library=measurements):
    """Check that <measurement> is a valid measurement."""
    if measurement not in measurements_library:
        raise TerminologyError(
            f"Provided measurement '{measurement}' is not defined in "
            "the measurements library (default analysis_neuro.measurements.measurements")


def validate_measured(measured_data, measurement, parameters, measurements=measurements):
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
            f"The measurement method {measurements[measurement]['method name']}"
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
            logging.debug(
                "Using method %s to measure %s from %s using %s",
                value, measurement, model, key)
            return partial(value, model, measurements_library=measurements_library)
    return None


def measure(model, measurement, parameters, measurements_library=measurements):
    """Measure the quantity measurement from a model.

    Arguments:
        model: a class implementing a method measuring measurement
        measurement: string representing a measurement type.
            should be one of the terms represented in
            analysis_neuro.measurements
        parameters: a DataFrame describing the parameters of the
            measurements to make. Use terminology from
            analysis_neuro.terminology to ensure consistency.
        measurements_library: a dict describing measurements for each term.
            Each key is a term describing the measured property, each value
            Generally only needed for testing purposes.
            is a dict describing the measurement procedure.
            This dict must contain an entry "method name".
            See analysis_neuro.measurements for an example
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
        return dframe[cols].groupby(cols, dropna=False).sum().reset_index()

    return _multicolumn_unique(observations, paramcols)

