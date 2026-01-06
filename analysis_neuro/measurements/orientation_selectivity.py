"""Orientation selectivity measurement methods."""

import numpy as np
import pandas as pd
from tqdm import tqdm

import analysis_neuro.terminology as terms


def _calculate_osi(dataframe):
    """Calculate orientation selectivity index from firing rates and orientations.

    Arguments:
        dataframe: DataFrame containing firing rates and stimulus orientations

    Returns:
        float: orientation selectivity index
    """
    rates = dataframe[terms.FIRING_RATE]
    orientations = dataframe[terms.STIM_ORIENTATION]
    return np.abs(
        np.sum(rates * np.exp(2 * 1j * np.deg2rad(orientations))) / np.sum(rates)
    )


def from_firing_rate(model, parameters, response_measurement=terms.FIRING_RATE):
    """Measure orientation selectivity on the basis of some response property (e.g. Firing rate).

    Arguments:
       model: an object which can measure the response_measurement
       parameters: a dataframe of measurement parameters, e.g. stimuli, cell populations
       response_measurement: (default: FIRING_RATE) the response property from which
           to calculate orientation selectivity.

    Returns:
        DataFrame with orientation selectivity values per cell
    """
    from analysis_neuro.measurements import measure

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
