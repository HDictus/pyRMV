"""Orientation selectivity measurement methods."""

import numpy as np
import pandas as pd
from tqdm import tqdm

import analysis_neuro.terminology as terms
from analysis_neuro.features import g_OSI_signal


# TODO: use g_OSI from features.py
def _calculate_osi(dataframe):
    """Calculate orientation selectivity index from firing rates and orientations.

    Arguments:
        dataframe: DataFrame containing firing rates and stimulus orientations

    Returns:
        float: orientation selectivity index
    """
    rates = dataframe[terms.FIRING_RATE]
    orientations = dataframe[terms.VISUAL_STIMULUS + terms.STIM_ORIENTATION]
    return np.abs(
        np.sum(rates * np.exp(2 * 1j * np.deg2rad(orientations))) / np.sum(rates)
    )


# TODO: wrap generalized from_response method
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
    out = []
    for _, row in tqdm(parameters.iterrows(), total=len(parameters)):
        stimuli_shown = row[terms.STIMULUS].df
        response = model.firing_rate(stimuli_shown.assign(**row)).reset_index()
        minfr = response.groupby(terms.CELL_ID)[terms.FIRING_RATE].min()
        response[terms.FIRING_RATE] -= minfr.loc[response[terms.CELL_ID]].values
        if len(response) == 0 or np.all(np.isnan(response[response_measurement])):
            continue
        selectivity = g_OSI_signal(
            response[response_measurement],
            response[terms.VISUAL_STIMULUS + terms.STIM_ORIENTATION],
            groupby=response[terms.CELL_ID]
        ).fillna(0)
        selectivity.name = terms.ORIENTATION_SELECTIVITY
        out.append(selectivity.reset_index().assign(**row))

    return pd.concat(out, axis=0)
