"""Methods for measuring properties.

All methods can be overwritten by a model. Those implemented here serve as defaults
That allow the computation of one measurement type from another.

For example, a method here may implement calculating density based on mass and volume.
"""
import numpy as np
import pandas as pd
from analysis_neuro import terms
from analysis_neuro import measurement_utils


def _calculate_osi(df):
    rates = df[terms.FIRING_RATE]
    orientations = df[terms.STIM_ORIENTATION]
    return np.abs(np.sum(rates * np.exp(2 * 1j * np.deg2rad(orientations))) / np.sum(rates))


# yeah this really should be an object
# TODO: also, this is inflexible: say we have a measurement on the basis of CA fluoresence instead... it will still defail to firing rate the way it is listed here
# I think we need to have a single function in the dict: this will either call the method, or
#    combine other methods. Maximum flexibility
def osi_firing_rate(model, parameters, measurements_library):
    """Measure orientation selectivity using firing rates."""   
    # we loop through the parameters at the moment.
    # there may be a more efficient way to do this with batch processing
    # but I haven't come up with it
    out = []
    for i, row in parameters.iterrows():
        stimuli_shown = row[terms.STIMULUS].df
        columns_both = [
            c for c in parameters.columns if c in stimuli_shown
            and row[c] not in ['optimal']
            ]
        if len(columns_both) > 0:
            stimuli_shown = stimuli_shown.set_index(columns_both).loc[
                row[columns_both]].reset_index()
        firing_rate = measurement_utils.measure(
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
            [c for c in firing_rate if c!=terms.FIRING_RATE])[
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
