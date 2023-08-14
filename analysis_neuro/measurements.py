from analysis_neuro import terms
from analysis_neuro import measurement_utils
from analysis_neuro import stimuli
import numpy as np


def _calculate_osi(df):
    rates = df[terms.FIRING_RATE]
    orientations = df[terms.STIM_ORIENTATION]
    return np.abs(np.sum(rates * np.exp(2 * 1j * np.deg2rad(orientations))) / np.sum(rates))


def _filter_params(measured, params):
    mutual_cols = [c for c in params if c in measured]
    index = [tuple(v) if len(v) > 1 else v[0] for v in params[mutual_cols].values]
    return measured.set_index(mutual_cols).loc[index].reset_index()


# yeah this really should be an object
# TODO: also, this is inflexible: say we have a measurement on the basis of CA fluoresence instead... it will still defail to firing rate the way it is listed here
# I think we need to have a single function in the dict: this will either call the method, or
#    combine other methods. Maximum flexibility
def osi_firing_rate(model, parameters, measurements_library):
    all_stimuli = measurement_utils.extract_parameters(stimuli.get(parameters))
    firing_rate = measurement_utils.measure(
        model, terms.FIRING_RATE, all_stimuli, measurements_library)
    firing_rate = _filter_params(firing_rate, parameters)
    osi = firing_rate.groupby(list(parameters.columns) + [terms.CELL_ID]).apply(_calculate_osi)
    return osi.rename(terms.ORIENTATION_SELECTIVITY).reset_index()


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
