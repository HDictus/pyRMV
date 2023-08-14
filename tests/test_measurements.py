from analysis_neuro import terms
import pandas as pd
import numpy as np
import analysis_neuro.measurements as measurements
import analysis_neuro.measurement_utils as measurement_utils
from analysis_neuro import stimuli


def test_measures_osi_with_firing_rate():

    class MockModel:

        def firing_rate(self, parameters):
            """Each neuron will have a rate of 1 if either temporal frequency or orientation are non-optimal.
            Else a 2.
            This way, each neuron is orientation-selective only at optimal tf
            """
            nrns = [0, 1, 2]
            pref_tf = [2, 8, 4]
            pref_ori = [0, 90, 270]
            out = []
            for i, row in parameters.iterrows():
                for nrn, tf, ori in zip(nrns, pref_tf, pref_ori):
                    if row[terms.TEMPORAL_FREQUENCY] == tf and row[terms.STIM_ORIENTATION] == ori:
                        out.append({terms.FIRING_RATE: 2, terms.CELL_ID: nrn, **row})
                    else:
                        out.append({terms.FIRING_RATE: 1, terms.CELL_ID: nrn, **row})
            return pd.DataFrame(out)

    rates = np.array([1, 1, 1, 2, 1, 1, 1, 1])
    oris = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    expected_OSI = np.abs(np.sum(rates * np.exp(2*1j * np.deg2rad(oris)))) / np.sum(rates)


    parameters = pd.DataFrame(
        {terms.STIMULUS: [stimuli.allen_brain_observatory.drifting_gratings]})
    measured = measurement_utils.measure(MockModel(), terms.ORIENTATION_SELECTIVITY, parameters)
    # measures by averaging over stimulus conditions
    assert list(measured.columns) == list(parameters.columns) + [terms.CELL_ID, terms.ORIENTATION_SELECTIVITY]
    assert all(measured[terms.ORIENTATION_SELECTIVITY] > 0)

    parameters[terms.TEMPORAL_FREQUENCY] = 1
    measured = measurement_utils.measure(MockModel(), terms.ORIENTATION_SELECTIVITY, parameters)
    # measures just for the specified temporal frequency
    assert list(measured.columns) == list(parameters.columns) + [terms.CELL_ID, terms.ORIENTATION_SELECTIVITY]
    assert np.allclose(measured[terms.ORIENTATION_SELECTIVITY], 0)

    parameters[terms.TEMPORAL_FREQUENCY] = 2
    measured = measurement_utils.measure(MockModel(), terms.ORIENTATION_SELECTIVITY, parameters)
    assert list(measured.columns) == list(parameters.columns) + [terms.CELL_ID, terms.ORIENTATION_SELECTIVITY]
    assert np.allclose(measured[terms.ORIENTATION_SELECTIVITY].values, [expected_OSI, 0, 0])

    parameters[terms.TEMPORAL_FREQUENCY] = 'optimal'
    # measures each cell at its optimal TF
    measured = measurement_utils.measure(MockModel(), terms.ORIENTATION_SELECTIVITY, parameters)
    assert list(measured.columns) == list(parameters.columns) + [terms.CELL_ID, terms.ORIENTATION_SELECTIVITY]
    assert np.allclose(measured[terms.ORIENTATION_SELECTIVITY], expected_OSI)

    parameters = pd.DataFrame(
        {terms.STIMULUS: [stimuli.allen_brain_observatory.drifting_gratings,
                          stimuli.allen_brain_observatory.drifting_gratings],
         terms.TEMPORAL_FREQUENCY: [2, 4]})
    measured = measurement_utils.measure(MockModel(), terms.ORIENTATION_SELECTIVITY, parameters)
    # a separate measurement for each unique TF value
    assert all((measured.set_index(terms.TEMPORAL_FREQUENCY).loc[2] > 0) == [True, False, False])
    assert all((measured.set_index(terms.TEMPORAL_FREQUENCY).loc[4] > 0) == [False, False, True])
