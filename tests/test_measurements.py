from analysis_neuro import terms
import pandas as pd
import analysis_neuro.measurements as measurements
import analysis_neuro.measurement_utils as measurement_utils
from analysis_neuro import stimuli


def test_measures_osi_with_firing_rate():

    class MockModel:

        def firing_rate(self, parameters):
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

    parameters = pd.DataFrame(
        {terms.STIMULUS: [stimuli.allen_brain_observatory.drifting_gratings]})
    measured = measurement_utils.measure(MockModel(), terms.ORIENTATION_SELECTIVITY, parameters)
    assert list(measured.columns) == list(parameters.columns) + [terms.CELL_ID, terms.ORIENTATION_SELECTIVITY]
    assert all(measured[terms.ORIENTATION_SELECTIVITY] > 0)

    parameters[terms.TEMPORAL_FREQUENCY] = 1
    measured = measurement_utils.measure(MockModel(), terms.ORIENTATION_SELECTIVITY, parameters)
    assert list(measured.columns) == list(parameters.columns) + [terms.CELL_ID, terms.ORIENTATION_SELECTIVITY]
    assert all(measured[terms.ORIENTATION_SELECTIVITY] == 0)

    parameters[terms.TEMPORAL_FREQUENCY] = 2
    measured = measurement_utils.measure(MockModel(), terms.ORIENTATION_SELECTIVITY, parameters)
    assert list(measured.columns) == list(parameters.columns) + [terms.CELL_ID, terms.ORIENTATION_SELECTIVITY]
    assert all(measured[terms.ORIENTATION_SELECTIVITY]  == [0.5, 0, 0])

    parameters[terms.TEMPORAL_FREQUENCY] = 'optimal'
    measured = measurement_utils.measure(MockModel(), terms.ORIENTATION_SELECTIVITY, parameters)
    assert list(measured.columns) == list(parameters.columns) + [terms.CELL_ID, terms.ORIENTATION_SELECTIVITY]
    assert all(measured[terms.ORIENTATION_SELECTIVITY] == 0.5)
