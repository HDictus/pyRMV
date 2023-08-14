from analysis_neuro import terms
import analysis_neuro.stimuli as stimuli
import pandas as pd


def test_retrieve_all_brain_observatory():

    allvalues = stimuli.get(
        pd.DataFrame(
            {terms.STIMULUS: stimuli.allen_brain_observatory.drifting_gratings,
             'other': ['a', 'b']}))

    dg_base = stimuli.stimuli[stimuli.allen_brain_observatory.drifting_gratings]

    pd.testing.assert_frame_equal(
        allvalues,
        pd.concat(
            [dg_base.assign(
                **{terms.STIMULUS: stimuli.allen_brain_observatory.drifting_gratings}
                , other='a'),
             dg_base.assign(
                 **{terms.STIMULUS: stimuli.allen_brain_observatory.drifting_gratings},
                 other='b')],
            axis=0))

    # test retrieval does not overwrite
    allvalues = stimuli.get(
        pd.DataFrame(
            {terms.STIMULUS: stimuli.allen_brain_observatory.drifting_gratings,
             terms.TEMPORAL_FREQUENCY: [2]}))
    assert all(allvalues[terms.TEMPORAL_FREQUENCY].values == dg_base[terms.TEMPORAL_FREQUENCY].values)
