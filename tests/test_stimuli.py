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
        pd.concat([dg_base.assign(other='a'),
                   dg_base.assign(other='b')], axis=0))



def test_subsets_specific_parameters():
    assert False


def test_ignores_optimal_parameters():
    assert False
