"""Parameterization of stimuli used in experiments.

The .stimuli dict has as keys the names of sets of stimuli, and as values a list stimuli.
Each stimulus is a dict with several parameters describing it.

We may later use this module to download e.g. stimulus movies from their source
"""
from analysis_neuro import terms

ALLEN_BRAIN_OBSERVATORY = "brain observatory 1.1"

# TODO: this isn't quite accurate
stimuli = {
    ALLEN_BRAIN_OBSERVATORY + "drifting gratings": [
        {terms.VISUAL_STIMULUS: 'drifting grating',
         terms.TEMPORAL_FREQUENCY: tf,
         terms.SPATIAL_FREQUENCY: 0.04,
         terms.CONTRAST: 0.8,
         terms.STIM_ORIENTATION: ori,
         terms.ANGLE_AZIMUTH: (-120, 120),
         terms.ANGLE_ELEVATION: (-60, 60)}
        for ori in range(0, 45, 360)
        for tf in [1, 2, 4, 8, 15]
    ],

    ALLEN_BRAIN_OBSERVATORY + "gray": [
        {terms.VISUAL_STIMULUS: 'gray'}
    ],
}

def retrieve_all(parameters):
    """Create a dataframe of all stimulus conditions for given parameters.

    In many cases a measurement is made using a set of stimuli instead of a single stimulus.
    Instead the parameters contain a name for the set of stimuli used.
    This function retrieves all of the individual stimuli for every stimulus set and
    returns it as a dataframe.
    """
    return
    
