from analysis_neuro import terms
import pandas as pd

ALLEN_BRAIN_OBSERVATORY = "brain observatory 1.1"
drifting_gratings = ALLEN_BRAIN_OBSERVATORY + 'drifting gratings'
gray = ALLEN_BRAIN_OBSERVATORY + 'gray'


# stimulus generation, measurement conversion, are all analysis-side stuff.
# but measurement conversion a model needs to be able to overwrite if it does not
# directly model the preceding measurement.\

# analysis should be passed a measurement method..
# if not it will ask model
# if model and analysis both have method, analysis ignores own
# PROBLEM: what if model hase general-purpose measurement method but should really use more specific one.

# TODO: this isn't quite accurate
stimuli = {
    drifting_gratings: pd.DataFrame([
        {terms.VISUAL_STIMULUS: 'sinusoidal grating', # TODO: should be some sort of lazy-loaded video?
         terms.TEMPORAL_FREQUENCY: tf,
         terms.SPATIAL_FREQUENCY: 0.04,
         terms.CONTRAST: 0.8,
         terms.STIM_ORIENTATION: ori,
         terms.ANGLE_AZIMUTH: (-120, 120),
         terms.ANGLE_ELEVATION: (-60, 60)}
        for ori in range(0, 360, 45)
        for tf in [1, 2, 4, 8, 15]
    ]),

    gray: pd.DataFrame([
        {terms.VISUAL_STIMULUS: 'gray'}
    ])
}
