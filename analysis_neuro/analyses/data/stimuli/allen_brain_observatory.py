from analysis_neuro import terms
import pandas as pd


drifting_gratings = pd.DataFrame([
        {terms.VISUAL_STIMULUS: 'sinusoidal grating', # TODO: should be some sort of lazy-loaded video?
         terms.TEMPORAL_FREQUENCY: tf,
         terms.SPATIAL_FREQUENCY: 0.04,
         terms.CONTRAST: 0.8,
         terms.STIM_ORIENTATION: ori,
         terms.ANGLE_AZIMUTH: (-120, 120),
         terms.ANGLE_ELEVATION: (-60, 60)}
        for ori in range(0, 360, 45)
        for tf in [1, 2, 4, 8, 15]
    ])


gray = pd.DataFrame({
        terms.VISUAL_STIMULUS: 'gray',
        terms.ANGLE_AZIMUTH: (-120, 120),
        terms.ANGLE_ELEVATION: (-60, 60)})

