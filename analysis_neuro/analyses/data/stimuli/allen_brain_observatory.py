"""Stimuli used in the Allen Brain Observatory experiments."""

import pandas as pd

import analysis_neuro.terminology as terms

drifting_gratings = pd.DataFrame(
    [
        {
            terms.VISUAL_STIMULUS: "sinusoidal grating",
            terms.TEMPORAL_FREQUENCY: tf,
            terms.SPATIAL_FREQUENCY: 0.04,
            terms.CONTRAST: 0.8,
            terms.STIM_ORIENTATION: ori,
            terms.ANGLE_AZIMUTH: (-120, 120),
            terms.ANGLE_ELEVATION: (-60, 60),
        }
        for ori in range(0, 360, 45)
        for tf in [1, 2, 4, 8, 15]
    ]
).pointer()


gray = pd.DataFrame(
    {
        terms.VISUAL_STIMULUS: "gray",
        # TODO: this was incorrectly formatted before, and not caught by tests
        #  how can we better enforce this and prevent difficulties?
        terms.ANGLE_AZIMUTH: [(-120, 120)],
        terms.ANGLE_ELEVATION: [(-60, 60)],
    }
).pointer()
