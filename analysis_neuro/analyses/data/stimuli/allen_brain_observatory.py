"""Stimuli used in the Allen Brain Observatory experiments."""

import pandas as pd

import analysis_neuro.terminology as terms

# TODO: consider describing as stimulus windows with set duration
#   
drifting_gratings = pd.DataFrame(
    [
        {
            terms.VISUAL_STIMULUS: "sinusoidal grating",
            terms.VISUAL_STIMULUS + terms.TEMPORAL_FREQUENCY: tf,
            terms.VISUAL_STIMULUS + terms.SPATIAL_FREQUENCY: 0.04,
            terms.VISUAL_STIMULUS + terms.CONTRAST: 0.8,
            terms.VISUAL_STIMULUS + terms.STIM_ORIENTATION: ori,
            terms.VISUAL_STIMULUS + terms.ANGLE_AZIMUTH: pd.Interval(-120, 120),
            terms.VISUAL_STIMULUS + terms.ANGLE_ELEVATION: pd.Interval(-60, 60),
            terms.VISUAL_STIMULUS + terms.DURATION: 2000
        }
        for ori in range(0, 360, 45)
        for tf in [1, 2, 4, 8, 15]
    ]
)


gray = pd.DataFrame(
    {
        terms.VISUAL_STIMULUS: "gray",
        # TODO: this was incorrectly formatted before, and not caught by tests
        #  how can we better enforce this and prevent difficulties?
        terms.VISUAL_STIMULUS + terms.ANGLE_AZIMUTH: [(-120, 120)],
        terms.VISUAL_STIMULUS + terms.ANGLE_ELEVATION: [(-60, 60)],
    }
)
