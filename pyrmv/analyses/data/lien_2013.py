"""Data extracted from @lien_tuned_2013."""
# pylint: disable=fixme
import pandas as pd
from pyrmv import terminology as terms

stimulus = pd.DataFrame({
    terms.VISUAL_STIMULUS: "bar grating",
    terms.VISUAL_STIMULUS + terms.CONTRAST: 1,
    terms.VISUAL_STIMULUS + terms.SPATIAL_FREQUENCY: 0.04,
    terms.VISUAL_STIMULUS + terms.TEMPORAL_FREQUENCY: 2,
    terms.VISUAL_STIMULUS + terms.ANGLE_AZIMUTH: pd.Interval(-120, 120),
    terms.VISUAL_STIMULUS + terms.ANGLE_ELEVATION: pd.Interval(-60, 60),
    terms.VISUAL_STIMULUS + terms.STIM_ORIENTATION: range(0, 360, 30),
})

thalamocortical_current = pd.DataFrame(
    {
        terms.SILENCED + terms.REGION: "VISp",
        terms.SPECIES: "mouse",
        terms.SAMPLE_SIZE: 42,
        terms.VOLTAGE_CLAMP: -70,
        terms.DATASET: "Lien2013",
        terms.REGION: "VISp",
        terms.LAYER: "L4",
        terms.SYNAPSE_CLASS: "EXC",
        # lien and scanziani noted that total current was independent of stimulus orientation
        # therefore, for this validation, only one orientation should be neccessary
        terms.STIMULUS: stimulus.iloc[:1].pointer(),
        terms.MEAN + terms.PATHWAY_CURRENT: [-0.046],
    }
)

fraction_excitation = pd.DataFrame(
    {
        terms.POSTSYNAPTIC + terms.REGION: "VISp",
        terms.POSTSYNAPTIC + terms.SYNAPSE_CLASS: "EXC",
        terms.PRESYNAPTIC + terms.REGION: "LGd",
        terms.MEAN + terms.FRACTION_EXCITATION_PER_CONNECTION: [0.012],
        terms.SAMPLE_SIZE: 14,
        terms.DATASET: "Lien2018",
    }
)
