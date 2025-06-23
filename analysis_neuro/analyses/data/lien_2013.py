import pandas as pd
from analysis_neuro import terminology as terms

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
        terms.STIMULUS: pd.DataFrame(
            {
                terms.VISUAL_STIMULUS: "bar grating",
                terms.CONTRAST: 1,
                terms.SPATIAL_FREQUENCY: 0.04,
                terms.TEMPORAL_FREQUENCY: 2,
                terms.ANGLE_AZIMUTH: [(-120, 120)],
                terms.ANGLE_ELEVATION: [(-60, 60)],
                terms.STIM_ORIENTATION: [0],
            }
        ).pointer(),
        terms.MEAN + terms.SOMATIC_CURRENT: [-0.046],
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
