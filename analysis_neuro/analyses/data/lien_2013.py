"""Data extracted from @lien_tuned_2013."""
import pandas as pd
from analysis_neuro import terminology as terms

stimulus = pd.DataFrame({
    terms.VISUAL_STIMULUS: "bar grating",
    terms.CONTRAST: 1,
    terms.SPATIAL_FREQUENCY: 0.04,
    terms.TEMPORAL_FREQUENCY: 2,
    terms.ANGLE_AZIMUTH: [(-120, 120)] * 12,
    terms.ANGLE_ELEVATION: [(-60, 60)] * 12, # TODO: use intervals, or MAX and MIN
    terms.STIM_ORIENTATION: range(0, 360, 30), # TODO: just to reuse ... meh
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

# Both tc current and osi_frequency_modulation need to be in terms of 
# PATHWAY_CURRENT
# current along a pathway during stimulation
# ji_relative also, though there the stimulus is optogenetic (and therefore ignored by our model, we do not use pathway current)
# for ji_relative, we should create the relative_to method
# we should have some way to reference it in the analysis itself: maybe something like
#    measurement=terms.RELATIVE + terms.PATHWAY_CURRENT
#    default_method=lambda model, params: relative_to(model, terms.PATHWAY_CURRENT, params)
#    on the other hand, this could also be related directly to the RELATIVE prefix somehow...
#    Term('relative', 'desc', 'method_name'=lambda suffix: 'relative' + suffix.method_name,'measurement_method'=relative_to   -> recieves model, params, suffixed