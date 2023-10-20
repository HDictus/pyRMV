"""Terminology used in validation.

Each term is used as a column header in dataframes.
The description specifies the format of the column's contents.

Terms can be combined using addition, e.g.
PRESYNAPTIC + MTYPE

terminology.measurements is a dict describing the methods associated
with terms that represent a measured quality.
e.g. terminology.measurements[terminology.CONNECTION_PROBABILITY]
-> {'method name': 'connection_probability'}
"""


ALL_TERMS = {}


class Term(str):
    """A string with an associated description defining it."""

    # pylint: disable=unused-argument
    def __new__(cls, term, *args, **kwargs):
        """Called on initialization before __init__.

        Arguments:
            term: single word label.
        """
        return super().__new__(cls, term)

    def __init__(self, term, description="No description provided", measurement_method=None):
        """Define a new term.

        Arguments:
            term: short label
            description: description of the term
            measurement_method: name of the method used to measure the property this term represents
                only needed if this term represents a measurable property.
        """
        ALL_TERMS[term] = self
        self.description = description
        self.measurement_method = measurement_method
        super().__init__()

    def describe(self):
        """Describe the format of this term."""
        return f"{self}: {self.description}"

    def __add__(self, added):
        """Combine term labels and descriptions."""
        description = [self.description.format(added=added)]
        if isinstance(added, Term):
            description.append(f"{added} : \n{added.description}")
        return self.__class__(str(self) + str(added), ("\n\n".join(description)))


SPECIES = Term(
    "species",
    description="The species of any subject(s) experimented on")

REGION = Term(
    "region",
    description="Acronym of a brain region according to AIBS atlas naming convention",
)
LAYER = Term(
    "layer",
    description="Layer of some brain region as a capitalized acronym: e.g. L1, L23, SP, VPL",
)
NEURON_OR_GLIA = Term(
    "neuron or glia?",
    description=("Either 'neuron' or 'glia', indicating which of the two broad"
                 " classes of cells to look at"),
)
MTYPE = Term(
    "mtype",
    description=(
        "Morphological type as a capitalized string, e.g. L1_SAC, L23_CHC, PC, LBC."
        " See analysis_neuro.mtypes.SUPPORTED_MTYPE_LABELS to see possible values"
    ),
)
SYNAPSE_CLASS = Term(
    "synapse class",
    description=(
        "The type of synapses a cell forms. Can be EXC, INH or MOD (representing modulatory)."
        "EXC synapses use glutamate, INH synapses use GABA, and all other synapses are MOD")
)
GENE_EXPRESSION = Term(
    "gene expression",
    description=(
        "Gene that the cell expresses."
    ),
)
SPIKING_CLASS = Term(
    "spiking class", description=("Spiking class of a neuron, eiher FS or RS")
)

CELL_DENSITY = Term(
    "cell density ($cells/mm^3$)", description="Number of cells per cubic millimetre",
    measurement_method='cell_density'
)
CELL_COUNT = Term(
    "cell count", description="Total number of cells",
    measurement_method='cell_count')
REGION_VOLUME = Term(
    "volume ($mm^3$)",
    description="Volume of the measured parts of the brain in cubic millimetres",
    measurement_method='region_volume'
)


DATASET = Term("dataset", description="The dataset these data belongs to.")
CITATION = Term(
    "citation",
    description=(
        "A citation for these data. Zotero bibtex format is reccommended: "
        "<first-authors-last-name>_<first-word-of-title>_<year>."
        "For example billeh_systematic_2020"
    ),
)
NOTES = Term("notes", description="Any notes related to the dataset.")
CELL_ID = Term(
    "gid",
    description=(
        "A unique identifier for a cell. When a measurement returns an"
        " observation for each of multiple cells, this column should be included"
    ),
)
TRIAL_ID = Term(
    "trial id",
    description=(
        "A unique identifier for each trial in an experiment. When a measurement returns "
        "an observation for each of multiple trials, this column should be included"
    ),
)


HEMISPHERE = Term(
    "hemisphere", description="Hemisphere of the brain, either 'left' or 'right'"
)

COLUMN_RADIUS = Term(
    "column radius (um)",
    description="Radius of a central column to sample from, in micrometers.",
)

SQERROR = Term(
    "squared error",
    description="Statistical metric. Squared error between two datasets",
)
STD = Term("std ", description="standard deviation")
TSTAT = Term("t-statistic", description="Statistical metric, student's t-statistic")
PVALUE = Term(
    "p-value",
    description=(
        "Statistical metric, p-value. "
        "Probability of equal or greater deviation from the expectation value "
        "assuming the tested hypothesis."
    ),
)
SAMPLE_SIZE = Term(
    "sample size",
    description=(
        "Size of a sample taken from a population. "
        "Use when individual samples are unavailable or impractical"
    ),
)


STIMULUS = Term(
    "stimulus",
    description="Set of stimuli shown to a subject. Add stimuli to analysis_neuro.stimuli "
    "and refer to their name with this parameter. For example, if an animal is shown the allen"
    " institute's brain observatory drifting gratings stimulus this may be "
    " \"brain_observatory 1.1drifting gratings\""
)

VISUAL_STIMULUS = Term(
    "visual stimulus type",
    description="The type of visual stimulus used in an experiment. "
    "See stimuli.py for stimuli associated with different datasets.",
)

ANGLE_AZIMUTH = Term(
    "stimulus width (degrees)",
    description=("A tuple: the range of horizontal angles that the stimulus covers"
                 " in the visual field"),
)
ANGLE_ELEVATION = Term(
    "stimulus height (degrees)",
    description=("A tuple: the range of vertical angles that the stimulus covers"
                 " in the visual field.")
)
FRAME_RATE = Term(
    "stimulus framerate (Hz)", description="The framerate of the simulus used, in Hertz"
)
SPATIAL_FREQUENCY = Term(
    "spatial frequency (cycles / degree)",
    description="for a spatially periodic stimulus in the visual field this represents the "
    "number of cycles that occur per degree moved in the visual field.")
CONTRAST = Term(
    "visual contrast",
    description="Contrast of a visual stimulus, a number between 0 and 1."
    " with 1.0 the darkest parts of the stimulus"
    " are completely black, and the lightest parts completely white.")

TEMPORAL_FREQUENCY = Term(
    "temporal frequency (Hz)",
    description="For a periodic stimulus this represents the number of cycles occurring"
    " per unit time. "
    "For a drifting gratings stimulus for instance, the temporal frequency will be "
    " the spatial frequency times the drift speed. "
    "If instead of a number this is set to 'optimal', it means that the measurement is "
    "performed for several temporal frequencies and only the temporal frequency with maximal "
    "response is used to calculate the measured variable")

START_TIME = Term(
    "start time (ms)",
    description="The time at which the stimulus commences, preceded by a blank stimulus",
)
END_TIME = Term("end time (ms)", description="The time at which the stimulus ceases")
RESOLUTION = Term(
    "resolution",
    description="Pixel dimensions of the visual stimulus. tuple of (width, height)",
)
STIM_ORIENTATION = Term(
    "stimulus orientation (degrees)",
    description=(
        "Orientation(s) of the stimulus used in the experiment. "
        "for drifting gratings, 0 degrees drifts to the right and increasing orientation "
        "is clockwise. If a tuple, describes the set of different orientations used for"
        " a particular measurement (for example, orientation selectivity"
    ),
)

CONNECTION_PROBABILITY = Term(
    "connection probability",
    description=(
        "The probability that any specific cell in the presynaptic population "
        "is connected to any random cell in the postsynaptic population"
    ),
    measurement_method="connection_probability"
)
INTERSOMATIC_DISTANCE = Term(
    "interesomatic distance (um)",
    description=("The distance between the centers of the soma of a pair of cells"),
    measurement_method="intersomatic_distance"
)
SYNAPSES_PER_CONNECTION = Term(
    "synapses per connection",
    description=("The number of synapses between a pair of connected cells"),
    measurement_method="synapses_per_connection"
)
NUM_SYNAPSES = Term(
    "Number of synapses",
    description="The total number of synapses along a pathway",
    measurement_method="num_synapses"
)

PSP_AMPLITUDE = Term(
    "PSP amplitude (mV)",
    description="The size of the change in potential of a post-synaptic cell when the presynaptic cell is stimulated",
    measurement_method='psp_amplitude'
)

RESPONSE_CORRELATION = Term(
    "response correlation",
    description="pearson correlation in the responses of a pair of cells.",
    measurement_method='response_correlation'
)

INCLUDE_UNCONNECTED = Term(
    "include unconnected",
    description="Whether unconnected pairs of cells are included as zeros in a connectivity measurement."
)

ORIENTATION_PREFERENCE_DIFFERENCE = Term(
    "$\Delta$ preferred orientation",
    description ="The angular difference in preferred orientation between pairs of cells" 
)

ORIENTATION_SELECTIVITY = Term(
    "orientation selectivity index",
    description=(
        "The level of selectivity a neuron has for particular stimulus orientations under "
        "the experimental conditions"
    ),
    measurement_method="orientation_selectivity"
)
FIRING_RATE = Term(
    "firing rate (Hz)",
    description=("Rate of firing under the experimental conditions, in Hertz"),
    measurement_method="firing_rate"
)

PRESYNAPTIC = Term(
    "Presynaptic ",
    description=("Prefix to apply for specifying presynaptic"
                 " cell populations. e.g. PRESYNAPTIC + MTYPE"),
)
POSTSYNAPTIC = Term(
    "Postsynaptic ",
    description=("Prefix to apply for specifying postsynaptic"
                 " cell populations. e.g. POSTSYNAPTIC + MTYPE"),
)
MIN = Term(
    "minimal ",
    description=("Prefix to apply for specifying minimal value"),
)

MAX = Term(
    "maximal ",
    description=("Prefix to apply for specifying maximal value"),
)


def describe(*terms):
    """Describe the given terms, or all terms if nothing passed."""
    if len(terms) == 0:
        terms = ALL_TERMS.keys()
    return "\n\n".join([ALL_TERMS[term].describe() for term in terms])
