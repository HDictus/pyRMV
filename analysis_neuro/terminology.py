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


_ALLTERMS = {}


class Term(str):
    """A string with an associated description defining it."""
    
    def __new__(cls, term, *args, **kwargs):
        """
        Arguments
        ----------
        `term`: single word label.
        """
        return super().__new__(cls, term)

    def __init__(self, term, description="No description provided", **kwargs):
        """
        Arguments
        ------------
        `term`: single word label
        `description`: description of the term
        any additional keyword arguments provided become attributes,
        allowing the combination of terminology in heirarchies
        """
        _ALLTERMS[term] = self
        self.description = description
        for kw, arg in kwargs.items():
            setattr(self, kw, arg)
        return super().__init__()

    def document(self, label=None):
        if label is None:
            label = self
        return "{}: {}" .format(label, self.description)

    def __add__(self, added):
        description = [self.description.format(added=added)]
        if isinstance(added, Term):
            description.append(f"{added} : \n{added.description}")
        return self.__class__(
            str(self) + str(added),
            ("\n\n".join(description)))

    
REGION = Term(
    "region",
    description="Acronym of a brain region according to AIBS atlas naming convention")
LAYER = Term(
    "layer",
    description="Layer of some brain region as a capitalized acronym: e.g. L1, L23, SP, VPL")
NEURON_OR_GLIA = Term(
    "neuron or glia?",
    description="Either 'neuron' or 'glia', indicating which of the two broad classes of cells to look at")
MTYPE = Term(
    "mtype",
    description=("Morphological type as a capitalized string, e.g. L1_SAC, L23_CHC, PC, LBC."
                 " See analysis_neuro.mtypes.SUPPORTED_MTYPE_LABELS to see possible values")
)
SPIKING_CLASS = Term(
    "spiking class",
    description=("Spiking class of a neuron, eiher FS or RS"))

CELL_DENSITY = Term(
    "cell density ($cells/mm^3$)",
    description="Number of cells per cubic millimetre")
CELL_COUNT = Term(
    "cell count",
    description="Total number of cells")
REGION_VOLUME = Term(
    "volume ($mm^3$)",
    description="Volume of the measured parts of the brain in cubic millimetres")


DATASET = Term(
    "dataset",
    descripion="The dataset these data belongs to.")
CITATION = Term(
    "citation",
    descripion=("A citation for these data. Zotero bibtex format is reccommended: "
                "<first-authors-last-name>_<first-word-of-title>_<year>."
                "For example billeh_systematic_2020"))
NOTES = Term(
    "notes",
    description="Any notes related to the dataset.")
CELL_ID = Term(
    'gid',
    description=("A unique identifier for a cell. When a measurement returns an"
                 " observation for each of multiple cells, this column should be included"))
TRIAL_ID = Term(
    'trial id',
    description=('A unique identifier for each trial in an experiment. When a measurement returns '
                 'an observation for each of multiple trials, this column should be included'))


HEMISPHERE = Term(
    "hemisphere",
    description="Hemisphere of the brain, either 'left' or 'right'")

COLUMN_RADIUS = Term(
    "column radius (um)",
    description="Radius of a central column to sample from, in micrometers.")

SQERROR = Term(
    "squared error",
    description="Statistical metric. Squared error between two datasets")
STD = Term(
    "std ",
    description="standard deviation")
TSTAT = Term(
    "t-statistic",
    description="Statistical metric, student's t-statistic")
PVALUE = Term(
    "p-value",
    description=("Statistical metric, p-value. "
                "Probability of equal or greater deviation from the expectation value "
                 "assuming the tested hypothesis."))
SAMPLE_SIZE = Term(
    "sample size",
    description=("Size of a sample taken from a population. "
                 "Use when individual samples are unavailable or impractical"))

VISUAL_STIMULUS = Term(
    "visual stimulus type",
    description="The type of visual stimulus used in an experiment, e.g. drifting grating")
STIMULUS_AZIMUTH = Term(
    "stimulus width (degrees)",
    descripion="The width of the visual stimulus used in degrees on the azimuth plane")
STIMULUS_ELEVATION = Term(
    "stimulus height (degrees)",
    description="The width of the visual stimulus used in degrees on the elevation plane")
FRAME_RATE = Term(
    "stimulus framerate (Hz)",
    description="The framerate of the simulus used, in Hertz")
DRIFT_SPEED = Term(
    "simulus drift speed (degrees/s)",
    description="The speed at which a visual stimulus drifts across the visual field")
START_TIME = Term(
    "start time (ms)",
    description="The time at which the stimulus commences, preceded by a blank stimulus")
END_TIME = Term(
    "end time (ms)",
    description="The time at which the stimulus ceases")
RESOLUTION = Term(
    "resolution",
    description="Pixel dimensions of the visual stimulus. tuple of (width, height)")
STIM_ORIENTATION = Term(
    "stimulus orientation (degrees)",
    description=(
        "Orientation(s) of the stimulus used in the experiment. "
        "for drifting gratings, 0 degrees drifts to the right and increasing orientation "
        "is clockwise. If a tuple, describes the set of different orientations used for"
        " a particular measurement (for example, orientation selectivity"))

CONNECTION_PROBABILITY = Term(
    "connection probability",
    description=("The probability that any specific cell in the presynaptic population "
                 "is connected to any random cell in the postsynaptic population"))
INTERSOMATIC_DISTANCE = Term(
    "interesomatic distance (um)",
    description=("The distance between the centers of the soma of a pair of cells"))
ORIENTATION_SELECTIVITY = Term(
    "orientation selectivity index",
    description=("The level of selectivity a neuron has for particular stimulus orientations under "
                 "the experimental conditions"))
FIRING_RATE = Term(
    "firing rate (Hz)",
    description=("Rate of firing under the experimental conditions, in Hertz"))

PRESYNAPTIC = Term(
    "Presynaptic ",
    description="Prefix to apply for specifying presynaptic cell populations. e.g. PRESYNAPTIC + MTYPE")
POSTSYNAPTIC = Term(
    "Postsynaptic ",
    description="Prefix to apply for specifying postsynaptic cell populations. e.g. POSTSYNAPTIC + MTYPE")


measurements = {
    CELL_DENSITY: {"method name": "cell_density"},
    CELL_COUNT: {"method name": "cell_count"},
    CONNECTION_PROBABILITY: {"method name": "connection_probability"},
    INTERSOMATIC_DISTANCE: {"method name": "intersomatic_distance"},
    ORIENTATION_SELECTIVITY: {"method name": "orientation_selectivity"},
    FIRING_RATE: {'method name': 'firing_rate'}
}


def describe(*terms):
    if len(terms) == 0:
        terms = _ALLTERMS.keys()
    return "\n\n".join([f"{term}: {_ALLTERMS[term].description}" for term in terms])
        
