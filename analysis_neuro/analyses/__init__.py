"""Collected analyses and validations."""
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from analysis_neuro import Analysis
from analysis_neuro import terminology as terms
from analysis_neuro import stats


schuz_density_1989 = Analysis(
    doc="""
    We evaluate the similarity of the total neuron density in the primary
    visual cortex of the model to in-vivo values by comparing to the
    mean density observed in @schuz_density_1989.
    Because Schuz does not report the individual densities for the three
    mice used, we cannot perform a statistical test.
    So instead we use a mean squared error to quantify the mismatch.
    We cannot render a verdict on the result, only assign a score.
    """,
    observations=pd.read_csv(
        Path(__file__).parent / "data" / "schuz_neuron_density_1989.csv", index_col=0
    ),
    measurement=terms.CELL_DENSITY,
    plotter=sns.barplot,
    stats=stats.squared_error,
)


keller_density_2018 = Analysis(
    observations=pd.read_csv(Path(__file__).parent / "data" / "keller_2018.csv"),
    measurement=terms.CELL_DENSITY,
    plotter=sns.barplot,
    stats=stats.TTest(),
    verdict=stats.PooledPvalueThreshold(threshold=0.05),
)


_collected_thalamus_data = pd.DataFrame(
    {
        terms.REGION: ["RT", "VPM", "VPL", "LGd", "LP"],
        terms.CELL_COUNT: [68749.57, 70919.19, 57466.91, 84434, 87476],
        terms.NEURON_OR_GLIA: "neuron",
        terms.CITATION: ["bertschy_internal_2021"] * 3
        + ["evangelio_thalamocortical_2018"] * 2,
        terms.NOTES: [
            ("https://bbpteam.epfl.ch/project/spaces/display/NEX/LNMC+Cell+density")
        ]
        * 3
        + [""] * 2,
        terms.DATASET: ["experiment"] * 3 + ["experiment"] * 2,
    }
)

thalamic_nuclei_cell_counts = Analysis(
    doc="""
    We compare the total cell counts in the thalamus model(s) to various data
    collected from the literature and in internal projects
    """,
    observations=_collected_thalamus_data,
    measurement=terms.CELL_COUNT,
    plotter=sns.barplot,
    stats=stats.squared_error,
)


for varname in dir():
    # pylint: disable=eval-used
    value = eval(varname)
    if isinstance(value, Analysis):
        value.__name__ = varname
