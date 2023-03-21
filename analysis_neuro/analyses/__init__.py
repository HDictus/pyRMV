"""Collected analyses and validations."""
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

from analysis_neuro import Analysis
from analysis_neuro import terminology as terms
from analysis_neuro import stats, plots
import analysis_neuro.analyses.data.jiang_distances as jiangd
from analysis_neuro.analyses.data.siegle_2019 import osi


DATADIR = files('analysis_neuro.analyses.data')


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
    observations=pd.read_csv(DATADIR.joinpath("schuz_neuron_density_1989.csv"), index_col=0),
    measurement=terms.CELL_DENSITY,
    plotter=sns.barplot,
    stats=stats.squared_error,
)


keller_density_2018 = Analysis(
    observations=pd.read_csv(DATADIR.joinpath("keller_2018.csv")),
    measurement=terms.CELL_DENSITY,
    plotter=sns.barplot,
    stats=stats.ttest,
    verdict=stats.PooledPValueThreshold(threshold=0.05),
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


jiang_connprob_2015 = Analysis(
    doc="""
    We compare connection probabilities to those observed by Jiang et al.
    """,
    observations=pd.read_csv(DATADIR.joinpath("jiang_connprob_2015.csv")),
    measurement=terms.CONNECTION_PROBABILITY,
    plotter=plots.crossplot,
    stats=stats.binom_test,
    verdict=stats.PooledPValueThreshold(0.05),
)


def _histogram(data, dependent, independent, compare):
    # pylint: disable=unused-argument
    fig = plt.figure()
    bins = np.linspace(data[dependent].min(), data[dependent].max(), 13)
    for label, dataset in data.groupby(compare):
        plt.hist(
            dataset[dependent], bins=bins, density=True, label=str(label), alpha=0.6
        )
    plt.legend()
    return fig


jiang_intersomatic_2015 = Analysis(
    doc="""
    We compare intersomatic distances observed for L5 pyramidal cells sampled
    in Jiang's paper (supplementary material, fig S13 B) to validate the
    assumption in jiang_connprob_2015 that a column size of 75 um accurately
    recreates the distance profiles of their study.
    """,
    measurement=terms.INTERSOMATIC_DISTANCE,
    observations=jiangd.jiang_intersomatic_2015,
    plotter=_histogram,
)


def _histogram_siegle(data, dependent, independent, compare):
    # pylint: disable=unused-argument
    out = {}
    for key, inddata in data.groupby(independent):
        fig = plt.figure()
        bins = np.linspace(inddata[dependent].min(), inddata[dependent].max(), 13)
        for label, dataset in inddata.groupby(compare):
            plt.hist(
                dataset[dependent], bins=bins, density=True, label=str(label), alpha=0.6
            )
            plt.legend()
        out[str(key)] = fig
    return out


siegle_osi_2019 = Analysis(
    doc="""
    We compare to the levels of orientation selectivity observed in
    Seigle et al. 2019""",
    measurement=terms.ORIENTATION_SELECTIVITY,
    observations=osi,
    plotter=_histogram_siegle,
)


pala_peterson_conprob_2015 = Analysis(
    doc="""
    We compare to the connectivity observed in
    Pala, Peterson 2015""",
    observations=pd.read_csv(DATADIR / "pala_peterson_conprob_2015.csv"),
    measurement=terms.CONNECTION_PROBABILITY,
    plotter=plots.crossplot,
    stats=stats.binom_test,
    verdict=stats.PooledPValueThreshold(0.05),
)


def mtype_to_mtype_connectivity(*models):
    """Visualize mtype to mtype connectivity matrices for one or more models.

    Generates heatmaps for connection probability, synapses per connection, and total synapses
    between all different mtypes within a 125um radius column. A separate set of heatmaps is created
    for each model.
    """
    mtypes = np.unique([mt for md in models for mt in md.mtype()[terms.MTYPE]])

    pathways = pd.DataFrame([{
        terms.PRESYNAPTIC + terms.MTYPE: pre_mtype,
        terms.POSTSYNAPTIC + terms.MTYPE: post_mtype,
        terms.COLUMN_RADIUS: 125}
        for pre_mtype in mtypes for post_mtype in mtypes])
    conn_prob_matrix = Analysis(
        measurement=terms.CONNECTION_PROBABILITY,
        observations=pathways,
        plotter=plots.pathway_heatmap,
        stats=stats.binom_test,
        verdict=stats.PooledPValueThreshold(0.05))
    syn_conn_matrix = conn_prob_matrix.with_fields(measurement=terms.SYNAPSES_PER_CONNECTION,
                                                   stats=lambda *a, **k: dict(
                                                       **stats.is_lognormal(*a, **k),
                                                       **stats.lognorm_ttest(*a, **k)),
                                                   verdict=stats.PooledPValueThreshold(0.05))
    num_syn_matrix = conn_prob_matrix.with_fields(measurement=terms.NUM_SYNAPSES,
                                                  stats=None,
                                                  verdict=None)
    return {terms.CONNECTION_PROBABILITY: conn_prob_matrix(*models),
            terms.SYNAPSES_PER_CONNECTION: syn_conn_matrix(*models),
            terms.NUM_SYNAPSES: num_syn_matrix(*models)}


for varname in dir():
    # pylint: disable=eval-used
    value = eval(varname)
    if isinstance(value, Analysis):
        value.__name__ = varname
