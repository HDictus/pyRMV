"""Collected analyses and validations."""

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

import importlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import analysis_neuro.analyses.data.jiang_distances as jiangd
from analysis_neuro import Analysis, plots, stats
from analysis_neuro import terminology as terms

DATADIR = files("analysis_neuro.analyses.data")


def _wide_barplot(x, y, hue):
    _, ax = plt.subplots(figsize=(len(np.unique(x)) * len(np.unique(hue)) / 3, 5))
    return sns.barplot(x=x, y=y, hue=hue, ax=ax)


ji_innervation_2016 = Analysis(
    observations=pd.read_csv(DATADIR.joinpath("ji_innervation_2016.csv"), index_col=0),
    measurement=terms.FRACTION_INNERVATED,
    stats=stats.binom_test,
    verdict=stats.PooledPValueThreshold(0.05),
    plotter=_wide_barplot,
)

ji_relative_2016 = Analysis(
    measurement=terms.RELATIVE_EXCITATION,
    observations=pd.read_csv(DATADIR.joinpath("ji_relative_2016.csv"), index_col=0),
    plotter=_wide_barplot,
    stats=stats.mann_whitney_u,
    verdict=stats.PooledPValueThreshold(0.05),
)


lien_fraction_excitation_2018 = Analysis(
    observations=importlib.import_module(
        "analysis_neuro.analyses.data.lien_2013"
    ).fraction_excitation,
    measurement=terms.FRACTION_EXCITATION_PER_CONNECTION,
    stats=stats.bootstrap_mean,
    plotter=plots.hist,
    verdict=stats.PooledPValueThreshold(0.05),
)


lien_thalamocortical_current_2013 = lien_fraction_excitation_2018.with_fields(
    doc="""
    We evaluate the strength of thalamocortical exitation in L4PCs by comparing to Lien et al. 2013
    This is likely to be an upper bound, as cortical silencing increases the activity of thalamocortical cells.

    In the experiment Lien and Scanziani reported that the mean current recorded was independent of stimulus direction.
    For this reason we only use one stimulus direction, so that models do not need to unnecessarily simulate
    multiple.
    """,
    observations=importlib.import_module(
        "analysis_neuro.analyses.data.lien_2013"
    ).thalamocortical_current,
    measurement=terms.SOMATIC_CURRENT,
)

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
    observations=pd.read_csv(DATADIR.joinpath("schuz_neuron_density_1989.csv")),
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
            "https://bbpteam.epfl.ch/project/spaces/display/NEX/LNMC+Cell+density"
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
        bins = np.linspace(inddata[dependent].min(), inddata[dependent].max(), 100)
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
    observations=importlib.import_module(
        "analysis_neuro.analyses.data.siegle_2019"
    ).osi,
    plotter=_histogram_siegle,
    stats=stats.mann_whitney_u,
    verdict=stats.PooledPValueThreshold(0.05),
)

siegle_dsi_2019 = siegle_osi.with_fields(
    doc="""
    We compare to the levels of direction selectivity observed in
    Seigle et al. 2019""",
    measurement=terms.DIRECTION_SELECTIVITY,
    observations=importlib.import_module(
        "analysis_neuro.analyses.data.siegle_2019"
    ).dsi
)

siegle_spontaneous_2019 = Analysis(
    doc="""
    We compare to the firing rate distribution for blank gray stimuli
    observed in Siegle et al. 2019""",
    measurement=terms.FIRING_RATE,
    observations=importlib.import_module(
        "analysis_neuro.analyses.data.siegle_2019"
    ).spontaneous,
    plotter=_histogram_siegle,
    stats=stats.mann_whitney_u,
    verdict=stats.PooledPValueThreshold(0.05),
)


ma_spontaneous_2010 = Analysis(
    observations=importlib.import_module(
        "analysis_neuro.analyses.data.ma_2010"
    ).spontaneous,
    measurement=terms.FIRING_RATE,
    plotter=plots.hist,
    stats=stats.bootstrap_mean,
    verdict=stats.PooledPValueThreshold(0.05),
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


def _cossell_respcorr(data, dependent, independent, compare=terms.DATASET):
    """Check whether 50% of psp strength is in 7% most correlated pairs"""
    # pylint: disable=too-many-locals,unused-argument
    hypotheses = {}
    n_pairs = 179 + 279 + 40 + 14 + 8
    sevenpct = int(np.floor(n_pairs * 0.07))
    n_samples = 1000
    # presently we assume only one set of parameters
    for label, data_for_dataset in data.groupby(compare):
        hypothesis = (
            f"The observation that the 7% pairs with highest {independent} "
            f"account for 50% of {dependent} "
            f"could be made from the dataset {label}."
        )
        fraction_accounted = []
        for _ in range(n_samples):
            sample = np.random.choice(np.arange(len(data_for_dataset)), size=n_pairs)
            sample_data = data_for_dataset.iloc[sample].sort_values(
                independent, ascending=False
            )
            top7pct = sample_data.iloc[:sevenpct]
            fraction = top7pct[dependent].sum() / sample_data[dependent].sum()
            fraction_accounted.append(fraction)
        fraction_accounted = np.array(fraction_accounted)
        meanfrac = np.mean(fraction_accounted)
        if meanfrac <= 0.5:
            pvalue = np.mean(fraction_accounted >= 0.5)
        else:
            pvalue = np.mean(fraction_accounted <= 0.5)
        hypotheses[hypothesis] = pd.DataFrame({terms.PVALUE: [pvalue]})
    return hypotheses


# pylint: disable=unused-argument
def _cossell_scatter(data, dependent, independent, compare):
    out = {}
    for label, dataset in data.groupby(compare):
        fig = plt.figure()
        plt.scatter(
            dataset[terms.RESPONSE_CORRELATION], dataset[terms.PSP_AMPLITUDE], alpha=0.1
        )
        points = np.linspace(-1, 1, 20)
        mean_psp = (
            dataset[terms.PSP_AMPLITUDE]
            .groupby(pd.cut(dataset[terms.RESPONSE_CORRELATION], points))
            .mean()
        )
        centers = [
            np.mean([interval.left, interval.right]) for interval in mean_psp.index
        ]
        plt.plot(centers, mean_psp.values, color="black")
        # a, b, c = np.polyfit(
        #    data[terms.RESPONSE_CORRELATION], data[terms.PSP_AMPLITUDE], deg=2
        # )
        # plt.plot(points, a * points**2 + b * points + c, color="black")
        out[label] = fig
    return out


cossell_correlation_psp_2015 = Analysis(
    doc="""
    Cossell et al. 2015 showed that the size of the PSP from a given excitatory connection
    in L23 of mouse visual cortex is strongly determined by the response correlation
    between these cells in response to natural stimuli.
    We do not directly compare to their experimental data, as this is not available.
    Rather, we check if their observation that the 7% most correlated pairs
    account for 50% of total psp strength is reproduced.
    """,
    observations=pd.DataFrame(
        {
            terms.REGION: "VISp",
            terms.LAYER: "L23",
            terms.SYNAPSE_CLASS: ["EXC"],
            terms.IMAGED_WIDTH: 263,
            terms.IMAGED_HEIGHT: 255,
            terms.MIN + terms.DEPTH: 135,
            terms.MAX + terms.DEPTH: 191,
            terms.STIMULUS: "natural-images",
            terms.DATASET: "Cossell et al. 2015",
            terms.INCLUDE_UNCONNECTED: True,
        }
    ),
    measurement=[terms.RESPONSE_CORRELATION, terms.PSP_AMPLITUDE],
    dependent=terms.PSP_AMPLITUDE,
    independent=terms.RESPONSE_CORRELATION,
    stats=_cossell_respcorr,
    plotter=_cossell_scatter,
    verdict=stats.PooledPValueThreshold(0.05),
)


cossell_connprob_corr_2015 = Analysis(
    measurement=terms.CONNECTION_PROBABILITY,
    observations=pd.DataFrame(
        {
            terms.MIN + terms.RESPONSE_CORRELATION: [-0.1, 0, 0.1, 0.2, 0.3],
            terms.MAX + terms.RESPONSE_CORRELATION: [0, 0.1, 0.2, 0.3, 0.4],
            terms.CONNECTION_PROBABILITY: [16 / 179, 37 / 279, 12 / 40, 5 / 14, 5 / 8],
            terms.SAMPLE_SIZE: [179, 279, 40, 14, 8],
            **cossell_correlation_psp_2015.observations.drop(
                columns=terms.INCLUDE_UNCONNECTED
            ).iloc[0],
        }
    ),
    plotter=sns.barplot,
    stats=stats.binom_test,
    verdict=stats.PooledPValueThreshold(0.05),
)


# pylint: disable=unused-argument
def _histogram_with_cossell_digitized(data, dependent, independent, compare):
    cossell_digitized = pd.read_csv(
        DATADIR.joinpath("cossell_response_correlation_2015.csv")
    ).values
    cossell_digitized[:, 1] /= np.trapz(
        cossell_digitized[:, 1], cossell_digitized[:, 0]
    )
    fig = plt.figure()
    # assumes no independent
    for label, dataset in data.groupby(compare):
        plt.hist(dataset[dependent], alpha=0.4, label=label, density=True)
    plt.plot(
        cossell_digitized[:, 0],
        cossell_digitized[:, 1],
        label="Cossell et al. 2015 (digitized)",
    )
    plt.legend()
    return {"hist": fig}


cossell_response_correlation_2015 = Analysis(
    # The paper does not provide actual numbers - only counts for bins
    # Therefore, the biodata is not in the analysis itself but loaded
    # separately by the plotter above
    observations=cossell_correlation_psp_2015.parameters,
    measurement=terms.RESPONSE_CORRELATION,
    plotter=_histogram_with_cossell_digitized,
)


lee_connprob_2016 = Analysis(
    measurement=terms.CONNECTION_PROBABILITY,
    observations=pd.DataFrame(
        {
            terms.DATASET: "Lee et al. 2016",
            terms.MIN + terms.ORIENTATION_PREFERENCE_DIFFERENCE: [0, 22.5, 45, 67.5],
            terms.MAX + terms.ORIENTATION_PREFERENCE_DIFFERENCE: [22.5, 45, 67.5, 90],
            terms.REGION: "VISp",
            terms.LAYER: "L23",
            terms.SYNAPSE_CLASS: "EXC",
            terms.COLUMN_RADIUS: 150,
            terms.SAMPLE_SIZE: [506, 458, 496, 520],
            terms.CONNECTION_PROBABILITY: [10 / 506, 11 / 458, 4 / 496, 4 / 520],
        }
    ),
    plotter=sns.barplot,
    stats=stats.binom_test,
    verdict=stats.PooledPValueThreshold(0.05),
)


def mtype_to_mtype_connprob(*models, radius=125):
    """Visualize the connection probabilty between all mtypes for one or more models.

    Will use all pairs of cells within the given radius (default 125 um). It will generate
    a heatplot of the connection probabilit and if provided two models will test whether their
    connection probabilities are statistically distinguishable.
    """
    mtypes = np.unique([mt for md in models for mt in md.mtype()[terms.MTYPE]])

    pathways = pd.DataFrame(
        [
            {
                terms.PRESYNAPTIC + terms.MTYPE: pre_mtype,
                terms.POSTSYNAPTIC + terms.MTYPE: post_mtype,
                terms.COLUMN_RADIUS: radius,
            }
            for pre_mtype in mtypes
            for post_mtype in mtypes
        ]
    )

    analysis = Analysis(
        measurement=terms.CONNECTION_PROBABILITY,
        observations=pathways,
        plotter=plots.pathway_heatmap,
        stats=stats.binom_test,
        verdict=stats.PooledPValueThreshold(0.05),
    )
    return analysis(*models)


def mtype_to_mtype_syn_per_conn(*models, radius=125):
    """Visualize the synapses per connection between all mtypes for one or more models.

    Will use all pairs of cells within the given radius (default 125 um). It will generate
    a heatplot of the mean synapses per connection and if provided two models
    will test whether their synapses per connection are lognormally distributed and
    statistically distinguishable.
    """
    mtypes = np.unique([mt for md in models for mt in md.mtype()[terms.MTYPE]])

    pathways = pd.DataFrame(
        [
            {
                terms.PRESYNAPTIC + terms.MTYPE: pre_mtype,
                terms.POSTSYNAPTIC + terms.MTYPE: post_mtype,
                terms.COLUMN_RADIUS: radius,
            }
            for pre_mtype in mtypes
            for post_mtype in mtypes
        ]
    )

    analysis = Analysis(
        measurement=terms.SYNAPSES_PER_CONNECTION,
        observations=pathways,
        plotter=plots.pathway_heatmap,
        stats=lambda *a, **k: {
            **stats.is_lognormal(*a, **k),
            **stats.lognorm_ttest(*a, **k),
        },
        verdict=stats.PooledPValueThreshold(0.05),
    )
    return analysis(*models)


def mtype_to_mtype_nsyn(*models, radius=125):
    """Visualize the total number of synapses along mtype pathways."""
    mtypes = np.unique([mt for md in models for mt in md.mtype()[terms.MTYPE]])

    pathways = pd.DataFrame(
        [
            {
                terms.PRESYNAPTIC + terms.MTYPE: pre_mtype,
                terms.POSTSYNAPTIC + terms.MTYPE: post_mtype,
                terms.COLUMN_RADIUS: radius,
            }
            for pre_mtype in mtypes
            for post_mtype in mtypes
        ]
    )

    analysis = Analysis(
        measurement=terms.NUM_SYNAPSES,
        observations=pathways,
        plotter=plots.pathway_heatmap,
    )
    return analysis(*models)


def mtype_to_mtype_connectivity(*models, radius=125):
    """Visualize mtype to mtype connectivity matrices for one or more models.

    Generates heatmaps for connection probability, synapses per connection, and total synapses
    between all different mtypes within a column of a set radius (in um).
    A separate set of heatmaps is created for each model.
    When multiple models are passed, their connection probability and synapses per connection
    are also
    """
    warnings.warn(
        "This analysis is deprecated, please use the individual analyses:"
        "mtype_to_mtype_connprob, mtype_to_mtype_syn_per_conn, mtype_to_mtype_nsyn",
        DeprecationWarning,
    )
    return {
        terms.CONNECTION_PROBABILITY: mtype_to_mtype_connprob(*models, radius=radius),
        terms.SYNAPSES_PER_CONNECTION: mtype_to_mtype_syn_per_conn(
            *models, radius=radius
        ),
        terms.NUM_SYNAPSES: mtype_to_mtype_nsyn(*models, radius=radius),
    }


for varname in dir():
    # pylint: disable=eval-used
    value = eval(varname)
    if isinstance(value, Analysis):
        value.__name__ = varname
