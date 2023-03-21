"""Tools for statistical hypothesis testing."""
import warnings
from typing import List
from scipy import stats
import numpy as np
import pandas as pd
from analysis_neuro import Assumption
from analysis_neuro import terminology as terms


def _iter_compare(data, compare):
    datasets = data.groupby(compare)
    already_compared = set()
    for label1, dataset1 in datasets:
        for label2, dataset2 in datasets:
            if (label2, label1) in already_compared:
                continue
            if label1 == label2:
                continue
            already_compared.add((label1, label2))
            yield label1, dataset1, label2, dataset2


def squared_error(data, dependent, independent, compare):
    """Compute the squared error between groups in data.

    Hypothesis:
       the compared values are similar between groups

    Metrics:
       squared error
    """
    hypotheses = {}

    def by_ind(dataset):
        return dataset.set_index(independent)[dependent]

    for label1, dataset1, label2, dataset2 in _iter_compare(data, compare):
        hypothesis = f"{dependent} is similar for {compare}" f" {label1} and {label2}"
        sqerr = (by_ind(dataset1) - by_ind(dataset2)) ** 2
        hypotheses[hypothesis] = sqerr.rename(terms.SQERROR).reset_index()
    return hypotheses


def _ttest_1samp(dataset1, dataset2, dependent):
    std = dataset1[terms.STD + dependent]
    sample_size = dataset1[terms.SAMPLE_SIZE]
    tstat = np.abs((dataset1[dependent].values - dataset2[dependent].values)) / (
        std.values / np.sqrt(sample_size.values)
    )
    pvalue = [stats.t.sf(tt, df=ss - 1) for tt, ss in zip(tstat, sample_size)]
    return tstat, pvalue


def _has_std_and_size(dataset, dependent):
    if terms.STD + dependent not in dataset:
        return False
    if np.isnan(dataset[terms.STD + dependent]).all():
        return False
    if terms.SAMPLE_SIZE not in dataset:
        return False
    return not np.isnan(dataset[terms.SAMPLE_SIZE]).all()


def _has_samples(dataset, dependent, independent):
    return (dataset.groupby(independent)[dependent].count() > 1).any()


def ttest(data, dependent, independent, compare):
    """Perform a two-tailed t-test between comparable datapoints.

    Assumptions:
        Samples are independent of one another.
        Samples are approximately normally distributed.

    Hypotheses:
        The samples have the same population mean.

    Metrics:
        T-statistic
        P-value
    For the time being, this t-test only supports comparisons where
    one of the datasets provides a standard deviation and sample size.
    """
    hypotheses = {}

    for label1, dataset1, label2, dataset2 in _iter_compare(data, compare):
        if _has_std_and_size(dataset1, dependent):
            if _has_std_and_size(dataset2, dependent):
                raise NotImplementedError(
                    "Comparison between two populations (2-sample t-test) "
                    "has not yet been implemented. Please make a pull-request."
                    "Current functionality supports 1-sample t-test"
                )
        else:
            if _has_std_and_size(dataset2, dependent):
                label1, label2 = (label2, label1)
                dataset1, dataset2 = (dataset2, dataset1)
            else:
                if _has_samples(dataset1, dependent, independent) or _has_samples(
                    dataset2, dependent, independent
                ):
                    raise NotImplementedError(
                        "Performing t-test for samples (instead of precomputed"
                        " standard deviation and sample size) is not yet "
                        "implemented. please make a pull-request"
                    )
                raise ValueError(
                    "The data provided do not contain the information "
                    "necessary to perform any kind of t-test"
                )

        hypothesis = (
            f"population mean of {dependent} for {label1} is "
            f"equal to the value of {dependent} for {label2}"
        )
        tstat, pvalue = _ttest_1samp(dataset1, dataset2, dependent)
        hypotheses[hypothesis] = (
            dataset1[independent]
            .reset_index(drop=True)
            .assign(**{terms.TSTAT: tstat, terms.PVALUE: pvalue})
        )

    return hypotheses


# pylint: disable=too-few-public-methods
class PooledPValueThreshold:
    """Pools the p-values of some statistical test across observations.

    The hypothesis fails if any p-value is below some threshold.
    This threshold is adjusted with a Bonferroni correction for the number of observations.
    """

    def __init__(self, threshold=0.05):
        """Initialize."""
        self.threshold = threshold

    def __call__(self, hypotheses, **kw):
        """Run."""
        verdicts = {}
        for hypothesis, data in hypotheses.items():
            pvalue = data[terms.PVALUE]
            bonferroni_threshold = self.threshold / data.shape[0]
            fail = (pvalue <= bonferroni_threshold).any()
            verdicts[hypothesis] = "Fail" if fail else "Pass"
        return verdicts


def _have_samples(data: pd.DataFrame):
    return terms.SAMPLE_SIZE in data.columns\
        and not np.isnan(data[terms.SAMPLE_SIZE]).all()


def binom_test(data: pd.DataFrame, dependent: str, independent: list, compare: str):
    """Test that a measured probability is the same between two values.

    Assumptions:
       Samples are independent.
       The sample with the larger sample size precisely represents the true,
         probability for that sample. If one sample has a sample size of NaN,
         then that one is assumed to represent the true probability.

    Hypothesis:
       The probability represented by the dependent variable in one population
       is equal to the dependent variable in the other.

    Arguments:
       data: the data to perform the test for
       dependent: the column containing the dependent variable, a probability
       independent: columns of independent variables
       compare: the column distinguishing the datasets to compare
    """

    def vector_binomtest(sampleprob, trials, probability):
        print(sampleprob, trials, probability)
        return [np.nan if any(np.isnan(np.float32([s, n, p]))) else
                stats.binomtest(int(s * n), int(n), p).pvalue
                for s, n, p in zip(sampleprob, trials, probability)]

    def binomtest(data1, data2, label1, label2):
        assume_1_accurate = np.logical_or(
            np.isnan(data1[terms.SAMPLE_SIZE].values),
            data1[terms.SAMPLE_SIZE].values > data2[terms.SAMPLE_SIZE].values)
        warn_assume_exact(data1[independent][assume_1_accurate], label1)
        warn_assume_exact(data2[independent][~assume_1_accurate], label2)

        pvalues = np.zeros(assume_1_accurate.shape)

        prob1 = data1[dependent][assume_1_accurate]
        num2 = data2[terms.SAMPLE_SIZE][assume_1_accurate]
        samp2 = data2[dependent][assume_1_accurate]

        pvalues[assume_1_accurate] = vector_binomtest(samp2, num2, prob1)

        prob2 = data2[dependent][~assume_1_accurate]
        num1 = data1[terms.SAMPLE_SIZE][~assume_1_accurate]
        samp1 = data1[dependent][~assume_1_accurate]

        pvalues[~assume_1_accurate] = vector_binomtest(samp1, num1, prob2)
        return pvalues

    def warn_assume_exact(for_params, label):
        warnings.warn(
            f"assuming the values of {dependent} for the {label} are the ground truth for {label}"
            f" for the following measurements: {for_params}",
            Assumption)

    hypotheses = {}
    for label1, dataset1, label2, dataset2 in _iter_compare(data, compare):

        hypothesis = (
            f"The probability {dependent} is the same for {label1} as for {label2}."
        )
        pvalues = binomtest(dataset1, dataset2, label1, label2)
        statistic = dataset1[independent]
        statistic[terms.PVALUE] = list(pvalues)
        hypotheses[hypothesis] = statistic

    return hypotheses


def is_lognormal(data, dependent, independent, compare):
    """Test that the dependent variable is lognormally distributed.

    One test for each value of the independent variables.

    Hypothesis:
       the dependent variable is lognormally distributed.
    """
    hypotheses = {}
    for label, dataframe in data.groupby(compare):
        pvals = dataframe.groupby(independent)[dependent].apply(
            lambda a: stats.normaltest(np.log(a)).pvalue)
        hypotheses[f"{dependent} is lognormally distributed for {label}"] =\
            pvals.reset_index().rename(columns={dependent: terms.PVALUE})
    return hypotheses


def lognorm_ttest(data: pd.DataFrame, dependent: str, independent: List[str], compare: str):
    """Run a t-test for lognormally distributed values.

    Arguments:
        data: A pandas dataframe containing the measured values. one column per variable
              and one row per sample
        dependent: the dependent variable, the value to compare across
        independent: the independent variables. Perfrom a separate comparison for each unique
            combination of these
        compare: variable to compare values between.

    Assumptions:
       the samples are lognormally distributed
       samples are independent

    Hypothesis:
       The dependent variable has the same mean value for both compared populations
    """

    def log_ttest(dataframe, label1, label2):
        return stats.ttest_ind(np.log(dataframe[label1]), np.log(dataframe[label2])).pvalue

    hypotheses = {}
    for label1, data1, label2, data2 in _iter_compare(data, compare):
        data1.set_index(independent, inplace=True)
        data2.set_index(independent, inplace=True)
        to_applyon = data1.rename(columns={dependent: label1})
        to_applyon[label2] = data2[dependent]
        p_value = to_applyon.reset_index().groupby(independent).apply(log_ttest, label1, label2)
        hypothesis = f'the population mean of {dependent} is the same for {label1} and {label2}'
        hypotheses[hypothesis] = p_value.reset_index().rename(columns={0: terms.PVALUE})
    return hypotheses
