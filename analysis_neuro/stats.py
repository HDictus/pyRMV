"""Tools for statistical hypothesis testing"""
from scipy import stats
import numpy as np
import pandas as pd
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
    """
    Compute the squared error between groups in data.

    Hypothesis:
       the compared values are similar between groups

    Metrics:
       squared error
    """

    hypotheses = {}
    
    def by_ind(dataset):
        return dataset.set_index(independent)[dependent]
    
    for label1, dataset1, label2, dataset2 in _iter_compare(data, compare):
        hypothesis = (f"{dependent} is similar for {compare}"
                      f" {label1} and {label2}")
        sqerr = (by_ind(dataset1) - by_ind(dataset2))**2
        hypotheses[hypothesis] = sqerr.rename(terms.SQERROR).reset_index()
    return hypotheses


def _ttest_1samp(dataset1, dataset2, dependent):
    std = dataset1[terms.STD + dependent]
    sample_size = dataset1[terms.SAMPLE_SIZE]
    t = np.abs((dataset1[dependent].values - dataset2[dependent].values))\
        / (std.values / np.sqrt(sample_size.values))
    p = [stats.t.sf(tt, df=ss-1) for tt, ss in zip(t, sample_size)]
    return t, p


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
    """
    Perform a two-tailed t-test between comparable datapoints.
    
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
                    "Current functionality supports 1-sample t-test")
        else:
            if _has_std_and_size(dataset2, dependent):
                label1, label2 = (label2, label1)
                dataset1, dataset2 = (dataset2, dataset1)
            else:
                if (_has_samples(dataset1, dependent, independent)\
                    or _has_samples(dataset2, dependent, independent)):
                    raise NotImplementedError(
                        "Performing t-test for samples (instead of precomputed"
                        " standard deviation and sample size) is not yet "
                        "implemented. please make a pull-request")
                else:
                    raise ValueError(
                        "The data provided do not contain the information "
                        "necessary to perform any kind of t-test")

        hypothesis = (f'population mean of {dependent} for {label1} is '
                      f'equal to the value of {dependent} for {label2}')
        t, p = _ttest_1samp(dataset1, dataset2, dependent)
        hypotheses[hypothesis] = dataset1[independent]\
            .reset_index(drop=True).assign(
                **{terms.TSTAT: t,
                   terms.PVALUE: p})
        
    return hypotheses

class PooledPValueThreshold:
    """Pools the p-values of some statistical test across observations.

    The hypothesis fails if any p-value is below some threshold.
    This threshold is adjusted with a Bonferroni correction for the number of observations.
    """

    def __init__(self, threshold=0.05):
        """Initialize."""
        self.threshold = threshold
        return

    def __call__(self, stats, **kw):
        """Run."""
        verdicts = {}
        for hypothesis, data in stats.items():
            p = data[terms.PVALUE]
            bonferroni_threshold = self.threshold / data.shape[0]
            fail = (p <= bonferroni_threshold).any()
            verdicts[hypothesis] = 'Fail' if fail else 'Pass'
        return verdicts
