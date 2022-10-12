"""Tools for statistical hypothesis testing"""
from analysis_neuro import terminology as terms


def squared_error(data, dependent, independent, compare):
    """
    Compute the squared error between groups in data.

    Hypothesis:
       the compared values are similar between groups

    Metrics:
       squared error
    """
    datasets = data.groupby(compare)
    hypotheses = {}
    already_compared = set()
    
    def by_ind(dataset):
        return dataset.set_index(independent)[dependent]
    
    for label1, dataset1 in datasets:
        for label2, dataset2 in datasets:
            if (label2, label1) in already_compared:
                continue
            if label1 == label2:
                continue
            already_compared.add((label1, label2))
            hypothesis = (f"{dependent} is similar for {compare}"
                          f" {label1} and {label2}")
            sqerr = (by_ind(dataset1) - by_ind(dataset2))**2
            hypotheses[hypothesis] = sqerr.rename(terms.SQERROR).reset_index()
    return hypotheses


class TTest:
    """
    Performs welch's t-test between the compared values

    Assumptions:
        the sample means of the compared values are normally distributed
        the individual samples are independent of each other

    Hypothesis:
        The true mean of the compared populations is equal

    Metrics:
        T-statistic, P-value
    """

    def __init__(self):
        """perform a t-test between compared values"""
        return

    def __call__(self, data, dependent, independent, compare):
        """see class documentation"""
        return


class PooledPvalueThreshold:
    """
    Pools the p-values of some statistical test across observations.

    The hypothesis fails if any p-value is below some threshold.
    This threshold is adjusted with a Bonferroni correction for the number of observations.
    """

    def __init__(self, threshold):
        return

    def __call__(self, stats, data, dependent, independent, compare):
        return
