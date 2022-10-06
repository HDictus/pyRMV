def squared_error(data, dependent, independent, compare):
    return


class TTest:
    """
    Performs welch's t-test between the compared values

    Assumptions:
        the sample means of the compared values are normally distributed
        the individual samples are independent of each other

    Hypothesis:
        The true mean of the compared populations is equal

    Returns:
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
