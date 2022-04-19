from scipy.stats import ttest_ind
from analysis_library import terminology as terms


class TTest:

    def __init__(self):
        """perform a t-test between compared values"""
        return

    def __call__(self, data, dependent, independent, compare):
        """
        Performs welch's t-test between the compared values

        Assumptions:
            the sample means of the compared values are normally distributed
            the individual samples are independent of each other

        Hypothesis:
            The true mean of the compared populations is equal
        """
        out = {}
        comparevalues = data[compare].unique()
        groups = data.groupby(independent)
        for value1 in comparevalues:
            for value2 in comparevalues:
                for group, dataframe in groups:
                    v1 = dataframe[dataframe[compare] == value1][dependent]
                    v2 = dataframe[dataframe[compare] == value2][dependent]
                    test_result = dict(
                        **{terms.PVALUE: ttest_ind(v1, v2, equal_var=False)},
                        **{key: value for key, value in zip(independent, group)})

                    out[f"the mean {dependent} of {value1} is equal to the mean of {dependent} of {value2}"] = test_result

        return out
