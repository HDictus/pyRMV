import pandas as pd
import numpy as np
import pytest as pyt
import scipy
import statsmodels.stats.proportion as proportion
from analysis_neuro import stats, terms, Assumption


def test_squared_error():
    data = pd.DataFrame(
        {
            "a": [1, 1, 2, 2, 3, 3] * 2,
            "b": [1, 2, 1, 2, 1, 2] * 2,
            "c": [1, 2, 3, 4, 5, 6] * 2,
            "msr": [1, 2, 3, 4, 5, 6, 1, 1, 2, 1, 1, 1],
            "comp": ["x"] * 6 + ["y"] * 6,
        }
    )
    squerr = stats.squared_error(
        data, dependent="msr", independent=["a", "b", "c"], compare="comp"
    )
    assert "msr is similar for comp x and y" in squerr
    assert len(squerr) == 1
    pd.testing.assert_frame_equal(
        squerr["msr is similar for comp x and y"],
        pd.DataFrame(
            {
                "a": [1, 1, 2, 2, 3, 3],
                "b": [1, 2, 1, 2, 1, 2],
                "c": [1, 2, 3, 4, 5, 6],
                terms.SQERROR: [0, 1, 1, 9, 16, 25],
            }
        ),
    )


def test_ttest_1samp():
    data = pd.DataFrame(
        {
            "a": [1, 1, 2] * 2,
            "b": [1, 2, 1] * 2,
            "c": [1, 2, 3] * 2,
            "msr": [1, 2, 3, 1, 4, 2],
            "comp": ["x"] * 3 + ["y"] * 3,
            terms.STD + "msr": [1, 2, 3, np.nan, np.nan, np.nan],
            terms.SAMPLE_SIZE: [3, 2, 3, np.nan, np.nan, np.nan],
        }
    )
    hypo = stats.ttest(
        data, dependent="msr", independent=["a", "b", "c"], compare="comp"
    )
    assert "population mean of msr for x is equal to the value of msr for y" in hypo

    exp_t_statistic = np.array([0, 2, 1]) / (
        np.array([1, 2, 3]) / np.sqrt(np.array([3, 2, 3]))
    )
    exp_p_value = []
    for t, n in zip(exp_t_statistic, [3, 2, 3]):
        exp_p_value.append(scipy.stats.t.sf(t, df=n - 1))

    pd.testing.assert_frame_equal(
        hypo["population mean of msr for x is equal to the value of msr for y"],
        pd.DataFrame(
            {
                "a": [1, 1, 2],
                "b": [1, 2, 1],
                "c": [1, 2, 3],
                terms.TSTAT: exp_t_statistic,
                terms.PVALUE: exp_p_value,
            }
        ),
    )
    return


def test_ttest_1samp_reverse():
    data = pd.DataFrame(
        {
            "a": [1, 1, 2] * 2,
            "b": [1, 2, 1] * 2,
            "c": [1, 2, 3] * 2,
            "msr": [1, 2, 3, 1, 4, 2],
            "comp": ["x"] * 3 + ["y"] * 3,
            terms.STD + "msr": [np.nan, np.nan, np.nan, 1, 2, 3],
            terms.SAMPLE_SIZE: [np.nan, np.nan, np.nan, 3, 2, 3],
        }
    )
    hypo = stats.ttest(
        data, dependent="msr", independent=["a", "b", "c"], compare="comp"
    )
    assert "population mean of msr for y is equal to the value of msr for x" in hypo

    exp_t_statistic = np.array([0, 2, 1]) / (
        np.array([1, 2, 3]) / np.sqrt(np.array([3, 2, 3]))
    )
    exp_p_value = []
    for t, n in zip(exp_t_statistic, [3, 2, 3]):
        exp_p_value.append(scipy.stats.t.sf(t, df=n - 1))

    pd.testing.assert_frame_equal(
        hypo["population mean of msr for y is equal to the value of msr for x"],
        pd.DataFrame(
            {
                "a": [1, 1, 2],
                "b": [1, 2, 1],
                "c": [1, 2, 3],
                terms.TSTAT: exp_t_statistic,
                terms.PVALUE: exp_p_value,
            }
        ),
    )
    return


def test_ttest_not_implemented():
    bothhavestd = pd.DataFrame(
        {
            "a": [1, 1] * 2,
            "b": [1, 2] * 2,
            "c": [1, 2] * 2,
            "msr": [1, 2, 4, 2],
            "comp": ["x"] * 2 + ["y"] * 2,
            terms.STD + "msr": [4, 3, 2, 3],
            terms.SAMPLE_SIZE: [6, 7, 2, 3],
        }
    )

    onehassamples = pd.DataFrame(
        {
            "a": [1, 1] * 4,
            "b": [1, 2] * 4,
            "c": [1, 2] * 4,
            "msr": [1, 2, 4, 2, 4, 3, 6, 7],
            "comp": ["x"] * 6 + ["y"] * 2,
        }
    )

    neitherhasanything = pd.DataFrame(
        {
            "a": [1, 1] * 2,
            "b": [1, 2] * 2,
            "c": [1, 2] * 2,
            "msr": [1, 2, 4, 5],
            "comp": ["x"] * 2 + ["y"] * 2,
        }
    )

    for data in (bothhavestd, onehassamples):
        with pyt.raises(NotImplementedError):
            stats.ttest(
                data, dependent="msr", independent=["a", "b", "c"], compare="comp"
            )

    with pyt.raises(ValueError):
        stats.ttest(
            neitherhasanything,
            dependent="msr",
            independent=["a", "b", "c"],
            compare="comp",
        )


def test_PooledPValueThreshold():
    hypotheses = {
        "some hypothesis": pd.DataFrame(
            {
                "a": [1, 2],
                terms.PVALUE: [0.11, 0.1],
            }
        )
    }
    verdicts = stats.PooledPValueThreshold(0.1)(hypotheses)
    assert verdicts["some hypothesis"] == "Pass"

    hypotheses = {
        "some hypothesis": pd.DataFrame(
            {
                "a": [1, 2],
                terms.PVALUE: [0.11, 0.05],
            }
        )
    }
    verdicts = stats.PooledPValueThreshold(0.1)(hypotheses)
    assert verdicts["some hypothesis"] == "Fail"

    hypotheses = {
        "some hypothesis": pd.DataFrame(
            {
                "a": [1, 2, 3, 4],
                terms.PVALUE: [0.026, 0.5, 0.1, 0.1],
            }
        )
    }
    verdicts = stats.PooledPValueThreshold(0.1)(hypotheses)
    assert verdicts["some hypothesis"] == "Pass"

    hypotheses = {
        "some hypothesis": pd.DataFrame(
            {
                "a": [1, 2, 3, 4],
                terms.PVALUE: [0.025, 0.5, 0.1, 0.1],
            }
        )
    }
    verdicts = stats.PooledPValueThreshold(0.1)(hypotheses)
    assert verdicts["some hypothesis"] == "Fail"


def test_binom_test_no_sample_size():
    data_no_samp = pd.DataFrame(
        {
            "a": [1, 1, 2] * 2,
            "b": [1, 2, 1] * 2,
            "c": [1, 2, 3] * 2,
            "msr": [0.5, 1.0, 0.25, 0.1, 0.2, 0.3],
            "comp": ["x"] * 3 + ["y"] * 3,
            terms.SAMPLE_SIZE: [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        })
    
    data_nan_samp = pd.DataFrame(
        {
            "a": [1, 1, 2] * 2,
            "b": [1, 2, 1] * 2,
            "c": [1, 2, 3] * 2,
            "msr": [0.5, 1.0, 0.25, 0.1, 0.2, 0.3],
            "comp": ["x"] * 3 + ["y"] * 3,
            terms.SAMPLE_SIZE: [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        })


    hypotheses = stats.binom_test(
        data_no_samp, dependent="msr", independent=["a", "b", "c"], compare="comp"
    )
    assert np.isnan(list(hypotheses.values())[0][terms.PVALUE].values).all()
                             
    
def test_binom_test_1_sample():
    data = pd.DataFrame(
        {
            "a": [1, 1, 2] * 2,
            "b": [1, 2, 1] * 2,
            "c": [1, 2, 3] * 2,
            "msr": [0.5, 1.0, 0.25, 0.1, 0.2, 0.3],
            "comp": ["x"] * 3 + ["y"] * 3,
            terms.SAMPLE_SIZE: [4, 2, 4, np.nan, np.nan, np.nan],
        }
    )
    expected = [
        scipy.stats.binomtest(k, n, p).pvalue
        for k, n, p in [(2, 4, 0.1), (2, 2, 0.2), (1, 4, 0.3)]
    ]
    hypotheses = stats.binom_test(
        data, dependent="msr", independent=["a", "b", "c"], compare="comp"
    )
    with pyt.warns(Assumption, match=(
            "The values of msr for y"
            " are the ground truth value for y")):
        assert "The probability msr is the same for x as for y." in hypotheses
        assert np.allclose(
            hypotheses["The probability msr is the same for x as for y."][
                terms.PVALUE
            ].values,
            expected,
        )
    return


def test_binom_test_nan():
    data = pd.DataFrame(
        {
            "a": [1, 1, 2] * 2,
            "b": [1, 2, 1] * 2,
            "c": [1, 2, 3] * 2,
            "msr": [0.5, 1.0, np.nan, 0.1, np.nan, 0.3],
            "comp": ["x"] * 3 + ["y"] * 3,
            terms.SAMPLE_SIZE: [4, 2, 4, np.nan, np.nan, np.nan],
        }
    )
    expected = [scipy.stats.binomtest(2, 4, 0.1).pvalue] + [np.nan, np.nan]
    hypotheses = stats.binom_test(
        data, dependent="msr", independent=["a", "b", "c"], compare="comp"
    )
    assert "The probability msr is the same for x as for y." in hypotheses
    assert np.allclose(
        np.nan_to_num(
            hypotheses["The probability msr is the same for x as for y."][
                terms.PVALUE
            ].values
        ),
        np.nan_to_num(expected),
    )
    return


def test_binom_test_two_sample():
    data = pd.DataFrame(
        {
            "a": [1, 1, 2] * 2,
            "b": [1, 2, 1] * 2,
            "c": [1, 2, 3] * 2,
            "msr": [0.5, 1.0, 0.25, 0.1, 0.2, 0.5],
            "comp": ["x"] * 3 + ["y"] * 3,
            terms.SAMPLE_SIZE: [12, 2, 4, 10, 7, 4],
        }
    )

    
    expected = [scipy.stats.binomtest(k, n, p).pvalue
                for k, n, p in (
                        (1, 10, 0.5),
                        (2, 2, 0.2),
                        (1, 4, 0.5))]
    hypotheses = stats.binom_test(
        data, dependent="msr", independent=["a", "b", "c"], compare="comp"
    )
    assert "The probability msr is the same for x as for y." in hypotheses
    assert np.allclose(
        np.nan_to_num(
            hypotheses["The probability msr is the same for x as for y."][
                terms.PVALUE
            ].values
        ),
        np.nan_to_num(expected),
    )
    return


def test_is_lognormal():
    nums = np.random.uniform(0, 100, size=(1000))
    data = pd.DataFrame(
        {'label': (['a'] * 500) + (['b'] * 500),
         'var': ([1] * 250) + ([2] * 250) + ([1] * 250) + ([2] * 250),
         'msr': nums})
    result = stats.is_lognormal(
        data, dependent='msr', independent=['var'], compare='label')
    aexp = pd.DataFrame({
        'var': [1, 2],
        terms.PVALUE: [scipy.stats.normaltest(np.log(nums[:250])).pvalue,
                       scipy.stats.normaltest(np.log(nums[250:500])).pvalue]})
    pd.testing.assert_frame_equal(result['msr is lognormally distributed for a'], aexp)
    bexp = pd.DataFrame({
        'var': [1, 2],
        terms.PVALUE: [scipy.stats.normaltest(np.log(nums[500:750])).pvalue,
                       scipy.stats.normaltest(np.log(nums[750:])).pvalue]})
    pd.testing.assert_frame_equal(result['msr is lognormally distributed for a'], bexp)
    

def test_lognorm_ttest():
    nums = np.random.uniform(0, 100, size=(1000))
    data = pd.DataFrame(
        {'label': (['a'] * 500) + (['b'] * 500),
         'var': ([1] * 250) + ([2] * 250) + ([1] * 250) + ([2] * 250),
         'msr': nums})

    result = stats.lognorm_ttest(
        data, dependent='msr', independent='var', compare='label')
    expectation = pd.DataFrame(
        {'var': [1, 2],
         terms.PVALUE: [scipy.stats.ttest_ind(np.log(nums[:250]), np.log(nums[500:750])).pvalue,
                        scipy.stats.ttest_ind(np.log(nums[250:500]), np.log(nums[750:])).pvalue]})

    pd.testing.assert_frame_equal(
        result['the population mean of msr is the same for a and b'],
        expectation)
