import pandas as pd
from analysis_neuro import stats
from analysis_neuro import terminology as terms


def test_squared_error():
    statistic = stats.squared_error(
        pd.DataFrame({
            'compare': [1, 1, 2, 2, 3, 3,],
            'ind': [1, 2, 1, 2, 1, 2],
            'dep': [4, 5, 3, 2, 4, 5]}),
        dependent='dep',
        independent='ind',
        compare='compare')
    expectation = {
        'dep is similar for compare 1 and 2': pd.DataFrame(
            {'ind': [1, 2], terms.SQERROR: [1, 9]}),
        'dep is similar for compare 1 and 3': pd.DataFrame(
            {'ind': [1, 2], terms.SQERROR: [0, 0]}),
        'dep is similar for compare 2 and 3': pd.DataFrame(
            {'ind': [1, 2], terms.SQERROR: [1, 9]})}
    for k, v in expectation.items():
        pd.testing.assert_frame_equal(statistic[k], v)
    assert len(statistic) == len(expectation)
    return
