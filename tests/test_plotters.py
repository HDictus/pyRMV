from analysis_library.plots import BarPlot
import pandas as pd

# TODO: plots should ideally support non-list ind/dep
#   to make them nicer to use on their own
# TODO: it probably doesn't make any sense to put a list for dependent
#   nah actually, you might want to plot both synapse density and axon density, for example
# TODO: but tidy data would require us to submit this as multiple datasets
#   OK., it is settled that these plotters need to support recieving multiple datasets
# default behavior, with multiple dependent variables would be to
#   make a new plot for each?
# TODO: discuss this in undecided.org
def test_barplot_single_dataset():
    data=pd.DataFrame({'a': [1, 2, 3, 4],
                       'b': [2, 3, 4, 5]})
    outd = BarPlot()(
        data=data,
        independent=['a'],
        dependent=['b'])
    fig = outd
    ax = fig.gca()
    assert ax.get_xlabel() == 'a'
    assert ax.get_ylabel() == 'b'


def test_barplot_compare_datasets():
    data=pd.DataFrame({'a': [1, 2, 1, 2],
                       'b': [2, 3, 4, 5],
                       'c': [0, 0, 1, 1]})
    outd = BarPlot()(
        data=data,
        independent=['a'],
        dependent=['b'],
        compare=['c'])
    ax = outd.gca()
    print(list(ax.get_xticklabels())[0].get_text())
    assert [l.get_text() for l in ax.get_xticklabels()] == ['1', '2']

def test_barplot_multiple_ind():
    data=pd.DataFrame({'a': [1, 2, 1, 2],
                       'b': [2, 3, 4, 5],
                       'c': [0, 0, 1, 1]})
    outd = BarPlot()(
        data=data,
        independent=['a', 'c'],
        dependent=['b'])
    ax = outd.gca()
    assert ax.get_xlabel() == 'a, c'
    assert [l.get_text() for l in ax.get_xticklabels()] ==\
        ['1, 0', '2, 0', '1, 1', '2, 1']
