from analysis_library.plots import BarPlot, LinePlot
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
def single_dataset_x_and_y(plotter):
    data=pd.DataFrame({'a': [1, 2, 3, 4],
                       'b': [2, 3, 4, 5]})
    outd = plotter(
        data=data,
        independent=['a'],
        dependent='b')
    fig = outd
    ax = fig.gca()
    assert ax.get_xlabel() == 'a'
    assert ax.get_ylabel() == 'b'


def compare_datasets(plotter):
    data = pd.DataFrame({'a': [1, 2, 1, 2],
                         'b': [2, 3, 4, 5],
                         'c': [0, 0, 1, 1]})
    outd = plotter(
        data=data,
        independent=['a'],
        dependent='b',
        compare='c')
    ax = outd.gca()
    print(list(ax.get_xticklabels())[0].get_text())
    assert [l.get_text() for l in ax.get_xticklabels()] == ['1', '2']


def multiple_independent_variables(plotter):
    data=pd.DataFrame({'a': [1, 2, 1, 2],
                       'b': [2, 3, 4, 5],
                       'c': [0, 0, 1, 1]})
    outd = plotter(
        data=data,
        independent=['a', 'c'],
        dependent='b')
    ax = outd.gca()

    assert ax.get_xlabel() == 'a, c'
    assert [l.get_text() for l in ax.get_xticklabels()] ==\
        ['1, 0', '2, 0', '1, 1', '2, 1']

def excludes_nonvarying_independent(plotter):
    data = pd.DataFrame({'a': [1, 1, 1, 1],
                         'b': [1, 2, 1, 2],
                         'c': [1, 2, 3, 4],
                         'dd': [3, 4, 5, 6],
                         'e': [2, 3, 2, 3]})
    plot = plotter(
        data=data,
        independent=['a', 'c', 'e'],
        dependent='dd',
        compare='b')

    ax = plot.gca()
    assert ax.get_xlabel() == 'c'
    assert [l.get_text() for l in ax.get_xticklabels()] ==\
        ['1', '2', '3', '4']


class TestBarPlot:

    def test_single_dataset_x_and_y(self):
        return single_dataset_x_and_y(BarPlot())

    def test_compare_datasets(self):
        return compare_datasets(BarPlot())

    def test_multiple_independent_variables(self):
        return multiple_independent_variables(BarPlot())

    def test_excludes_nonvarying_independent(self):
        return excludes_nonvarying_independent(BarPlot())


class TestLinePlot:

    def test_single_dataset_x_and_y(self):
        return single_dataset_x_and_y(LinePlot())

    def test_compare_datasets(self):
        data = pd.DataFrame({'a': [1, 2, 1, 2],
                             'b': [2, 3, 4, 5],
                             'c': [0, 0, 1, 1]})
        outd = LinePlot()(
            data=data,
            independent=['a'],
            dependent='b',
            compare='c')
        ax = outd.gca()
        assert [l.get_label() for l in ax.get_lines()
                if "_child" not in l.get_label()  # necessary for seaborn
                ] == ['0', '1']

    def test_multiple_independent_variables(self):
        return multiple_independent_variables(LinePlot())

    def test_excludes_nonvarying_independent(self):
        return excludes_nonvarying_independent(LinePlot())

    def test_averages_range_parameters(self):
        data = pd.DataFrame({'a': [(0, 2), (2, 3), (3, 7), (7, 9)],
                             'b': [2, 3, 4, 5]})
        fig = LinePlot()(
            data=data, independent=['a'], dependent='b')
        line = fig.gca().get_lines()[0]
        xd, yd = line.get_data()
        print(xd, yd)
        assert xd == ['1', '2.5', '5', '8']
