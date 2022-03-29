import seaborn as sns
import matplotlib.pyplot as plt



# TODO: should probably act on an axis by default: s.t. plots_by, subplots_by
#   can be handled at the analysis level...


class BarPlot:

    # TODO: wait, wtf are we passing compare as None, again?
    def __call__(self, data, independent, dependent, compare=None):
        f, ax = plt.subplots()
        xcolumns, constants, compare_owned = _varying_independent_only(
            data, independent, compare)
        f = _bar_plot(data, xcolumns, dependent, hue=compare, ax=ax)
        return f


# TODO: what to do if nothing varies? i.e. comparing just one value
# TODO: ensure this supports mutliple ind/compare
# TODO: multiple compares
def _varying_independent_only(data, independent, compare):
    """TODO: mention the tidy data paper in this docsting"""
    if compare is not None:
        varied = []
        for d, group in data.groupby(compare):
            varied_within, constant, _ = _varying_independent_only(
                group, independent, None)
            for col in varied_within:
                if col not in varied:
                    varied.append(col)
        return varied, [], []

    varied = [c for c in independent if len(data[c].unique()) > 1]
    constant = [c for c in independent if c not in varied]
    return varied, constant, []


def _bar_plot(measurements, independent, dependent, ax=None, hue=None):
    measurements, independent_label = _concatenate_independent(
        measurements, independent)
    # TODO: this should not be necessary
    # TODO: these difficulties imply writing a plotter under the current
    # system is too complicated.
    # maybe there should be a base class which
    #  does a lot of stuff for you, but can be overwritten
    #measurements = measurements.drop(columns=independent)
    # others = [col for col in measurements if col not in
    #           (dependent, independent
    # measurements = pd.concat([measurement, data], axis=0).groupby(
    #     self.independent)
    # ax = measurements.plot(kind='bar', y=self.dependent)# , color='dataset')
    axes = sns.barplot(data=measurements,
                       x=independent_label, y=dependent,
                       hue=hue, ax=ax)

    # TODO: multiple dependents on a single plot is awkward...
    #  I think if you measure multiple things, you should specify which is
    #  dependent in the analysis...
    fig = axes.get_figure()
    return fig


class LinePlot:

    # TODO: wait, wtf are we passing compare as None, again?
    def __call__(self, data, independent, dependent, compare=None):
        f, ax = plt.subplots()
        xcolumns, constants, compare_owned = _varying_independent_only(
            data, independent, compare)
        f = _line_plot(data, x=xcolumns, y=dependent, hue=compare, ax=ax)
        return f


def _line_plot(data, x, y, hue=None, ax=None):
    data, x = _concatenate_independent(data, x)
    axes = sns.lineplot(data=data,
                        x=x, y=y, hue=hue, ax=ax)
    fig = axes.get_figure()
    return fig


def _concatenate_independent(measurements, independent):
    independent_label = ", ".join(independent)

    if isinstance(independent, list):
        if len(independent) == 1:
            return measurements, independent[0]
    else:
        return measurements, independent

    measurements = measurements.assign(
        **{independent_label:
           [", ".join([str(v) for v in row.values])
            for i, row in measurements[independent].iterrows()]})
    return measurements, independent_label
