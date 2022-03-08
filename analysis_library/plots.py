import seaborn as sns



# TODO: should probably act on an axis by default: s.t. plots_by, subplots_by
#   can be handled at the analysis level...


class BarPlot:

    # TODO: wait, wtf are we passing compare as None, again?
    def __call__(self, data, independent, dependent, compare=(None, )):
        xcolumns, constants, compare_owned = _varying_independent_only(
            data, independent, compare[0])
        print(xcolumns)
        f = _bar_plot(data, xcolumns, dependent, hue=compare[0])
        return f


# TODO: what to do if nothing varies? i.e. comparing just one value
# TODO: ensure this supports mutliple ind/compare
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

    print([(c, len(data[c].unique())) for c in independent])
    varied = [c for c in independent if len(data[c].unique()) > 1]
    constant = [c for c in independent if c not in varied]
    return varied, constant, []

def _bar_plot(measurements, independent, dependent, ax=None, hue=None):

    independent_label = ", ".join(independent)
    measurements = measurements.assign(
        **{independent_label:
           [", ".join([str(v) for v in row.values])
            for i, row in measurements[independent].iterrows()]})
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
                       x=independent_label, y=dependent[0],
                       hue=hue, ax=ax)

    # TODO: multiple dependents on a single plot is awkward...
    #  I think if you measure multiple things, you should specify which is
    #  dependent in the analysis...
    fig = axes.get_figure()
    return fig
