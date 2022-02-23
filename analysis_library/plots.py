import seaborn as sns



# TODO: should probably act on an axis by default: s.t. plots_by, subplots_by
#   can be handled at the analysis level...


class BarPlot:

    def __call__(self, data, independent, dependent, compare=(None, )):
        f = _bar_plot(data, independent, dependent, hue=compare[0])
        return f


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
