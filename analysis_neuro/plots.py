"""Plotting tools."""
import matplotlib.pyplot as plt


def crossplot(data, dependent, independent, compare):
    """Compare datasets with a crossplot.

    Plot the dependent variable for each of the compared datasets
    along a different axis, showing the level of agreement between them by
    how close they are to the diagonal.
    """
    # pylint: disable=invalid-name
    # pylint: disable=unused-argument
    comparevalues = data[compare].unique()

    if len(comparevalues) < 2:
        raise ValueError(
            "a crossplot only makes sense when comparing data. "
            "It is not defined for less than 2 datasets."
        )
    if len(comparevalues) > 2:
        raise NotImplementedError(
            "currently crossplot comparisons between more than 2 "
            "datasets is not implemented, please make a pull request"
        )

    data1 = data[data[compare] == comparevalues[0]]
    data2 = data[data[compare] == comparevalues[1]]
    x = data1[dependent].values

    y = data2[dependent].values

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    plt.xlabel(comparevalues[0])
    plt.ylabel(comparevalues[1])
    plt.title(dependent)
    minimum = max(ax.get_xlim()[0], ax.get_ylim()[0])
    maximum = min(ax.get_xlim()[1], ax.get_ylim()[1])
    plt.plot((minimum, maximum), (minimum, maximum), color="gray", linestyle="--")
    return fig
