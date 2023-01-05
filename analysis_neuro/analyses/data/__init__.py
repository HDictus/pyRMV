"""Storage and generation of experimental data."""
import numpy as np


def data_from_histogram(
        bin_edges, bin_tops, horizontal_range, vertical_range
):
    """Generate artificial data based on a histogram.

    Each bin will have N datapoints located at its center, where N is the
    number of points reported in that bin.

    Arguments:
        bin_edges: the sides of each bin reported, in pixels in the image
        bin_tops: the tops of each bin, in pixels in the image
        horizontal_range: tuple of min and max values on the x axis
        vertical_range: tuple of min and max values on the y axis

    Returns:
       list of numbers, each representing one sample of the distribution
    """
    diff = np.max(bin_edges) - np.min(bin_edges[0])
    horizontal_sz = horizontal_range[1] - horizontal_range[0]
    horizontal_per_pixel = horizontal_sz / diff
    xvalues = [
        horizontal_range[0] + (be - bin_edges[0]) * horizontal_per_pixel for be in bin_edges
    ]

    binmeans = [(e1 + e2) * 0.5 for e1, e2 in zip(xvalues[:-1], xvalues[1:])]

    vertical_sz = vertical_range[1] - vertical_range[0]
    toppest = np.min(bin_tops)
    count_per_pixel = vertical_sz / (toppest - np.max(bin_tops))
    yvalues = [
        int(vertical_range[0] + np.round((bp - bin_tops[0]) * count_per_pixel))
        for bp in bin_tops
    ]
    return [binmeans[i] for i, n in enumerate(yvalues) for _ in range(n)]
