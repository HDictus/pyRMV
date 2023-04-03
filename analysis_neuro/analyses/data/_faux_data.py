"""Construct datasets based on plots.

These  cannot represent the actual data, but reproduce its properties.
"""
import numpy as np


def data_from_histogram(bin_edges, bin_tops, horizontal_range, vertical_range):
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

    def extract_bin_mean_values(bin_edges, horizontal_range):
        diff = np.max(bin_edges) - np.min(bin_edges[0])
        horizontal_axis_size = horizontal_range[1] - horizontal_range[0]
        horizontal_per_pixel = horizontal_axis_size / diff
        xvalues = [
            horizontal_range[0] + (bin_edge - bin_edges[0]) * horizontal_per_pixel
            for bin_edge in bin_edges
        ]
        binmeans = [(e1 + e2) * 0.5 for e1, e2 in zip(xvalues[:-1], xvalues[1:])]
        return binmeans

    def extract_entries_per_bin(bin_tops, vertical_range):
        vertical_sz = vertical_range[1] - vertical_range[0]
        toppest = np.min(bin_tops)
        count_per_pixel = vertical_sz / (toppest - np.max(bin_tops))
        yvalues = [
            int(vertical_range[0] + np.round((bin_top - bin_tops[0]) * count_per_pixel))
            for bin_top in bin_tops
        ]
        return yvalues


    binmeans = extract_bin_mean_values(bin_edges)
    entries_per_bin = extract_entries_per_bin(bin_tops, vertical_range)

    out_data = []
    # create num_entries values at each bin center
    for i, num_entries in enumerate(entries_per_bin):
        out_data = out_data + [binmeans[i]] * num_entries
    return out_data
