import numpy as np


def data_from_histogram(bin_edges, bin_tops,
                        horizontal_min,
                        horizontal_max,
                        vertical_min,
                        vertical_max):
    """Generate artificial data based on a histogram.

    Each bin will have N datapoints located at its center, where N is the 
    number of points reported in that bin.
    
    Arguments:
        bin_edges: the sides of each bin reported, in pixels in the image
        bin_tops: the tops of each bin, in pixels in the image
        horizontal/vertical min/max : minimum and maximum values on the 
            plot's axis of the edges/tops.
    
    Returns:
       list of numbers, each representing one sample of the distribution 
    """
    diff = np.max(bin_edges) - np.min(bin_edges[0])
    horizontal_range = horizontal_max - horizontal_min
    horizontal_per_pixel = horizontal_range / diff
    xvalues = [horizontal_min + (be-bin_edges[0]) * horizontal_per_pixel for be in bin_edges]

    binmeans = [(e1 + e2) * 0.5 for e1, e2 in zip(xvalues[:-1], xvalues[1:])]
    
    vertical_range = vertical_max - vertical_min
    toppest = np.min(bin_tops)
    count_per_pixel = vertical_range / (toppest - np.max(bin_tops))
    yvalues = [int(vertical_min + np.round((bp - bin_tops[0]) * count_per_pixel)) for bp in bin_tops]
    print(yvalues)
    return [binmeans[i] for i, n in enumerate(yvalues) for _ in range(n)]
