from pyrmv.analyses.data import data_from_histogram


def test_artificial_hist():
    bin_edges = [1, 2, 3, 4, 5, 6]
    bin_tops = [1000, 900, 800, 600, 500]
    data = data_from_histogram(bin_edges, bin_tops, (0, 5), (0, 5))
    assert data == [
        1.5,  # bin 2
        2.5, 2.5,  # bin 3,
        3.5, 3.5, 3.5, 3.5,  # bin 4,
        4.5, 4.5, 4.5, 4.5, 4.5,  # bin 5
    ]
