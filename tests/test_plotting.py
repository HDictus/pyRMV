import pandas as pd
import numpy as np
import pyrmv.terminology as terms
import matplotlib.pyplot as plt
from pyrmv import plots
import pytest


def test_pathway_heatmap_adjusts_size():
    figs = plots.pathway_heatmap(
        pd.DataFrame({terms.PRESYNAPTIC + 'something': ['a', 'a', 'b', 'b'],
                      terms.POSTSYNAPTIC + 'something': ['a', 'b', 'a', 'b'],
                      'measured': [1, 2, 3, 4],
                      'dataset': 'fake'}),
        dependent='measured',
        independent=[terms.PRESYNAPTIC + 'something',
                     terms.POSTSYNAPTIC + 'something'],
        compare='dataset')
    assert type(figs['fake']) == plt.Figure
    assert tuple(figs['fake'].get_size_inches()) == (3, 3)
    
    figs = plots.pathway_heatmap(
        pd.DataFrame([
            {terms.PRESYNAPTIC + 'something': pre,
             terms.POSTSYNAPTIC + 'something': post,
             'measured': 6,
             'dataset': 'fake'}
            for pre in range(75) for post in range(60)]),
        dependent='measured',
        independent=[terms.PRESYNAPTIC + 'something',
                     terms.POSTSYNAPTIC + 'something'],
        compare='dataset')
    assert type(figs['fake']) == plt.Figure
    # 0.25 inches per row is generally enough to see all the labels
    assert tuple(figs['fake'].get_size_inches()) == (75/4, 60/4)
    

def test_crossplot_handles_multiple_compare():
    data = pd.DataFrame({
        'measured': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'param': [3, 3, 3, 3, 3, 3, 3, 3, 3],
        'compare': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c']
    })
    out = plots.crossplot(data, 'measured', ['param'], 'compare')
    assert len(out) == 3


# Tests for new general purpose plotters

def test_wide_barplot_basic():
    """Test basic wide barplot functionality."""
    x = ['A', 'B', 'A', 'B', 'A', 'B']
    y = [1, 2, 3, 4, 5, 6]
    hue = ['X', 'X', 'Y', 'Y', 'Z', 'Z']
    
    ax = plots.wide_barplot(x, y, hue)
    assert isinstance(ax, plt.Axes)
    
    # Test figure size scaling
    expected_width = len(np.unique(x)) * len(np.unique(hue)) / 3
    assert ax.figure.get_size_inches()[0] == expected_width
    assert ax.figure.get_size_inches()[1] == 5


def test_wide_barplot_no_hue():
    """Test wide barplot without hue parameter."""
    x = ['A', 'B', 'C']
    y = [1, 2, 3]
    hue = [None, None, None]
    
    ax = plots.wide_barplot(x, y, hue)
    assert isinstance(ax, plt.Axes)


def test_wide_barplot_single_category():
    """Test wide barplot with single category."""
    x = ['A'] * 3
    y = [1, 2, 3]
    hue = ['X', 'Y', 'Z']
    
    ax = plots.wide_barplot(x, y, hue)
    assert isinstance(ax, plt.Axes)


def test_hist_basic():
    """Test basic hist functionality."""
    np.random.seed(42)
    data = pd.DataFrame({
        'values': np.random.normal(0, 1, 100),
        'dataset': ['experiment', 'model'] * 50
    })
    
    figs = plots.hist(data, 'values', [], 'dataset')
    assert len(figs) == 1
    assert isinstance(list(figs.values())[0], plt.Figure)


def test_hist_with_grouping():
    """Test hist with grouping variables."""
    np.random.seed(42)
    data = pd.DataFrame({
        'values': np.random.normal(0, 1, 100),
        'group': ['A'] * 50 + ['B'] * 50,
        'dataset': ['experiment', 'model'] * 50
    })
    
    figs = plots.hist(data, 'values', ['group'], 'dataset')
    assert len(figs) == 2
    assert all(isinstance(fig, plt.Figure) for fig in figs.values())


def test_hist_standard_bins():
    """Test hist with standard bin count."""
    np.random.seed(42)
    data = pd.DataFrame({
        'values': np.random.normal(0, 1, 100),
        'dataset': ['test'] * 100
    })
    
    figs = plots.hist(data, 'values', [], 'dataset')
    assert len(figs) == 1
    assert isinstance(list(figs.values())[0], plt.Figure)


def test_hist_with_experimental_mean():
    """Test hist with experimental mean values."""
    np.random.seed(42)
    data = pd.DataFrame({
        'values': np.random.normal(0, 1, 100),
        'dataset': ['test'] * 100
    })
    data[terms.MEAN + 'values'] = 0.5
    
    figs = plots.hist(data, 'values', [], 'dataset')
    assert len(figs) == 1
    assert isinstance(list(figs.values())[0], plt.Figure)






def test_hist_integration():
    """Integration test for hist with grouping."""
    np.random.seed(42)
    data = pd.DataFrame({
        'y': np.random.normal(0, 1, 60),
        'x': ['A', 'B', 'C'] * 20,
        'dataset': ['experiment', 'model'] * 30
    })
    
    figs = plots.hist(data, 'y', ['x'], 'dataset')
    assert len(figs) == 3
    assert all(isinstance(f, plt.Figure) for f in figs.values())


def test_crossplot_basic():
    """Test basic crossplot functionality."""
    data = pd.DataFrame({
        'independent': [1, 2, 3] * 4,
        'dependent': [1, 4, 9, 2, 5, 10, 1.5, 4.5, 9.5, 2.5, 5.5, 10.5],
        'dataset': ['A'] * 6 + ['B'] * 6
    })
    
    figs = plots.crossplot(data, 'dependent', ['independent'], 'dataset')
    assert len(figs) == 1
    assert isinstance(list(figs.values())[0], plt.Figure)


def test_crossplot_multiple_datasets():
    """Test crossplot with multiple datasets."""
    data = pd.DataFrame({
        'independent': [1, 2, 3] * 6,
        'dependent': [1, 4, 9] * 6,
        'dataset': ['A'] * 6 + ['B'] * 6 + ['C'] * 6
    })
    
    figs = plots.crossplot(data, 'dependent', ['independent'], 'dataset')
    assert len(figs) == 3


def test_crossplot_single_dataset():
    """Test crossplot with single dataset (should return empty)."""
    data = pd.DataFrame({
        'independent': [1, 2, 3],
        'dependent': [1, 4, 9],
        'dataset': ['A'] * 3
    })
    
    figs = plots.crossplot(data, 'dependent', ['independent'], 'dataset')
    assert len(figs) == 0


def test_pathway_barplot_basic():
    """Test basic pathway barplot functionality."""
    data = pd.DataFrame({
        terms.PRESYNAPTIC + terms.LAYER: ['L1', 'L2', 'L1', 'L2'],
        terms.POSTSYNAPTIC + terms.LAYER: ['L1', 'L1', 'L2', 'L2'],
        terms.MIN + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE: [0, 0, 50, 50],
        terms.MAX + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE: [50, 50, 100, 100],
        terms.MIN + terms.HORIZONTAL + terms.INTERSOMATIC_DISTANCE: [0, 25, 0, 25],
        'measurement': [0.1, 0.2, 0.3, 0.4],
        'dataset': ['test'] * 4
    })
    
    independent = [
        terms.PRESYNAPTIC + terms.LAYER,
        terms.POSTSYNAPTIC + terms.LAYER,
        terms.MIN + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE,
        terms.MAX + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE,
        terms.MIN + terms.HORIZONTAL + terms.INTERSOMATIC_DISTANCE
    ]
    
    figs = plots.pathway_barplot(data, 'measurement', independent, 'dataset')
    assert len(figs) >= 1
    assert all(isinstance(fig, plt.Figure) for fig in figs.values())


def test_pathway_barplot_multiple_pathways():
    """Test pathway barplot with multiple pathway combinations."""
    data = pd.DataFrame({
        terms.PRESYNAPTIC + terms.LAYER: ['L1', 'L2'] * 4,
        terms.POSTSYNAPTIC + terms.LAYER: ['L1', 'L1', 'L2', 'L2'] * 2,
        terms.MIN + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE: [0, 0, 50, 50] * 2,
        terms.MAX + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE: [50, 50, 100, 100] * 2,
        terms.MIN + terms.HORIZONTAL + terms.INTERSOMATIC_DISTANCE: [0, 25] * 4,
        'measurement': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'dataset': ['experiment', 'model'] * 4
    })
    
    independent = [
        terms.PRESYNAPTIC + terms.LAYER,
        terms.POSTSYNAPTIC + terms.LAYER,
        terms.MIN + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE,
        terms.MAX + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE,
        terms.MIN + terms.HORIZONTAL + terms.INTERSOMATIC_DISTANCE
    ]
    
    figs = plots.pathway_barplot(data, 'measurement', independent, 'dataset')
    assert len(figs) >= 1
    assert all(isinstance(fig, plt.Figure) for fig in figs.values())


def test_scatter_with_binned_mean_basic():
    """Test basic scatter with binned mean functionality."""
    np.random.seed(42)
    x = np.random.uniform(-1, 1, 100)
    y = x * 0.5 + np.random.normal(0, 0.1, 100)
    data = pd.DataFrame({
        'x': x,
        'y': y,
        'dataset': ['test'] * 100
    })
    
    figs = plots.scatter_with_binned_mean(data, 'y', ['x'], 'dataset')
    assert len(figs) == 1
    assert isinstance(figs['test'], plt.Figure)


def test_scatter_with_binned_mean_standard_bins():
    """Test scatter with binned mean with standard bin count."""
    np.random.seed(42)
    x = np.random.uniform(-1, 1, 100)
    y = x * 0.5 + np.random.normal(0, 0.1, 100)
    data = pd.DataFrame({
        'x': x,
        'y': y,
        'dataset': ['test'] * 100
    })
    
    figs = plots.scatter_with_binned_mean(data, 'y', ['x'], 'dataset')
    assert len(figs) == 1
    assert isinstance(figs['test'], plt.Figure)


def test_scatter_with_binned_mean_multiple_datasets():
    """Test scatter with binned mean for multiple datasets."""
    np.random.seed(42)
    x = np.random.uniform(-1, 1, 100)
    y1 = x[:50] * 0.5 + np.random.normal(0, 0.1, 50)
    y2 = x[:50] * 0.3 + np.random.normal(0, 0.1, 50)
    
    data = pd.DataFrame({
        'x': np.concatenate([x[:50], x[:50]]),
        'y': np.concatenate([y1, y2]),
        'dataset': ['experiment'] * 50 + ['model'] * 50
    })
    
    figs = plots.scatter_with_binned_mean(data, 'y', ['x'], 'dataset')
    assert len(figs) == 2
    assert all(isinstance(fig, plt.Figure) for fig in figs.values())


def test_scatter_with_binned_mean_no_independent():
    """Test scatter with binned mean with no independent variables."""
    data = pd.DataFrame({
        'y': [1, 2, 3],
        'dataset': ['test'] * 3
    })
    
    figs = plots.scatter_with_binned_mean(data, 'y', [], 'dataset')
    assert len(figs) == 0


@pytest.fixture
def sample_data():
    """Fixture providing sample data for integration tests."""
    np.random.seed(42)
    return pd.DataFrame({
        'x': ['A', 'B', 'C'] * 20,
        'y': np.random.normal(0, 1, 60),
        'hue': ['X', 'Y'] * 30,
        'dataset': ['experiment', 'model'] * 30
    })


def test_wide_barplot_integration(sample_data):
    """Integration test for wide barplot."""
    x = sample_data['x'].values
    y = sample_data['y'].values  
    hue = sample_data['hue'].values
    
    ax = plots.wide_barplot(x, y, hue)
    assert isinstance(ax, plt.Axes)
    
