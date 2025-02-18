import pandas as pd
import analysis_neuro.terminology as terms
import matplotlib.pyplot as plt
from analysis_neuro import plots


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
        'param': 3,
        'compare': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c']
    })
    out = plots.crossplot(data, 'measured', ['param'], 'compare')
    assert len(out) == 3