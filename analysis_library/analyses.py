from .measurements import CellDensity
from .plots import BarPlot
from analysis_core import Analysis
import analysis_library.terminology as terms

cell_density = Analysis(
    measurement=CellDensity(),
    parameters={terms.LAYER: ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']},
    plot=BarPlot())
