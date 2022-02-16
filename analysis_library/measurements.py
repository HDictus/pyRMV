from . import terminology as terms
from analysis_core.protocols import Model


class CellDensity:
    """
    We measure the cell density of the model in cells per $cm^3$
    based on the total cells and the total volume (within parameters)
    """

    measured_phenomena = terms.CELL_DENSITY

    def __call__(self, model: Model, parameters: dict):
        """
        measure the cell density of a model under given parameters
        """
        print(model)
        return {terms.CELL_DENSITY:
                model.num_cells(parameters) / model.region_volume(parameters)}
