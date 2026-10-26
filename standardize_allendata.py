import pandas as pd
cell_types_data = pd.read_csv("../cell_types_specimen_details.csv")
from pyrmv import terminology as terms
standardized = pd.DataFrame()
standardized[terms.LAYER] = ["L23" if l == "2/3" else f"L{l}" for l in cell_types_data.structure__layer]
standardized[terms.REGION] = cell_types_data.structure_parent__acronym
standardized[terms.SPECIES] = cell_types_data.donor__species
# TODO: how do we deal with these mouse lines where more than one gene is in the name
# is RFP expressed when both are present or just one?
# how do we list this in a dataframe?
# (suggestion: Gene1&Gene2 and Gene1/Gene2)
standardized[terms.GENE_EXPRESSION] = [cell_types_data.line

standardized[terms.SUBJECT] = cell_types_data.donor__id
standardized[terms.GENE_EXPRESSION] = cell_types
