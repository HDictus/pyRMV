"""Loading of data from Seigle et al. 2019"""

from pathlib import Path
import pandas as pd
from analysis_neuro import terminology as terms


# tuples are required for the data in seigle et al. Set them here after loading.
osi = pd.read_csv(Path(__file__).parent / 'siegle_osi_2019.csv', index_col=0)
osi[terms.STIM_ORIENTATION] = [(0, 45, 90, 135, 180, 225, 270, 315)
                                   for i in range(osi.shape[0])]
osi[terms.RESOLUTION] = [(240, 120) for i in range(osi.shape[0])]
