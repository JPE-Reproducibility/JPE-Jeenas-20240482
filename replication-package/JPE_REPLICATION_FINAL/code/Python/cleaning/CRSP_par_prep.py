# Python script to load CRSP and COMPUSTAT merged database
# Order by gvkey, and split into 60 files to be later read in parallelized procedure to compute distance-to-default

import pandas as pd
import numpy as np
from pathlib import Path

# Define the root directory, robust to interactive or file execution
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
except NameError:
    PROJECT_ROOT = Path.cwd()

# Import and split the merged data
# "nlines_i" is the number of lines to be read in each iteration
nlines_i = 10000000
approx_tot_lines = 90000001
N_iters  = int(np.ceil(approx_tot_lines/nlines_i))

Nout = 60 # How many split files to create

# Loop over data and compile into one big database
CRSP_target_folder = PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/CRSP_split'
# Load first split as base
crsp_data = pd.read_csv(CRSP_target_folder / f"CRSP_data_DD_int_{0}.csv")
for ii in range(1, N_iters):
    crsp_data_in = pd.read_csv(CRSP_target_folder / f"CRSP_data_DD_int_{ii}.csv")
    # Combine to existing data
    crsp_data = pd.concat([crsp_data, crsp_data_in], axis=0, ignore_index=True)
    del crsp_data_in

# Transform gvkey to category
crsp_data['gvkey'] = crsp_data.gvkey.astype('int').astype('category')
# If there are some literal copies of some rows, drop these
crsp_data   = crsp_data[~crsp_data.duplicated(['gvkey','date','TICKER','PERMNO'], keep='first')]
# Reorder data by gvkey and date
crsp_data   = crsp_data.sort_values(by=['gvkey', 'date'])

# Count different gvkeys in order to partition data so that gvkeys do not get split
gv_list = crsp_data.groupby('gvkey').date.count().reset_index()
gv_list = gv_list.gvkey
Nf      = len(gv_list)
nper    = int(np.ceil(Nf/Nout))
Nout_true = int(np.ceil(Nf/nper))

# Save all in pieces
for ii in range(0,Nout_true):
    gv_sel = gv_list.iloc[range(ii*nper,min((ii+1)*nper, Nf))]
    crsp_data_sel = crsp_data[crsp_data.gvkey.isin(gv_sel)]
    crsp_data_sel.to_csv(CRSP_target_folder / f"CRSP_data_DD_int_forpar_{ii}.csv", index=False)
