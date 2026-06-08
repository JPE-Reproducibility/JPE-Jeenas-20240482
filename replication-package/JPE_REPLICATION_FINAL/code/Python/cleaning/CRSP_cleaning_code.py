# Python script to drop irrelevant variables in FULL CRSP data to prepare for work on distance-to-default

import pandas as pd
import numpy as np
from pathlib import Path

# Define the root directory, robust to interactive or file execution
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
except NameError:
    PROJECT_ROOT = Path.cwd()

# Import CRSP data, keep relevant variables and save as separate files of length "nlines_i"
file_loc = PROJECT_ROOT / 'proprietary-data-not-for-publication/raw/CRSP_data_raw.csv'
# "nlines_i" is the number of lines to be read in each iteration
nlines_i = 10000000
approx_tot_lines = 90000001 # As almost 91 million lines in raw data, add the extra "1" here, so that 10 splits of 10 million lines get created
N_iters  = int(np.ceil(approx_tot_lines/nlines_i))

# Load all the data, drop variables, and write as separate files
# Create target folder
target_folder = PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/CRSP_split'
target_folder.mkdir(parents=True, exist_ok=True)

for ii in range(N_iters):
    data_temp = pd.read_csv(file_loc, nrows=nlines_i, skiprows=range(1, ii*nlines_i+1))
    data_temp = data_temp.drop(['SHRCLS','TSYMBOL','PRIMEXCH','TRDSTAT','SECSTAT','PAYDT','DIVAMT','BIDLO','ASKHI','VOL','OPENPRC','NUMTRD','RETX'],axis=1)
    # Save the smaller database only including identifiers, returns, prices, and shares outstanding
    data_temp.to_csv(target_folder / f"CRSP_data_clean_{ii}.csv", index=False)
