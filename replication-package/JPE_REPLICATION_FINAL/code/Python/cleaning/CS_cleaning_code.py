# Python script to do first cleaning of raw Compustat dataset

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Define the root directory, robust to interactive or file execution
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
except NameError:
    PROJECT_ROOT = Path.cwd()

# Load data
data_raw = pd.read_csv(PROJECT_ROOT / 'proprietary-data-not-for-publication/raw/CS_data_raw.csv')

# Only consider companies whose "fic" (Foreign Incorporation Code) is "USA"
data_raw = data_raw[(data_raw.fic=='USA') & (data_raw.curcdq=='USD')]
# And drop some variables not used
data_raw = data_raw.drop(['fic','curcdq','indfmt','consol','popsrc','datafmt','costat'],axis=1)

# Drop financial firms
data_raw = data_raw[~((data_raw.sic <= 6999) & (data_raw.sic >= 6000))]
# Drop utilities
data_raw = data_raw[~((data_raw.sic <= 4999) & (data_raw.sic >= 4900))]
# Drop quasi-governmental sector
data_raw = data_raw[~(data_raw.sic >= 9000)]

# Only firm-date observations with measured, strictly positive "Total Assets"
data_raw = data_raw[data_raw['atq'] > 0]

# Drop some variables not used
data_raw = data_raw.drop(['apq','lltq'],axis=1)
# Also drop all footnotes not used
data_raw = data_raw.drop(['apq_fn1', 'aqpq_fn', 'atq_fn1', 'cshoq_fn', 'dlcq_fn1', 'dlttq_fn1', 'dpq_fn1', 'ibq_fn1', 'txtq_fn1', 'xrdq_fn', 'capxy_fn1'],axis=1)

# Create target folder
target_folder = PROJECT_ROOT / 'proprietary-data-not-for-publication/processed'
target_folder.mkdir(parents=True, exist_ok=True)
# Save data
data_raw.to_csv(PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/CS_data_clean.csv')