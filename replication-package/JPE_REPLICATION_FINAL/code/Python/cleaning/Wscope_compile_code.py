# Script for compiling "WSUSX" Worldscope files downloaded from Datastream from Excel sheet format to a CSV

import pandas as pd
import numpy as np
from pathlib import Path

# Define the root directory, robust to interactive or file execution
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
except NameError:
    PROJECT_ROOT = Path.cwd()

# Load raw Worldscope data
Wscope_data_filename = PROJECT_ROOT / 'proprietary-data-not-for-publication/raw/Wscope_data_raw.xlsx'
data_raw = pd.read_excel(Wscope_data_filename, sheet_name='Sheet1', header=0)
# Loop to add onto data_raw
for ii in range(2,25):
    data_temp = pd.read_excel(Wscope_data_filename, sheet_name='Sheet'+str(ii), header=0)
    data_raw  = pd.concat([data_raw, data_temp], ignore_index=True)

# Rename columns
data_raw = data_raw.rename(columns={'WC05601': 'tick', 'WC06004': 'cusip', 'WC18272': 'fnddate', 'WC18273': 'incdate'}, inplace=False)
# Reformat - CUSIP
data_raw.cusip = data_raw.cusip.astype(str)
data_raw['cusip_len'] = data_raw.cusip.str.len()
for nn in range(1,4):
    data_raw.loc[data_raw.cusip_len==9-nn,'cusip'] = '0'*nn + data_raw.loc[data_raw.cusip_len==9-nn,'cusip']
# To all outside (6,9) symbols in cusip, set missing (not plausibly usable as issuer or security CUSIPs)
data_raw.loc[(data_raw.cusip_len > 9) | (data_raw.cusip_len < 6),'cusip'] = np.nan
# Reformat - INCDATE
data_raw.loc[data_raw.incdate.isnull(),'incdate'] = 99 # To convert to integers, need to set missing ones to 99
data_raw['incdate_str'] = data_raw.incdate.astype(int).astype(str)
data_raw['incdate_len'] = data_raw.incdate_str.str.len()
data_raw.loc[data_raw.incdate_len==6, 'incdate_str'] = data_raw.loc[data_raw.incdate_len==6, 'incdate_str'] + '15' # If day missing, set it to 15th
data_raw.loc[data_raw.incdate_len==4, 'incdate_str'] = data_raw.loc[data_raw.incdate_len==4, 'incdate_str'] + '0630' # If month also missing, set the middle of the year (June 30th)
data_raw.loc[data_raw.incdate_len==2, 'incdate_str'] = np.nan # Missing
# Finally, convert to number
data_raw['incdate_num'] = (data_raw.incdate_str.str[0:4].astype(float) + (data_raw.incdate_str.str[4:6].astype(float)-1)/12 + (data_raw.incdate_str.str[6:8].astype(float)-1)/365)

# Reformat - FNDDATE
data_raw.loc[data_raw.fnddate.isnull(),'fnddate'] = 99 # To convert to integers, need to set missing ones to 99
data_raw['fnddate_str'] = data_raw.fnddate.astype(int).astype(str)
data_raw['fnddate_len'] = data_raw.fnddate_str.str.len()
data_raw.loc[data_raw.fnddate_len==6, 'fnddate_str'] = data_raw.loc[data_raw.fnddate_len==6, 'fnddate_str'] + '15' # If day missing, set it to 15th
data_raw.loc[data_raw.fnddate_len==4, 'fnddate_str'] = data_raw.loc[data_raw.fnddate_len==4, 'fnddate_str'] + '0630' # If month also missing, set the middle of the year (June 30th)
data_raw.loc[data_raw.fnddate_len==2, 'fnddate_str'] = np.nan # Missing
# Finally, convert to number
data_raw['fnddate_num'] = (data_raw.fnddate_str.str[0:4].astype(float) + (data_raw.fnddate_str.str[4:6].astype(float)-1)/12 + (data_raw.fnddate_str.str[6:8].astype(float)-1)/365)

# Drop unused variables
data_raw = data_raw.drop(['cusip_len', 'incdate', 'incdate_str', 'incdate_len', 'fnddate', 'fnddate_str', 'fnddate_len'],axis=1)

# Save output, ensuring that relevant folder exists
processed_dir = PROJECT_ROOT / "proprietary-data-not-for-publication" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

data_raw.to_csv(processed_dir / 'Wscope_data_compiled.csv')

