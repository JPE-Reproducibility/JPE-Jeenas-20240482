# Script for working on FOFA data from the Federal Reserve

import pandas as pd
import numpy as np
from pathlib import Path

# Define the root directory, robust to interactive or file execution
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd()

####### LOADING DATA #########
data = pd.read_csv(PROJECT_ROOT / 'data/raw/FRB_Z1.csv')
# Drop first few header rows
data = data.drop(range(5))
data.rename(columns = {'Series Description':'Year'}, inplace = True)
data = data.astype('float64')
data.index = data.Year
data = data.drop('Year',axis=1)

# Select time-period
start_data = 1945
end_data = 2015
data = data[(data.index >= start_data) * (data.index <= end_data)]

####### COMPUTING VARIABLES #########
####
# Consider all entries possibly included in "cheq" in Compustat
####
# Redefine variable names for brevity
data['Nonfin Corp; CHDC'] = data['Nonfinancial corporate business; checkable deposits and currency; asset']
data['Nonfin Corp; TSdep'] = data['Nonfinancial corporate business; total time and savings deposits; asset']
data['Nonfin Corp; MMF'] = data['Nonfinancial corporate business; money market mutual fund shares; asset']
data['Nonfin Corp; SecRepo'] = data['Nonfinancial corporate business; security repurchase agreements; asset']
data['Nonfin Corp; ComPap'] = data['Nonfinancial corporate business; commercial paper; asset']
data['Nonfin Corp; Treasury'] = data['Nonfinancial corporate business; Treasury securities; asset']
data['Nonfin Corp; AGSE'] = data['Nonfinancial corporate business; agency- and GSE-backed securities; asset']
data['Nonfin Corp; MunSec'] = data['Nonfinancial corporate business; municipal securities; asset']
data['Nonfin Corp; Mutf'] = data['Nonfinancial corporate business; mutual fund shares; asset']

data['Nonfin Corp; Cash_total'] = data['Nonfin Corp; CHDC'] + data['Nonfin Corp; TSdep'] + data['Nonfin Corp; MMF'] + data['Nonfin Corp; SecRepo'] + data['Nonfin Corp; ComPap'] + data['Nonfin Corp; Treasury'] + data['Nonfin Corp; AGSE'] + data['Nonfin Corp; MunSec'] + data['Nonfin Corp; Mutf']

# Compute all the shares
data['Nonfin Corp; CHDCdCash'] = data['Nonfin Corp; CHDC'] / data['Nonfin Corp; Cash_total']
data['Nonfin Corp; TSdepdCash'] = data['Nonfin Corp; TSdep'] / data['Nonfin Corp; Cash_total']
data['Nonfin Corp; MMFdCash'] = data['Nonfin Corp; MMF'] / data['Nonfin Corp; Cash_total']
data['Nonfin Corp; SecRepodCash'] = data['Nonfin Corp; SecRepo'] / data['Nonfin Corp; Cash_total']
data['Nonfin Corp; ComPapdCash'] = data['Nonfin Corp; ComPap'] / data['Nonfin Corp; Cash_total']
data['Nonfin Corp; TreasurydCash'] = data['Nonfin Corp; Treasury'] / data['Nonfin Corp; Cash_total']
data['Nonfin Corp; AGSEdCash'] = data['Nonfin Corp; AGSE'] / data['Nonfin Corp; Cash_total']
data['Nonfin Corp; MunSecdCash'] = data['Nonfin Corp; MunSec'] / data['Nonfin Corp; Cash_total']
data['Nonfin Corp; MutfdCash'] = data['Nonfin Corp; Mutf'] / data['Nonfin Corp; Cash_total']

# Shares data
liq_shares = data[['Nonfin Corp; CHDCdCash', 'Nonfin Corp; TSdepdCash', 'Nonfin Corp; MMFdCash', 'Nonfin Corp; SecRepodCash', 'Nonfin Corp; ComPapdCash', 'Nonfin Corp; TreasurydCash', 'Nonfin Corp; AGSEdCash', 'Nonfin Corp; MunSecdCash', 'Nonfin Corp; MutfdCash']]
base_liq_shares  = 100*liq_shares.loc[1990:2008,:].mean()
pre90_liq_shares = 100*liq_shares.loc[1985:1989,:].mean()

# Return base shares as latex table
asset_map = {
    "CHDCdCash": "Checkable deposits and currency",
    "TSdepdCash": "Time and savings deposits",
    "MMFdCash": "Money market fund shares",
    "SecRepodCash": "Security repurchase agreements",
    "ComPapdCash": "Commercial paper",
    "TreasurydCash": "Treasury securities",
    "AGSEdCash": "Agency- and GSE-backed securities",
    "MunSecdCash": "Municipal securities",
    "MutfdCash": "Mutual fund shares",
}
lines = []
for idx, val in base_liq_shares.items():
    code = idx.split(";")[1].strip()      # remove "Nonfin Corp;"
    asset_name = asset_map.get(code, code)
    lines.append(f"        {asset_name} & {val:.1f}\\% \\\\")
latex_table = (
    "\\begin{table}[htb]\n\n"
    "\t\\setlength\\tabcolsep{5.2pt}\n"
    "\t\\centering\n"
    "\t\\caption{Asset shares in the US nonfinancial corporate sector's liquid assets portfolio} "
    "\\label{Tab_Appx_Data_FoFashares}\n"
    "\t\\begin{tabular}{lc}\n"
    "\t\t\\hline \\hline\n"
    "\t\tAsset & Share \\\\\n"
    "\t\t\\hline\n"
    + "\n".join(lines) + "\n"
    "\t\t\\hline \\hline\n"
    "\t\\end{tabular}\n\n"
    "\t\\vspace{2pt}\n"
    "\t\\begin{minipage}{\\textwidth}\n"
    "\t\t{\\footnotesize \\emph{Source:} Annual Flow of Funds Accounts, time-average shares between 1990--2008.}\n"
    "\t\\end{minipage}\n"
    "\\end{table}"
)

# Set output targets
tables_dir = PROJECT_ROOT / "output" / "tables"
other_dir = PROJECT_ROOT / "output" / "other"
interim_dir = PROJECT_ROOT / "interim_output"
# And make sure corresponding folders exist, just in case
tables_dir.mkdir(parents=True, exist_ok=True)
other_dir.mkdir(parents=True, exist_ok=True)
interim_dir.mkdir(parents=True, exist_ok=True)

# Print into table file
with open(tables_dir / "TabA5_liqshares.tex", "w") as f:
    f.write(latex_table)

# Also, print the shares for calibrating the pre-1990 model
with open(other_dir / "Text_AppxB10p1_liqshares.txt", "w") as f:
    f.write(f"Pre-1990, the share of checkable deposits and currency in the nonfinancial corporate liquidity portfolio was {round(pre90_liq_shares['Nonfin Corp; CHDCdCash'],1)}%. The share of time and savings deposits was {round(pre90_liq_shares['Nonfin Corp; TSdepdCash'],1)}%. This implies an rf-to-rm pass-through of {round((100-pre90_liq_shares['Nonfin Corp; CHDCdCash'] - 0.5*pre90_liq_shares['Nonfin Corp; TSdepdCash'])/100,3)}.")

# Construct aggregate liquidity ratio to return to model code
data['Nonfin Corp; cheat_rat'] = data['Nonfin Corp; Cash_total']/data['Nonfinancial corporate business; total assets']
# Package up and save the liquidity ratio time series as csv
data['Nonfin Corp; cheat_rat'].to_csv(interim_dir / "FOFA_agg_cheat_t.csv")
