# Python script to merge (cleaned, split) CRSP data with Compustat balance sheet data

import pandas as pd
import numpy as np
from pathlib import Path

# Define the root directory, robust to interactive or file execution
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
except NameError:
    PROJECT_ROOT = Path.cwd()

# Parameters
int_QtoD_ind = True # Indicator for later linearly interpolating quarterly balance sheet data to daily

# Parameters related to splitting CRSP data
# "nlines_i" is the number of lines to be read in each iteration
nlines_i = 10000000
approx_tot_lines = 90000001
N_iters  = int(np.ceil(approx_tot_lines/nlines_i))

#####
# IMPORT CCM DATA -- CRSP-Compustat links
#####
ccm_data = pd.read_csv(PROJECT_ROOT / 'proprietary-data-not-for-publication/raw/CCM_data_raw.csv')
ccm_data = ccm_data.rename(columns={'GVKEY': 'gvkey'}, inplace=False)
# Only keep primary ('P','C') securities
ccm_data = ccm_data[(ccm_data.LINKPRIM=='C') | (ccm_data.LINKPRIM=='P')]
# There are 555 duplicate combinations of (LPERMNO,datadate) because of firms changing financial year timing. Thus, the same datadate can have TWO observations: one for the old financial quarter, and one for the new one. Exactly one of each pair seems to have datacqtr missing, so drop all these ones to arrive at a set of unique (LPERMNO,datadate). Among the duplicates, the number of unique LPERMNO's and gvkey's is identical, so this choice should not affect the matching of LPERMNO to gvkey.
ccm_data = ccm_data[~((ccm_data.duplicated(['gvkey','datadate'],keep=False)) & pd.isnull(ccm_data.datacqtr))]
# Do multiindex
ccm_data.set_index(['gvkey', 'datadate'], inplace=True, drop=False)

#####
# COMPUSTAT
#####
cstat_data  = pd.read_csv(PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/CS_data_clean.csv')
# Drop index column created by writing .csv
cstat_data  = cstat_data.drop(['Unnamed: 0'],axis=1)
# Immediately drop all irrelevant variables, except debt and stock price data
cstat_data = cstat_data.drop(['actq', 'aqpq', 'atq', 'ceqq', 'cheq', 'chq', 'dpq', 'ibq', 'intanq', 'invtq', 'ivltq', 'ivstq', 'lctq', 'ltq', 'oiadpq', 'ppegtq', 'ppentq', 'rectq', 'rectrq', 'req', 'saleq', 'txdbq', 'txtq', 'wcapq', 'xintq', 'xrdq', 'aqcy', 'capxy', 'dltisy', 'dltry', 'dvpy', 'dvy', 'prstkcy', 'sstky', 'ppentq_fn1', 'rectrq_fn1', 'req_fn1', 'saleq_fn1', 'xintq_fn1', 'sic'],axis=1)

# Change gvkey into a category variable
cstat_data.gvkey = cstat_data.gvkey.astype('category')
# Check for duplicates and drop the ones for which "datacqtr" == NaN -- as for CCM; also these have less data
cstat_data = cstat_data[~((cstat_data.duplicated(['gvkey','datadate'],keep=False)) & pd.isnull(cstat_data.datacqtr))]
# There are still some LITERAL duplicates left (due to appearance of footnotes), drop these
cstat_data = cstat_data[~cstat_data.duplicated(['gvkey','datadate'],keep='first')]
# Organizing data
cstat_data['dateval'] = pd.to_datetime(cstat_data['datadate'],format='%Y%m%d')
# Only keep the CSTAT observation dates that are in CCM
cstat_data = cstat_data[cstat_data.datadate.isin(ccm_data.datadate.unique())]
# Merge the LPERMNO to CSTAT
cstat_data = cstat_data.join(ccm_data['LPERMNO'].astype('Int64'), on=['gvkey','datadate'])

######
# CRSP
######
CRSP_target_folder = PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/CRSP_split'

# Load split CRSP files
for ii in range(N_iters):
    crsp_data = pd.read_csv(CRSP_target_folder / f"CRSP_data_clean_{ii}.csv")
    # Create CRSP date
    crsp_data['dateval'] = pd.to_datetime(crsp_data['date'],format='%Y%m%d')
    crsp_data['PERMNO'] = pd.to_numeric(crsp_data['PERMNO'], errors='coerce').astype('Int64')
    # Change PERMNO into category
    crsp_data.PERMNO = crsp_data.PERMNO.astype('category')

    ######
    # MERGING
    ######
    ### If FYR is in (12,9,6,3), the measurement date ("datadate") should be the end of the previous quarter
    crsp_data['merge_date'] = (crsp_data['dateval'] + pd.tseries.offsets.DateOffset(days=1)) - pd.tseries.offsets.QuarterEnd()
    if int_QtoD_ind:
        # If interpolate later, align gvkey "properly"
        crsp_data = crsp_data.merge(cstat_data[cstat_data['fyr'].isin([12,9,6,3])][['LPERMNO', 'dateval', 'gvkey']], how='left', left_on = ['PERMNO','merge_date'], right_on = ['LPERMNO','dateval'], suffixes = ('','_cstat'))
    else:
        # If don't interpolate, align all variables "properly"
        crsp_data = crsp_data.merge(cstat_data[cstat_data['fyr'].isin([12,9,6,3])], how='left', left_on = ['PERMNO','merge_date'], right_on = ['LPERMNO','dateval'], suffixes = ('','_cstat'))

    ### If FYR is in (11,8,5,2), the measurement dates are "quarter end - 1 month". To compute measurement date, take (current quarter's end - 1 month). If the resulting observation date is past the current "date_val", subtract an additional quarter
    crsp_data['merge_date'] = crsp_data['dateval'] + pd.tseries.offsets.QuarterEnd()
    crsp_data['merge_date'] = crsp_data['merge_date'] - pd.tseries.offsets.MonthEnd()
    # Get indicator that determines which dates need to be adjusted back
    mask_dates = crsp_data['merge_date'] > crsp_data['dateval']
    crsp_data.loc[mask_dates,'merge_date'] = crsp_data.loc[mask_dates,'merge_date'] - pd.tseries.offsets.QuarterEnd()
    crsp_data.loc[mask_dates,'merge_date'] = crsp_data.loc[mask_dates,'merge_date'] - pd.tseries.offsets.MonthEnd()
    if int_QtoD_ind:
        # If interpolate later, align gvkey "properly"
        crsp_data = crsp_data.merge(cstat_data[cstat_data['fyr'].isin([11,8,5,2])][['LPERMNO', 'dateval', 'gvkey']], how='left', left_on = ['PERMNO','merge_date'], right_on = ['LPERMNO','dateval'], suffixes = ('','_f2'))
    else:
        # If don't interpolate, align all variables "properly"
        crsp_data = crsp_data.merge(cstat_data[cstat_data['fyr'].isin([11,8,5,2])], how='left', left_on = ['PERMNO','merge_date'], right_on = ['LPERMNO','dateval'], suffixes = ('','_f2'))
    # Get rid of duplicate columns
    for varname in crsp_data.columns:
        if varname[-3:] == '_f2':
            crsp_data[varname[:-3]].fillna(crsp_data[varname], inplace = True)
            del crsp_data[varname]

    ### If FYR is in (10,7,4,1), the measurement dates are "quarter end + 1 month". To compute measurement date, take (last quarter's end + 1 month). If the resulting observation date is past the current "date_val", subtract an additional quarter + add 1 month
    crsp_data['merge_date'] = crsp_data['dateval'] - pd.tseries.offsets.QuarterEnd()
    crsp_data['merge_date'] = crsp_data['merge_date'] + pd.tseries.offsets.MonthEnd()
    # Get indicator that determines which dates need to be adjusted back
    mask_dates = crsp_data['merge_date'] > crsp_data['dateval']
    crsp_data.loc[mask_dates,'merge_date'] = crsp_data.loc[mask_dates,'merge_date'] - 2*pd.tseries.offsets.QuarterEnd()
    crsp_data.loc[mask_dates,'merge_date'] = crsp_data.loc[mask_dates,'merge_date'] + pd.tseries.offsets.MonthEnd()
    if int_QtoD_ind:
        # If interpolate later, align gvkey
        crsp_data = crsp_data.merge(cstat_data[cstat_data['fyr'].isin([10,7,4,1])][['LPERMNO', 'dateval', 'gvkey']], how='left', left_on = ['PERMNO','merge_date'], right_on = ['LPERMNO','dateval'], suffixes = ('','_f3'))
    else:
        # If don't interpolate, align all variables
        crsp_data = crsp_data.merge(cstat_data[cstat_data['fyr'].isin([10,7,4,1])], how='left', left_on = ['PERMNO','merge_date'], right_on = ['LPERMNO','dateval'], suffixes = ('','_f3'))
    # Get rid of duplicate columns
    for varname in crsp_data.columns:
        if varname[-3:] == '_f3':
            crsp_data[varname[:-3]].fillna(crsp_data[varname], inplace = True)
            del crsp_data[varname]

    # If only loaded gvkey above, load now quarterly observations of all other variables
    if int_QtoD_ind:
        crsp_data['merge_date'] = crsp_data['dateval']
        crsp_data = crsp_data.merge(cstat_data, how='left', left_on = ['PERMNO','merge_date'], right_on = ['LPERMNO','dateval'], suffixes = ('','_f4'))
        # Get rid of duplicate columns
        for varname in crsp_data.columns:
            if varname[-3:] == '_f4':
                crsp_data[varname[:-3]].fillna(crsp_data[varname], inplace = True)
                del crsp_data[varname]

    ###
    # Computing variables for DD calculations
    ###
    # NOTE: COMPUSTAT cshoq is in MILLIONS of stocks and all other variables (most importantly, debt) are measured in MILLIONS of dollars; CRSP SHROUT is in THOUSANDS -- so we divide the CRSP SHROUT by 1000 to make the resulting market equity value comparable with Compustat data
    crsp_data['DD_debt'] = crsp_data.dlcq + 0.5*crsp_data.dlttq
    crsp_data['mkt_eqy'] = (crsp_data.SHROUT / 1000) * np.abs(crsp_data.PRC)

    # Drop observations missing match from COMPUSTAT
    crsp_data = crsp_data[~pd.isnull(crsp_data.gvkey)]

    # Drop all irrelevant variables from the compiled dataset
    crsp_data = crsp_data.drop(['CUSIP', 'PRC', 'SHROUT', 'dateval', 'merge_date', 'datadate', 'fyearq', 'fqtr', 'fyr', 'tic', 'conm', 'datacqtr', 'datafqtr', 'cshoq', 'dlcq', 'dlttq', 'prccq', 'dateval_cstat', 'LPERMNO', 'vwretd', 'ewretd'],axis=1)

    # Finally write up data
    if int_QtoD_ind:
        crsp_data.to_csv(CRSP_target_folder / f"CRSP_data_DD_int_{ii}.csv", index=False)
    else:
        crsp_data.to_csv(CRSP_target_folder / f"CRSP_data_DD_{ii}.csv", index=False)
