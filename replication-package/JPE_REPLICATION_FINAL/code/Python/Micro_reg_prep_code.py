# Python script for preparing cleaned datasets for main regression analysis, merging Compustat with computed distance-to-default and incorporation date data

import pandas as pd
import numpy as np
from pathlib import Path

# Define the root directory, robust to interactive or file execution
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd()

# Whether construct capital stock from perpetual inventory or not
k_stock_ind = True

# Parameters
start_data = 1986.00 # Because sffr starts in 1990, only use late data
end_data = 2017.00

data = pd.read_csv(PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/CS_data_clean.csv')
# Drop index column created by writing .csv
data = data.drop(['Unnamed: 0'],axis=1)

# Change gvkey into a category variable
data.gvkey = data.gvkey.astype('category')

# Define fin quarter "datafqtr" as numeric, with 1980Q1 = 1980.25, Q2=.50 etc
data['fqtr_num'] = [float(dd[:4]) + 0.25*(float(dd[5])) for dd in data.datafqtr]

# Translate numeric financial quarter "fqtr_num" using financial year end month "fyr" into numeric calendar quarter "cqtr_num"
# To speed up, create indicators for "fyr" groups, to add to "datafqtr_num" to get "cqtr_num"
data['dattr_ind'] = np.zeros(data.shape[0])
data.loc[data.fyr.isin([11,12,1]), 'dattr_ind'] = 0
data.loc[data.fyr.isin([2,3,4]),'dattr_ind'] = 1
data.loc[data.fyr.isin([5]),'dattr_ind'] = 2
data.loc[data.fyr.isin([6,7]),'dattr_ind'] = -2
data.loc[data.fyr.isin([8,9,10]),'dattr_ind'] = -1
data['cqtr_num'] = data.fqtr_num + 0.25*data.dattr_ind
del data['dattr_ind']

# Check for duplicates and drop the ones for which "datacqtr" == NaN -- these have less coverage of data
data = data[~((data.duplicated(['gvkey','cqtr_num'],keep=False)) & pd.isnull(data.datacqtr))]
# As some duplicates remain, keep the remaining first one
data = data[~(data.duplicated(['gvkey','cqtr_num'],keep='first'))]

# Load Fed Funds Rate data
ffr_raw = pd.read_excel(PROJECT_ROOT / 'data/raw/FEDFUNDS_Q.xls', sheet_name='FRED Graph', header=10, index_col=0, usecols = 'A,B')
# NOTE: Timing of FFR -- 2000-01-01 is the average over Q1 (Jan-Mar) of 2000
ffr_raw.index = np.arange(1954.75, 2016.75 + 0.25, 0.25)

ffr_raw.columns = ['ffr']
# And also add the FFR observation into the main dataframe
data['ffr'] = ffr_raw['ffr'].reindex(data['cqtr_num']).values
# First difference of FFR as well
ffr_raw['dffr'] = ffr_raw.ffr.diff(1)
data['dffr'] = ffr_raw['dffr'].reindex(data['cqtr_num']).values

# Also load GDP to include GDP growth
# Real GDP
GDP_table = pd.read_excel(PROJECT_ROOT / 'data/raw/GDPr_Q_BEA.xls', sheet_name='Sheet0', header=5, usecols = 'A:IV', skiprows=0, skipfooter=0)
GDPr_raw = pd.DataFrame(GDP_table.loc[1].iloc[2:].astype(float).to_numpy(), index=np.linspace(1947.25, 2010.50, int((2010.50 - 1947.25)/0.25 + 1)), columns=["gdp_real"])
GDPr_raw['dlgdp_real'] = np.log(GDPr_raw.gdp_real).diff(1)
data["dlgdp_real"] = GDPr_raw["dlgdp_real"].reindex(data["cqtr_num"]).to_numpy()

# Load GVA Deflator (2009=100)
GVADef_table1 = pd.read_excel(PROJECT_ROOT / 'data/raw/GVADEF_Q_BEA.xls', sheet_name='Sheet0', header=5, usecols = 'A:IV', skiprows=0, skipfooter=0)
GVADef_raw1 = pd.DataFrame(GVADef_table1.loc[3].iloc[2:].astype(float).to_numpy(), index=np.linspace(1947.25, 2010.50, int((2010.50 - 1947.25)/0.25 + 1)), columns=["GVADef"])
GVADef_table2 = pd.read_excel(PROJECT_ROOT / 'data/raw/GVADEF_Q_BEA.xls', sheet_name='Sheet1', header=5, usecols = 'A:AD', skiprows=0, skipfooter=0)
GVADef_raw2 = pd.DataFrame(GVADef_table2.loc[3].iloc[2:].astype(float).to_numpy(), index=np.linspace(2010.75, 2017.50, int((2017.50 - 2010.75)/0.25 + 1)), columns=["GVADef"])
GVADef_raw = pd.concat([GVADef_raw1, GVADef_raw2])
data['GVADef']  = GVADef_raw['GVADef'].reindex(data['cqtr_num']).to_numpy()

# Load full ffr shocks ("sffr") dataset from Jarocinski and Karadi (2020)
sffr_raw = pd.read_csv(PROJECT_ROOT / 'data/raw/JK_data_fig4.csv')
sffr_raw.index = pd.to_datetime(sffr_raw.year.astype(str)+sffr_raw.period.astype(str),format='%Y%m')
# Alternative constructions of sffr
# Also compute quarterly shocks to adjust for shock timing (_wd, "weighted"), but not used in paper
sffr_raw['quart'] = sffr_raw.index.quarter
sffr_raw['quartend'] = sffr_raw.index - pd.tseries.offsets.DateOffset(days=1) + pd.tseries.offsets.QuarterEnd()
sffr_raw['quartstart'] = sffr_raw.index + pd.tseries.offsets.DateOffset(days=1) - pd.tseries.offsets.QuarterBegin(startingMonth=1)
sffr_raw['quartlength'] = (sffr_raw.quartend - sffr_raw.quartstart).dt.days
sffr_raw['difquartend'] = [(end-date).days for end,date in zip(sffr_raw.quartend,sffr_raw.index)]
sffr_raw['MPshockSign_curquart'] = sffr_raw.MPshockSign*(sffr_raw.difquartend)/sffr_raw.quartlength
sffr_raw['MPshockSign_nextquart'] = (sffr_raw.MPshockSign*(sffr_raw.quartlength-sffr_raw.difquartend)/sffr_raw.quartlength)
# Compute simple quarterly sums
# Robust across old and new version of pandas
try: 
    sffr_q = sffr_raw.resample('QE-DEC').sum(numeric_only=True)
except Exception:
    sffr_q = sffr_raw.resample('Q-DEC').sum()
# Weighted shocks
sffr_q['MPshockSign_wd'] = sffr_q.MPshockSign_curquart + sffr_q.MPshockSign_nextquart.shift(1)
sffr_q.loc[sffr_q.index[0], 'MPshockSign_wd'] = sffr_q['MPshockSign_curquart'].iloc[0]

# Rename dates as convention above
sffr_q['date_num'] = np.linspace(1990.25, 2017.00, int((2017.00 - 1990.25)/0.25 + 1))
sffr_q.index = sffr_q.date_num
# Select only the used sample
sffr_q = sffr_q[(sffr_q.index >= start_data+0.25) & (sffr_q.index <= end_data)]

# And also add the FFR surprise observation into the main dataframe
data['sffr']        = sffr_q['MPshockSign'].reindex(data['cqtr_num']).to_numpy()
data['sffr_wd']     = sffr_q['MPshockSign_wd'].reindex(data['cqtr_num']).to_numpy()
data['sffr_JK']     = sffr_q['MPshockSign'].reindex(data['cqtr_num']).to_numpy()
data['sffr_JKpmt']  = sffr_q['MPshockPoorman'].reindex(data['cqtr_num']).to_numpy()
data['sffr_ff4']    = sffr_q['ff4_hf'].reindex(data['cqtr_num']).to_numpy()

# Compute real PPE
data['ppentq_real'] = 100 * data.ppentq / data.GVADef
data['ppegtq_real'] = 100 * data.ppegtq / data.GVADef

# Select data with positive "PROPERTY, PLANT AND EQUIPMENT (NET)" for constructing capital stocks using the perpetual inventory method
print("Dropping {} firm-quarters with non-positive \'Property, plant and equipment (Net)\', out of remaining {}.".format((~(data.ppentq > 0)).sum(),data.shape[0]))
data = data[data.ppentq > 0]

######## Compute financial ratios
# Debt
data['dttq'] = data.dlttq + data.dlcq
# Compute leverage
data['lev_rat'] = data.dttq/data.atq
# CAPEX
# Quarterly CAPEX measure
data['capxq'] = data.capxy - data.capxy.shift(1)
data.loc[data['fqtr']==1, 'capxq'] = data.loc[data['fqtr']==1, 'capxy']
mask_fc = (data.gvkey != data.gvkey.shift(1)) | ((data.cqtr_num-data.cqtr_num.shift(1)) != 0.25)
data.loc[mask_fc,'capxq'] = np.nan                      
data['capxq_real'] = 100*data.capxq / data.GVADef                                   
# Market value of common equity
data['mv_ceqq'] = data.cshoq*data.prccq
# Market value of assets
data['mv_atq'] = data.atq + data.mv_ceqq - data.ceqq
# Tobin's Q
data['tobin_q'] = data.mv_atq / data.atq

# Construct capital stock -- because iterating backwards on the law of motion for the perpetual inventory method yields a telescoping sum, missed rows in a firms' series (i.e., an unbalanced panel) are not an issue
def perp_inv_cap(group):
    T_i = group.shape[0]
    k_samp = np.zeros(T_i)
    k_g = np.array(group['ppegtq_real'])
    k_n = np.array(group['ppentq_real'])
    k_samp[0] = k_g[0]
    for tt in range(1,T_i):
        k_samp[tt] = k_samp[tt-1] + (k_n[tt]-k_n[tt-1])

    return k_samp

if k_stock_ind:
    first_ppegtq_ind = data.groupby('gvkey').ppegtq.transform(lambda x: x.first_valid_index())
    ind_values = data.groupby('gvkey').ppegtq.transform(lambda x: x.index)
    # Drop all firms which never have ppegtq measured
    data = data[~pd.isnull(first_ppegtq_ind)]
    ind_values = ind_values[~pd.isnull(first_ppegtq_ind)]
    first_ppegtq_ind = first_ppegtq_ind[~pd.isnull(first_ppegtq_ind)]
    # Drop all firm-quarters which are below their first valid ppegtq
    data = data[ind_values >= first_ppegtq_ind]
    # Construct capital stock
    data.gvkey = data.gvkey.cat.remove_unused_categories()
    xx = data.groupby('gvkey').apply(perp_inv_cap)
    data['k_stock'] = np.concatenate(xx.values)

#####
### FURTHER DATA SELECTION AFTER SELECTING SAMPLE PERIOD (1985Q4--2016Q4)
#####
# Select sample
data = data[data.cqtr_num.between(start_data,end_data)]
print("After selecting the sample {}--{} there are {} firm-quarters remaining.".format(start_data,end_data,data.shape[0]))
# Drop data with missing and negative Debt in Current Liabilities
print("Dropping {} firm-quarters with negative and {} with missing \'Debt in Current Liabilities\', out of remaining {}.".format((data.dlcq < 0).sum(),pd.isnull(data.dlcq).sum(),data.shape[0]))
data = data[data.dlcq >= 0.0]
# Drop data with missing and negative Long-Term Debt
print("Dropping {} firm-quarters with negative and {} with missing \'Long Term Debt -- Total\', out of remaining {}.".format((data.dlttq < 0).sum(),pd.isnull(data.dlttq).sum(),data.shape[0]))
data = data[data.dlttq >= 0.0]
# Drop data with missing and negative Cash and Short Term Investments
print("Dropping {} firm-quarters with negative and {} with missing \'Cash and Short Term Investments\', out of remaining {}.".format((data.cheq < 0).sum(),pd.isnull(data.cheq).sum(),data.shape[0]))
data = data[data.cheq >= 0.0]
# Drop data with missing and non-positive Sales
print("Dropping {} firm-quarters with non-positive and {} with missing \'Sales\', out of remaining {}.".format((data.saleq <= 0).sum(),pd.isnull(data.saleq).sum(),data.shape[0]))
data = data[data.saleq > 0.0]

###
# Drop variables that are definitely not used
###
data = data.drop(['tic','aqpq','ceqq','chq','cshoq','ivltq','ivstq','txtq','wcapq','xrdq','dvpy','prccq','mv_ceqq','mv_atq'],axis=1)
data = data.drop(['datafqtr','saleq_fn1'],axis=1)

# Deal with age from WORLDSCOPE, first matching GVKEY to CUSIP
if True:
    ccm_data = pd.read_csv(PROJECT_ROOT / 'proprietary-data-not-for-publication/raw/CCM_data_raw.csv') 
    ccm_data = ccm_data.rename(columns={'GVKEY': 'gvkey'}, inplace=False)
    # Only keep primary ('P','C') securities
    ccm_data = ccm_data[(ccm_data.LINKPRIM=='C') | (ccm_data.LINKPRIM=='P')]
    # As GVKEYs and CUSIPs are mapped as a bijection, can drop duplicates
    ccm_data = ccm_data[~((ccm_data.duplicated(['gvkey','datadate'],keep=False)) & pd.isnull(ccm_data.datacqtr))]
    # Do multiindex on CCM
    ccm_data.set_index(['gvkey', 'datadate'], inplace=True, drop=False)
    # Merge the CUSIP to the CSTAT data
    data    = data.join(ccm_data[['cusip','conm']], on=['gvkey','datadate'], rsuffix='_ccm')

    # Fill up CUSIPs in missing firm-quarters by observed values in other quarters
    g = data.groupby('gvkey')
    g['cusip'].transform(lambda s: '99' if pd.isnull(s).all() == True else s.loc[s.first_valid_index()])
    data['cusip_filled'] = g['cusip'].transform(lambda s: '99' if pd.isnull(s).all() == True else s.loc[s.first_valid_index()])

    # Load WSCOPE data
    wsc_data = pd.read_csv(PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/Wscope_data_compiled.csv')
    # Merge the CUSIP to the CSTAT data
    data    = data.merge(wsc_data[['NAME','cusip','incdate_num','fnddate_num']], left_on='cusip_filled', right_on='cusip', how="left", suffixes=('_cs', '_wsc'))

    # Drop further unused variables
    data = data.drop(['conm_ccm', 'NAME','cusip_wsc'],axis=1)
    # Rename CUSIP back
    data = data.rename(columns={'cusip_cs': 'cusip'}, inplace=False)

####
# After having done all the above, also load KMV Distance-to-Default data from CRSP daily data (in many tiny slices) and match based on end of calendar quarter observation. To do this, find the most recent observation of DD in any calendar quarter and match it to the Compstat data based on the calendar quarter.
# Merge based on "merge_date" in "data" which is the actual calendar quarter end date. Just need to construct the corresponding ones also in each CRSP data slice.
####
if True:
    CRSP_target_folder = PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/CRSP_split'
    # Create merge_date based on calendar quarter -- was dropped just before, so do it again
    data['merge_date'] = pd.to_datetime(data['datacqtr']) + pd.offsets.QuarterEnd(0)
    # Iterate through the CRSP data slices with DD data
    Ncrsp = 60
    for ii in range(Ncrsp):
        crsp_data_in = pd.read_csv(CRSP_target_folder / f"CRSP_data_DD_int_done_{ii}.csv")
        # Create CRSP date
        crsp_data_in['dateval'] = pd.to_datetime(crsp_data_in['date'],format='%Y%m%d')
        # Create "quarter end" observation (call it "merge_date")
        crsp_data_in['merge_date'] = crsp_data_in['dateval'] + pd.offsets.QuarterEnd(0)
        # This gives the same "merge date" to all daily observations in a quarter.
        # Drop all "gvkey-merge date" duplicates, leaving only the last one.
        crsp_data_in = crsp_data_in[~crsp_data_in.duplicated(['gvkey','merge_date'], keep='last')]
        # Finally, merge the resulting database to the Compustat data
        data = data.merge(crsp_data_in[['gvkey','Va','DD','PD','merge_date']], how='left', left_on = ['gvkey','merge_date'], right_on = ['gvkey','merge_date'], suffixes=('','_2'))
        # Combine (Va,DD,PD) columns into one
        if ii>0:
            data['Va'] = data['Va'].combine_first(data['Va_2'])
            data['DD'] = data['DD'].combine_first(data['DD_2'])
            data['PD'] = data['PD'].combine_first(data['PD_2'])
            # Drop the extra created columns
            del data['Va_2'], data['DD_2'], data['PD_2']

    # Also, in the end, drop Va and PD also, which we don't really need
    del data['Va'], data['PD']

#######
# SAVE DATA FILE
#######

data.to_csv(PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/Micro_reg_data_compiled.csv')