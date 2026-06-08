# Python script to load CRSP and COMPUSTAT merged database that already includes (NOMINAL) market equity and the conventionally used face value of debt
# Run in array in HPC, or in parallel locally, loading data piece corresponding to "slurmcounter"
import pandas as pd
import numpy as np
import scipy.optimize as scoptim
from scipy.stats import norm
import time
import sys
from pathlib import Path

# Define the root directory, robust to interactive or file execution
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd()

slurmcounter = int(sys.argv[1])
# Load data corresponding to slurmcounter
CRSP_target_folder = PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/CRSP_split'
crsp_data = pd.read_csv(CRSP_target_folder / f"CRSP_data_DD_int_forpar_{slurmcounter}.csv")

# Transform gvkey to category
crsp_data['gvkey'] = crsp_data.gvkey.astype('int').astype('category')
# Create CRSP date
crsp_data['dateval'] = pd.to_datetime(crsp_data['date'],format='%Y%m%d')

# Analyze holes in gvkey-based data
crsp_data['datedif'] = (crsp_data.dateval - crsp_data.dateval.shift(1)).dt.days # Difference in days
# Drop datedif where gvkey changed
mask_fc = crsp_data.gvkey != crsp_data.gvkey.shift(1)
crsp_data.loc[mask_fc,'datedif'] = np.nan

# Keep only firm segments with ≥252 daily observations to ensure a full one-year window of continuous data for KMV/Merton distance-to-default estimation.
# Insert point to break data
crsp_data['break_ind'] = False
crsp_data.loc[crsp_data.datedif > 7, 'break_ind'] = True
# Count up how many breaks have happened in whole dataset
crsp_data['break_count'] = crsp_data.break_ind.cumsum()
# Based on breaks, create new gvkey's
crsp_data['gvkey_b']    = crsp_data.gvkey.astype(str) + '_' + crsp_data.break_count.astype(str)
crsp_data['gvkey_b']    = crsp_data.gvkey_b.astype('category')
# Count how many times each "new" gvkey appears and drop any with less than 252 observations
gvb_counts = crsp_data.groupby('gvkey_b').gvkey.count().reset_index()
gvb_drop_list = gvb_counts.loc[gvb_counts.gvkey < 252, 'gvkey_b']
crsp_data   = crsp_data[~crsp_data.gvkey_b.isin(gvb_drop_list)]
# Drop unused levels
crsp_data['gvkey_b']   = crsp_data.gvkey_b.cat.remove_unused_categories()
crsp_data['gvkey']   = crsp_data.gvkey.cat.remove_unused_categories()
# And drop unused columns
crsp_data   = crsp_data.drop(['break_ind', 'break_count'],axis=1)

# Load risk free rate
GS1_raw         = pd.read_csv(PROJECT_ROOT / 'data/raw/DGS1.csv')
GS1_raw.columns = ['DATE', 'r_f']
GS1_raw.loc[GS1_raw.r_f=='.' ,'r_f']       = np.nan
GS1_raw.r_f     = GS1_raw.r_f.astype('float')/100
GS1_raw['r_f_int'] = GS1_raw.r_f.interpolate(method='linear', axis=0, limit=999, limit_direction='forward')
GS1_raw.DATE    = pd.to_datetime(GS1_raw.DATE)
GS1_raw.index   = GS1_raw.DATE
# Only keep observations before latest GDPDef observation (2020-06-01)
GS1_raw = GS1_raw[GS1_raw['DATE'] <= '2020-06-01']
del GS1_raw['DATE']
# Merge to CRSP data
crsp_data['r_f'] = np.array(GS1_raw.r_f_int[crsp_data['dateval']])

# Load GDP Deflator (2015=100)
GDPDef_raw = pd.read_csv(PROJECT_ROOT / 'data/raw/GDPDEF_Q.csv')
GDPDef_raw.columns = ['DATE', 'GDPDef']
# Impose midpoint of quarter (add 45 days) to GDP Deflator measurement
GDPDef_raw.DATE = pd.to_datetime(GDPDef_raw.DATE) + pd.DateOffset(days=45)
# Generate daily series of GDP Deflator by interpolation
GDPDef_daily    = pd.DataFrame({'DATE': pd.date_range(start='1960-01-01', end='2020-06-01')})
GDPDef_daily    = GDPDef_daily.merge(GDPDef_raw, how='left', left_on=['DATE'], right_on=['DATE'])
GDPDef_daily['GDPDef_int'] = GDPDef_daily.GDPDef.interpolate(method='linear', axis=0, limit=999, limit_direction='forward')
GDPDef_daily.index = GDPDef_daily.DATE
# Merge to CRSP data
crsp_data['GDPDef'] = np.array(GDPDef_daily.GDPDef_int[crsp_data['dateval']])

# Use the daily GDP deflator data and daily Treasury rate to compute the implied daily real rate
GDPDef_daily['GDPDef_infl'] = (GDPDef_daily.GDPDef_int / GDPDef_daily.GDPDef_int.shift(1))**365 - 1
# Merge to GS1 data to compute real rate
GS1_raw['GDPDef_infl']  = np.array(GDPDef_daily.GDPDef_infl[GS1_raw.index])
GS1_raw['rr_f']         = GS1_raw['r_f_int'] - GS1_raw['GDPDef_infl']
# Merge real rate to CRSP
crsp_data['rr_f'] = np.array(GS1_raw.rr_f[crsp_data['dateval']])

# Deflate main variables
if True:
    crsp_data['mkt_eqy'] = 100 * crsp_data['mkt_eqy'] / crsp_data['GDPDef']
    crsp_data['DD_debt'] = 100 * crsp_data['DD_debt'] / crsp_data['GDPDef']

# Delete all unnecessary frames
del GDPDef_raw, GS1_raw, GDPDef_daily
del crsp_data['GDPDef']

# Define residual function for computing Va based on BS pricing; where x=asset value, sig_a=asset volatlity, d=face value of debt, ve=market equity value, r=risk free rate, Td=debt maturity in days
def BS_Va_dev(x, sig_a, d, ve, r, Td):
    d1 = (np.log(x/d) + (r + 0.5*sig_a**2)*Td) / (sig_a*np.sqrt(Td))
    d2 = d1 - sig_a * np.sqrt(Td)
    dev = ve - (x*norm.cdf(d1) - np.exp(-r*Td)*d*norm.cdf(d2))
    return dev

# Write loop to circle through all firms
crsp_data['Va'] = np.nan
crsp_data['DD'] = np.nan
crsp_data['PD'] = np.nan

gv_list = crsp_data.groupby('gvkey_b').gvkey.count().reset_index()
gv_list = gv_list.gvkey_b

ntot    = len(gv_list)
ncount  = 0
start_tclock = time.time()
for gv_name in gv_list:
    ncount       = ncount + 1
    elapsed_tclock = time.time() - start_tclock
    print("Computing DD for firm "+gv_name+", "+str(ncount)+"/"+str(ntot)+"="+str(round(ncount/ntot,3))+". Time: "+str(round(elapsed_tclock,2))+"s. ETA: "+str(round((1/3600)*(ntot-ncount)*elapsed_tclock/ncount,2))+"h.")
    f_data = crsp_data[crsp_data.gvkey_b == gv_name]
    # Interpolate debt and market value variables
    f_data['DD_debt_int'] = f_data.DD_debt.interpolate(method='linear', axis=0, limit=9999, limit_direction='forward')
    f_data['mkt_eqy_int'] = f_data.mkt_eqy.interpolate(method='linear', axis=0, limit=30, limit_direction='forward')

    # If equity or debt data is completely missing, skip this gvkey
    if ((~pd.isnull(f_data.mkt_eqy_int)).sum() == 0) | ((~pd.isnull(f_data.DD_debt_int)).sum() == 0):
        continue

    # Work on KMV DD model
    min_hist_vals = 252
    win           = 252
    # Start point of data
    start_time = min_hist_vals
    timesteps  = range(min_hist_vals, f_data.shape[0]+1)

    # Calculate historical equity return and volatility
    lr_e            = np.log(f_data.mkt_eqy_int / f_data.mkt_eqy_int.shift(1))
    sigma_e         = lr_e.rolling(win-1).std()

    # Array for storing results
    Va_out = np.empty(f_data.shape[0])
    mu_out = np.zeros(f_data.shape[0])
    DD_out = np.zeros(f_data.shape[0])
    PD_out = np.zeros(f_data.shape[0])
    T      = 252*1 # Applied debt maturity (1 year)
    # Set initial guess for firm value equal to the equity value (Ve)
    Va_out = f_data['mkt_eqy_int'].copy()

    # Iterate through the time series
    # Initialize sigma_a
    sigma_a = np.nan
    for i_tt, tt in enumerate(timesteps):
        # Check if the company is levered in tt
        if f_data.iloc[tt-1]['DD_debt_int'] > 1e-10:
            # Based on incoming Va data, compute initial sigma_a
            Va_per = Va_out.iloc[(tt-win):tt].copy()
            Va_ret = np.log(Va_per / Va_per.shift(1))
            # Compute new sigma_a only if starting with data for new firm. Otherwise, because only the last entry in the Va series is changing, the previous sigma_a is a very good guess.
            if pd.isnull(sigma_a):
                sigma_a = np.std(Va_ret)
            else:
                pass

            # If first observation, update all past firm values as well
            if i_tt == 0:
                subset_timesteps = range(tt-min_hist_vals, tt)
            else:
                subset_timesteps = [tt-1]

            # Iterate on values of Va
            nn = 0
            while nn < 10:
                nn = nn + 1
                # Loop over timesteps, calculating Va using the current guess for sigma_a
                for t_sub in subset_timesteps:
                    # print(t_sub)
                    r_f     = (1 + f_data.iloc[t_sub]['rr_f'])**(1.0/365) - 1
                    Ve      = f_data.iloc[t_sub]['mkt_eqy_int']
                    debt_val= f_data.iloc[t_sub]['DD_debt_int']
                    sol     = scoptim.root(lambda xx: BS_Va_dev(xx, sigma_a, debt_val, Ve, r_f, T), Va_out.iloc[t_sub])
                    Va_out.iloc[t_sub] = sol['x'][0]

                # Update sigma_a based on most recent Va values
                sigma_a_old = sigma_a
                Va_per = Va_out.iloc[(tt-win):tt].copy()
                Va_ret = np.log(Va_per / Va_per.shift(1))
                sigma_a = np.std(Va_ret)

                # Given most recent Va and sigma_a values, also compute and impose mu_a
                mu_a_cur = np.mean(Va_ret) + 0.5*sigma_a**2
                mu_out[subset_timesteps] = mu_a_cur
                DD_out[subset_timesteps] = (np.log(Va_out.iloc[subset_timesteps]/f_data.iloc[subset_timesteps]['DD_debt_int']) + (mu_out[subset_timesteps] - 0.5*sigma_a**2)*T ) / (sigma_a * np.sqrt(T))

                if abs(sigma_a_old - sigma_a) < 1e-4:
                    break

        else:
            pass # Leave Va_out equal to Ve

    # Having computed DD and everything else, compute probability of default
    PD_out  = norm.cdf(-DD_out)

    # Append to main dataframe
    crsp_data.loc[crsp_data.gvkey_b == gv_name, 'Va'] = Va_out
    crsp_data.loc[crsp_data.gvkey_b == gv_name, 'DD'] = DD_out
    crsp_data.loc[crsp_data.gvkey_b == gv_name, 'PD'] = PD_out

# Finally, save the results, first dropping unused columns
crsp_data   = crsp_data.drop(['dateval', 'datedif', 'r_f', 'rr_f'],axis=1)
crsp_data.to_csv(CRSP_target_folder / f"CRSP_data_DD_int_done_{slurmcounter}.csv", index=False)