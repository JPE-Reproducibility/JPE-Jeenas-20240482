## Filepaths Analysis Details

**/Users/florianoswald/actions-runner/_work/JPE-Jeenas-20240482/JPE-Jeenas-20240482/replication-package/JPE_REPLICATION_FINAL/code/Python/cleaning/CS_cleaning_code.py**

- Line 15, unix : data_raw = pd.read_csv(PROJECT_ROOT / 'proprietary-data-not-for-publication/raw/CS_data_raw.csv')
- Line 38, unix : target_folder = PROJECT_ROOT / 'proprietary-data-not-for-publication/processed'
- Line 41, unix : data_raw.to_csv(PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/CS_data_clean.csv')

**/Users/florianoswald/actions-runner/_work/JPE-Jeenas-20240482/JPE-Jeenas-20240482/replication-package/JPE_REPLICATION_FINAL/code/Julia/Julia_model_code/calib_functions.jl**

- Line 63, unix : Ltot    = Ltot/sum(Ltot)
- Line 74, unix : gammatil_ss = modl.gamma/modl.Pi
- Line 248, unix : n_0_dN  = n_0/aeq.Na
- Line 383, unix : age_avg_pub     = 20.0 - 1.0 + 1.0/modl.eta;
- Line 426, unix : # Specify the indicator that picks the H/L groups
- Line 427, unix : # Split based on liquidity (m/a) ratio percentiles
- Line 437, unix : # Unpack the solution matrices for SS behavior into vector/matrix form, as necessary
- Line 570, unix : # For computing true b/k ratio at market debt values, compute the Rc_ss

**/Users/florianoswald/actions-runner/_work/JPE-Jeenas-20240482/JPE-Jeenas-20240482/replication-package/JPE_REPLICATION_FINAL/code/R/Micro_model_reg_code.r**

- Line 31, unix : modl_Pi     = 1.02^(1.0/4)
- Line 32, unix : modl_gamma  = 1-1/16
- Line 33, unix : gammatil_ss = modl_gamma/modl_Pi
- Line 80, unix : p_data['dltisqat_rat'] = p_data$dltisq/rlag(p_data$atq,1)
- Line 81, unix : p_data['ndltisqat_rat'] = p_data$ndltisq/rlag(p_data$atq,1)
- Line 82, unix : p_data['kinvat_rat'] = p_data$kinv/rlag(p_data$atq,1)
- Line 83, unix : p_data['invat_rat'] = p_data$inv/rlag(p_data$atq,1)
- Line 84, unix : p_data['dcheat_rat'] = p_data$dche/rlag(p_data$atq,1)
- Line 85, unix : p_data['dvqat_rat'] = p_data$dvq/rlag(p_data$atq,1)

**/Users/florianoswald/actions-runner/_work/JPE-Jeenas-20240482/JPE-Jeenas-20240482/replication-package/JPE_REPLICATION_FINAL/code/Python/cleaning/CRSP_cleaning_code.py**

- Line 14, unix : file_loc = PROJECT_ROOT / 'proprietary-data-not-for-publication/raw/CRSP_data_raw.csv'
- Line 18, unix : N_iters  = int(np.ceil(approx_tot_lines/nlines_i))
- Line 22, unix : target_folder = PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/CRSP_split'

**/Users/florianoswald/actions-runner/_work/JPE-Jeenas-20240482/JPE-Jeenas-20240482/replication-package/JPE_REPLICATION_FINAL/code/Python/Micro_reg_prep_code.py**

- Line 20, unix : data = pd.read_csv(PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/CS_data_clean.csv')
- Line 47, unix : ffr_raw = pd.read_excel(PROJECT_ROOT / 'data/raw/FEDFUNDS_Q.xls', sheet_name='FRED Graph', header=10, index_col=0, usecols = 'A,B')
- Line 60, unix : GDP_table = pd.read_excel(PROJECT_ROOT / 'data/raw/GDPr_Q_BEA.xls', sheet_name='Sheet0', header=5, usecols = 'A:IV', skiprows=0, skipfooter=0)
- Line 66, unix : GVADef_table1 = pd.read_excel(PROJECT_ROOT / 'data/raw/GVADEF_Q_BEA.xls', sheet_name='Sheet0', header=5, usecols = 'A:IV', skiprows=0, skipfooter=0)
- Line 68, unix : GVADef_table2 = pd.read_excel(PROJECT_ROOT / 'data/raw/GVADEF_Q_BEA.xls', sheet_name='Sheet1', header=5, usecols = 'A:AD', skiprows=0, skipfooter=0)
- Line 74, unix : sffr_raw = pd.read_csv(PROJECT_ROOT / 'data/raw/JK_data_fig4.csv')
- Line 120, unix : data['lev_rat'] = data.dttq/data.atq
- Line 188, unix : ccm_data = pd.read_csv(PROJECT_ROOT / 'proprietary-data-not-for-publication/raw/CCM_data_raw.csv')
- Line 205, unix : wsc_data = pd.read_csv(PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/Wscope_data_compiled.csv')
- Line 219, unix : CRSP_target_folder = PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/CRSP_split'
- Line 250, unix : data.to_csv(PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/Micro_reg_data_compiled.csv')

**/Users/florianoswald/actions-runner/_work/JPE-Jeenas-20240482/JPE-Jeenas-20240482/replication-package/JPE_REPLICATION_FINAL/code/Python/cleaning/CRSP_par_prep.py**

- Line 18, unix : N_iters  = int(np.ceil(approx_tot_lines/nlines_i))
- Line 23, unix : CRSP_target_folder = PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/CRSP_split'
- Line 43, unix : nper    = int(np.ceil(Nf/Nout))
- Line 44, unix : Nout_true = int(np.ceil(Nf/nper))

**/Users/florianoswald/actions-runner/_work/JPE-Jeenas-20240482/JPE-Jeenas-20240482/replication-package/JPE_REPLICATION_FINAL/code/Julia/Julia_model_code/Master_model_work.jl**

- Line 19, unix : ### Calibration/parameter setting block
- Line 21, unix : beta_set    = (1.0/(1.0+0.02))^(1.0/4.0)
- Line 22, unix : Pi_set      = 1.02^(1.0/4.0)         # Gross inflation
- Line 23, unix : rf_imp      = Pi_set/beta_set - 1.0 # Risk-free nominal rate
- Line 33, unix : gamma_set   = 1-1/16    # Long-term debt decay parameter
- Line 40, unix : # Capital price elasticity to I/K
- Line 227, unix : gammatil_ss = modl.gamma/modl.Pi # Steady state adjusted gamma parameter
- Line 229, unix : # Note that the following figures are in the (m/k, b/k) space
- Line 245, unix : # Leave out the small number of extremely large (high k) firms that would, for the very highest values of m/k that this figure plots, require extrapolation to m positions never observed in the population nor covered by the rectangular model state space
- Line 282, unix : Ba_mk00_bk00 = Ba_sq[1,1] # m/k=0.0, b/k=0.0
- Line 283, unix : Ba_mk00_bk02 = Ba_sq[1,6] # m/k=0.0, b/k=0.2
- Line 284, unix : Ba_mk02_bk02 = Ba_sq[6,6] # m/k=0.2, b/k=0.2
- Line 344, unix : Table_Untarg_row6 = mean(mda_paths[:,1].==0)    # m/a ratios at zero
- Line 347, unix : Table_Untarg_row1 = std(mda_paths[:,1]) # sd(m/a)
- Line 348, unix : # Correlation of m/a and log size
- Line 356, unix : # Correlation of m/a and i/k
- Line 379, unix : # Create group indicators of H/L leverage and liquidity, based on initial state
- Line 491, unix : # And compute the public firms' M/A ratio (to report in Appendix B.11.1 text)
- Line 586, unix : # Given the baseline solution, draw random set of firms, simulate panel in SS and IRF and compute/plot regression coef on liq ratio -- Figure 6, also, Figure B.2, Figure B.3a, Figure B.3b, Figure B.4
- Line 615, unix : # Given the baseline solution, draw random set of firms, simulate panel in SS and IRF and compute/plot regression coef on liq ratio -- Figure B.11a
- Line 637, unix : modl_hr_cf.Pi       = 1.0342^(1.0/4.0) # (Based on 3mTB difference: 1985-89/2: 3.42%)
- Line 641, unix : modl_hr_cf.rf = modl_hr_cf.Pi/modl_hr_cf.beta - 1.0
- Line 717, unix : airf_g_hr_cf, irfsoln_g_hr_cf = solve_eqIRF_NKMg_xsh(0.000, zeta_col, spread_col, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts, MU_col_0=MU_in, pQ_col_0=pQ_in, w_ss=modl_hr_cf.psi/aeq_hr_cf.p);
- Line 756, unix : airf_g_hr_cf_basephimresp, irfsoln_g_hr_cf_basephimresp = solve_eqIRF_NKMg_xsh(0.000, zeta_col, spread_col, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts, MU_col_0=MU_in, pQ_col_0=pQ_in, w_ss=modl_hr_cf.psi/aeq_hr_cf.p);
- Line 776, unix : airf_g_hr_cf_basephimresp_noGE, irfsoln_g_hr_cf_basephimresp_noGE = solve_eqIRF_NKMg_xsh(0.000, zeta_col, spread_col, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts, MU_col_0=airf_g_hr_cf.MU_col, pQ_col_0=airf_g_hr_cf.pQ_col, w_ss=modl_hr_cf.psi/aeq_hr_cf.p)
- Line 806, unix : # Analysis of group-specific responses in base vs hr_cf economies, conditional on 25th percentile cutoff in the m/a-space
- Line 1305, unix : global airf_g_hr_cf_rmsh, irfsoln_g_hr_cf_rmsh = solve_eqIRF_NKMg_xsh_rmsh(0.000, zeta_col, spread_col, rmsh_col, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts, MU_col_0=MU_in, pQ_col_0=pQ_in, w_ss=modl_hr_cf.psi/aeq_hr_cf.p);
- Line 1339, unix : # Lines to replicate core analysis with phi_k=1/8, adjusting/overwriting model objects from above (Appendix B.11.2)
- Line 1379, unix : modl.phi_k = 1/8
- Line 1380, unix : modl_hr_cf.phi_k = 1/8
- Line 1411, unix : global airf_g_hr_cf, irfsoln_g_hr_cf = solve_eqIRF_NKMg_xsh(0.000, zeta_col, spread_col, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts, MU_col_0=MU_in, pQ_col_0=pQ_in, w_ss=modl_hr_cf.psi/aeq_hr_cf.p);
- Line 1417, unix : global airf_g_hr_cf_basephimresp, irfsoln_g_hr_cf_basephimresp = solve_eqIRF_NKMg_xsh(0.000, zeta_col, spread_col, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts, MU_col_0=MU_in, pQ_col_0=pQ_in, w_ss=modl_hr_cf.psi/aeq_hr_cf.p);
- Line 1425, unix : global airf_g_hr_cf_basephimresp_noGE, irfsoln_g_hr_cf_basephimresp_noGE = solve_eqIRF_NKMg_xsh(0.000, zeta_col, spread_col, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts, MU_col_0=MU_in, pQ_col_0=pQ_in, w_ss=modl_hr_cf.psi/aeq_hr_cf.p)
- Line 1460, unix : # Given the baseline solution, draw random set of firms, simulate panel in SS and IRF and compute/plot regression coef on liq ratio -- Figure B.13a
- Line 1476, unix : # Run decomposition of channels for the phi_k=1/8 case

**/Users/florianoswald/actions-runner/_work/JPE-Jeenas-20240482/JPE-Jeenas-20240482/replication-package/JPE_REPLICATION_FINAL/code/Python/FOFA_analysis_code.py**

- Line 14, unix : data = pd.read_csv(PROJECT_ROOT / 'data/raw/FRB_Z1.csv')
- Line 80, windows : "\t\\centering\n"
- Line 86, windows : "\t\t\\hline\n"

**/Users/florianoswald/actions-runner/_work/JPE-Jeenas-20240482/JPE-Jeenas-20240482/replication-package/JPE_REPLICATION_FINAL/code/R/Micro_reg_code.r**

- Line 210, unix : # Ensure output/tables folder exists
- Line 294, unix : p_data['kinvat_rat'] = p_data$kinv/rlag(p_data$atq_real,1)
- Line 530, unix : p_data$dvqat_rat = p_data$dvq/rlag(p_data$atq,1)
- Line 1062, unix : # Split HIGH/LOW cash and leverage in sample, based on medians
- Line 1078, unix : # Also, define large/small firms within SIC3-industry

**/Users/florianoswald/actions-runner/_work/JPE-Jeenas-20240482/JPE-Jeenas-20240482/replication-package/JPE_REPLICATION_FINAL/code/Python/TB3_analysis_code.py**

- Line 14, unix : data = pd.read_csv(PROJECT_ROOT / 'data/raw/TB3MS.csv')

**/Users/florianoswald/actions-runner/_work/JPE-Jeenas-20240482/JPE-Jeenas-20240482/replication-package/JPE_REPLICATION_FINAL/code/Python/DD_work_run.py**

- Line 19, unix : CRSP_target_folder = PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/CRSP_split'
- Line 33, unix : # Keep only firm segments with ≥252 daily observations to ensure a full one-year window of continuous data for KMV/Merton distance-to-default estimation.
- Line 53, unix : GS1_raw         = pd.read_csv(PROJECT_ROOT / 'data/raw/DGS1.csv')
- Line 67, unix : GDPDef_raw = pd.read_csv(PROJECT_ROOT / 'data/raw/GDPDEF_Q.csv')

**/Users/florianoswald/actions-runner/_work/JPE-Jeenas-20240482/JPE-Jeenas-20240482/replication-package/JPE_REPLICATION_FINAL/code/Python/cleaning/CStoCRSP_merge.py**

- Line 20, unix : N_iters  = int(np.ceil(approx_tot_lines/nlines_i))
- Line 25, unix : ccm_data = pd.read_csv(PROJECT_ROOT / 'proprietary-data-not-for-publication/raw/CCM_data_raw.csv')
- Line 37, unix : cstat_data  = pd.read_csv(PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/CS_data_clean.csv')
- Line 59, unix : CRSP_target_folder = PROJECT_ROOT / 'proprietary-data-not-for-publication/processed/CRSP_split'

**/Users/florianoswald/actions-runner/_work/JPE-Jeenas-20240482/JPE-Jeenas-20240482/replication-package/JPE_REPLICATION_FINAL/code/Julia/Julia_model_code/auxiliary_model_functions.jl**

- Line 122, unix : k_grid0::Array{Float64,1}                # To save initial grid for computing basis matrices b/c cubic spline extends grid
- Line 124, unix : m_grid0::Array{Float64,1}                # To save initial grid for computing basis matrices b/c cubic spline extends grid
- Line 225, unix : # Firms' responses/simulations container
- Line 410, unix : k_grid      = range(sspace.k_min^curv[1], sspace.k_max^curv[1], length=Nk).^(1.0/curv[1])
- Line 412, unix : m_grid      = range((sspace.m_min-sspace.m_min)^curv[2], (sspace.m_max-sspace.m_min)^curv[2], length=Nm).^(1.0/curv[2]) .+ sspace.m_min
- Line 414, unix : b_grid      = range((sspace.b_min-sspace.b_min)^curv[3], (sspace.b_max-sspace.b_min)^curv[3], length=Nb).^(1.0/curv[3]) .+ sspace.b_min # Because b-values are negative, shift them all to a positive range)
- Line 417, unix : sspace.a_grid      = range((sspace.a_min-sspace.a_min)^curvaux[1], (sspace.a_max-sspace.a_min)^curvaux[1], length=Na).^(1.0/curvaux[1]) .+ sspace.a_min
- Line 418, unix : sspace.atil_grid   = range((sspace.atil_min-sspace.atil_min)^curvaux[2], (sspace.atil_max-sspace.atil_min)^curvaux[2], length=Natil).^(1.0/curvaux[2]) .+ sspace.atil_min
- Line 419, unix : sspace.btil_grid   = range((sspace.btil_min-sspace.btil_min)^curvaux[3], (sspace.btil_max-sspace.btil_min)^curvaux[3], length=Nbtil).^(1.0/curvaux[3]) .+ sspace.btil_min
- Line 420, unix : sspace.mtil_grid   = range((sspace.mtil_min-sspace.mtil_min)^curvaux[4], (sspace.mtil_max-sspace.mtil_min)^curvaux[4], length=Nmtil).^(1.0/curvaux[4]) .+ sspace.mtil_min
- Line 422, unix : sspace.mu_grid     = range(sspace.mu_max^curvlm[1], sspace.mu_min^curvlm[1], length=Nmu).^(1.0/curvlm[1]) # FLIPPED ORDER
- Line 423, unix : sspace.muk_grid    = range(sspace.muk_max^curvlm[2], sspace.muk_min^curvlm[2], length=Nmuk).^(1.0/curvlm[2]) # FLIPPED ORDER
- Line 424, unix : sspace.chiS_grid   = range(sspace.chiS_min^curvlm[3], sspace.chiS_max^curvlm[3], length=NchiS).^(1.0/curvlm[3]) # NOT FLIPPED ORDER -- it's a SAVING constraint
- Line 425, unix : sspace.chiB_grid   = range(sspace.chiB_max^curvlm[4], sspace.chiB_min^curvlm[4], length=NchiB).^(1.0/curvlm[4]) # NOT FLIPPED ORDER
- Line 446, unix : k_gridf         = range(sspace.k_minf^curvf[1], sspace.k_maxf^curvf[1], length=Nkf).^(1.0/curvf[1])
- Line 447, unix : m_gridf         = range((sspace.m_minf-sspace.m_minf)^curvf[2], (sspace.m_maxf-sspace.m_minf)^curvf[2], length=Nmf).^(1.0/curvf[2]) .+ sspace.m_minf
- Line 448, unix : b_gridf         = range((sspace.b_minf-sspace.b_minf)^curvf[3], (sspace.b_maxf-sspace.b_minf)^curvf[3], length=Nbf).^(1.0/curvf[3]) .+ sspace.b_minf
- Line 509, unix : gammatil_ss = modl.gamma/modl.Pi
- Line 529, unix : # And policies for returning final output/labor etc results
- Line 923, unix : # The object to interpolate is V0ebdk_c_itp=V0eb/k'
- Line 1481, unix : # And map these into V1 and its derivatives based on the envelope conditions. NOTE: lambda1N=V2Nmtil/k
- Line 1725, unix : chiS_grid_used   = range(sspace.chiS_min^sspace.curvlm[3], chiS_max_prime^sspace.curvlm[3], length=sspace.NchiS).^(1.0/sspace.curvlm[3])
- Line 1883, unix : # Create interpolants for the continuation value function derivatives V3Am, V3Abtil/k, and V3A, conditional on k (and z); and for completeness, also V3Ak
- Line 1891, unix : # Compute the value of the V3Am/V3Abtil ratio at m'=m_min
- Line 2062, unix : # And map these into V1 and its derivatives based on the envelope conditions; lambda1A = V2Aatil/k
- Line 2543, unix : gammatil_ss = modl.gamma/modl.Pi
- Line 2547, unix : b_iss_at_rat_paths_SS = b_iss_paths_SS ./ (sim_paths_SS.k_path[:,1:(end-1)] + sim_paths_SS.m_path[:,1:(end-1)]) # Real issuance/assets ratio
- Line 2551, unix : b_iss_at_rat_paths_IRF = b_iss_paths_IRF ./ (sim_paths_IRF.k_path[:,1:(end-1)] + sim_paths_IRF.m_path[:,1:(end-1)]) # Real issuance/assets ratio
- Line 2673, unix : gammatil_ss = modl.gamma/modl.Pi
- Line 2853, unix : gammatil_ss = modl.gamma/modl.Pi
- Line 3014, unix : gammatil_ss = modl.gamma/modl.Pi
- Line 3253, unix : # Compute implied I/K aggregate adjustment costs:
- Line 3259, unix : pQ_col_upd[tt] = (IdK_cost_cur/modl.delta)^modl.phi_k
- Line 3316, unix : gammatil_ss = modl.gamma/modl.Pi
- Line 3557, unix : # Compute implied I/K aggregate adjustment costs:
- Line 3563, unix : pQ_col_upd[tt] = (IdK_cost_cur/modl.delta)^modl.phi_k
- Line 3625, unix : gammatil_ss = modl.gamma/modl.Pi
- Line 3876, unix : # Compute implied I/K aggregate adjustment costs:

**/Users/florianoswald/actions-runner/_work/JPE-Jeenas-20240482/JPE-Jeenas-20240482/replication-package/JPE_REPLICATION_FINAL/code/Python/cleaning/Wscope_compile_code.py**

- Line 14, unix : Wscope_data_filename = PROJECT_ROOT / 'proprietary-data-not-for-publication/raw/Wscope_data_raw.xlsx'

