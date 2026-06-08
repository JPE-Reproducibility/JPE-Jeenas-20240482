# Julia code used for solving and analyzing the structural model in the paper "Firm Balance Sheet Liquidity, Monetary Policy Shocks, and Investment Dynamics" by Priit Jeenas

# Define the root directory, robust to interactive or file execution
PROJECT_ROOT = try
    normpath(joinpath(@__DIR__, "..", "..", ".."))
catch
    pwd()
end
cd(PROJECT_ROOT)
# Load auxiliary functions
include(joinpath(PROJECT_ROOT, "code", "Julia", "Julia_model_code", "auxiliary_model_functions.jl"))
using .auxiliary_model_functions

# Load relevant packages
using LaTeXStrings, Interpolations, Roots, QuantEcon, CompEcon, SparseArrays, LinearAlgebra, JLD, CSV, Distributions, DataFrames, GLM, Random, Printf, Binscatters
using Plots, AverageShiftedHistograms
Plots.gr()

### Calibration/parameter setting block
# Set parameter values
beta_set    = (1.0/(1.0+0.02))^(1.0/4.0)
Pi_set      = 1.02^(1.0/4.0)         # Gross inflation
rf_imp      = Pi_set/beta_set - 1.0 # Risk-free nominal rate
phi_m_set        = 1.0-0.255    # SS "exposure" of nominal rm to nominal rf
phi_mresp_set    = 0.618        # IRF "exposure" of nominal rm to nominal rf
rm_imp      = phi_m_set*rf_imp
# Set any spread on r^b-r^f=0
bspread_set  = 0.0                  # Spread of rb over rf
rb_imp      = rf_imp + bspread_set  # Rate on borrowing

delta_set   = 0.025     # Capital depreciation
eta_set     = 0.022     # Quarterly firm death probability
gamma_set   = 1-1/16    # Long-term debt decay parameter
varpi_set   = 1.0       # Household's CRRA parameter
psi_set     = 1.2       # Place-holder for household's labor disutility (overwritten in w_ss=1 normalization)
nu_set      = 0.64      # Labor exponent in production function
alpha_set   = 0.23      # Capital exponent in production function
xi_lb_set   = 0.0       # Lower bound on uniform debt issuance cost xi
rhoz_set    = 0.9       # TFP AR(1) persistence
# Capital price elasticity to I/K
phi_k_set   = 0.25
# Taylor rule responsiveness
phi_pi_set  = 1.25
# Phillips curve slope
kappa_p_set = 0.1
# Normalization of SS wage
w_0         = 1.0
# Corporate tax rate
tau_set         = 0.20 # following Whited and Nikolov (2014)

###
# Internally calibrated parameter values
###
theta_set, xi_ub_set, kappa_set, k_0_set, sige_set, z_0_shift_set, phi_w_set = 0.422,    0.0615,   0.1575,   1.84,     0.245,    0.24,   0.3776    
# Note, importantly, that "phi_w" in the model code equals 1-ϕ_w from the paper's notation.

# Introduce based on operational cash needs, before also setting some exogenously parameterized steady state prices
MU_ss, w_ss = 10.0/(10.0-1.0), w_0      # Gross markup, real wage -- NOTE: These "parameters" are hard-coded in a variety of subprocedures, e.g., in benchmark steady state solutions, unless explicitly overwritten
mdk_0_temp  = max( 0.0, (MU_ss^(-1.0/(1.0-nu_set))) * ((nu_set/w_ss)^(nu_set/(1.0-nu_set))) * (k_0_set^(alpha_set/(1.0-nu_set) - 1.0)) * (phi_w_set - (1.0-nu_set)) )
m_0_set         = k_0_set*mdk_0_temp

b_0_set         = 0.00
zh_set          = 4.00      # Level of additional "very z draw" (not used)
pzh_set         = 0.0       # Probability of additional "very z draw" (not used)
# xi distribution
xi_dist_set     = "uniform"

# Number of points in z Markov chain
Nz_set = 5
# For grid sizes, compute the z_grid here
P, z_grid, Pssz = setup_MarkovZ(Nz_set, sige_set, rhoz_set,1)
### End calibration block

# SETTING GRIDS
# Grid limits
if true
    k_min_set, k_max_set  = 0.4, 180.0          # Bounds on firms' k-space
    m_min_set, m_max_set  = 0.0, 0.15*k_max_set # Bounds on firms' m-space
    b_min_set, b_max_set = -theta_set, 0.0      # Bounds on firms' space b which in the code refers to the "theory object" Rc(t-1)*(-b)/k, where Rc(t-1) = (1+rb(t)), with rb(t) and Pi(t) as expected of the corresponding variables by agents in the economy at period t-1 along any perfect foresight equilibrium path (or steady state)

    # End-points of distribution histogram in Young (2010) style approach
    k_minf_set, k_maxf_set = k_min_set, k_max_set
    m_minf_set, m_maxf_set = m_min_set, m_max_set
    b_minf_set, b_maxf_set = b_min_set, b_max_set
end
# Widened auxiliary grid limits for IRFs
if true
    # Widen grids based on interval for prices outside of steady state
    rm_min_aux, rm_max_aux = rm_imp-0.0025, rm_imp+0.0025
    rb_min_aux, rb_max_aux = rb_imp-0.0025, rb_imp+0.0025
    pQ_min_aux, pQ_max_aux = 0.95, 1.05
    Pi_min_aux, Pi_max_aux = Pi_set, Pi_set
    Rc_min_aux, Rc_max_aux = (1.0+rb_min_aux)/(pQ_max_aux*Pi_max_aux), (1.0+rb_max_aux)/(pQ_min_aux*Pi_min_aux)
    w_min_aux, w_max_aux   = 1.0, 1.0
    A_val_max_aux = (1.0-tau_set)*(1.0-nu_set)*((nu_set/w_min_aux)^(nu_set/(1.0-nu_set)))

    # a -- refers to firm's available ("liquid") financial resources (per capital) at the beginning of "adjuster (A)" problem
    a_min_set, a_max_set       = ((1.0+(1.0-tau_set)*rm_min_aux)/Pi_max_aux)*m_min_set/k_max_set + ((1.0+(1.0-tau_set)*rb_max_aux)/(Rc_min_aux*Pi_min_aux))*b_min_set - 1e-8, ((1.0+(1.0-tau_set)*rm_max_aux)/Pi_min_aux)*m_max_set/k_min_set + ((1.0+(1.0-tau_set)*rb_min_aux)/(Rc_max_aux*Pi_max_aux))*b_max_set + A_val_max_aux*z_grid[end]*k_min_set^(alpha_set/(1.0-nu_set)-1) + 1e-4 

    # atil -- refers to A's available financial resources (per capital) after paying dividends
    atil_min_set, atil_max_set = a_min_set - 0.01, a_max_set + 1e-4 
    # btil -- refers to A's available financial resources (per capital) after choosing m'
    btil_min_set, btil_max_set = atil_min_set - 0.01, atil_max_set + 1e-4

    # mtil -- refers to N's available financial resources (per capital) after paying dividends, shifted accounting for capital adjustment costs and financial constraints
    # Compute minimal mtil that supports an empty choice set for N in steady state
    mtil_umin = -(1-(1-tau_set)*delta_set) 
    mtil_min_set, mtil_max_set = mtil_umin,  ((1.0+(1.0-tau_set)*rm_max_aux)/Pi_min_aux)*m_max_set/k_min_set + ((1.0+(1.0-tau_set)*rb_min_aux)/Pi_max_aux - gamma_set/Pi_min_aux)*b_max_set - m_min_set/k_max_set + pQ_min_aux*(1.0/theta_set)*(gamma_set/Pi_max_aux)*b_max_set + A_val_max_aux*z_grid[end]*k_min_set^(alpha_set/(1.0-nu_set)-1) + 1e-4
end
# Lagrange multiplier grid limits
if true
    mu_min_set, mu_max_set      = 0.0, 0.6  # mu -- Lagrange multiplier on (m' ≥ 0) for N
    muk_min_set, muk_max_set    = 0.0, 0.8  # muk -- Lagrange multiplier on (b' ≥ -θ) for N
    chiS_min_set, chiS_max_set  = 0.0, 0.9  # chiS -- Lagrange multiplier on (b' ≤ 0) for A
    chiB_min_set, chiB_max_set  = 0.0, 20.0 # chiB -- Lagrange multiplier on (b' ≥ -θ) for A
end

# Numbers of grid points
n_set  = [35, 35, 40, Nz_set] # (k,m,b,z)
nf_set = [35, 35, 40, Nz_set]
naux_set    = [70, 70, 70, 100] # (a,atil,btil,mtil)
nlm_set     = [30, 30, 30, 30] # (mu, muk,chiS,chiB)

# Concentrate more gridpoints at extreme values using curv-parameter
if true
    curv_set    = [0.35,0.5,0.9]     # (k,m,b,z)
    curvf_set   = curv_set          # (k,m,b,z)
    curvaux_set = [0.3,0.3,0.3,0.5] # (a,atil,btil,mtil)
    curvlm_set  = [1.0,1.0,1.0,0.5]     # (mu,muk,chiS,chiB)
end

# Entrants' distribution
z_0_dist_set = lognorm_discrshift(-z_0_shift_set-0.5*(sige_set^2)/(1.0-rhoz_set^2), sige_set/sqrt(1.0-rhoz_set^2), log.(z_grid))

# Give the model a name
modlname = "mod1"

# Create folder for model interim output
mkpath(joinpath(PROJECT_ROOT, "interim_output", "model_interim_output"))

# Create empty text file to save some model output
mkpath(joinpath(PROJECT_ROOT, "interim_output", "model_interim_output", "savedout"))
txtout_path = joinpath(PROJECT_ROOT, "interim_output", "model_interim_output", "savedout", modlname * "_output.txt")
ofile = open(txtout_path, "w")
close(ofile)

# Set all solution params
opts = Options(Nbell=600, tolV=1e-5, phi_V=[0.3,0.4,0.4,0.3], itermaxL=500, tolL=1e-8, tolp=1e-5, phi_peq=0.9, tol_p_col=2*1e-5, phi_pQ_acl=1.0, phi_p=0.95, project_root=PROJECT_ROOT);
sspace = StateSpace(spliorder=[1,1,1,1], n=n_set, nf=nf_set, naux=naux_set, nlm=nlm_set, k_min=k_min_set, k_max=k_max_set, m_min=m_min_set, m_max=m_max_set, b_min=b_min_set, b_max=b_max_set, k_minf=k_minf_set, k_maxf=k_maxf_set, m_minf=m_minf_set, m_maxf=m_maxf_set, b_minf=b_minf_set, b_maxf=b_maxf_set, curv=curv_set, curvf=curvf_set, a_min=a_min_set, a_max=a_max_set, atil_min=atil_min_set, atil_max=atil_max_set, btil_min=btil_min_set, btil_max=btil_max_set, mtil_min=mtil_min_set, mtil_max=mtil_max_set, mu_min=mu_min_set, mu_max=mu_max_set, muk_min=muk_min_set, muk_max=muk_max_set, chiS_min=chiS_min_set, chiS_max=chiS_max_set, chiB_min=chiB_min_set, chiB_max=chiB_max_set, curvaux=curvaux_set, curvlm=curvlm_set);
modl = Model(beta=beta_set, psi=psi_set, varpi=varpi_set, rhoz=rhoz_set, sige=sige_set, A=1.0, alpha=alpha_set, nu=nu_set, delta=delta_set, bspread=bspread_set, Pi=Pi_set, theta = theta_set, gamma=gamma_set, kappa=kappa_set, eta=eta_set, xi_lb=xi_lb_set, xi_ub=xi_ub_set, xi_dist=xi_dist_set, tau=tau_set, phi_w=phi_w_set, k_0=k_0_set, m_0=m_0_set, b_0=b_0_set, z_0_dist=z_0_dist_set, phi_k=phi_k_set, phi_pi=phi_pi_set, kappa_p=kappa_p_set, phi_m=phi_m_set, phi_mresp=phi_mresp_set, zh=zh_set, pzh=pzh_set, name=modlname);

manprintln("Setup", txtout_path)
modl, sspace = setup!(modl, sspace, opts);
manprintln("Setup complete", txtout_path)

# NOTE: Make sure that the last b gridpoint is truly zero
sspace.b_grid[end] = 0.0
sspace.b_gridf[end] = 0.0
sspace.sf = gridmake(sspace.k_gridf,sspace.m_gridf,sspace.b_gridf,sspace.z_gridf);

# Make folder for saved solution checkpoints
savedsoln_dir = joinpath(PROJECT_ROOT, "interim_output", "model_interim_output", "savedsolns")
mkpath(savedsoln_dir)
# If the baseline FirmSolution checkpoint does not yet exist, solve and save it
if !isfile(joinpath(savedsoln_dir, "fsoln_base.jld"))
    fsoln = solve_V_EGM(modl,sspace,opts);
    save(joinpath(savedsoln_dir, "fsoln_base.jld"),  "fsoln", fsoln)
end
# Reload the saved checkpoint and verify convergence, so later stages always continue from the stored baseline solution
fsoln_load = load(joinpath(savedsoln_dir, "fsoln_base.jld"), "fsoln");
fsoln = solve_V_EGM(modl,sspace,opts, fsoln_in=fsoln_load);

# Solve steady state aggregate equilibrium
aeq = solve_Leq(fsoln, modl, sspace, opts);
# Generate "public" firm distribution
Lin_pub = gen_L_Tage(20, aeq.Lin, aeq.Qend, modl, sspace, opts);

# Define font style early
manl_ftsize = 13
plot_font = "serif-roman"
default(fontfamily=plot_font)

# Back up base solutions
aeq_base, fsoln_base = aeq, fsoln;

# Prepare folders for collecting model output, in case do not yet exist
mkpath(joinpath(PROJECT_ROOT, "output", "figures", "model_figures"))
mkpath(joinpath(PROJECT_ROOT, "output", "other"))
mkpath(joinpath(PROJECT_ROOT, "output", "tables"))

###
# Run calibration procedures, to report values of targets of interest
###
include(joinpath(PROJECT_ROOT, "code", "Julia", "Julia_model_code", "calib_functions.jl"))

# Set of moments from full population
targ_calib      = calib_nosimcalc(fsoln, aeq.Lin, modl, sspace, opts);
# Set of moments from "public" firm subsample
targ_calib_pub  = calib_nosimcalc(fsoln, Lin_pub, modl, sspace, opts);
# Characterize age groups in population
age_g_vals = calib_age_nosimcalc(fsoln, aeq, modl, sspace, opts);

# Print output (Table B.1)
open(joinpath(PROJECT_ROOT, "output", "tables", "_partof_TabB1_model_calibmoments.txt"), "w") do io
    println(io, "### Model calibration moments ###")
    println(io, rpad("Aggregate Debt/Assets:", 28), round(-targ_calib_pub[2], digits=3))
    println(io, rpad("Aggregate Cash/Assets:", 28), round(targ_calib_pub[1], digits=3))
    println(io, rpad("acor(i_a/k):", 28), round(targ_calib_pub[8], digits=2))
    println(io, rpad("sig(i_a/k):", 28), @sprintf("%.3f", targ_calib_pub[7])) # Explicitly writing the third, zero decimal as in earlier versions of Draft
    println(io, rpad("freq(D=1):", 28), round(targ_calib_pub[3], digits=3))
    println(io, rpad("E[i_a/k]:", 28), round(targ_calib[6], digits=2))
    println(io, rpad("E[n_0]/E[n]:", 28), round(age_g_vals[1], digits=2))
end
# Print output (Footnote 46)
open(joinpath(PROJECT_ROOT, "output", "other", "Footnote46_justification.txt"), "w") do io
    println(io, "The calibrated model generates an average observed ratio of the issuance cost to funds raised of about $(round(100 * targ_calib_pub[4],digits=1))%.")
end


###
# Average firm heatmap over (m,b), to assess probability of issuance -- Figure 4a
###

# Steady state capital price and debt position adjustment term in firm state space
pQ_ss       = 1.0 
Rc_ss       = (1.0+modl.rb)/(modl.Pi*pQ_ss)
gammatil_ss = modl.gamma/modl.Pi # Steady state adjusted gamma parameter

# Note that the following figures are in the (m/k, b/k) space
n_hmap = 11
size_scaler = 0.9
# Fix grids
mk_cur_lims = [0.0, 0.4];
bneg_cur_lims = [0.0, 0.4];
mkgrid_hmap = range(mk_cur_lims[1], mk_cur_lims[2], length=n_hmap);
bneggrid_hmap = range(bneg_cur_lims[1], bneg_cur_lims[2], length=n_hmap);
mkgrid_hmapf, bneggrid_hmapf= kron(ones(n_hmap), mkgrid_hmap), kron(bneggrid_hmap, ones(n_hmap));

# Sample N firms
Nfirms_set = 50000;
# Draw initial distribution
Random.seed!(999)
# Distribution from given Lin
ii_samp = wsample(1:sspace.Nsf, aeq.Lin, Nfirms_set);
# Leave out the small number of extremely large (high k) firms that would, for the very highest values of m/k that this figure plots, require extrapolation to m positions never observed in the population nor covered by the rectangular model state space
kdim_ulimit = sspace.m_grid[end] / mk_cur_lims[2]
ii_samp = ii_samp[sspace.sf[ii_samp, 1] .<= kdim_ulimit ] 
Nfirms = length(ii_samp)
se_init = sspace.sf[ii_samp, 1:3]
zi_col_init= repeat(1:sspace.nf[4], inner=sspace.nf[1]*sspace.nf[2]*sspace.nf[3])[ii_samp];
# Set up collector of issuance probabilities
Ba_sq_col = zeros(n_hmap, n_hmap, Nfirms);
# Create interpolants of policies to compute issuance probabilities, ensuring that adjustment actually refers to issuance
Ba_prob_itp  = interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), fsoln.Ba_prob, Gridded(Linear()));
kpA_itp     = interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), fsoln.kpA, Gridded(Linear()));
bpA_itp     = interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), fsoln.bpA, Gridded(Linear()));
for iif in 1:Nfirms
    k_cur   = se_init[iif,1];
    zi_cur  = zi_col_init[iif];
    mgrid_hmapf_cur = mkgrid_hmapf*k_cur;
    s_hmap      = [k_cur*ones(n_hmap^2) mgrid_hmapf_cur -bneggrid_hmapf*Rc_ss sspace.z_grid[Int.(zi_cur*ones(n_hmap^2))]];
    Ba_prob_hmap = [Ba_prob_itp(s_hmap[ii,:]...) for ii=1:n_hmap^2];
    kpA_hmap     = [kpA_itp(s_hmap[ii,:]...)     for ii=1:n_hmap^2];
    bpA_hmap     = [bpA_itp(s_hmap[ii,:]...)     for ii=1:n_hmap^2];
    # Check if would actually issue when adjust
    b_tr_hmap    = s_hmap[:,1] .* s_hmap[:,3] ./ Rc_ss; # Current incoming b level
    dissA_hmap   = (kpA_hmap .* bpA_hmap ./ Rc_ss) .< (gammatil_ss .* b_tr_hmap .- 1e-12); # Indicator if adjuster's chosen b is more negative than b' implied by no adjustment
    dissprob_hmap = dissA_hmap .* Ba_prob_hmap;
    Ba_sq_cur        = reshape(dissprob_hmap,n_hmap,n_hmap);
    # Save this in the collector
    Ba_sq_col[:,:,iif]   = copy(Ba_sq_cur);
end
# Take average over the third dimension
Ba_sq = mean(Ba_sq_col, dims=3)[:,:,1]
# Adjust upper limit
clims_man = (0,0.5)
# Plot
Plots.heatmap(bneggrid_hmap, mkgrid_hmap, Ba_sq, xlabel="Debt/Capital", ylabel="Cash/Capital", clims=clims_man, size=(size_scaler*520,size_scaler*450), xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize, c = cgrad(:Blues), dpi=600, right_margin = 4Plots.mm)
Plots.plot!([0,bneg_cur_lims[2]], [0,bneg_cur_lims[2]*(1.0+(1.0-modl.tau)*modl.rb)/(1.0+(1.0-modl.tau)*modl.rm)], linecolor=:black, legend=:none, linestyle=:dash, linewidth=1.5)
savefig(joinpath(PROJECT_ROOT, "output", "figures", "model_figures", "Fig4a_"*modl.name*"_polHM_Ba_ind_avgfirm.pdf"))
# Report numbers in text (Section 4.2)
Ba_mk00_bk00 = Ba_sq[1,1] # m/k=0.0, b/k=0.0
Ba_mk00_bk02 = Ba_sq[1,6] # m/k=0.0, b/k=0.2
Ba_mk02_bk02 = Ba_sq[6,6] # m/k=0.2, b/k=0.2
Ba_mk01_bk00 = 0.5*(Ba_sq[3,1] + Ba_sq[4,1])    # m/k=0.1, b/k=0.0 (Take mean, because m/k=0.1 is exactly at boundary of cells)
Ba_mk01_bk02 = 0.5*(Ba_sq[3,6] + Ba_sq[4,6])    # m/k=0.1, b/k=0.2 (Take mean, because m/k=0.1 is exactly at boundary of cells)
# Print output
open(joinpath(PROJECT_ROOT, "output", "other", "Text_Sec4p2.txt"), "w") do io
    println(io, "On heatmap (Fig 4), at debt-to-capital ratio of 0.2, increasing cash-to-capital from 0 to 0.2 moves probability from $(round(Ba_mk00_bk02,digits=2)) to $(round(Ba_mk02_bk02,digits=2)).")
    println(io, "On heatmap (Fig 4), at cash-to-capital ratio of 0.1, increasing debt-to-capital from 0 to 0.2 moves probability from $(round(Ba_mk01_bk00,digits=2)) to $(round(Ba_mk01_bk02,digits=2)).")
    println(io, "On heatmap (Fig 4), going from cash- and debt-to-capital ratios both at 0.0, to both at 0.2 moves probability from $(round(Ba_mk00_bk00,digits=2)) to $(round(Ba_mk02_bk02,digits=2)).")
end


###
# Lines to locally analyze a representative sample drawn from the stationary distribution of firms
###
# Sample N firms
Nfirms=50000;
T=10;
# Public firms
Lin_arg = Lin_pub;
if true
    zi_col, xi_cdf_col =  Array{Int64,2}(undef, Nfirms,T), zeros(Nfirms, T);
    mcz         = MarkovChain(sspace.P);

    # Draw initial distribution
    Random.seed!(999)
    # Distribution from given Lin
    ii_samp = wsample(1:sspace.Nsf, Lin_arg, Nfirms);
    se_init = sspace.sf[ii_samp, 1:3];
    xi_cdf_col = rand(Nfirms,T);
    zi_col_init= repeat(1:sspace.nf[4], inner=sspace.nf[1]*sspace.nf[2]*sspace.nf[3])[ii_samp];
    # Simulate MChains
    for ii in 1:Nfirms
        zi_col[ii,:]   = simulate(mcz,T,init=zi_col_init[ii]);
    end

    # Simulate paths in IRF and SS
    sim_paths_SS  = sim_firm_SS(se_init, zi_col, xi_cdf_col, fsoln, true, modl, sspace, opts);
end
# Analyze debt issuance probabilities
# To compute actual issuance when adjusting, need to evaluate the adjuster's policy
bpA_paths_SS = zeros(size(sim_paths_SS.Ba_prob_path));
kpA_paths_SS = zeros(size(sim_paths_SS.Ba_prob_path));
# Create interpolants of the adjuster's policy functions
kpA_itp = interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), fsoln.kpA, Gridded(Linear()));
bpA_itp = interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), fsoln.bpA, Gridded(Linear()));
for tt=1:T
    for ii=1:Nfirms
        bpA_paths_SS[ii,tt] = bpA_itp(sim_paths_SS.k_path[ii,tt], sim_paths_SS.m_path[ii,tt], sim_paths_SS.b_path[ii,tt], sspace.z_grid[zi_col[ii,tt]]);
        kpA_paths_SS[ii,tt] = kpA_itp(sim_paths_SS.k_path[ii,tt], sim_paths_SS.m_path[ii,tt], sim_paths_SS.b_path[ii,tt], sspace.z_grid[zi_col[ii,tt]]);
    end
end
# True level of b in the simulated paths
b_tr_paths_SS = sim_paths_SS.k_path.*sim_paths_SS.b_path/ Rc_ss;
# Based on the "counterfactual adjuster" chosen path, determine if would have adjusted
dissA_paths_SS = (kpA_paths_SS.*bpA_paths_SS/Rc_ss) .< (gammatil_ss * b_tr_paths_SS .-1e-12);
# Given this, can finally determine paths for actual issuance probabilities
diss_prob_paths_SS = dissA_paths_SS .* sim_paths_SS.Ba_prob_path;

# Characterize untargeted moments locally
mda_paths = sim_paths_SS.m_path./(sim_paths_SS.m_path + sim_paths_SS.k_path);
Table_Untarg_row6 = mean(mda_paths[:,1].==0)    # m/a ratios at zero
Table_Untarg_row7 = mean(mda_paths[:,1].<1e-3)  # m/a ratios below 0.001
# median(mda_paths, dims=1)
Table_Untarg_row1 = std(mda_paths[:,1]) # sd(m/a)
# Correlation of m/a and log size
Table_Untarg_row2 = cor(mda_paths[:,1], log.(sim_paths_SS.k_path[:,1] + sim_paths_SS.m_path[:,1]))
# Issuance probability of large (>50th perc firms) -- from calibration code (inferred from population probability and <= 50th perc firms' probability)
Table_Untarg_row9  = targ_calib_pub[9]      # Small firms' issuance probability, returned by calibration moments above
Table_Untarg_row10 = targ_calib_pub[3]/0.5 - Table_Untarg_row9
# Dividend payers' share
Table_Untarg_row11 = sum(sum(sim_paths_SS.d_path[:,1:4], dims=2) .> 0)/Nfirms # Per year

# Correlation of m/a and i/k
ik_paths = (sim_paths_SS.k_path[:,2:end] - sim_paths_SS.k_path[:,1:(end-1)])./sim_paths_SS.k_path[:,1:(end-1)];
Table_Untarg_row3  = cor(mda_paths[:,1], ik_paths[:,4]) # 4q lag

# Simple calculations for theory Appendix B.5.3
bk_paths=-b_tr_paths_SS./(sim_paths_SS.k_path);
Bdif=(modl.gamma/modl.Pi)*(1/(1-modl.delta))*mean(bk_paths[:,1]); # Difference B^A-B^N at mean b/k
# Compute implied calligraphic Y^A and Y^N paths, as defined in Appendix B.5.3
YcalA_paths = sim_paths_SS.Yn_path + ((1+(1-modl.tau)*modl.rb)/modl.Pi) * b_tr_paths_SS + ((1+(1-modl.tau)*modl.rm)/modl.Pi)*sim_paths_SS.m_path
BcalA_paths = -YcalA_paths ./ ((1-modl.delta)*sim_paths_SS.k_path);
YcalN_paths = sim_paths_SS.Yn_path + ((1+(1-modl.tau)*modl.rb)/modl.Pi - modl.gamma/modl.Pi) * b_tr_paths_SS + ((1+(1-modl.tau)*modl.rm)/modl.Pi)*sim_paths_SS.m_path;
BcalN_paths = -YcalN_paths ./ ((1-modl.delta)*sim_paths_SS.k_path);
# Implied "constrained" elasticities AT means
mean_BcalA, mean_BcalN = mean(BcalA_paths[:,1]), mean(BcalN_paths[:,1])
epsA_atmeans, epsN_atmeans = mean_BcalA/(1-mean_BcalA), mean_BcalN/(1-mean_BcalN)
# Write output
open(joinpath(PROJECT_ROOT, "output", "other", "Text_AppxB5p3.txt"), "w") do io
    println(io, "Mean b/k:      $(round(mean(bk_paths[:,1]), digits=3))")
    println(io, "Mean B^A-B^N:  $(round(Bdif, digits=3))")
    println(io, "dlog(k)/dlog(Q) for A, at mean financial position:  $(round(epsA_atmeans, digits=3))")
    println(io, "dlog(k)/dlog(Q) for N, at mean financial position:  $(round(epsN_atmeans, digits=3))")
end

# Create group indicators of H/L leverage and liquidity, based on initial state
cheat_paths_SS = sim_paths_SS.m_path./(sim_paths_SS.m_path + sim_paths_SS.k_path);
lev_paths_SS   = -b_tr_paths_SS./(sim_paths_SS.m_path + sim_paths_SS.k_path);
med_cheat_rat = median(cheat_paths_SS[:,1])
med_lev_rat   = median(lev_paths_SS[:,1])
# Group indicators
Hlev_ind      = lev_paths_SS[:,1] .> med_lev_rat
Hcheat_ind    = cheat_paths_SS[:,1] .> med_cheat_rat
HH_ind        = Hcheat_ind .& Hlev_ind
LL_ind        = (.!Hcheat_ind) .& (.!Hlev_ind)
HL_ind        = Hcheat_ind .& (.!Hlev_ind)
LH_ind        =(.!Hcheat_ind) .& Hlev_ind

# Analyze the group-specific characteristics
# Mean leverage across liquidity groups
Table_Untarg_row4 = mean(lev_paths_SS[HL_ind .| HH_ind,1])
Table_Untarg_row5 = mean(lev_paths_SS[LH_ind .| LL_ind,1])
# Combine and print the untargeted moments table vector, corresponding to the model (Table B.2)
Table_Untarg_row8  = targ_calib_pub[5]      # Skew(log(k)), returned by calibration moments above
open(joinpath(PROJECT_ROOT, "output", "tables", "_partof_TabB2_nums_untargmoms_model.txt"), "w") do io
    for x in round.([Table_Untarg_row1, Table_Untarg_row2, Table_Untarg_row3, Table_Untarg_row4, Table_Untarg_row5, Table_Untarg_row6, Table_Untarg_row7, Table_Untarg_row8, Table_Untarg_row9, Table_Untarg_row10, Table_Untarg_row11], digits=3)
        println(io, x)
    end
end

# Plot histograms of model- and data-implied debt issuance probabilitites  -- Figure 4b
manl_ftsize = 12
# First, read empirical group-specfic frequencies
emp_dissprobs_lines = readlines(joinpath(PROJECT_ROOT, "interim_output", "Fig4b_nums_groupspec_Dfreq.txt"))
row1 = split(strip(emp_dissprobs_lines[2]))
row2 = split(strip(emp_dissprobs_lines[3]))
emp_dissprob_HL, emp_dissprob_HH, emp_dissprob_LL, emp_dissprob_LH = parse(Float64, row1[3]), parse(Float64, row1[4]), parse(Float64, row2[3]), parse(Float64, row2[4])
# Plot all together
colblue, colred, colgreen, colorange, colpurple = "#0080FF", "#FE2E2E", "#088A08", "#DF7401", "#8258FA"
colgrey = "#505050"
ylim_man = (0,12)
ppHL=ash(diss_prob_paths_SS[HL_ind,1], rng=0:.01:1)
Plots.plot(ppHL, xlims=(0.0, 1.0), ylims=ylim_man, hist=false, title="High liq, low lev", layout=(2,2), subplot=1, color=colgrey, label=:none, xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-1, size=(600,400))
Plots.vline!([emp_dissprob_HL], color=colblue, linewidth=2.0, linestyle=:dash, subplot=1, label="Data")
Plots.vline!([mean(diss_prob_paths_SS[HL_ind,1])], color=colred, linewidth=2.0, linestyle=:dot, subplot=1, label="Model")
ppHH=ash(diss_prob_paths_SS[HH_ind,1], rng=0:.01:1)
Plots.plot!(ppHH, xlims=(0.0, 1.0), ylims=ylim_man, hist=false, title="High liq, high lev", legend=:none, layout=(2,2), subplot=2, color=colgrey, xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-1)
Plots.vline!([emp_dissprob_HH], color=colblue, linewidth=2.0, linestyle=:dash, subplot=2)
Plots.vline!([mean(diss_prob_paths_SS[HH_ind,1])], color=colred, linewidth=2.0, linestyle=:dot, subplot=2)
ppLL=ash(diss_prob_paths_SS[LL_ind,1], rng=0:.01:1)
Plots.plot!(ppLL, xlims=(0.0, 1.0), ylims=ylim_man, hist=false, title="Low liq, low lev", legend=:none, layout=(2,2), subplot=3, color=colgrey, xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-1)
Plots.vline!([mean(diss_prob_paths_SS[LL_ind,1])], color=colred, linewidth=2.0, linestyle=:dot, subplot=3)
Plots.vline!([emp_dissprob_LL], color=colblue, linewidth=2.0, linestyle=:dash, subplot=3)
ppLH=ash(diss_prob_paths_SS[LH_ind,1], rng=0:.01:1)
Plots.plot!(ppLH, xlims=(0.0, 1.0), ylims=ylim_man, hist=false, title="Low liq, high lev", legend=:none, layout=(2,2), subplot=4, color=colgrey, xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-1)
Plots.vline!([mean(diss_prob_paths_SS[LH_ind,1])], color=colred, linewidth=2.0, linestyle=:dot, subplot=4)
Plots.vline!([emp_dissprob_LH], color=colblue, linewidth=2.0, linestyle=:dash, subplot=4)
Plots.savefig(joinpath(PROJECT_ROOT, "output", "figures", "model_figures", "Fig4b_"*modl.name*"_tdiss_prob_ss_dens.pdf"))

# Characterize firm life-cycle (Figure B.1) and average public vs private firm (Table B.5)
char_lifecycle(120, fsoln_base, aeq_base, modl, sspace, opts)
char_pubvspriv(Lin_pub, fsoln_base, aeq_base, modl, sspace, opts)

###
# Simulate another, longer sample and save sim_paths_SS into a csv of a balanced panel of firm-level data, to be analyzed in R code for steady state micro-level regressions
###
Nfirms_long=20000;
T_long=40;
zi_col_long, xi_cdf_col_long =  Array{Int64,2}(undef, Nfirms_long,T_long), zeros(Nfirms_long, T_long);
# Draw initial distribution
Random.seed!(999)
# Distribution from given Lin
ii_samp_long = wsample(1:sspace.Nsf, Lin_arg, Nfirms_long);
se_init_long = sspace.sf[ii_samp_long, 1:3];
xi_cdf_col_long = rand(Nfirms_long,T_long);
zi_col_init_long= repeat(1:sspace.nf[4], inner=sspace.nf[1]*sspace.nf[2]*sspace.nf[3])[ii_samp_long];
# Simulate MChains
for ii in 1:Nfirms_long
    zi_col_long[ii,:]   = simulate(mcz,T_long,init=zi_col_init_long[ii]);
end
sim_paths_SS_long  = sim_firm_SS(se_init_long, zi_col_long, xi_cdf_col_long, fsoln, true, modl, sspace, opts);

# To proxy Tobin's q, compute V0e values
V0e_paths_SS_long = zeros(size(sim_paths_SS_long.k_path));
# Create interpolants of the continuation value function
V0e_itp = interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), fsoln.V0e, Gridded(Linear()));
for tt=1:T_long
    for ii=1:Nfirms_long
        V0e_paths_SS_long[ii,tt] = V0e_itp(sim_paths_SS_long.k_path[ii,tt], sim_paths_SS_long.m_path[ii,tt], sim_paths_SS_long.b_path[ii,tt], sspace.z_grid[zi_col_long[ii,tt]]);
    end
end
# True level of b in the simulated paths
b_tr_paths_SS_long = sim_paths_SS_long.k_path.*sim_paths_SS_long.b_path/ Rc_ss;
Tobinq_paths_SS_long = (V0e_paths_SS_long - ((1.0+modl.rb)/modl.Pi)*b_tr_paths_SS_long) ./ (pQ_ss*sim_paths_SS_long.k_path + sim_paths_SS_long.m_path);

df_sim_paths = DataFrame(gvkey=vec(repeat(1:Nfirms_long,1,T_long)'), cqtr_num=vec(repeat((1:T_long)',Nfirms_long,1)'), k_stock = vec(sim_paths_SS_long.k_path'), dttq = -vec((sim_paths_SS_long.k_path').*(sim_paths_SS_long.b_path')/ Rc_ss), cheq = vec(sim_paths_SS_long.m_path'), dvq=vec(sim_paths_SS_long.d_path'), Ba_mod=vec(sim_paths_SS_long.Ba_path'), Ba_prob_mod=vec(sim_paths_SS_long.Ba_prob_path'), saleq=vec(sim_paths_SS_long.Y_path'), tobin_q = vec(Tobinq_paths_SS_long');)
CSV.write(joinpath(PROJECT_ROOT, "interim_output", "model_interim_output", "df_model_panel.csv"), df_sim_paths)

###
# Before continuing with monetary shock analysis, also construct phiw0 (no working capital constraint) counterfactual model and solution
###
# Set up phiw0 counterfactual model
modl_base_phiw0 = deepcopy(modl); 
modl_base_phiw0.phi_w = 0.0;
modl_base_phiw0.m_0 = 0.0;
# If the corresponding FirmSolution checkpoint does not yet exist, solve and save it, starting from base solution as guess. Otherwise, just load the saved solution.
if !isfile(joinpath(savedsoln_dir, "fsoln_base_phiw0.jld"))
    fsoln_base_phiw0 = solve_V_EGM(modl_base_phiw0,sspace,opts, fsoln_in=fsoln_base);
    save(joinpath(savedsoln_dir, "fsoln_base_phiw0.jld"),  "fsoln", fsoln_base_phiw0)
else
    fsoln_base_phiw0 = load(joinpath(savedsoln_dir, "fsoln_base_phiw0.jld"), "fsoln");
end

# Solve steady state aggregate equilibrium
aeq_base_phiw0 = solve_Leq(fsoln_base_phiw0, modl_base_phiw0, sspace, opts);
# Also, generate "public" firms distribution in base_phiw0 economy
Lin_pub_phiw0 = gen_L_Tage(20, aeq_base_phiw0.Lin, aeq_base_phiw0.Qend, modl_base_phiw0, sspace, opts);
# And compute the public firms' M/A ratio (to report in Appendix B.11.1 text)
Ma_pub_phiw0 = Lin_pub_phiw0' * sspace.sf[:,2];
Ka_pub_phiw0 = Lin_pub_phiw0' * sspace.sf[:,1];
# Print output
open(joinpath(PROJECT_ROOT, "output", "other", "Text_AppxB11p1.txt"), "w") do io
    println(io, "In the economy without a working capital constraint, the SS aggregate cash-to-assets ratio among public firms is $(round(100 * (Ma_pub_phiw0 / (Ma_pub_phiw0 + Ka_pub_phiw0)),digits=1))%.")
end


####################################
# Monetary shock analysis
####################################

# Set up length of considered transition path
T_g        = 61
# Note timing. The entry corresponding to t in zeta_col and spread_col will be applied to rf_{t+1} and rb_{t+1}. Thus, to introduce monetary shock announcement at impact, the first entry in zeta_col must be positive
# AR1 shock to zeta
# Autocorrelation of shock of 0.61, from Kaplan et al. (2018):
zeta_col = gen_ar1(T_g,1,T_g,0.0,0.61, 25/(4*10000))
# Constant exposure of spread to rb, set to zero:
spread_col = 0.0*zeta_col + (modl.rb-modl.rf)*ones(T_g) # No shock to rb-rf spread

# Create placeholders for variables created in loop
airf_g_base, irfsoln_g_base = nothing, nothing;
airf_g_base_phiw0, irfsoln_g_base_phiw0 = nothing, nothing;
MU_in, pQ_in = nothing, nothing;

for IRF_indic in ["base", "base_phiw0"]
    # Introduce pre-saved guesses for MU and pQ paths
    local MU_in, pQ_in
    if true
        if IRF_indic == "base"
            # Pre-saved guess, converged at 2*1e-5:
            MU_in = [1.1130866099943457, 1.112251350496448, 1.111808374978569, 1.1115315600814624, 1.1113633845628463, 1.1112607973083806, 1.1111986503180524, 1.111161258368187, 1.111138822056545, 1.111123787893173, 1.1111149169951196, 1.1111093887308319, 1.1111062288564328, 1.1111045291781683, 1.1111037279505243, 1.111103496969165, 1.1111036339108629, 1.111104004158294, 1.1111045196481042, 1.1111051145050994, 1.1111059793411358, 1.11110677315427, 1.111107549151496, 1.111108272492327, 1.111108914897949, 1.1111094539487298, 1.1111098729163107, 1.1111101621887882, 1.1111103200778418, 1.111110341313243, 1.1111116129082232, 1.1111121043689127, 1.111112522621286, 1.111112868008675, 1.1111131321799061, 1.111113308977509, 1.1111133963583126, 1.1111133932669455, 1.1111133024593969, 1.1111131208273182, 1.1111139781231736, 1.1111140508712425, 1.1111140302918059, 1.1111139148411366, 1.1111136955661227, 1.111113366011554, 1.1111129215361077, 1.1111123597405503, 1.1111116805473584, 1.1111108866102892, 1.1111099834449734, 1.111108979671518, 1.111107887249346, 1.1111067212899572, 1.1111055001207466, 1.111104245850011, 1.1111029844773526, 1.1111017330669732, 1.111100563249125, 1.1110995452614838, 1.1111111111111112];
            pQ_in = [0.999086203911522, 0.9994581317119203, 0.9996808769061724, 0.9998175157141206, 0.9999016004964538, 0.9999536346166266, 0.999985979782095, 1.0000061832569114, 1.0000188252990245, 1.00002746416085, 1.0000327148433197, 1.0000363826220249, 1.0000387400463076, 1.0000402343611745, 1.0000411502686664, 1.000041679862906, 1.0000419414508657, 1.0000420057134027, 1.0000419117496773, 1.0000416834803352, 1.0000413516097095, 1.0000409066289844, 1.0000403654582242, 1.0000397339688027, 1.0000390164828776, 1.0000382182221823, 1.0000373460013923, 1.000036408388354, 1.0000354149490127, 1.000034373448414, 1.000033373286255, 1.0000322665467294, 1.0000311130302941, 1.0000299184004342, 1.0000286879547355, 1.0000274277400734, 1.0000261447719652, 1.0000248466539121, 1.0000235412468952, 1.000022234335408, 1.0000209979399852, 1.0000197006480498, 1.0000184012187008, 1.0000171050818993, 1.000015817110074, 1.0000145427544738, 1.0000132880829706, 1.000012059591223, 1.0000108641565173, 1.0000097089025906, 1.0000086011032923, 1.0000075479615138, 1.0000065564940295, 1.0000056331748295, 1.0000047840189314, 1.0000040130734285, 1.0000033221361935, 1.0000027156920785, 1.0000021579932263, 1.0000015083199798, 1.0];
        elseif IRF_indic == "base_phiw0"
            # Pre-saved guess, converged at 2*1e-5:
            MU_in = [1.1130635517095424, 1.1122677323204533, 1.1118095184770165, 1.1115328464592251, 1.1113655451221145, 1.111263884848738, 1.111202284527729, 1.1111650519332412, 1.1111425743674073, 1.1111287622663533, 1.1111203194022734, 1.1111152168942078, 1.111112084756059, 1.1111101410749982, 1.1111089182677456, 1.111108139711291, 1.111107639424547, 1.1111073176348745, 1.1111071150870877, 1.1111069962308873, 1.111106994784516, 1.1111070378010406, 1.1111071321391637, 1.1111072694166186, 1.1111074432712351, 1.111107648720682, 1.1111078803842587, 1.1111081348416132, 1.1111084128694835, 1.1111087053095823, 1.1111092935545164, 1.1111097849741944, 1.111110268261662, 1.1111107907385314, 1.1111112890636121, 1.1111118017423314, 1.1111122682454952, 1.1111127241404077, 1.111113104615061, 1.1111134487568854, 1.111113985628319, 1.111114349990297, 1.1111146344214544, 1.1111148161681121, 1.1111148782027025, 1.1111148003620894, 1.1111145646485003, 1.1111141555508106, 1.1111135614368288, 1.1111127749677316, 1.1111117931930774, 1.1111106181972008, 1.111109257951456, 1.1111077268376954, 1.1111060461582087, 1.111104244178175, 1.1111023536932407, 1.1111004139641454, 1.1110984767657237, 1.1110965964715729, 1.1111111111111112];
            pQ_in = [0.9990803971021168, 0.9994503331927836, 0.9996758387669575, 0.9998145634974747, 0.9999004683668716, 0.9999541565301543, 0.9999878928036291, 1.0000091283499453, 1.0000225461350047, 1.000031120180564, 1.0000365429225548, 1.0000399412557381, 1.0000420174919606, 1.0000432103667312, 1.0000438026299683, 1.0000439825263359, 1.0000438755062564, 1.0000435641234857, 1.0000431035186694, 1.00004253182988, 1.0000418801211464, 1.0000411637687965, 1.00004039849778, 1.0000395943124851, 1.000038757977897, 1.0000378940719818, 1.0000370057464196, 1.0000360953079659, 1.0000351649862067, 1.00003422050652, 1.0000332987953258, 1.0000323484436262, 1.000031394940757, 1.000030405203004, 1.0000294096330773, 1.0000283761124924, 1.0000273334307117, 1.000026251182662, 1.0000251593129905, 1.0000240233244533, 1.0000228741695167, 1.0000216728037723, 1.0000204435807278, 1.0000191917215433, 1.0000179191309917, 1.0000166319135266, 1.000015336791385, 1.0000140413807135, 1.0000127538294294, 1.0000114830635787, 1.0000102390585468, 1.000009032588368, 1.0000078747493486, 1.0000067765559195, 1.0000057483876363, 1.0000047989989522, 1.0000039371394962, 1.0000031682115025, 1.0000024758718147, 1.0000017138369863, 1.0];
        end

        # Adjust lengths of guesses to T_g
        if length(MU_in) < T_g; MU_in = vcat(MU_in, MU_ss*ones(T_g-length(MU_in))); end
        if length(MU_in) > T_g; MU_in = MU_in[1:T_g]; end
        if length(pQ_in) < T_g; pQ_in = vcat(pQ_in, ones(T_g-length(pQ_in))); end
        if length(pQ_in) > T_g; pQ_in = pQ_in[1:T_g]; end
    end

    # Recheck setup and run
    opts.phi_p      = 0.96      # Significantly dampen updating of price paths 
    opts.phi_pQ_acl = 0.5       # Further dampen updating of pQ path relative to MU path

    # Solve IRF path, unless saved file already exists (in which case, load the solution from file)
    if (IRF_indic == "base") & isfile(joinpath(savedsoln_dir, "irf_base.jld"))
        airf_g, irfsoln_g = load(joinpath(savedsoln_dir, "irf_base.jld"), "airf_g", "irfsoln_g");
    elseif (IRF_indic == "base_phiw0") & isfile(joinpath(savedsoln_dir, "irf_base_phiw0.jld"))
        airf_g, irfsoln_g = load(joinpath(savedsoln_dir, "irf_base_phiw0.jld"), "airf_g", "irfsoln_g");
    elseif (IRF_indic == "base")
        airf_g, irfsoln_g = solve_eqIRF_NKMg_xsh(0.000, zeta_col, spread_col, fsoln, aeq, modl, sspace, opts, MU_col_0=MU_in, pQ_col_0=pQ_in);
        # Save solution to file
        save(joinpath(savedsoln_dir, "irf_base.jld"),  "airf_g", airf_g,  "irfsoln_g", irfsoln_g)
    elseif (IRF_indic == "base_phiw0")
        airf_g, irfsoln_g = solve_eqIRF_NKMg_xsh(0.000, zeta_col, spread_col, fsoln_base_phiw0, aeq_base_phiw0, modl_base_phiw0, sspace, opts, MU_col_0=MU_in, pQ_col_0=pQ_in);
        # Save solution to file
        save(joinpath(savedsoln_dir, "irf_base_phiw0.jld"),  "airf_g", airf_g,  "irfsoln_g", irfsoln_g)
    end

    # Back up solutions (in running environment) and run any additional procedures required with the current "IRF_indic"
    if IRF_indic == "base"
        # Back up
        global airf_g_base, irfsoln_g_base = airf_g, irfsoln_g;
        # After backing up, drop generic airf_g, irfsoln_g to save RAM
        airf_g, irfsoln_g = nothing, nothing

        # Format aggregate IRF figures -- Figure 5
        # Construct aggregate investment path for plotting
        Ia_col_base = airf_g_base.Ka_col[2:end]-(1.0-modl.delta)*airf_g_base.Ka_col[1:(end-1)]
        # Plot
        Tmax=20
        local manl_ftsize = 13
        fg = Plots.plot(0:Tmax, [4*100*(airf_g_base.rf_col[1:Tmax+1].-modl.rf), 4*100*(airf_g_base.Pi_col[1:Tmax+1].-modl.Pi), 100*(log.(airf_g_base.MU_col[1:Tmax+1]).-log(MU_ss)), 100*(log.(airf_g_base.Ya_col[1:Tmax+1]).-log(aeq_base.Ya)), 100*(log.(Ia_col_base[1:Tmax+1]).-log(modl.delta*aeq_base.Ka)), 100*(log.(airf_g_base.Ka_col[1:Tmax+1]).-log(aeq_base.Ka))], layout=(2,3), title=["Interest rates" "Inflation, \$Q\$" "Markup" L"Y, n, c" "Investment" "Capital" ], label=[L"r^{f}" L"\Pi" :none L"Y" L"I" :none], legend=[:bottomright :bottomright :none :bottomright :none :none], size=(800,460), linewidth=2, linecolor=:blue, xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-2, xticks=0:5:(length(airf_g_base.rf_col)-1))
        fg = Plots.plot!(0:Tmax, 4*100*(airf_g_base.rb_col[1:Tmax+1].-modl.rb), subplot=1, label=L"r^{b}", linewidth=2, linecolor=:red, linestyle=:dash)
        fg = Plots.plot!(0:Tmax, 4*100*(airf_g_base.rm_col[1:Tmax+1].-modl.rm), subplot=1, label=L"r^{m}", linewidth=2, linecolor=:green, linestyle=:dot)
        fg = Plots.plot!(0:Tmax, 100*(log.(airf_g_base.Na_col[1:Tmax+1]).-log(aeq_base.Na)), subplot=4, label=L"n", linewidth=2, linecolor=:red, linestyle=:dash)
        fg = Plots.plot!(0:Tmax, 100*(log.(airf_g_base.Ca_col[1:Tmax+1]).-log(aeq_base.Ca)), subplot=4, label=L"c", linewidth=2, linecolor=:green, linestyle=:dot)
        fg = Plots.plot!(0:Tmax, 100*(log.(airf_g_base.pQ_col[1:Tmax+1]).-log(pQ_ss)), subplot=2, label=L"Q", linewidth=2, linecolor=:red, linestyle=:dash)
        fg = hline!([0 0 0 0 0 0], subplot=1:6, color=:black, linestyle=:dash, label=:none)
        ylims!(fg[1], -0.050, 0.043)
        Plots.savefig(fg, joinpath(PROJECT_ROOT, "output", "figures", "model_figures", "Fig5_"*modl.name*"IRFaggs_main_rnarrow.pdf"))

        # Save some key impulse responses to csv file to be compared to empirics
        IRF_df = DataFrame(rb = 4*100*(airf_g_base.rb_col[1:Tmax+1].-modl.rb), Y = 100*(log.(airf_g_base.Ya_col[1:Tmax+1]).-log(aeq_base.Ya)), I = 100*(log.(Ia_col_base[1:Tmax+1]).-log(modl.delta*aeq_base.Ka)), Q = 100*(log.(airf_g_base.pQ_col[1:Tmax+1]).-log(pQ_ss)))
        CSV.write(joinpath(PROJECT_ROOT, "interim_output", "model_interim_output", "IRFs_base.csv"), IRF_df)

        if true
            # Given the baseline solution, draw random set of firms, simulate panel in SS and IRF and compute/plot regression coef on liq ratio -- Figure 6, also, Figure B.2, Figure B.3a, Figure B.3b, Figure B.4
            Random.seed!(999)
            # Public firms
            sim_IRFSS_anl_rand(50000,20, 60, Lin_pub, fsoln, irfsoln_g_base, true, true, "_base_pub", modl, sspace, opts)
        end

    elseif IRF_indic == "base_phiw0"
        # Back up
        global airf_g_base_phiw0, irfsoln_g_base_phiw0 = airf_g, irfsoln_g;
        # After backing up, drop generic airf_g, irfsoln_g to save RAM
        airf_g, irfsoln_g = nothing, nothing;

        # Format aggregate IRF figures -- Figure B.10
        # Construct aggregate investment path for plotting
        Ia_col_base = airf_g_base_phiw0.Ka_col[2:end]-(1.0-modl_base_phiw0.delta)*airf_g_base_phiw0.Ka_col[1:(end-1)]
        # Plot
        Tmax=20
        local manl_ftsize = 13
        fg = Plots.plot(0:Tmax, [4*100*(airf_g_base_phiw0.rf_col[1:Tmax+1].-modl_base_phiw0.rf), 4*100*(airf_g_base_phiw0.Pi_col[1:Tmax+1].-modl_base_phiw0.Pi), 100*(log.(airf_g_base_phiw0.MU_col[1:Tmax+1]).-log(MU_ss)), 100*(log.(airf_g_base_phiw0.Ya_col[1:Tmax+1]).-log(aeq_base_phiw0.Ya)), 100*(log.(Ia_col_base[1:Tmax+1]).-log(modl_base_phiw0.delta*aeq_base_phiw0.Ka)), 100*(log.(airf_g_base_phiw0.Ka_col[1:Tmax+1]).-log(aeq_base_phiw0.Ka))], layout=(2,3), title=["Interest rates" "Inflation, \$Q\$" "Markup" L"Y, n, c" "Investment" "Capital" ], label=[L"r^{f}" L"\Pi" :none L"Y" L"I" :none], legend=[:bottomright :bottomright :none :bottomright :none :none], size=(800,460), linewidth=2, linecolor=:blue, xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-2, xticks=0:5:(length(airf_g_base_phiw0.rf_col)-1))
        fg = Plots.plot!(0:Tmax, 4*100*(airf_g_base_phiw0.rb_col[1:Tmax+1].-modl_base_phiw0.rb), subplot=1, label=L"r^{b}", linewidth=2, linecolor=:red, linestyle=:dash)
        fg = Plots.plot!(0:Tmax, 4*100*(airf_g_base_phiw0.rm_col[1:Tmax+1].-modl_base_phiw0.rm), subplot=1, label=L"r^{m}", linewidth=2, linecolor=:green, linestyle=:dot)
        fg = Plots.plot!(0:Tmax, 100*(log.(airf_g_base_phiw0.Na_col[1:Tmax+1]).-log(aeq_base_phiw0.Na)), subplot=4, label=L"n", linewidth=2, linecolor=:red, linestyle=:dash)
        fg = Plots.plot!(0:Tmax, 100*(log.(airf_g_base_phiw0.Ca_col[1:Tmax+1]).-log(aeq_base_phiw0.Ca)), subplot=4, label=L"c", linewidth=2, linecolor=:green, linestyle=:dot)
        fg = Plots.plot!(0:Tmax, 100*(log.(airf_g_base_phiw0.pQ_col[1:Tmax+1]).-log(pQ_ss)), subplot=2, label=L"Q", linewidth=2, linecolor=:red, linestyle=:dash)
        fg = hline!([0 0 0 0 0 0], subplot=1:6, color=:black, linestyle=:dash, label=:none)
        ylims!(fg[1], -0.050, 0.043)
        Plots.savefig(fg, joinpath(PROJECT_ROOT, "output", "figures", "model_figures", "FigB10_"*modl.name*"IRFaggs_main_phiw0_rnarrow.pdf"))

        if true
            # Given the baseline solution, draw random set of firms, simulate panel in SS and IRF and compute/plot regression coef on liq ratio -- Figure B.11a
            Random.seed!(999)
            # Public firms
            sim_IRFSS_anl_rand(50000,20, 60, Lin_pub_phiw0, fsoln_base_phiw0, irfsoln_g_base_phiw0, true, true, "_phiw0_pub", modl_base_phiw0, sspace, opts)
        end
        # Forget irfsoln to save RAM
        irfsoln_g_base_phiw0 = nothing
    end
end
GC.gc() # -- Run garbage collection for RAM


###################################
# Lines to analyze counterfactual SS outcomes (corresponding to the Pre-90 calibration, Section 5)
###################################

######
# Steady state analysis
######
# Create copy of baseline model
modl_hr_cf = deepcopy(modl);
# Calibration of ("high rate") counterfactual
modl_hr_cf.Pi       = 1.0342^(1.0/4.0) # (Based on 3mTB difference: 1985-89/2: 3.42%)
modl_hr_cf.phi_m    = 1.0-0.4256 # SS exposure of nominal rm to nominal rf (1985-89: 1-0.4256)
modl_hr_cf.phi_mresp= 0.5*0.2445 + (1-0.2445-0.4256)*1 # IRF exposure of nominal rm to nominal rf (1985-89: 0.45215)
# Impose values
modl_hr_cf.rf = modl_hr_cf.Pi/modl_hr_cf.beta - 1.0
modl_hr_cf.rb = modl_hr_cf.rf + bspread_set
modl_hr_cf.rm = modl_hr_cf.phi_m*modl_hr_cf.rf 

# Solve new SS
opts.phi_V = [0.2,0.3,0.3,0.2] # Set more aggressive updating

# If the corresponding firm solution and aggregate solution checkpoint does not yet exist, solve and save it, starting from base firm solution as guess. Otherwise, just skip and load the saved solution below.
if !isfile(joinpath(savedsoln_dir, "fsoln_aeq_hr_cf.jld"))
    # Solve for steady state, given baseline model psi value: must iterate for equilibrium wage (equivalently, "p" which is the marginal utility of consumption)
    p_guessh = 0.7110165697537438; # Start from correct initial guess
    # Start Bellman iterations, with initial guess for value function from the baseline model, and iterate on p=u'(c) and the implied wage until goods market clears, so that the labor disutility parameter psi is the same across base and hr_cf economies (i.e., it is otherwise the same economy at different points in time, with different nominal rates).
    fsoln_hr_cf, aeq_hr_cf = solve_Leq_endp(p_guessh, modl_hr_cf, sspace, opts, fsoln_in=fsoln_base);
    # Save results
    save(joinpath(savedsoln_dir, "fsoln_aeq_hr_cf.jld"),  "fsoln", fsoln_hr_cf, "aeq", aeq_hr_cf);
end    
# Load saved solution, impose p guess and run again to double-check that the loaded solution is converged
fsoln_hr_cf_load, aeq_hr_cf_load = load(joinpath(savedsoln_dir, "fsoln_aeq_hr_cf.jld"), "fsoln", "aeq");
p_guessh = aeq_hr_cf_load.p
fsoln_hr_cf, aeq_hr_cf = solve_Leq_endp(p_guessh, modl_hr_cf, sspace, opts, fsoln_in=fsoln_hr_cf_load);

# Load empirical data on liquidity ratios from FOFA and plot
df_cheat_raw    = DataFrame(CSV.File(joinpath(PROJECT_ROOT, "interim_output", "FOFA_agg_cheat_t.csv")));
df_TB3MS_raw    = DataFrame(CSV.File(joinpath(PROJECT_ROOT, "data", "raw", "TB3MS.csv"), limit=90));
df_TB3MS_raw[!,"Year"] = 1973:2025;
df_CStat_cheat_raw = DataFrame(CSV.File(joinpath(PROJECT_ROOT, "interim_output","CS_agg_cheat_t.csv")));
df_Cstat_annl = df_CStat_cheat_raw[1:4:end,:]
Cstat_pre90_q7525_rat = median(df_Cstat_annl[1:5,"cheat_p75"] ./ df_Cstat_annl[1:5,"cheat_p25"]) # Median p75/p25 ratio pre-1990
Cstat_post90_q7525_rat = median(df_Cstat_annl[6:end,"cheat_p75"] ./ df_Cstat_annl[6:end,"cheat_p25"]) # Median p75/p25 ratio post-1990
# Report numbers stated in text, in Section 5.1
open(joinpath(PROJECT_ROOT, "output", "other", "Text_Sec5p1.txt"), "w") do io
    println(io, "Model aggregate liquidity ratio pre-90: $(round(aeq_hr_cf.Ma/(aeq_hr_cf.Ma+aeq_hr_cf.Ka),digits=3))")
    println(io, "Model aggregate liquidity ratio post-90: $(round(aeq.Ma/(aeq.Ma+aeq.Ka),digits=3))")
    println(io, "FoFA aggregate liquidity ratio in 1989: $(round(df_cheat_raw[df_cheat_raw[!,"Year"].==1989, "Nonfin Corp; cheat_rat"][1],digits=3))")
    println(io, "FoFA aggregate liquidity ratio in 2005: $(round(df_cheat_raw[df_cheat_raw[!,"Year"].==2005, "Nonfin Corp; cheat_rat"][1],digits=3))")
    println(io, "Compustat aggregate liquidity ratio increase 1989 -> 2005: $(round(df_Cstat_annl[df_Cstat_annl[!,"cqtr_num"].==2006, "cheat"][1] - df_Cstat_annl[df_Cstat_annl[!,"cqtr_num"].==1990, "cheat"][1],digits=3))")
    println(io, "Change in model aggregate debt-to-capital ratio pre-to-post-90: $(round(-aeq.Ba/aeq.Ka + aeq_hr_cf.Ba/aeq_hr_cf.Ka,digits=3))")
end

# Plot -- Figure B.6
manl_ftsize_mhat = 18
plot_start, plot_end = 1980, 2008
rowsel_ch = plot_start .<= df_cheat_raw[!,"Year"] .<= plot_end
rowsel_tb = plot_start .<= df_TB3MS_raw[!,"Year"] .<= plot_end
Plots.plot(df_cheat_raw[rowsel_ch,"Year"], df_cheat_raw[rowsel_ch,"Nonfin Corp; cheat_rat"], label="Liq ratio (Data, FoFA)", xtickfontsize=manl_ftsize_mhat-1, ytickfontsize=manl_ftsize_mhat-1, yguidefontsize=manl_ftsize_mhat, xguidefontsize=manl_ftsize_mhat, legendfontsize=manl_ftsize_mhat-2, size=(940,600), color=:blue, linewidth=3, ylabel="Liquidity ratio", xlabel="Year", right_margin = 4Plots.mm, left_margin = 4Plots.mm, bottom_margin = 4Plots.mm, legend=:topright, ylims=[0.025, 0.071])
Plots.plot!([1990,2008], (aeq.Ma/(aeq.Ma+aeq.Ka))*ones(2), label="Liq ratio (Model, post-90)", color=:green, linewidth=2, linestyle=:dot)
Plots.plot!([1980,1990], (aeq_hr_cf.Ma/(aeq_hr_cf.Ma+aeq_hr_cf.Ka))*ones(2), label="Liq ratio (Model, pre-90)", color=:red, linewidth=2, linestyle=:dash)
plot!([1990],[0], linecolor=:black, linewidth=2, label="3m Treasury rate", linestyle=:dashdotdot)
Plots.plot!(twinx(), df_TB3MS_raw[rowsel_tb,"Year"], df_TB3MS_raw[rowsel_tb,"TB3MS"]/100, linewidth=2, label=:none, ytickfontsize=manl_ftsize_mhat-1, yguidefontsize=manl_ftsize_mhat, ylabel="Interest rate", linecolor=:black, linestyle=:dashdotdot)
Plots.savefig(joinpath(PROJECT_ROOT, "output", "figures", "model_figures", "FigB6_cheat_rat_hr_cf.pdf"))


######
# Monetary shock analysis in counterfactual Pre-90 calibration
######

# Conditional on the solution captured by (fsoln_hr_cf, aeq_hr_cf), solve IRF to monetary shock in general equilibrium
if true
    # Introduce pre-saved guesses for MU and pQ paths
    if true
        # Pre-saved guess, converged at 2*1e-5:
        MU_in = [1.1131106764724292, 1.1122472391544107, 1.111802539920827, 1.1115289377185793, 1.111362383713051, 1.1112612595472997, 1.1112000708219676, 1.1111631707830234, 1.1111409593259416, 1.1111274946179215, 1.111119328422317, 1.1111144338889691, 1.111111464189509, 1.1111096430043603, 1.1111085090848807, 1.111107789457195, 1.111107323293229, 1.1111070164977606, 1.111106814884671, 1.1111066874073892, 1.1111066578949278, 1.111106671771726, 1.1111067341857976, 1.1111068382580724, 1.111106980072504, 1.1111071554483931, 1.1111073609323818, 1.111107593742537, 1.111107852291529, 1.1111081336828816, 1.1111086729936124, 1.1111091569065048, 1.1111096698464507, 1.1111102055542932, 1.1111107525786956, 1.1111112986557907, 1.1111118312767083, 1.111112337465375, 1.111112804428305, 1.111113217724813, 1.111113756694245, 1.111114125334448, 1.1111143965950467, 1.1111145556256654, 1.1111145848127661, 1.1111144683761334, 1.111114192807567, 1.111113747709774, 1.1111131262073555, 1.1111123258434645, 1.111111348834926, 1.111110203166567, 1.1111089018338605, 1.1111074665605445, 1.1111059238469636, 1.1111043101668368, 1.1111026675592663, 1.1111010367271221, 1.1110995132979735, 1.1110983660511267, 1.1111111111111112];
        pQ_in = [0.9990687617893698, 0.9994439586669669, 0.9996732485655991, 0.999813561497536, 0.9999005407340761, 0.9999548188149906, 0.9999889771967473, 1.0000105456465027, 1.0000242119276301, 1.0000328905146463, 1.0000383606100736, 1.0000417673644515, 1.0000438203845365, 1.0000449684115365, 1.0000455022868135, 1.0000456153908277, 1.0000454356140733, 1.000045047723161, 1.000044509446022, 1.0000438615075935, 1.0000431357076176, 1.0000423504607892, 1.0000415217426646, 1.0000406619822215, 1.0000397787046054, 1.0000388768775847, 1.0000379602575213, 1.0000370315569838, 1.000036092605041, 1.0000351442013784, 1.0000341990432522, 1.000033233166856, 1.0000322519006077, 1.0000312535451517, 1.0000302361655466, 1.000029198056344, 1.0000281378770561, 1.0000270547393675, 1.0000259482294165, 1.0000248182015512, 1.0000236752605494, 1.0000224996539664, 1.0000212997184572, 1.0000200778698327, 1.0000188372388439, 1.0000175820159602, 1.0000163174979975, 1.0000150500518665, 1.0000137871337593, 1.0000125370773687, 1.0000113092378413, 1.0000101133245636, 1.000008959942158, 1.0000078589330779, 1.0000068202384287, 1.0000058498311446, 1.0000049500319237, 1.0000041240701545, 1.0000033255443201, 1.0000023214146976, 1.0];

        # Adjust lengths
        if length(MU_in) < T_g; MU_in = vcat(MU_in, MU_ss*ones(T_g-length(MU_in))); end
        if length(MU_in) > T_g; MU_in = MU_in[1:T_g]; end
        if length(pQ_in) < T_g; pQ_in = vcat(pQ_in, ones(T_g-length(pQ_in))); end
        if length(pQ_in) > T_g; pQ_in = pQ_in[1:T_g]; end
    end

    # Solve IRF path, unless saved file already exists
    if isfile(joinpath(savedsoln_dir, "irf_hr_cf.jld"))
        # Load if saved
        airf_g_hr_cf, irfsoln_g_hr_cf = load(joinpath(savedsoln_dir, "irf_hr_cf.jld"), "airf_g", "irfsoln_g");
    else
        airf_g_hr_cf, irfsoln_g_hr_cf = solve_eqIRF_NKMg_xsh(0.000, zeta_col, spread_col, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts, MU_col_0=MU_in, pQ_col_0=pQ_in, w_ss=modl_hr_cf.psi/aeq_hr_cf.p);
        # Save to file
        save(joinpath(savedsoln_dir, "irf_hr_cf.jld"),  "airf_g", airf_g_hr_cf,  "irfsoln_g", irfsoln_g_hr_cf);
    end

    # Plot IRF solutions for the hr_cf
    if true
        # Format aggregate IRF figures -- Figure B.7
        # Construct Agg investment
        Ia_col_base = airf_g_hr_cf.Ka_col[2:end]-(1.0-modl_hr_cf.delta)*airf_g_hr_cf.Ka_col[1:(end-1)]
        # Plot
        Tmax=20
        manl_ftsize = 13
        fg = Plots.plot(0:Tmax, [4*100*(airf_g_hr_cf.rf_col[1:Tmax+1].-modl_hr_cf.rf), 4*100*(airf_g_hr_cf.Pi_col[1:Tmax+1].-modl_hr_cf.Pi), 100*(log.(airf_g_hr_cf.MU_col[1:Tmax+1]).-log(MU_ss)), 100*(log.(airf_g_hr_cf.Ya_col[1:Tmax+1]).-log(aeq_hr_cf.Ya)), 100*(log.(Ia_col_base[1:Tmax+1]).-log(modl_hr_cf.delta*aeq_hr_cf.Ka)), 100*(log.(airf_g_hr_cf.Ka_col[1:Tmax+1]).-log(aeq_hr_cf.Ka))], layout=(2,3), title=["Interest rates" "Inflation, \$Q\$" "Markup" L"Y, n, c" "Investment" "Capital" ], label=[L"r^{f}" L"\Pi" :none L"Y" L"I" :none], legend=[:bottomright :bottomright :none :bottomright :none :none], size=(800,460), linewidth=2, linecolor=:blue, xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-2, xticks=0:5:(length(airf_g_hr_cf.rf_col)-1))
        fg = Plots.plot!(0:Tmax, 4*100*(airf_g_hr_cf.rb_col[1:Tmax+1].-modl_hr_cf.rb), subplot=1, label=L"r^{b}", linewidth=2, linecolor=:red, linestyle=:dash)
        fg = Plots.plot!(0:Tmax, 4*100*(airf_g_hr_cf.rm_col[1:Tmax+1].-modl_hr_cf.rm), subplot=1, label=L"r^{m}", linewidth=2, linecolor=:green, linestyle=:dot)
        fg = Plots.plot!(0:Tmax, 100*(log.(airf_g_hr_cf.Na_col[1:Tmax+1]).-log(aeq_hr_cf.Na)), subplot=4, label=L"n", linewidth=2, linecolor=:red, linestyle=:dash)
        fg = Plots.plot!(0:Tmax, 100*(log.(airf_g_hr_cf.Ca_col[1:Tmax+1]).-log(aeq_hr_cf.Ca)), subplot=4, label=L"c", linewidth=2, linecolor=:green, linestyle=:dot)
        fg = Plots.plot!(0:Tmax, 100*(log.(airf_g_hr_cf.pQ_col[1:Tmax+1]).-log(pQ_ss)), subplot=2, label=L"Q", linewidth=2, linecolor=:red, linestyle=:dash)
        fg = hline!([0 0 0 0 0 0], subplot=1:6, color=:black, linestyle=:dash, label=:none)
        ylims!(fg[1], -0.050, 0.043)
        Plots.savefig(fg, joinpath(PROJECT_ROOT, "output", "figures", "model_figures", "FigB7_"*modl_hr_cf.name*"IRFaggs_hr_cf_rnarrow.pdf"))
    end

    # Also, right after solve the IRF for the Pre-90-phi^m_H economy, which is the "Pre-90 model", temporarily imposing the "base" phi_mresp, with the model version referred to as "hr_cf_basephimresp"
    # Introduce pre-saved guesses for MU and pQ paths
    if true
        # Pre-saved guess, converged at 2*1e-5:
        MU_in = [1.1131180754440642, 1.1122495292489236, 1.1118026685397149, 1.1115280788359199, 1.1113610539822436, 1.1112598039022445, 1.1111986706119374, 1.1111619125093646, 1.1111398856390615, 1.1111266441299232, 1.1111187011196193, 1.1111139920539996, 1.1111111807862621, 1.1111094893358515, 1.1111084561684923, 1.1111078090255622, 1.1111073889065017, 1.1111071044833678, 1.1111069046416533, 1.111106762774601, 1.1111066934638825, 1.1111066603890223, 1.111106669633579, 1.1111067204356584, 1.1111068121397722, 1.1111069435203151, 1.1111071140674555, 1.1111073235361262, 1.1111075722055004, 1.1111078590243362, 1.111108354938006, 1.1111088376221139, 1.1111093631386189, 1.1111099248436698, 1.1111105113300737, 1.1111111098071964, 1.1111117065977636, 1.1111122870661034, 1.1111128361867975, 1.111113337324045, 1.1111139139137947, 1.1111143484589716, 1.1111146826077531, 1.1111148980813563, 1.1111149746355709, 1.1111148938422966, 1.1111146396902416, 1.111114199519487, 1.1111135645898929, 1.11111273111712, 1.1111117006046203, 1.1111104813555606, 1.1111090874760539, 1.1111075430970374, 1.1111058779476528, 1.1111041341027985, 1.1111023599871641, 1.1111006040060973, 1.1110989792827313, 1.1110978127174906, 1.1111111111111112];
        pQ_in = [0.9990621934203209, 0.999440617746825, 0.9996717954299696, 0.9998130280781543, 0.9999004964437161, 0.9999550440468052, 0.9999893519062487, 1.0000110050002813, 1.000024720547764, 1.0000334314691819, 1.0000389244499825, 1.0000423460579153, 1.000044408703197, 1.000045562098924, 1.0000460972166936, 1.0000462072776277, 1.0000460201382109, 1.0000456207517732, 1.0000450674452488, 1.0000444010568077, 1.0000436545337694, 1.000042847435453, 1.0000419960236007, 1.0000411120049586, 1.0000402038102332, 1.0000392773642959, 1.000038336817254, 1.00003738497981, 1.0000364235728012, 1.000035453182297, 1.0000344826803067, 1.0000334944278961, 1.0000324920482602, 1.000031473817042, 1.0000304377969083, 1.0000293822369348, 1.0000283056996846, 1.0000272071632792, 1.000026086065052, 1.0000249421685636, 1.0000237831979104, 1.0000225953567419, 1.0000213856064768, 1.0000201567395768, 1.000018912362744, 1.0000176571360588, 1.0000163968038867, 1.0000151381145903, 1.0000138888235475, 1.0000126574204453, 1.0000114532519273, 1.0000102856752382, 1.000009164700223, 1.0000080989051139, 1.0000070966459564, 1.000006161221589, 1.0000052915795503, 1.000004487521925, 1.0000036909835552, 1.0000026342543087, 1.0];
    end

    # If solution file exists, load it
    if isfile(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp.jld"))
        airf_g_hr_cf_basephimresp = load(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp.jld"), "airf_g");
    else
        # Impose base phi_mresp temporarily on the pre-90 economy
        bu_phi_mresp = copy(modl_hr_cf.phi_mresp); # Back up default pre-90 value
        modl_hr_cf.phi_mresp = modl.phi_mresp;
        airf_g_hr_cf_basephimresp, irfsoln_g_hr_cf_basephimresp = solve_eqIRF_NKMg_xsh(0.000, zeta_col, spread_col, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts, MU_col_0=MU_in, pQ_col_0=pQ_in, w_ss=modl_hr_cf.psi/aeq_hr_cf.p);
        # Save to file
        save(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp.jld"),  "airf_g", airf_g_hr_cf_basephimresp,  "irfsoln_g", irfsoln_g_hr_cf_basephimresp)
        # Reset phi_mresp to the Pre-90 value
        modl_hr_cf.phi_mresp = bu_phi_mresp;
        # Forget irfsoln to save RAM
        irfsoln_g_hr_cf_basephimresp = nothing;
    end
end

# Compute IRFs to monetary shock in partial equilibrium (noGE), by loading specific price paths into the GE solver, but not iterating on the price paths (setting opts.tolp_col to a high number)
# First, the PE effect of introducing basephimresp to hr_cf economy
if true
    if isfile(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp_noGE.jld"))
        airf_g_hr_cf_basephimresp_noGE = load(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp_noGE.jld"), "airf_g");
    else
        # Impose base phi_mresp temporarily on the pre-90 economy
        bu_phi_mresp = copy(modl_hr_cf.phi_mresp); # Back up default pre-90 value
        modl_hr_cf.phi_mresp = modl.phi_mresp;
        opts.tol_p_col = 2*1e5 # To ensure no iterations on price paths, locally
        airf_g_hr_cf_basephimresp_noGE, irfsoln_g_hr_cf_basephimresp_noGE = solve_eqIRF_NKMg_xsh(0.000, zeta_col, spread_col, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts, MU_col_0=airf_g_hr_cf.MU_col, pQ_col_0=airf_g_hr_cf.pQ_col, w_ss=modl_hr_cf.psi/aeq_hr_cf.p)
        opts.tol_p_col = 2*1e-5 # Reset tolerance
        # Save to file
        save(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp_noGE.jld"),  "airf_g", airf_g_hr_cf_basephimresp_noGE,  "irfsoln_g", irfsoln_g_hr_cf_basephimresp_noGE)
        # Reset phi_mresp
        modl_hr_cf.phi_mresp = bu_phi_mresp;
        # Forget irfsoln to save RAM
        irfsoln_g_hr_cf_basephimresp_noGE = nothing;
    end
end
# Second, the PE effect of going to the Post-90 economy, but imposing Pre-90 price IRF paths
if true
    if isfile(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE.jld"))
        airf_g_base_hr_cf_basephimresp_noGE = load(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE.jld"), "airf_g");
    else
        # Use base economy model, with base phi_mresp
        opts.tol_p_col = 2*1e5 # To ensure no iterations on price paths, locally
        airf_g_base_hr_cf_basephimresp_noGE, irfsoln_g_base_hr_cf_basephimresp_noGE = solve_eqIRF_NKMg_xsh(0.000, zeta_col, spread_col, fsoln, aeq, modl, sspace, opts, MU_col_0=airf_g_hr_cf.MU_col, pQ_col_0=airf_g_hr_cf.pQ_col);
        opts.tol_p_col = 2*1e-5 # Reset tolerance
        # Save to file
        save(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE.jld"),  "airf_g", airf_g_base_hr_cf_basephimresp_noGE,  "irfsoln_g", irfsoln_g_base_hr_cf_basephimresp_noGE)
        # Forget irfsoln to save RAM
        irfsoln_g_base_hr_cf_basephimresp_noGE = nothing;
    end
end


# Lines for generating Table 2
if true

    # Analysis of group-specific responses in base vs hr_cf economies, conditional on 25th percentile cutoff in the m/a-space
    ldev_Ka_base, Ka_SSshare_base = gen_eqIRF_groupspecs(5, 0.25, 0.25, irfsoln_g_base, fsoln_base, aeq_base, modl, sspace, opts);
    ldev_Ka_hr_cf, Ka_SSshare_hr_cf = gen_eqIRF_groupspecs(5, 0.25, 0.25, irfsoln_g_hr_cf, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts);

    # Compute the contribution shares
    Ka_L_contr_share_base = (ldev_Ka_base[:,3].*Ka_SSshare_base[:,3]) ./ ldev_Ka_base[:,1];
    Ka_L_contr_share_base[1] = 0.0;
    Ka_L_contr_share_hr_cf = (ldev_Ka_hr_cf[:,3].*Ka_SSshare_hr_cf[:,3]) ./ ldev_Ka_hr_cf[:,1];
    Ka_L_contr_share_hr_cf[1] = 0.0;

    # Compute population liquidity ratio percentiles
    ma_gridf = sspace.sf[:,2] ./ (sspace.sf[:,1] + sspace.sf[:,2])
    liq_p25_base, liq_p75_base   = discr_percs(ma_gridf, aeq_base.Lin, [0.25, 0.75])
    liq_p25_hr_cf, liq_p75_hr_cf = discr_percs(ma_gridf, aeq_hr_cf.Lin, [0.25, 0.75])

    # Numbers for Table 2
    pre_data_q7525   = round(Cstat_pre90_q7525_rat, digits=2)
    post_data_q7525  = round(Cstat_post90_q7525_rat, digits=2)
    pre_model_q7525  = round(liq_p75_hr_cf / liq_p25_hr_cf, digits=2)
    post_model_q7525 = round(liq_p75_base / liq_p25_base, digits=2)
    pre_kdiff  = round(ldev_Ka_hr_cf[5,3] - ldev_Ka_hr_cf[5,2], digits=3)
    post_kdiff = round(ldev_Ka_base[5,3] - ldev_Ka_base[5,2], digits=3)
    pre_share  = round(Ka_L_contr_share_hr_cf[5], digits=3)
    post_share = round(Ka_L_contr_share_base[5], digits=3)
    # Print out Table 2
    open(joinpath(PROJECT_ROOT, "output", "tables", "Tab2_HRCF_hety.tex"), "w") do io
        print(io, """
        \\begin{table}[htbp]
        \\caption{Liquidity ratios and investment responses in cross-section pre- and post-1990}
        \\vspace{-14pt}
        \\begin{center}
        \\begin{tabular}{>{\\raggedright}p{20mm} | >{\\centering}p{28mm}  >{\\centering\\arraybackslash}p{28mm} | >{\\centering\\arraybackslash}p{33mm}>{\\centering\\arraybackslash}p{33mm} }
        \\hline \\hline
        & \$q_{\\ell}^{75}/q_{\\ell}^{25}\$, data & \$q_{\\ell}^{75}/q_{\\ell}^{25}\$, model
        & \$\\hat{K}_{4|\\ell_{i} \\leq q_{\\ell}^{25}} - \\hat{K}_{4|\\ell_{i}>q_{\\ell}^{25}}\$ 
        & \$\\ell_{i}\\leq q_{\\ell}^{25}\$ sh.\\ in \$\\hat{K}_{4}\$ \\\\
        \\hline
        Pre-1990 & $(@sprintf("%.2f", pre_data_q7525)) & $(@sprintf("%.2f", pre_model_q7525)) & $(@sprintf("%.3f", pre_kdiff)) & $(@sprintf("%.3f", pre_share)) \\\\
        Post-1990 & $(@sprintf("%.2f", post_data_q7525)) & $(@sprintf("%.2f", post_model_q7525)) & $(@sprintf("%.3f", post_kdiff)) & $(@sprintf("%.3f", post_share)) \\\\
        \\hline \\hline
        \\end{tabular}
        \\label{tab_HRCF_hety}

        \\vspace{4pt}
        \\begin{minipage}{\\textwidth}
        {\\footnotesize \\emph{Notes:} \$q_{\\ell}^{\\alpha}\$ is the \$\\alpha\$-th percentile of the firms' liquidity ratio cross-section. \$q_{\\ell}^{75}/q_{\\ell}^{25}\$ in the data computed based on Compustat cross-section by quarter, reporting across-time medians for 1985--89 (pre-1990) and 1990--2008 (post-1990); in model based on the corresponding steady state populations.
        \$\\hat{K}_{4 | l_{i} \\leq q_{\\ell}^{25}}\$ is the model-implied
        (log) response of total capital one year after \$\\varepsilon_{0}^{f}\$ for firms with \$\\ell_{i,0} \\leq q_{l}^{25}\$;
        \$\\hat{K}_{4 | l_{i} > q_{\\ell}^{25}}\$ refers to the rest.
        The last column reports
        the contribution of the bottom-fourth firms in the aggregate \$K_{4}\$ response,
        computed as \$\\left(\\hat{K}_{4| l_{i} \\leq q_{\\ell}^{25}} / \\hat{K}_{4}\\right) \\cdot (K_{4,SS| l_{i} \\leq q_{\\ell}^{25}} / K_{SS})\$, with \$K_{4,SS| l_{i} \\leq q_{\\ell}^{25}}\$ the group's \$t=4\$ capital in steady state.
        }
        \\end{minipage}
        \\end{center}

        \\vspace{-8pt}
        \\end{table}
        """)
    end
end
# After this, irfsoln_g_hr_cf no longer needed
irfsoln_g_hr_cf = nothing;
GC.gc()

########
# Transmission channel decomposition analysis
########
# Precompute all the possible channel decompositions used throughout the paper and save them as interim outputs

### Analyze Q-channel
# Base model
if !isfile(joinpath(savedsoln_dir, "irf_base_decomp_Q.jld"))
    airf_g_base_decomp_Q, irfsoln_g_base_decomp_Q = solve_eqIRF_decomp("Q", airf_g_base, fsoln, aeq, modl, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_base_decomp_Q.jld"),  "airf_g", airf_g_base_decomp_Q,  "irfsoln_g", irfsoln_g_base_decomp_Q)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_base_decomp_Q, irfsoln_g_base_decomp_Q = nothing, nothing;
end
# For hr_cf
if !isfile(joinpath(savedsoln_dir, "irf_hr_cf_decomp_Q.jld"))
    airf_g_hr_cf_decomp_Q, irfsoln_g_hr_cf_decomp_Q = solve_eqIRF_decomp("Q", airf_g_hr_cf, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_hr_cf_decomp_Q.jld"),  "airf_g", airf_g_hr_cf_decomp_Q,  "irfsoln_g", irfsoln_g_hr_cf_decomp_Q)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_hr_cf_decomp_Q, irfsoln_g_hr_cf_decomp_Q = nothing, nothing;
end
# base_phiw0
if !isfile(joinpath(savedsoln_dir, "irf_base_phiw0_decomp_Q.jld"))
    airf_g_base_phiw0_decomp_Q, irfsoln_g_base_phiw0_decomp_Q = solve_eqIRF_decomp("Q", airf_g_base_phiw0, fsoln_base_phiw0, aeq_base_phiw0, modl_base_phiw0, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_base_phiw0_decomp_Q.jld"),  "airf_g", airf_g_base_phiw0_decomp_Q,  "irfsoln_g", irfsoln_g_base_phiw0_decomp_Q)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_base_phiw0_decomp_Q, irfsoln_g_base_phiw0_decomp_Q  = nothing, nothing;
end
# Also, for PE effect: base_hr_cf_basephimresp_noGE
if !isfile(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE_decomp_Q.jld"))
    airf_g_base_hr_cf_basephimresp_noGE_decomp_Q, irfsoln_g_base_hr_cf_basephimresp_noGE_decomp_Q = solve_eqIRF_decomp("Q", airf_g_base_hr_cf_basephimresp_noGE, fsoln, aeq, modl, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE_decomp_Q.jld"),  "airf_g", airf_g_base_hr_cf_basephimresp_noGE_decomp_Q,  "irfsoln_g", irfsoln_g_base_hr_cf_basephimresp_noGE_decomp_Q)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_base_hr_cf_basephimresp_noGE_decomp_Q, irfsoln_g_base_hr_cf_basephimresp_noGE_decomp_Q = nothing, nothing;
end


### Analyze rbtaureal-channel
# Base model
if isfile(joinpath(savedsoln_dir, "irf_base_decomp_rbtaureal.jld"))
    # Load if saved
    airf_g_base_decomp_rbtaureal, irfsoln_g_base_decomp_rbtaureal = load(joinpath(savedsoln_dir, "irf_base_decomp_rbtaureal.jld"), "airf_g", "irfsoln_g");
else
    airf_g_base_decomp_rbtaureal, irfsoln_g_base_decomp_rbtaureal = solve_eqIRF_decomp("rbtaureal", airf_g_base, fsoln, aeq, modl, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_base_decomp_rbtaureal.jld"),  "airf_g", airf_g_base_decomp_rbtaureal,  "irfsoln_g", irfsoln_g_base_decomp_rbtaureal);
end
# Run regression coefs for rbtaureal, and then forget irfsoln_g_base_decomp_rbtaureal
if true
    # Given the baseline solution, draw random set of firms, simulate panel in SS and IRF and compute regression coef on liq ratio
    Random.seed!(999)
    # Also, need to introduce "base rm_col" for regression coefficient scale adjustment
    irfsoln_g_base_decomp_rbtaureal.rm_col = airf_g_base.rm_col;
    # Public firms
    sim_IRFSS_anl_rand(50000,20, 60, Lin_pub, fsoln, irfsoln_g_base_decomp_rbtaureal, true, true, "_rbtaureal_pub", modl, sspace, opts)
end
# And forget in order to save RAM, to be loaded as needed later
airf_g_base_decomp_rbtaureal, irfsoln_g_base_decomp_rbtaureal = nothing, nothing;
GC.gc();
# For hr_cf
if !isfile(joinpath(savedsoln_dir, "irf_hr_cf_decomp_rbtaureal.jld"))
    airf_g_hr_cf_decomp_rbtaureal, irfsoln_g_hr_cf_decomp_rbtaureal = solve_eqIRF_decomp("rbtaureal", airf_g_hr_cf, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_hr_cf_decomp_rbtaureal.jld"),  "airf_g", airf_g_hr_cf_decomp_rbtaureal,  "irfsoln_g", irfsoln_g_hr_cf_decomp_rbtaureal)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_hr_cf_decomp_rbtaureal, irfsoln_g_hr_cf_decomp_rbtaureal = nothing, nothing;
end
# base_phiw0
if !isfile(joinpath(savedsoln_dir, "irf_base_phiw0_decomp_rbtaureal.jld"))
    airf_g_base_phiw0_decomp_rbtaureal, irfsoln_g_base_phiw0_decomp_rbtaureal = solve_eqIRF_decomp("rbtaureal", airf_g_base_phiw0, fsoln_base_phiw0, aeq_base_phiw0, modl_base_phiw0, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_base_phiw0_decomp_rbtaureal.jld"),  "airf_g", airf_g_base_phiw0_decomp_rbtaureal,  "irfsoln_g", irfsoln_g_base_phiw0_decomp_rbtaureal)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_base_phiw0_decomp_rbtaureal, irfsoln_g_base_phiw0_decomp_rbtaureal = nothing, nothing;
end

### Analyze rmtaureal-channel
# Base model
if !isfile(joinpath(savedsoln_dir, "irf_base_decomp_rmtaureal.jld"))
    airf_g_base_decomp_rmtaureal, irfsoln_g_base_decomp_rmtaureal = solve_eqIRF_decomp("rmtaureal", airf_g_base, fsoln, aeq, modl, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_base_decomp_rmtaureal.jld"),  "airf_g", airf_g_base_decomp_rmtaureal,  "irfsoln_g", irfsoln_g_base_decomp_rmtaureal)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_base_decomp_rmtaureal, irfsoln_g_base_decomp_rmtaureal = nothing, nothing;
end
# For hr_cf
if !isfile(joinpath(savedsoln_dir, "irf_hr_cf_decomp_rmtaureal.jld"))
    airf_g_hr_cf_decomp_rmtaureal, irfsoln_g_hr_cf_decomp_rmtaureal = solve_eqIRF_decomp("rmtaureal", airf_g_hr_cf, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_hr_cf_decomp_rmtaureal.jld"),  "airf_g", airf_g_hr_cf_decomp_rmtaureal,  "irfsoln_g", irfsoln_g_hr_cf_decomp_rmtaureal)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_hr_cf_decomp_rmtaureal, irfsoln_g_hr_cf_decomp_rmtaureal = nothing, nothing;
end
# For hr_cf_basephimresp
if !isfile(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp_decomp_rmtaureal.jld"))
    airf_g_hr_cf_basephimresp_decomp_rmtaureal, irfsoln_g_hr_cf_basephimresp_decomp_rmtaureal = solve_eqIRF_decomp("rmtaureal", airf_g_hr_cf_basephimresp, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp_decomp_rmtaureal.jld"),  "airf_g", airf_g_hr_cf_basephimresp_decomp_rmtaureal,  "irfsoln_g", irfsoln_g_hr_cf_basephimresp_decomp_rmtaureal)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_hr_cf_basephimresp_decomp_rmtaureal, irfsoln_g_hr_cf_basephimresp_decomp_rmtaureal = nothing, nothing;
end
# base_phiw0
if !isfile(joinpath(savedsoln_dir, "irf_base_phiw0_decomp_rmtaureal.jld"))
    airf_g_base_phiw0_decomp_rmtaureal, irfsoln_g_base_phiw0_decomp_rmtaureal = solve_eqIRF_decomp("rmtaureal", airf_g_base_phiw0, fsoln_base_phiw0, aeq_base_phiw0, modl_base_phiw0, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_base_phiw0_decomp_rmtaureal.jld"),  "airf_g", airf_g_base_phiw0_decomp_rmtaureal,  "irfsoln_g", irfsoln_g_base_phiw0_decomp_rmtaureal)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_base_phiw0_decomp_rmtaureal, irfsoln_g_base_phiw0_decomp_rmtaureal = nothing, nothing;
end

### Analyze rsPiQ (real rb+rm+Q)-channel for heterogeneous response regression
# Base model
if isfile(joinpath(savedsoln_dir, "irf_base_decomp_rsPiQ.jld"))
    # Load if saved
    airf_g_base_decomp_rsPiQ, irfsoln_g_base_decomp_rsPiQ = load(joinpath(savedsoln_dir, "irf_base_decomp_rsPiQ.jld"), "airf_g", "irfsoln_g");
else
    airf_g_base_decomp_rsPiQ, irfsoln_g_base_decomp_rsPiQ = solve_eqIRF_decomp("rsPiQ", airf_g_base, fsoln, aeq, modl, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_base_decomp_rsPiQ.jld"),  "airf_g", airf_g_base_decomp_rsPiQ,  "irfsoln_g", irfsoln_g_base_decomp_rsPiQ);
end
# Run regression coefs for rsPiQ, and forget irfsoln_g_base_decomp_rsPiQ
if true
    # Given the baseline solution, draw random set of firms, simulate panel in SS and IRF and compute regression coef on liq ratio
    Random.seed!(999)
    # Also, need to introduce "base rm_col" for regression coefficient scale adjustment
    irfsoln_g_base_decomp_rsPiQ.rm_col = airf_g_base.rm_col;
    # Public firms
    sim_IRFSS_anl_rand(50000,20, 60, Lin_pub, fsoln, irfsoln_g_base_decomp_rsPiQ, true, true, "_rsPiQ_pub", modl, sspace, opts)
end
# And forget in order to save RAM, to be loaded as needed later
airf_g_base_decomp_rsPiQ, irfsoln_g_base_decomp_rsPiQ = nothing, nothing;
GC.gc();

### Analyze M-channel
# Base model
if !isfile(joinpath(savedsoln_dir, "irf_base_decomp_M.jld"))
    airf_g_base_decomp_M, irfsoln_g_base_decomp_M = solve_eqIRF_decomp("M", airf_g_base, fsoln, aeq, modl, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_base_decomp_M.jld"),  "airf_g", airf_g_base_decomp_M,  "irfsoln_g", irfsoln_g_base_decomp_M)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_base_decomp_M, irfsoln_g_base_decomp_M = nothing, nothing;
end
# base_phiw0
if !isfile(joinpath(savedsoln_dir, "irf_base_phiw0_decomp_M.jld"))
    airf_g_base_phiw0_decomp_M, irfsoln_g_base_phiw0_decomp_M = solve_eqIRF_decomp("M", airf_g_base_phiw0, fsoln_base_phiw0, aeq_base_phiw0, modl_base_phiw0, sspace, opts);
    # Save full firm solution
    save(joinpath(savedsoln_dir, "irf_base_phiw0_decomp_M.jld"),  "airf_g", airf_g_base_phiw0_decomp_M,  "irfsoln_g", irfsoln_g_base_phiw0_decomp_M)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_base_phiw0_decomp_M, irfsoln_g_base_phiw0_decomp_M = nothing, nothing;
end

### Analyze MU-channel
# Base model
if !isfile(joinpath(savedsoln_dir, "irf_base_decomp_MU.jld"))
    airf_g_base_decomp_MU, irfsoln_g_base_decomp_MU = solve_eqIRF_decomp("MU", airf_g_base, fsoln, aeq, modl, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_base_decomp_MU.jld"),  "airf_g", airf_g_base_decomp_MU,  "irfsoln_g", irfsoln_g_base_decomp_MU)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_base_decomp_MU, irfsoln_g_base_decomp_MU = nothing, nothing;
end
# For hr_cf
if !isfile(joinpath(savedsoln_dir, "irf_hr_cf_decomp_MU.jld"))
    airf_g_hr_cf_decomp_MU, irfsoln_g_hr_cf_decomp_MU = solve_eqIRF_decomp("MU", airf_g_hr_cf, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts);
    # Save firm solution
    save(joinpath(savedsoln_dir, "irf_hr_cf_decomp_MU.jld"),  "airf_g", airf_g_hr_cf_decomp_MU,  "irfsoln_g", irfsoln_g_hr_cf_decomp_MU)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_hr_cf_decomp_MU, irfsoln_g_hr_cf_decomp_MU = nothing, nothing;
end
# base_phiw0
if !isfile(joinpath(savedsoln_dir, "irf_base_phiw0_decomp_MU.jld"))
    airf_g_base_phiw0_decomp_MU, irfsoln_g_base_phiw0_decomp_MU = solve_eqIRF_decomp("MU", airf_g_base_phiw0, fsoln_base_phiw0, aeq_base_phiw0, modl_base_phiw0, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_base_phiw0_decomp_MU.jld"),  "airf_g", airf_g_base_phiw0_decomp_MU,  "irfsoln_g", irfsoln_g_base_phiw0_decomp_MU)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_base_phiw0_decomp_MU, irfsoln_g_base_phiw0_decomp_MU = nothing, nothing;
end

################ Done computing decompositions

# Lines for generating Figure B.5
# Go through the three channels depicted for leverage, and save the bins values
lev_Lin_in = Lin_pub;
nbins_lev   = 10;
tval_lev    = 5;
lhs_lev = Term(Symbol("lk_dev_t$(tval_lev)"));
rhs_lev = Term(Symbol("lev"));
form_lev = lhs_lev ~ rhs_lev;
lev_yvals_decomp_Q, lev_yvals_decomp_rmtaureal, lev_yvals_decomp_rbtaureal = zeros(nbins_lev), zeros(nbins_lev), zeros(nbins_lev); 
# decomp_Q
airf_g_base_decomp_Q, irfsoln_g_base_decomp_Q = load(joinpath(savedsoln_dir, "irf_base_decomp_Q.jld"), "airf_g", "irfsoln_g");
data_lev = sim_IRFSS_respbinscat(50000, 5, lev_Lin_in, fsoln, irfsoln_g_base_decomp_Q, true, modl, sspace, opts);
bins_temp = binscatter(data_lev, form_lev, nbins_lev);
lev_yvals_decomp_Q  = bins_temp.series_list[1][:y] 
airf_g_base_decomp_Q, irfsoln_g_base_decomp_Q = Nothing, Nothing; # Delete from memory to save RAM
# decomp_rmtaureal
airf_g_base_decomp_rmtaureal, irfsoln_g_base_decomp_rmtaureal = load(joinpath(savedsoln_dir, "irf_base_decomp_rmtaureal.jld"), "airf_g", "irfsoln_g");
data_lev = sim_IRFSS_respbinscat(50000, 5, lev_Lin_in, fsoln, irfsoln_g_base_decomp_rmtaureal, true, modl, sspace, opts);
bins_temp = binscatter(data_lev, form_lev, nbins_lev);
lev_yvals_decomp_rmtaureal  = bins_temp.series_list[1][:y] 
airf_g_base_decomp_rmtaureal, irfsoln_g_base_decomp_rmtaureal = Nothing, Nothing; # Delete from memory to save RAM
# decomp_rbtaureal
airf_g_base_decomp_rbtaureal, irfsoln_g_base_decomp_rbtaureal = load(joinpath(savedsoln_dir, "irf_base_decomp_rbtaureal.jld"), "airf_g", "irfsoln_g");
data_lev = sim_IRFSS_respbinscat(50000, 5, lev_Lin_in, fsoln, irfsoln_g_base_decomp_rbtaureal, true, modl, sspace, opts);
bins_temp = binscatter(data_lev, form_lev, nbins_lev);
lev_yvals_decomp_rbtaureal  = bins_temp.series_list[1][:y] 
airf_g_base_decomp_rbtaureal, irfsoln_g_base_decomp_rbtaureal = Nothing, Nothing; # Delete from memory to save RAM
# Plot everything together in Figure B.5
manl_ftsize = 13
lev_plotx = 1:nbins_lev;
markersize_here = 4;
fg=Plots.plot(lev_plotx, lev_yvals_decomp_rbtaureal, label="Real \$r^{b}\$", linewidth=1, linecolor=:red, size=(600,400), xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-2, xlabel = "Leverage decile", ylabel = "Percent", m=(:circle, :red, markersize_here))
fg=Plots.plot!(lev_plotx, lev_yvals_decomp_rmtaureal, label="Real \$r^{m}\$", linewidth=1, linecolor=:teal, linestyle=:dot, m=(:circle, :teal, markersize_here))
fg=Plots.plot!(lev_plotx, lev_yvals_decomp_Q, label="\$Q\$", linewidth=1, linecolor=:green, linestyle=:dashdot, m=(:circle, :green, markersize_here))
fg = hline!([0], color=:black, linestyle=:dash, label=:none, linewidth = 0.5, xlims=[minimum(lev_plotx), maximum(lev_plotx)])
Plots.savefig(fg, joinpath(PROJECT_ROOT, "output", "figures", "model_figures", "FigB5_"*modl.name*"_lev_channels_pub.pdf"))
GC.gc()

# Plot decomposition of heterogeneous response coefficients -- Figure 7b
reg_coefs_base      = load(joinpath(PROJECT_ROOT, "interim_output", "model_interim_output", "anl_regcfs_vals_tempcheat_base_pub.jld"), "mod_coefs");
reg_coefs_rbtaureal = load(joinpath(PROJECT_ROOT, "interim_output", "model_interim_output", "anl_regcfs_vals_tempcheat_rbtaureal_pub.jld"), "mod_coefs");
reg_coefs_rsPiQ     = load(joinpath(PROJECT_ROOT, "interim_output", "model_interim_output", "anl_regcfs_vals_tempcheat_rsPiQ_pub.jld"), "mod_coefs");

# Plot comparison base model (capital accumulation)
manl_ftsize = 13
fg=Plots.plot(0:(length(reg_coefs_base)-1), reg_coefs_base, label="Total", ylims=(-2.8, 0.30), linewidth=3, linecolor=:blue, size=(600,400), xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-2, xlabel = "Quarters (h)", ylabel = "Percent", xticks=0:2:(length(reg_coefs_base)-1))
fg=Plots.plot!(0:(length(reg_coefs_base)-1), reg_coefs_rbtaureal, label="Real \$r^{b}\$", linewidth=2, linecolor=:red, linestyle=:dash)
fg=Plots.plot!(0:(length(reg_coefs_base)-1), reg_coefs_rsPiQ, label="Real \$r^{b} + r^{m} +Q\$", linewidth=2, linecolor=:teal, linestyle=:dashdot)
fg = hline!([0], color=:black, linestyle=:dash, label=:none, linewidth = 0.5, xlims=[0, length(reg_coefs_base)-1])
fg = Plots.plot!(0:(length(reg_coefs_base)-1), 0*(0:(length(reg_coefs_base)-1)), fillrange = reg_coefs_rbtaureal, fillalpha = 0.20, c = :red, label = :none, linealpha=0)
fg = Plots.plot!(0:(length(reg_coefs_base)-1), reg_coefs_rbtaureal, fillrange = reg_coefs_rsPiQ, fillalpha = 0.20, c = :teal, label = :none, linealpha=0)
Plots.savefig(fg, joinpath(PROJECT_ROOT, "output", "figures", "model_figures", "Fig7b_"*modl.name*"_regcfs_decomp_channels_base.pdf"))


# Plot decomposition of channels (to capital accumulation) in aggregate
if true
    # First, load relevant decompositions here
    airf_g_base = load(joinpath(savedsoln_dir, "irf_base.jld"), "airf_g");
    airf_g_base_decomp_rbtaureal = load(joinpath(savedsoln_dir, "irf_base_decomp_rbtaureal.jld"), "airf_g");
    airf_g_base_decomp_rmtaureal = load(joinpath(savedsoln_dir, "irf_base_decomp_rmtaureal.jld"), "airf_g");
    airf_g_base_decomp_Q  = load(joinpath(savedsoln_dir, "irf_base_decomp_Q.jld"), "airf_g");
    airf_g_base_decomp_MU = load(joinpath(savedsoln_dir, "irf_base_decomp_MU.jld"), "airf_g");
    airf_g_base_decomp_M  = load(joinpath(savedsoln_dir, "irf_base_decomp_M.jld"), "airf_g");

    # Plot comparison base model (capital accumulation) -- Figure 7a
    T_plot = 21;
    manl_ftsize = 13
    fg=Plots.plot(0:(T_plot-1), 100*log.(airf_g_base.Ka_col[1:T_plot]./aeq_base.Ka), label="Total", ylims=(-0.18, 0.30), linewidth=3, linecolor=:blue, size=(600,400), xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-2, xlabel = "\$t\$", ylabel = "Percent")
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_base_decomp_rbtaureal.Ka_col[1:T_plot]./aeq_base.Ka), label="Real \$r^{b}\$", linewidth=2, linecolor=:red, linestyle=:dash)
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_base_decomp_rmtaureal.Ka_col[1:T_plot]./aeq_base.Ka), label="Real \$r^{m}\$", linewidth=2, linecolor=:teal, linestyle=:dot)
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_base_decomp_M.Ka_col[1:T_plot]./aeq_base.Ka), label="\$M\$ (SDF)", linewidth=1.5, linecolor=:chocolate3, linestyle=:solid)
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_base_decomp_Q.Ka_col[1:T_plot]./aeq_base.Ka), label="\$Q\$", linewidth=2, linecolor=:green, linestyle=:dashdot)
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_base_decomp_MU.Ka_col[1:T_plot]./aeq_base.Ka), label="\$\\mathcal{M}\$", linewidth=1.5, linecolor=:purple3, linestyle=:solid)
    fg = hline!([0], color=:black, linestyle=:dash, label=:none, linewidth = 0.5, xlims=[0, T_plot-1])
    Plots.savefig(fg, joinpath(PROJECT_ROOT, "output", "figures", "model_figures", "Fig7a_"*modl.name*"_IRFKa_decomp_channels_base.pdf"));

    # Repeat for hr_cf model
    # First, load relevant decompositions here
    airf_g_hr_cf = load(joinpath(savedsoln_dir, "irf_hr_cf.jld"), "airf_g");
    airf_g_hr_cf_decomp_rbtaureal = load(joinpath(savedsoln_dir, "irf_hr_cf_decomp_rbtaureal.jld"), "airf_g");
    airf_g_hr_cf_decomp_rmtaureal = load(joinpath(savedsoln_dir, "irf_hr_cf_decomp_rmtaureal.jld"), "airf_g");
    airf_g_hr_cf_decomp_Q  = load(joinpath(savedsoln_dir, "irf_hr_cf_decomp_Q.jld"), "airf_g");
    airf_g_hr_cf_decomp_MU = load(joinpath(savedsoln_dir, "irf_hr_cf_decomp_MU.jld"), "airf_g");

    # Plot comparison base model (capital accumulation) -- Figure B.8
    T_plot = 21;
    manl_ftsize = 13
    fg=Plots.plot(0:(T_plot-1), 100*log.(airf_g_hr_cf.Ka_col[1:T_plot]./aeq_hr_cf.Ka), label="Total", ylims=(-0.18, 0.30), linewidth=3, linecolor=:blue, size=(600,400), xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-2, xlabel = "\$t\$", ylabel = "Percent")
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_hr_cf_decomp_rbtaureal.Ka_col[1:T_plot]./aeq_hr_cf.Ka), label="Real \$r^{b}\$", linewidth=2, linecolor=:red, linestyle=:dash)
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_hr_cf_decomp_rmtaureal.Ka_col[1:T_plot]./aeq_hr_cf.Ka), label="Real \$r^{m}\$", linewidth=2, linecolor=:teal, linestyle=:dot)
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_hr_cf_decomp_Q.Ka_col[1:T_plot]./aeq_hr_cf.Ka), label="\$Q\$", linewidth=2, linecolor=:green, linestyle=:dashdot)
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_hr_cf_decomp_MU.Ka_col[1:T_plot]./aeq_hr_cf.Ka), label="\$\\mathcal{M}\$", linewidth=1.5, linecolor=:purple3, linestyle=:solid)
    fg = hline!([0], color=:black, linestyle=:dash, label=:none, linewidth = 0.5, xlims=[0, T_plot-1])
    Plots.savefig(fg, joinpath(PROJECT_ROOT, "output", "figures", "model_figures", "FigB8_"*modl.name*"_IRFKa_decomp_channels_hr_cf.pdf"))

    # Repeat loading for hr_cf_basephimresp model
    airf_g_hr_cf_basephimresp = load(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp.jld"), "airf_g");
    airf_g_hr_cf_basephimresp_decomp_rmtaureal = load(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp_decomp_rmtaureal.jld"), "airf_g");

    # Also load PE solutions
    airf_g_hr_cf_basephimresp_noGE = load(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp_noGE.jld"), "airf_g");
    airf_g_base_hr_cf_basephimresp_noGE = load(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE.jld"), "airf_g");

    # Compare at some specific horizon
    # K4 -- Figure 8a
    hor_K = 5;
    fg=Plots.plot(1:3, 100*[log.(airf_g_hr_cf.Ka_col[hor_K]./aeq_hr_cf.Ka) log.(airf_g_hr_cf_basephimresp.Ka_col[hor_K]./aeq_hr_cf.Ka) log.(airf_g_base.Ka_col[hor_K]./aeq_base.Ka)]', label="\$\\hat{K}_{4}^{GE}\$", ylims=(-0.0230, -0.0125), linewidth=1, linecolor=:blue, m=(:diamond, :blue, 6), size=(600,400), xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-2, ylabel = "Percent", legend=:topleft, xticks=(1:3, ["Pre-90", "Pre-90-\$\\phi_{H}^{m}\$", "Post-90"]), xlims = [0.8, 3.2], linestyle=:dash)
    fg=Plots.plot!(1:3, 100*[log.(airf_g_hr_cf.Ka_col[hor_K]./aeq_hr_cf.Ka) log.(airf_g_hr_cf_basephimresp_noGE.Ka_col[hor_K]./aeq_hr_cf.Ka) log.(airf_g_base_hr_cf_basephimresp_noGE.Ka_col[hor_K]./aeq_base.Ka)]', label="\$\\hat{K}_{4}^{PE}\$", linewidth=1, linecolor=:green, linestyle=:dot, m=(:circle, :green, 6))
    Plots.savefig(fg, joinpath(PROJECT_ROOT, "output", "figures", "model_figures", "Fig8a_"*modl.name*"_IRFKa4_comp.pdf"))
    
    # Now, compare rmtaureal channel across the three relevant models
    # Plot comparison -- Figure 8b
    T_plot = 9;
    manl_ftsize = 13
    fg=Plots.plot(0:(T_plot-1), 100*log.(airf_g_hr_cf_decomp_rmtaureal.Ka_col[1:T_plot]./aeq_hr_cf.Ka), label="Pre-90", ylims=(-0.078, 0.01), linewidth=2, linecolor=:midnightblue, size=(600,400), xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-2, xlabel = "\$t\$", ylabel = "Percent", linestyle = :dash)
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_hr_cf_basephimresp_decomp_rmtaureal.Ka_col[1:T_plot]./aeq_hr_cf.Ka), label="Pre-90-\$\\phi_{H}^{m}\$", linewidth=2, linecolor=:steelblue2, linestyle=:dashdot)
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_base_decomp_rmtaureal.Ka_col[1:T_plot]./aeq_base.Ka), label="Post-90", linewidth=2, linecolor=:blue, linestyle=:solid)
    fg = hline!([0], color=:black, linestyle=:dash, label=:none, linewidth = 0.5, xlims=[0, T_plot-1])
    Plots.savefig(fg, joinpath(PROJECT_ROOT, "output", "figures", "model_figures", "Fig8b_"*modl.name*"_IRFKa_decomp_rmtaureal_comp.pdf"))

    # Compute responses relative to hr_cf
    Ka_relresp_hr_cf_basephimresp = log.(airf_g_hr_cf_basephimresp_decomp_rmtaureal.Ka_col[1:T_plot]./aeq_hr_cf.Ka) ./ log.(airf_g_hr_cf_decomp_rmtaureal.Ka_col[1:T_plot]./aeq_hr_cf.Ka)
    Ka_relresp_base = log.(airf_g_base_decomp_rmtaureal.Ka_col[1:T_plot]./aeq_base.Ka) ./ log.(airf_g_hr_cf_decomp_rmtaureal.Ka_col[1:T_plot]./aeq_hr_cf.Ka)

    # Finally, compare the rmtaureal to relative importance of rbtaureal
    airf_g_hr_cf_decomp_rbtaureal = load(joinpath(savedsoln_dir, "irf_hr_cf_decomp_rbtaureal.jld"), "airf_g");
    airf_g_base_decomp_rbtaureal = load(joinpath(savedsoln_dir, "irf_base_decomp_rbtaureal.jld"), "airf_g");

    # Consider the peak horizon of rm effect, and round the shares to the nearest fourth, as stated in Section 5.2 of the paper
    t_rmpeak = 3;
    pre90_rmrb_peak  = log.(airf_g_hr_cf_decomp_rmtaureal.Ka_col[t_rmpeak]./aeq_hr_cf.Ka) ./ log.(airf_g_hr_cf_decomp_rbtaureal.Ka_col[t_rmpeak]./aeq_hr_cf.Ka);
    post90_rmrb_peak = log.(airf_g_base_decomp_rmtaureal.Ka_col[t_rmpeak]./aeq_base.Ka) ./ log.(airf_g_base_decomp_rbtaureal.Ka_col[t_rmpeak]./aeq_base.Ka);
    pre90_rmrb_peak_ratl  = rationalize(round(pre90_rmrb_peak*4)/4)
    post90_rmrb_peak_ratl = rationalize(round(post90_rmrb_peak*4)/4)
    # Print relevant output
    open(joinpath(PROJECT_ROOT, "output", "other", "Text_Sec5p2_rmrb_peak.txt"), "w") do io
        println(io, "Pre-90: at its peak effect horizon (t=$(t_rmpeak-1)), the rm channel was about $(round(pre90_rmrb_peak,digits=2)), or $(numerator(pre90_rmrb_peak_ratl))/$(denominator(pre90_rmrb_peak_ratl)) as important as rb.")
        println(io, "Post-90: at its peak effect horizon (t=$(t_rmpeak-1)), the rm channel was about $(round(post90_rmrb_peak,digits=2)), or $(numerator(post90_rmrb_peak_ratl))/$(denominator(post90_rmrb_peak_ratl)) as important as rb.")
    end
end

# Compute the relative role of the different channels in explaining the (post-1990) gap, as reported in Section 5.2 of paper
if true
    # Load all the relevant agg responses
    # GE and channels for base
    airf_g_base = load(joinpath(savedsoln_dir, "irf_base.jld"), "airf_g");
    airf_g_base_decomp_rbtaureal = load(joinpath(savedsoln_dir, "irf_base_decomp_rbtaureal.jld"), "airf_g");
    airf_g_base_decomp_rmtaureal = load(joinpath(savedsoln_dir, "irf_base_decomp_rmtaureal.jld"), "airf_g");
    airf_g_base_decomp_Q  =  load(joinpath(savedsoln_dir, "irf_base_decomp_Q.jld"), "airf_g");
    # noGE (PE) and channels for the PE case
    airf_g_base_hr_cf_basephimresp_noGE = load(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE.jld"), "airf_g");
    airf_g_base_hr_cf_basephimresp_noGE_decomp_Q =  load(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE_decomp_Q.jld"), "airf_g");

    # Compute full gap in K4
    GEnoGE_K4_gap = 100*log(airf_g_base.Ka_col[5] / aeq.Ka) - 100*log(airf_g_base_hr_cf_basephimresp_noGE.Ka_col[5] / aeq.Ka) 
    # And the part explained by the Q channel, as reported in Section 5.2
    GEnoGE_K4_Q_part = 100*log(airf_g_base_decomp_Q.Ka_col[5] / aeq.Ka) - 100*log(airf_g_base_hr_cf_basephimresp_noGE_decomp_Q.Ka_col[5] / aeq.Ka) 
    # And finally, the corresponding share
    GEnoGE_K4_Q_share = GEnoGE_K4_Q_part / GEnoGE_K4_gap
    open(joinpath(PROJECT_ROOT, "output", "other", "Text_Sec5p2_GEPEshare.txt"), "w") do io
        println(io, "GE-PE gap share explained by Q channel: $(round(GEnoGE_K4_Q_share, digits=2))")
    end

    # Compute the size of increase in Ma, relative to initial NW (as expressed in footnote 57)
    NWagg_hr_cf = aeq_hr_cf.Ma + aeq_hr_cf.Ka + aeq_hr_cf.Ba
    Ma_diff     = aeq.Ma - aeq_hr_cf.Ma
    Ma_diff_relNW = Ma_diff / NWagg_hr_cf;
    open(joinpath(PROJECT_ROOT, "output", "other", "Footnote57_justification.txt"), "w") do io
        println(io, "Aggregate M increase pre-to-post-90 as share of net worth: $(round(100*Ma_diff_relNW, digits=1))%.")
    end
end

# Plot decomposition of channels (to capital accumulation) in base_phiw0
if true
    # First, load relevant decompositions here
    airf_g_base_phiw0 = load(joinpath(savedsoln_dir, "irf_base_phiw0.jld"), "airf_g");
    airf_g_base_phiw0_decomp_rbtaureal = load(joinpath(savedsoln_dir, "irf_base_phiw0_decomp_rbtaureal.jld"), "airf_g");
    airf_g_base_phiw0_decomp_rmtaureal = load(joinpath(savedsoln_dir, "irf_base_phiw0_decomp_rmtaureal.jld"), "airf_g");
    airf_g_base_phiw0_decomp_Q  = load(joinpath(savedsoln_dir, "irf_base_phiw0_decomp_Q.jld"), "airf_g");
    airf_g_base_phiw0_decomp_MU = load(joinpath(savedsoln_dir, "irf_base_phiw0_decomp_MU.jld"), "airf_g");
    airf_g_base_phiw0_decomp_M  = load(joinpath(savedsoln_dir, "irf_base_phiw0_decomp_M.jld"), "airf_g");

    # Plot comparison base model (capital accumulation) -- Figure B.11b
    T_plot = 21;
    manl_ftsize = 13
    fg=Plots.plot(0:(T_plot-1), 100*log.(airf_g_base_phiw0.Ka_col[1:T_plot]./aeq_base_phiw0.Ka), label="Total", ylims=(-0.18, 0.30), linewidth=3, linecolor=:blue, size=(600,400), xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-2, xlabel = "\$t\$", ylabel = "Percent")
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_base_phiw0_decomp_rbtaureal.Ka_col[1:T_plot]./aeq_base_phiw0.Ka), label="Real \$r^{b}\$", linewidth=2, linecolor=:red, linestyle=:dash)
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_base_phiw0_decomp_rmtaureal.Ka_col[1:T_plot]./aeq_base_phiw0.Ka), label="Real \$r^{m}\$", linewidth=2, linecolor=:teal, linestyle=:dot)
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_base_phiw0_decomp_M.Ka_col[1:T_plot]./aeq_base_phiw0.Ka), label="\$M\$ (SDF)", linewidth=1.5, linecolor=:chocolate3, linestyle=:solid)
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_base_phiw0_decomp_Q.Ka_col[1:T_plot]./aeq_base_phiw0.Ka), label="\$Q\$", linewidth=2, linecolor=:green, linestyle=:dashdot)
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_base_phiw0_decomp_MU.Ka_col[1:T_plot]./aeq_base_phiw0.Ka), label="\$\\mathcal{M}\$", linewidth=1.5, linecolor=:purple3, linestyle=:solid)
    fg = hline!([0], color=:black, linestyle=:dash, label=:none, linewidth = 0.5, xlims=[0, T_plot-1])
    Plots.savefig(fg, joinpath(PROJECT_ROOT, "output", "figures", "model_figures", "FigB11b_"*modl.name*"_phiw0_IRFKa_decomp_channels_base.pdf"))
end

# Finished analysis of monetary shock analysis in base model. Forget airf_* objects to save RAM.
airf_g_base_decomp_MU, airf_g_base_decomp_Q, airf_g_base_decomp_rbtaureal, airf_g_base_decomp_rmtaureal, airf_g_base_decomp_M  = nothing, nothing, nothing, nothing, nothing;
airf_g_hr_cf_decomp_MU, airf_g_hr_cf_decomp_Q, airf_g_hr_cf_decomp_rbtaureal, airf_g_hr_cf_decomp_rmtaureal  = nothing, nothing, nothing, nothing;
airf_g_hr_cf_basephimresp_decomp_rmtaureal = nothing;
airf_g_base_phiw0_decomp_MU, airf_g_base_phiw0_decomp_Q, airf_g_base_phiw0_decomp_rbtaureal, airf_g_base_phiw0_decomp_rmtaureal, airf_g_base_phiw0_decomp_M  = nothing, nothing, nothing, nothing, nothing;
airf_g_base_hr_cf_basephimresp_noGE_decomp_Q  = nothing;
GC.gc();


###
# rm-shock analysis (as stated in Section 5.2 and Appendix B.10.3)
###

# Consider an rm shock (rm up, while rf, i.e., Taylor rule not explicitly shocked, meaning zeta=0)
zeta_col   = 0.0*ones(T_g)
spread_col = (modl.rb-modl.rf)*ones(T_g)
# AR1 shock to rm
rfzscaler = 4.0 # set to =1, for 25bp (ann), to =4 for 100bp (ann)
# Autocorrelation of shock of 0.61, like for baseline monetary shock:
rmsh_col = gen_ar1(T_g,1,T_g,0.0,0.61,rfzscaler*25/(4*10000))

# Create placeholders for relevant variables created in loop
airf_g_base_rmsh, airf_g_hr_cf_rmsh = nothing, nothing;
MU_in, pQ_in = nothing, nothing;

# Introduce guesses
for IRF_indic in ["base", "hr_cf"]
    # Set guesses
    local MU_in, pQ_in
    if IRF_indic == "base"
        # Pre-saved guess, converged at 2*1e-5:
        MU_in = [1.1112884444821767, 1.1115437094549998, 1.111356220743115, 1.1112781961473797, 1.1112042486371794, 1.1111649619357125, 1.1111394723402233, 1.111123724246413, 1.1111160939923057, 1.1111109583181262, 1.1111096458046545, 1.1111090179737968, 1.1111086951573552, 1.1111097997439774, 1.1111102741767422, 1.1111106676855602, 1.111111003964312, 1.1111112569196009, 1.1111114450096475, 1.1111115973492358, 1.1111117261778212, 1.1111118333425583, 1.1111119229182627, 1.1111119984726112, 1.1111120612093508, 1.1111121108650002, 1.1111121459145261, 1.1111121645445179, 1.11111216489288, 1.1111121449937797, 1.1111121018596022, 1.1111120304985143, 1.111111927663183, 1.111111792882516, 1.1111116260576905, 1.1111114282958543, 1.1111112022690965, 1.1111109522853082, 1.1111106843368572, 1.1111104060696027, 1.1111101267694459, 1.111109857245271, 1.1111096096836859, 1.1111093973772284, 1.1111092344588438, 1.1111091354534006, 1.1111091148816896, 1.1111091866568412, 1.111109363574763, 1.1111096565727914, 1.1111100741211206, 1.1111106214607742, 1.1111112999327366, 1.1111121063239597, 1.111113032443486, 1.1111140640348613, 1.1111151815245486, 1.1111163621306295, 1.1111175595260971, 1.1111186897508911, 1.1111111111111112];
        pQ_in = [0.9985690452999758, 0.9995876415434033, 0.9999868737157599, 1.0001466219630062, 1.000208640783462, 1.000227805743152, 1.0002231197562987, 1.000205985238265, 1.0001830765074426, 1.0001606611580005, 1.00013893114508, 1.0001190861001348, 1.0001021603513103, 1.0000867531079916, 1.0000734350505405, 1.0000619722878201, 1.000052154265667, 1.000043750082929, 1.0000365396909203, 1.0000303342456773, 1.000024987516208, 1.0000203827748577, 1.0000164190364431, 1.0000130074012021, 1.0000100701697145, 1.0000075401398776, 1.000005359413954, 1.0000034782004223, 1.0000018541405575, 1.0000004518519343, 0.9999992421065325, 0.9999982008327526, 0.9999973083831877, 0.9999965486859338, 0.9999959083861829, 0.9999953763985622, 0.999994943513236, 0.9999946020750732, 0.9999943456774758, 0.9999941689059278, 0.9999940670841998, 0.9999940360483436, 0.9999940719126372, 0.9999941708629109, 0.9999943289635149, 0.9999945419829629, 0.999994805207587, 0.9999951133309493, 0.9999954603018896, 0.9999958392827984, 0.9999962425659424, 0.9999966616241323, 0.9999970871100954, 0.999997509043632, 0.9999979167953471, 0.9999982999822018, 0.9999986484301641, 0.9999989504486985, 0.9999992158922897, 0.9999995057565026, 1.0];
    elseif IRF_indic == "hr_cf"
        # Pre-saved guess, converged at 2*1e-5:
        MU_in = [1.1111392407561183, 1.1114942828457197, 1.1114278909804607, 1.1113390716626776, 1.1112624853403599, 1.1112101244679553, 1.1111751092969304, 1.111148866440442, 1.1111384597094076, 1.111126693883493, 1.1111206824667406, 1.1111158115406603, 1.1111123150596762, 1.111110810555906, 1.1111102659675784, 1.1111099978123005, 1.1111097652624418, 1.1111097752233388, 1.1111098801249695, 1.111110029911847, 1.1111101935529712, 1.1111103562044973, 1.111110512842701, 1.1111106614085842, 1.111110800614296, 1.111110929359892, 1.1111110469732, 1.1111111538521754, 1.1111112506881284, 1.111111338107922, 1.1111114164005036, 1.1111114853919621, 1.111111544513426, 1.1111115929333024, 1.1111116294781347, 1.1111116529774934, 1.1111116623228026, 1.1111116565536825, 1.1111116349722379, 1.1111115972636774, 1.111111543584622, 1.1111114746950497, 1.1111113919762454, 1.1111112974714534, 1.1111111939805183, 1.1111110848236245, 1.1111109739484875, 1.1111108657729183, 1.1111107650046126, 1.111110676450824, 1.1111106047656019, 1.1111105542518436, 1.1111105284259992, 1.1111105300012636, 1.1111105603048346, 1.1111106196068123, 1.1111107069025765, 1.1111108179398628, 1.1111109545748832, 1.1111111604215838, 1.1111111111111112];
        pQ_in = [0.9989160964022861, 0.9996991461435228, 0.9999719940816792, 1.0000830306138317, 1.0001213201189205, 1.0001286684294557, 1.0001213904075201, 1.0001094416409522, 1.0000957912599155, 1.0000839074496135, 1.0000728364517384, 1.0000635319135995, 1.0000560357573904, 1.0000497158069832, 1.0000441311781825, 1.0000393490244777, 1.0000353386828065, 1.0000318933110623, 1.0000288997068676, 1.000026289930776, 1.0000240000586393, 1.0000219762720937, 1.0000201755795184, 1.000018564937545, 1.0000171178395774, 1.000015812568046, 1.0000146301837778, 1.0000135540464177, 1.0000125700104137, 1.0000116661299834, 1.0000108322261498, 1.0000100596187695, 1.0000093409992867, 1.0000086702126734, 1.000008042076924, 1.000007452268477, 1.0000068972099443, 1.0000063739754543, 1.0000058802135616, 1.0000054140793808, 1.0000049741679289, 1.000004559456727, 1.000004169229814, 1.0000038030080525, 1.0000034604593129, 1.0000031413288673, 1.0000028453864027, 1.000002572321044, 1.0000023216568361, 1.0000020926875839, 1.0000018844056886, 1.0000016954406405, 1.000001524025467, 1.0000013679204907, 1.0000012243660001, 1.0000010900004008, 1.00000096062589, 1.0000008316316245, 1.0000006937510362, 1.0000005161323835, 1.0];
    end

    # Adjust lengths of guesses
    if length(MU_in) < T_g; MU_in = vcat(MU_in, MU_ss*ones(T_g-length(MU_in))); end
    if length(MU_in) > T_g; MU_in = MU_in[1:T_g]; end
    if length(pQ_in) < T_g; pQ_in = vcat(pQ_in, ones(T_g-length(pQ_in))); end
    if length(pQ_in) > T_g; pQ_in = pQ_in[1:T_g]; end

    # Load solutions, if exist. Otherwise, solve IRF.
    if (IRF_indic=="base") & isfile(joinpath(savedsoln_dir, "irf_base_rmsh_100bp.jld"))
        global airf_g_base_rmsh = load(joinpath(savedsoln_dir, "irf_base_rmsh_100bp.jld"), "airf_g");
    elseif (IRF_indic=="hr_cf") & isfile(joinpath(savedsoln_dir, "irf_hr_cf_rmsh_100bp.jld"))
        global airf_g_hr_cf_rmsh = load(joinpath(savedsoln_dir, "irf_hr_cf_rmsh_100bp.jld"), "airf_g");
    elseif (IRF_indic == "base")
        global airf_g_base_rmsh, irfsoln_g_base_rmsh = solve_eqIRF_NKMg_xsh_rmsh(0.000, zeta_col, spread_col, rmsh_col, fsoln, aeq, modl, sspace, opts, MU_col_0=MU_in, pQ_col_0=pQ_in);
        # Save to file
        save(joinpath(savedsoln_dir, "irf_base_rmsh_100bp.jld"),  "airf_g", airf_g_base_rmsh,  "irfsoln_g", irfsoln_g_base_rmsh)
        # Since irfsoln is not used, forget it
        global irfsoln_g_base_rmsh = nothing;
    elseif (IRF_indic == "hr_cf")
        global airf_g_hr_cf_rmsh, irfsoln_g_hr_cf_rmsh = solve_eqIRF_NKMg_xsh_rmsh(0.000, zeta_col, spread_col, rmsh_col, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts, MU_col_0=MU_in, pQ_col_0=pQ_in, w_ss=modl_hr_cf.psi/aeq_hr_cf.p);
        # Save to file
        save(joinpath(savedsoln_dir, "irf_hr_cf_rmsh_100bp.jld"),  "airf_g", airf_g_hr_cf_rmsh,  "irfsoln_g", irfsoln_g_hr_cf_rmsh)
        # Since irfsoln is not used, forget it
        global irfsoln_g_hr_cf_rmsh = nothing;
    end

end

# Construct agg investment series
Ia_col_base_rmsh = airf_g_base_rmsh.Ka_col[2:end]-(1.0-modl.delta)*airf_g_base_rmsh.Ka_col[1:(end-1)]
Ia_col_hr_cf_rmsh = airf_g_hr_cf_rmsh.Ka_col[2:end]-(1.0-modl.delta)*airf_g_hr_cf_rmsh.Ka_col[1:(end-1)]

# Construct comparison of one specific variable across the three relevant models (in GE) -- Figure B.9
Tmax = 8;
manl_ftsize = 13
# Plot
fg = Plots.plot(0:Tmax, [100*(log.(airf_g_hr_cf_rmsh.Ya_col[1:Tmax+1]).-log(aeq_hr_cf.Ya)), 100*(log.(Ia_col_hr_cf_rmsh[1:Tmax+1]).-log(modl.delta*aeq_hr_cf.Ka)), 100*(log.(airf_g_hr_cf_rmsh.Ka_col[1:Tmax+1]).-log(aeq_hr_cf.Ka))], layout=(1,3), title=[L"Y" "Investment" "Capital"], label=["Pre-90" :none :none ], legend=[:bottomright :none :none ], size=(800,230), linewidth=2, linecolor=:midnightblue, xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-2, linestyle = :dash)
fg = Plots.plot!(0:Tmax, 100*(log.(airf_g_base_rmsh.Ya_col[1:Tmax+1]).-log(aeq_base.Ya)), label="Post-90", subplot=1, linewidth=2, linecolor=:blue, linestyle=:solid)
fg = Plots.plot!(0:Tmax, 100*(log.(Ia_col_base_rmsh[1:Tmax+1]).-log(modl.delta*aeq_base.Ka)), subplot=2, linewidth=2, linecolor=:blue, linestyle=:solid)
fg = Plots.plot!(0:Tmax, 100*(log.(airf_g_base_rmsh.Ka_col[1:Tmax+1]).-log(aeq_base.Ka)), subplot=3, linewidth=2, linecolor=:blue, linestyle=:solid)
fg = hline!([0 0 0 0 0 0], subplot=1:3, color=:black, linestyle=:dash, label=:none)
ylims!(fg[1], -0.073, 0.016)
ylims!(fg[2], -0.61, 0.09)
ylims!(fg[3], -0.02, 0.001)
Plots.savefig(fg, joinpath(PROJECT_ROOT, "output", "figures", "model_figures", "FigB9_"*modl.name*"_rmsh_comp_100bp.pdf"))
# Print text
open(joinpath(PROJECT_ROOT, "output", "other", "Text_Sec5p2_rmshresps.txt"), "w") do io
    println(io, "Pre-1990 impact response in Ia to rm: $(round(100 * (log(Ia_col_hr_cf_rmsh[1]) - log(modl.delta * aeq_hr_cf.Ka)), digits=2))%")
    println(io, "Post-1990 impact response in Ia to rm: $(round(100 * (log(Ia_col_base_rmsh[1]) - log(modl.delta * aeq_base.Ka)), digits=2))%")
end


#######
# Lines to replicate core analysis with phi_k=1/8, adjusting/overwriting model objects from above (Appendix B.11.2)
#######
if true
    # AR1 rm
    zeta_col = gen_ar1(T_g,1,T_g,0.0,0.61,25/(4*10000))
    spread_col = 0.0*zeta_col + (modl.rb-modl.rf)*ones(T_g)
end

# Compute baseline, hr_cf, and hr_cf_basephimresp GE, and PE responses
# Loop over "IRF indicators": ["base", "hr_cf", "hr_cf_basephimresp", "hr_cf_basephimresp_noGE", "base_hr_cf_basephimresp_noGE"]
# Overwrite corresponding existing solution objects
for IRF_indic in ["base", "hr_cf", "hr_cf_basephimresp", "hr_cf_basephimresp_noGE", "base_hr_cf_basephimresp_noGE"]
    # Set guesses
    local MU_in, pQ_in
    if true
        if IRF_indic == "base"
            # Pre-saved guess, converged at 2*1e-5:
            MU_in = [1.11331125856659, 1.1123808510417992, 1.1118822876503427, 1.1115719748456274, 1.1113847410514515, 1.111271234519829, 1.111202277710887, 1.111161281183344, 1.1111374500051925, 1.1111235984563983, 1.1111151493940323, 1.1111101729443416, 1.1111083524761782, 1.1111078082876507, 1.1111062213369804, 1.111106150564832, 1.1111053676722797, 1.111105067326226, 1.11110497709138, 1.1111050190633973, 1.111105118382892, 1.1111052491395372, 1.111105396096621, 1.1111055565806929, 1.1111057330081857, 1.111105926583755, 1.1111061363877088, 1.1111063623615212, 1.1111066081384773, 1.1111068785491902, 1.1111071883302155, 1.1111075257944054, 1.1111078922946156, 1.111108286978118, 1.1111087069529013, 1.1111091481714432, 1.1111096056790672, 1.1111100728808834, 1.1111105414102695, 1.1111110009882468, 1.1111114489761913, 1.111111860126262, 1.1111122221784686, 1.1111125183948705, 1.1111127308746906, 1.1111128412669236, 1.111112831342298, 1.1111126835760523, 1.1111123819087527, 1.111111912518722, 1.1111112648500208, 1.1111104323017003, 1.111109413642325, 1.1111082131240153, 1.1111068420828867, 1.1111053195429001, 1.1111036747627752, 1.1111019258412587, 1.1111001873359998, 1.1110985100651336, 1.1111111111111112];
            pQ_in = [0.9991645212952247, 0.9995230045493761, 0.9997349451497635, 0.9998633345173974, 0.9999410230357242, 0.999987842653463, 1.000015928217913, 1.0000325620530552, 1.0000421677528055, 1.0000474470654146, 1.000049910385457, 1.0000506721651545, 1.0000504076548153, 1.0000494443533796, 1.0000486280735235, 1.0000472833518463, 1.0000460842421828, 1.0000447208693661, 1.000043290726687, 1.0000418201103232, 1.0000403421424129, 1.000038877904861, 1.000037442102674, 1.0000360419449181, 1.000034680492473, 1.0000333595909274, 1.0000320809120298, 1.000030845220315, 1.00002965138887, 1.0000284969099993, 1.0000273796503032, 1.000026295821376, 1.0000252431395138, 1.000024218014002, 1.000023216855327, 1.0000222358014659, 1.0000212706866662, 1.0000203173723818, 1.0000193719006292, 1.0000184304952506, 1.0000174902817567, 1.0000165472902947, 1.0000155993505986, 1.0000146447811546, 1.0000136827962556, 1.0000127136045844, 1.000011738572359, 1.000010760261484, 1.0000097825936545, 1.0000088108196665, 1.0000078516201576, 1.0000069129686655, 1.0000060041635694, 1.0000051354525985, 1.0000043182117584, 1.0000035633294664, 1.0000028815004887, 1.000002285601586, 1.0000017543629072, 1.0000011622406588, 1.0];
        elseif IRF_indic == "hr_cf"
            # Pre-saved guess, converged at 2*1e-5:
            MU_in = [1.1133536571971518, 1.112405614598436, 1.1118963617902466, 1.1115835082155803, 1.1113932022095072, 1.111277379790177, 1.1112069377176548, 1.1111644694359233, 1.1111391232536156, 1.111124041483334, 1.111114890352732, 1.1111093990603806, 1.1111064686386976, 1.1111048763725744, 1.1111040712965332, 1.1111037259001735, 1.1111036433336792, 1.1111037066578802, 1.1111038463791778, 1.1111040226551647, 1.111104213799172, 1.1111044081963823, 1.1111046017278883, 1.1111047941918042, 1.1111049879510762, 1.1111051868407582, 1.1111053955215633, 1.1111056189716775, 1.1111058620777805, 1.1111061292877353, 1.1111064284294194, 1.1111067579385159, 1.1111071200504932, 1.111107514823279, 1.1111079388623133, 1.1111083901532046, 1.1111088643858675, 1.1111093547960182, 1.1111098525494318, 1.111110346826045, 1.1111108282546607, 1.1111112785611956, 1.111111682727551, 1.1111120235760505, 1.1111122830824447, 1.111112442972138, 1.1111124852924859, 1.111112393077856, 1.1111121511336794, 1.111111746834582, 1.1111111707277894, 1.111110418381754, 1.1111094895964584, 1.1111083922245621, 1.1111071401925252, 1.1111057594246307, 1.111104286918137, 1.1111027482037086, 1.1111012599072512, 1.1111000052114823, 1.1111111111111112];
            pQ_in = [0.9991570529023017, 0.9995177064999327, 0.9997335561820649, 0.9998634393867041, 0.9999423166859044, 0.9999901405934489, 1.0000190080983895, 1.0000361759178968, 1.0000461104926843, 1.0000515684518365, 1.0000542015920706, 1.0000550902855814, 1.0000549193205919, 1.0000540699316203, 1.0000527999311024, 1.0000512774103325, 1.0000496106266452, 1.0000478687720002, 1.0000460968170908, 1.0000443249123163, 1.0000425738866452, 1.000040858114786, 1.0000391873899912, 1.000037568069275, 1.0000360038891147, 1.0000344966515873, 1.000033046696222, 1.0000316532640214, 1.0000303147320255, 1.0000290287772309, 1.000027792729882, 1.0000266029627254, 1.0000254559037394, 1.0000243476718234, 1.0000232745308164, 1.0000222316225835, 1.0000212142278906, 1.0000202178623723, 1.0000192382334219, 1.0000182712678831, 1.000017313409722, 1.0000163610570971, 1.0000154115586193, 1.0000144628805936, 1.0000135138078363, 1.0000125640497435, 1.0000116143939985, 1.0000106667426596, 1.0000097242727728, 1.000008791361255, 1.0000078738298706, 1.000006978527234, 1.000006113958013, 1.000005288893958, 1.0000045134565572, 1.0000037957245091, 1.0000031424803955, 1.0000025614840764, 1.0000020270303405, 1.0000013764242248, 1.0];
        elseif IRF_indic == "hr_cf_basephimresp"
            # Pre-saved guess, converged at 2*1e-5:
            MU_in = [1.1133601591283357, 1.112408964589489, 1.1118995237528102, 1.1115863279657479, 1.1113955895043932, 1.1112794985013006, 1.1112089103132217, 1.1111662497315185, 1.1111406596982714, 1.1111253603323301, 1.1111161417863271, 1.111110646531355, 1.111107606808965, 1.1111059325617723, 1.1111050670684905, 1.1111046753619673, 1.1111045545624658, 1.1111045833534403, 1.1111046892134546, 1.1111048299495991, 1.1111049818873873, 1.1111051321747167, 1.1111052757827513, 1.1111054119919999, 1.1111055431780787, 1.111105673670164, 1.1111058089255965, 1.1111059553380789, 1.1111061192962521, 1.1111063062433444, 1.1111065251588712, 1.111106775141856, 1.111107061133969, 1.1111073860454495, 1.1111077504839424, 1.1111081527771716, 1.1111085891796246, 1.1111090536816655, 1.1111095379553886, 1.1111100312153637, 1.1111105222543827, 1.1111109934468082, 1.1111114278503813, 1.11111180630997, 1.111112108401601, 1.1111123130494431, 1.1111123992545124, 1.1111123469071396, 1.1111121375936286, 1.1111117558426085, 1.1111111897390473, 1.111110433146501, 1.1111094850283367, 1.1111083540164288, 1.1111070557293028, 1.1111056205347636, 1.1111040913943337, 1.1111025017675902, 1.1111009855454401, 1.1110997953537252, 1.1111111111111112];
            pQ_in = [0.9991500580768077, 0.9995129728276506, 0.9997305863780785, 0.9998616332347239, 0.999941375228503, 0.9999898800856644, 1.00001928683242, 1.0000368684375585, 1.000047114556164, 1.0000528086310279, 1.0000556240090395, 1.0000566442581154, 1.0000565519704852, 1.0000557465223674, 1.0000544924941215, 1.000052962909612, 1.000051270293765, 1.0000494875618722, 1.0000476628871633, 1.000045829148325, 1.0000440094430898, 1.000042220156458, 1.0000404726863206, 1.0000387747745227, 1.0000371313038516, 1.0000355450011889, 1.000034016938386, 1.0000325468939577, 1.0000311336346148, 1.0000297753970593, 1.000028469319819, 1.0000272131560806, 1.0000260030336006, 1.0000248345484868, 1.0000237033753272, 1.000022605141198, 1.0000215354358921, 1.0000204899103007, 1.0000194643839415, 1.0000184549553486, 1.0000174582301506, 1.0000164710406003, 1.0000154910869308, 1.0000145167683494, 1.000013547356299, 1.0000125830691706, 1.0000116251853504, 1.0000106760560747, 1.0000097392304041, 1.0000088193056702, 1.0000079221397769, 1.0000070542379491, 1.000006223440248, 1.000005437134084, 1.0000047034997082, 1.000004027587913, 1.0000034122496857, 1.0000028609538412, 1.0000023370825728, 1.000001649856855, 1.0];
        elseif (IRF_indic == "base_hr_cf_basephimresp_noGE") | (IRF_indic == "hr_cf_basephimresp_noGE")
            MU_in = airf_g_hr_cf.MU_col;
            pQ_in = airf_g_hr_cf.pQ_col;
        end

        # Adjust lengths
        if length(MU_in) < T_g; MU_in = vcat(MU_in, MU_ss*ones(T_g-length(MU_in))); end
        if length(MU_in) > T_g; MU_in = MU_in[1:T_g]; end
        if length(pQ_in) < T_g; pQ_in = vcat(pQ_in, ones(T_g-length(pQ_in))); end
        if length(pQ_in) > T_g; pQ_in = pQ_in[1:T_g]; end
    end

    # Overwrite capital price elasticity parameters
    modl.phi_k = 1/8
    modl_hr_cf.phi_k = 1/8
    # phim_resp set accordingly
    if (IRF_indic == "base") | (IRF_indic == "base_hr_cf_basephimresp_noGE")
        modl.phi_mresp = 0.618
    elseif (IRF_indic == "hr_cf")
        modl_hr_cf.phi_mresp = 0.5*0.2445 + (1-0.2445-0.4256)*1
    elseif (IRF_indic == "hr_cf_basephimresp") | (IRF_indic == "hr_cf_basephimresp_noGE")
        modl_hr_cf.phi_mresp = 0.618
    end

    # Solve IRF path, unless saved file already exists
    if (IRF_indic == "base") & isfile(joinpath(savedsoln_dir, "irf_base_phik1d8.jld"))
        # Load if saved
        global airf_g_base, irfsoln_g_base = load(joinpath(savedsoln_dir, "irf_base_phik1d8.jld"), "airf_g", "irfsoln_g");
    elseif (IRF_indic == "hr_cf") & isfile(joinpath(savedsoln_dir, "irf_hr_cf_phik1d8.jld"))
        # Load if saved
        global airf_g_hr_cf = load(joinpath(savedsoln_dir, "irf_hr_cf_phik1d8.jld"), "airf_g");
    elseif (IRF_indic == "hr_cf_basephimresp") & isfile(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp_phik1d8.jld"))
        # Load if saved
        global airf_g_hr_cf_basephimresp = load(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp_phik1d8.jld"), "airf_g");
    elseif (IRF_indic == "hr_cf_basephimresp_noGE") & isfile(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp_noGE_phik1d8.jld"))
        # Load if saved
        global airf_g_hr_cf_basephimresp_noGE = load(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp_noGE_phik1d8.jld"), "airf_g");
    elseif (IRF_indic == "base_hr_cf_basephimresp_noGE") & isfile(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE_phik1d8.jld"))
        # Load if saved
        global airf_g_base_hr_cf_basephimresp_noGE = load(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE_phik1d8.jld"), "airf_g");
    elseif (IRF_indic == "base")
        global airf_g_base, irfsoln_g_base = solve_eqIRF_NKMg_xsh(0.000, zeta_col, spread_col, fsoln, aeq, modl, sspace, opts, MU_col_0=MU_in, pQ_col_0=pQ_in);
        # Save to file
        save(joinpath(savedsoln_dir, "irf_base_phik1d8.jld"),  "airf_g", airf_g_base,  "irfsoln_g", irfsoln_g_base)
    elseif (IRF_indic == "hr_cf")
        global airf_g_hr_cf, irfsoln_g_hr_cf = solve_eqIRF_NKMg_xsh(0.000, zeta_col, spread_col, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts, MU_col_0=MU_in, pQ_col_0=pQ_in, w_ss=modl_hr_cf.psi/aeq_hr_cf.p);
        # Save to file
        save(joinpath(savedsoln_dir, "irf_hr_cf_phik1d8.jld"),  "airf_g", airf_g_hr_cf,  "irfsoln_g", irfsoln_g_hr_cf);
        # Forget irfsoln
        global irfsoln_g_hr_cf = nothing;
    elseif (IRF_indic == "hr_cf_basephimresp")
        global airf_g_hr_cf_basephimresp, irfsoln_g_hr_cf_basephimresp = solve_eqIRF_NKMg_xsh(0.000, zeta_col, spread_col, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts, MU_col_0=MU_in, pQ_col_0=pQ_in, w_ss=modl_hr_cf.psi/aeq_hr_cf.p);
        # Save to file
        save(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp_phik1d8.jld"),  "airf_g", airf_g_hr_cf_basephimresp,  "irfsoln_g", irfsoln_g_hr_cf_basephimresp)
        # Forget irfsoln
        global irfsoln_g_hr_cf_basephimresp = nothing;
    elseif (IRF_indic == "hr_cf_basephimresp_noGE")
        manprintln("Running hr_cf_basephimresp_noGE IRF calculation...", txtout_path)
        opts.tol_p_col = 2*1e5
        global airf_g_hr_cf_basephimresp_noGE, irfsoln_g_hr_cf_basephimresp_noGE = solve_eqIRF_NKMg_xsh(0.000, zeta_col, spread_col, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts, MU_col_0=MU_in, pQ_col_0=pQ_in, w_ss=modl_hr_cf.psi/aeq_hr_cf.p)
        opts.tol_p_col = 2*1e-5
        # Save to file
        save(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp_noGE_phik1d8.jld"),  "airf_g", airf_g_hr_cf_basephimresp_noGE,  "irfsoln_g", irfsoln_g_hr_cf_basephimresp_noGE)
        # Forget irfsoln
        global irfsoln_g_hr_cf_basephimresp_noGE = nothing;
    elseif (IRF_indic == "base_hr_cf_basephimresp_noGE")
        manprintln("Running base_hr_cf_basephimresp_noGE IRF calculation...", txtout_path)
        opts.tol_p_col = 2*1e5
        global airf_g_base_hr_cf_basephimresp_noGE, irfsoln_g_base_hr_cf_basephimresp_noGE = solve_eqIRF_NKMg_xsh(0.000, zeta_col, spread_col, fsoln, aeq, modl, sspace, opts, MU_col_0=MU_in, pQ_col_0=pQ_in);
        opts.tol_p_col = 2*1e-5
        # Save to file
        save(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE_phik1d8.jld"),  "airf_g", airf_g_base_hr_cf_basephimresp_noGE,  "irfsoln_g", irfsoln_g_base_hr_cf_basephimresp_noGE)
        # Forget irfsoln
        global irfsoln_g_base_hr_cf_basephimresp_noGE = nothing;
    end
end

# Format aggregate IRF figures -- Figure B.12
# Construct Agg investment
Ia_col_base = airf_g_base.Ka_col[2:end]-(1.0-modl.delta)*airf_g_base.Ka_col[1:(end-1)]
# Plot
Tmax=20
manl_ftsize = 13
fg = Plots.plot(0:Tmax, [4*100*(airf_g_base.rf_col[1:Tmax+1].-modl.rf), 4*100*(airf_g_base.Pi_col[1:Tmax+1].-modl.Pi), 100*(log.(airf_g_base.MU_col[1:Tmax+1]).-log(MU_ss)), 100*(log.(airf_g_base.Ya_col[1:Tmax+1]).-log(aeq_base.Ya)), 100*(log.(Ia_col_base[1:Tmax+1]).-log(modl.delta*aeq_base.Ka)), 100*(log.(airf_g_base.Ka_col[1:Tmax+1]).-log(aeq_base.Ka))], layout=(2,3), title=["Interest rates" "Inflation, \$Q\$" "Markup" L"Y, n, c" "Investment" "Capital" ], label=[L"r^{f}" L"\Pi" :none L"Y" L"I" :none], legend=[:bottomright :bottomright :none :bottomright :none :none], size=(800,460), linewidth=2, linecolor=:blue, xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-2, xticks=0:5:(length(airf_g_base.rf_col)-1))
fg = Plots.plot!(0:Tmax, 4*100*(airf_g_base.rb_col[1:Tmax+1].-modl.rb), subplot=1, label=L"r^{b}", linewidth=2, linecolor=:red, linestyle=:dash)
fg = Plots.plot!(0:Tmax, 4*100*(airf_g_base.rm_col[1:Tmax+1].-modl.rm), subplot=1, label=L"r^{m}", linewidth=2, linecolor=:green, linestyle=:dot)
fg = Plots.plot!(0:Tmax, 100*(log.(airf_g_base.Na_col[1:Tmax+1]).-log(aeq_base.Na)), subplot=4, label=L"n", linewidth=2, linecolor=:red, linestyle=:dash)
fg = Plots.plot!(0:Tmax, 100*(log.(airf_g_base.Ca_col[1:Tmax+1]).-log(aeq_base.Ca)), subplot=4, label=L"c", linewidth=2, linecolor=:green, linestyle=:dot)
fg = Plots.plot!(0:Tmax, 100*(log.(airf_g_base.pQ_col[1:Tmax+1]).-log(pQ_ss)), subplot=2, label=L"Q", linewidth=2, linecolor=:red, linestyle=:dash)
fg = hline!([0 0 0 0 0 0], subplot=1:6, color=:black, linestyle=:dash, label=:none)
ylims!(fg[1], -0.050, 0.043)
Plots.savefig(fg, joinpath(PROJECT_ROOT, "output", "figures", "model_figures", "FigB12_"*modl.name*"IRFaggs_main_rnarrow_phik1d8.pdf"))

if true
    # Given the baseline solution, draw random set of firms, simulate panel in SS and IRF and compute/plot regression coef on liq ratio -- Figure B.13a
    Random.seed!(999)
    # Public firms
    sim_IRFSS_anl_rand(50000,20, 60, Lin_pub, fsoln, irfsoln_g_base, true, true, "_phik1d8_pub", modl, sspace, opts)
end
# Forget irfsoln_g
irfsoln_g_base = nothing;
GC.gc();

# Compare at some specific horizon
# K4 -- Figure B.14a
hor_K = 5;
fg=Plots.plot(1:3, 100*[log.(airf_g_hr_cf.Ka_col[hor_K]./aeq_hr_cf.Ka) log.(airf_g_hr_cf_basephimresp.Ka_col[hor_K]./aeq_hr_cf.Ka) log.(airf_g_base.Ka_col[hor_K]./aeq_base.Ka)]', label="\$\\hat{K}_{4}^{GE}\$", ylims=(-0.0353-(0.023-0.0202), -0.0353+(-0.0125+0.0202)), linewidth=1, linecolor=:blue, m=(:diamond, :blue, 6), size=(600,400), xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-2, ylabel = "Percent", legend=:topleft, xticks=(1:3, ["Pre-90", "Pre-90-\$\\phi_{H}^{m}\$", "Post-90"]), xlims = [0.8, 3.2], linestyle=:dash)
fg=Plots.plot!(1:3, 100*[log.(airf_g_hr_cf.Ka_col[hor_K]./aeq_hr_cf.Ka) log.(airf_g_hr_cf_basephimresp_noGE.Ka_col[hor_K]./aeq_hr_cf.Ka) log.(airf_g_base_hr_cf_basephimresp_noGE.Ka_col[hor_K]./aeq_base.Ka)]', label="\$\\hat{K}_{4}^{PE}\$", linewidth=1, linecolor=:green, linestyle=:dot, m=(:circle, :green, 6))
Plots.savefig(fg, joinpath(PROJECT_ROOT, "output", "figures", "model_figures", "FigB14a_"*modl.name*"_IRFKa4_comp_phik1d8.pdf"))

# Run decomposition of channels for the phi_k=1/8 case
### Analyze rmtaureal-channel
# Base model
if !isfile(joinpath(savedsoln_dir, "irf_base_decomp_rmtaureal_phik1d8.jld"))
    airf_g_base_decomp_rmtaureal, irfsoln_g_base_decomp_rmtaureal = solve_eqIRF_decomp("rmtaureal", airf_g_base, fsoln, aeq, modl, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_base_decomp_rmtaureal_phik1d8.jld"),  "airf_g", airf_g_base_decomp_rmtaureal,  "irfsoln_g", irfsoln_g_base_decomp_rmtaureal)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_base_decomp_rmtaureal, irfsoln_g_base_decomp_rmtaureal = nothing, nothing;
end
# For hr_cf
if !isfile(joinpath(savedsoln_dir, "irf_hr_cf_decomp_rmtaureal_phik1d8.jld"))
    airf_g_hr_cf_decomp_rmtaureal, irfsoln_g_hr_cf_decomp_rmtaureal = solve_eqIRF_decomp("rmtaureal", airf_g_hr_cf, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_hr_cf_decomp_rmtaureal_phik1d8.jld"),  "airf_g", airf_g_hr_cf_decomp_rmtaureal,  "irfsoln_g", irfsoln_g_hr_cf_decomp_rmtaureal)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_hr_cf_decomp_rmtaureal, irfsoln_g_hr_cf_decomp_rmtaureal = nothing, nothing;
end
# For hr_cf_basephimresp
if !isfile(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp_decomp_rmtaureal_phik1d8.jld"))
    airf_g_hr_cf_basephimresp_decomp_rmtaureal, irfsoln_g_hr_cf_basephimresp_decomp_rmtaureal = solve_eqIRF_decomp("rmtaureal", airf_g_hr_cf_basephimresp, fsoln_hr_cf, aeq_hr_cf, modl_hr_cf, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp_decomp_rmtaureal_phik1d8.jld"),  "airf_g", airf_g_hr_cf_basephimresp_decomp_rmtaureal,  "irfsoln_g", irfsoln_g_hr_cf_basephimresp_decomp_rmtaureal)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_hr_cf_basephimresp_decomp_rmtaureal, irfsoln_g_hr_cf_basephimresp_decomp_rmtaureal = nothing, nothing;
end
# Also, for PE effect: base_hr_cf_basephimresp_noGE
if !isfile(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE_decomp_rmtaureal_phik1d8.jld"))
    airf_g_base_hr_cf_basephimresp_noGE_decomp_rmtaureal, irfsoln_g_base_hr_cf_basephimresp_noGE_decomp_rmtaureal = solve_eqIRF_decomp("rmtaureal", airf_g_base_hr_cf_basephimresp_noGE, fsoln, aeq, modl, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE_decomp_rmtaureal_phik1d8.jld"),  "airf_g", airf_g_base_hr_cf_basephimresp_noGE_decomp_rmtaureal,  "irfsoln_g", irfsoln_g_base_hr_cf_basephimresp_noGE_decomp_rmtaureal)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_base_hr_cf_basephimresp_noGE_decomp_rmtaureal, irfsoln_g_base_hr_cf_basephimresp_noGE_decomp_rmtaureal = nothing, nothing;
end

### Analyze rbtaureal-channel
# Base model
if !isfile(joinpath(savedsoln_dir, "irf_base_decomp_rbtaureal_phik1d8.jld"))
    airf_g_base_decomp_rbtaureal, irfsoln_g_base_decomp_rbtaureal = solve_eqIRF_decomp("rbtaureal", airf_g_base, fsoln, aeq, modl, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_base_decomp_rbtaureal_phik1d8.jld"),  "airf_g", airf_g_base_decomp_rbtaureal,  "irfsoln_g", irfsoln_g_base_decomp_rbtaureal)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_base_decomp_rbtaureal, irfsoln_g_base_decomp_rbtaureal = nothing, nothing;
end
# Also, for PE effect: base_hr_cf_basephimresp_noGE
if !isfile(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE_decomp_rbtaureal_phik1d8.jld"))
    airf_g_base_hr_cf_basephimresp_noGE_decomp_rbtaureal, irfsoln_g_base_hr_cf_basephimresp_noGE_decomp_rbtaureal = solve_eqIRF_decomp("rbtaureal", airf_g_base_hr_cf_basephimresp_noGE, fsoln, aeq, modl, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE_decomp_rbtaureal_phik1d8.jld"),  "airf_g", airf_g_base_hr_cf_basephimresp_noGE_decomp_rbtaureal,  "irfsoln_g", irfsoln_g_base_hr_cf_basephimresp_noGE_decomp_rbtaureal)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_base_hr_cf_basephimresp_noGE_decomp_rbtaureal, irfsoln_g_base_hr_cf_basephimresp_noGE_decomp_rbtaureal = nothing, nothing;
end

### Analyze Q-channel
# Base model
if !isfile(joinpath(savedsoln_dir, "irf_base_decomp_Q_phik1d8.jld"))
    airf_g_base_decomp_Q, irfsoln_g_base_decomp_Q = solve_eqIRF_decomp("Q", airf_g_base, fsoln, aeq, modl, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_base_decomp_Q_phik1d8.jld"),  "airf_g", airf_g_base_decomp_Q,  "irfsoln_g", irfsoln_g_base_decomp_Q)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_base_decomp_Q, irfsoln_g_base_decomp_Q = nothing, nothing;
end
# Also, for PE effect: base_hr_cf_basephimresp_noGE
if !isfile(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE_decomp_Q_phik1d8.jld"))
    airf_g_base_hr_cf_basephimresp_noGE_decomp_Q, irfsoln_g_base_hr_cf_basephimresp_noGE_decomp_Q = solve_eqIRF_decomp("Q", airf_g_base_hr_cf_basephimresp_noGE, fsoln, aeq, modl, sspace, opts);
    # Save full solution
    save(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE_decomp_Q_phik1d8.jld"),  "airf_g", airf_g_base_hr_cf_basephimresp_noGE_decomp_Q,  "irfsoln_g", irfsoln_g_base_hr_cf_basephimresp_noGE_decomp_Q)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_base_hr_cf_basephimresp_noGE_decomp_Q, irfsoln_g_base_hr_cf_basephimresp_noGE_decomp_Q = nothing, nothing;
end

### Analyze MU-channel
# Base model
if !isfile(joinpath(savedsoln_dir, "irf_base_decomp_MU_phik1d8.jld"))
    airf_g_base_decomp_MU, irfsoln_g_base_decomp_MU = solve_eqIRF_decomp("MU", airf_g_base, fsoln, aeq, modl, sspace, opts);
    # Save full firm solution
    save(joinpath(savedsoln_dir, "irf_base_decomp_MU_phik1d8.jld"),  "airf_g", airf_g_base_decomp_MU,  "irfsoln_g", irfsoln_g_base_decomp_MU)
    # And forget in order to save RAM, to be loaded as needed later
    airf_g_base_decomp_MU, irfsoln_g_base_decomp_MU = nothing, nothing;
end

# Plot decomposition of channels (to capital accumulation) in aggregate
if true
    # First, load relevant decompositions here
    airf_g_base_decomp_rbtaureal = load(joinpath(savedsoln_dir, "irf_base_decomp_rbtaureal_phik1d8.jld"), "airf_g");
    airf_g_base_decomp_rmtaureal = load(joinpath(savedsoln_dir, "irf_base_decomp_rmtaureal_phik1d8.jld"), "airf_g");
    airf_g_base_decomp_Q = load(joinpath(savedsoln_dir, "irf_base_decomp_Q_phik1d8.jld"), "airf_g");
    airf_g_base_decomp_MU = load(joinpath(savedsoln_dir, "irf_base_decomp_MU_phik1d8.jld"), "airf_g");

    # Plot comparison base model (capital accumulation) -- Figure B.13b
    T_plot = 21;
    manl_ftsize = 13
    fg=Plots.plot(0:(T_plot-1), 100*log.(airf_g_base.Ka_col[1:T_plot]./aeq_base.Ka), label="Total", ylims=(-0.18, 0.30), linewidth=3, linecolor=:blue, size=(600,400), xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-2, xlabel = "\$t\$", ylabel = "Percent")
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_base_decomp_rbtaureal.Ka_col[1:T_plot]./aeq_base.Ka), label="Real \$r^{b}\$", linewidth=2, linecolor=:red, linestyle=:dash)
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_base_decomp_rmtaureal.Ka_col[1:T_plot]./aeq_base.Ka), label="Real \$r^{m}\$", linewidth=2, linecolor=:teal, linestyle=:dot)
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_base_decomp_Q.Ka_col[1:T_plot]./aeq_base.Ka), label="\$Q\$", linewidth=2, linecolor=:green, linestyle=:dashdot)
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_base_decomp_MU.Ka_col[1:T_plot]./aeq_base.Ka), label="\$\\mathcal{M}\$", linewidth=1.5, linecolor=:purple3, linestyle=:solid)
    fg = hline!([0], color=:black, linestyle=:dash, label=:none, linewidth = 0.5, xlims=[0, T_plot-1])
    Plots.savefig(fg, joinpath(PROJECT_ROOT, "output", "figures", "model_figures", "FigB13b_"*modl.name*"_IRFKa_decomp_channels_base_phik1d8.pdf"))
end

# Instead, compare rmtaureal channels across the three relevant models
if true
    # Load paths
    airf_g_hr_cf_decomp_rmtaureal = load(joinpath(savedsoln_dir, "irf_hr_cf_decomp_rmtaureal_phik1d8.jld"), "airf_g");
    airf_g_hr_cf_basephimresp_decomp_rmtaureal = load(joinpath(savedsoln_dir, "irf_hr_cf_basephimresp_decomp_rmtaureal_phik1d8.jld"), "airf_g");

    # Plot comparison -- Figure B.14b
    T_plot = 9;
    manl_ftsize = 13
    fg=Plots.plot(0:(T_plot-1), 100*log.(airf_g_hr_cf_decomp_rmtaureal.Ka_col[1:T_plot]./aeq_hr_cf.Ka), label="Pre-90", ylims=(-0.078, 0.01), linewidth=2, linecolor=:midnightblue, size=(600,400), xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-2, xlabel = "\$t\$", ylabel = "Percent", linestyle = :dash)
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_hr_cf_basephimresp_decomp_rmtaureal.Ka_col[1:T_plot]./aeq_hr_cf.Ka), label="Pre-90-\$\\phi_{H}^{m}\$", linewidth=2, linecolor=:steelblue2, linestyle=:dashdot)
    fg=Plots.plot!(0:(T_plot-1), 100*log.(airf_g_base_decomp_rmtaureal.Ka_col[1:T_plot]./aeq_base.Ka), label="Post-90", linewidth=2, linecolor=:blue, linestyle=:solid)
    fg = hline!([0], color=:black, linestyle=:dash, label=:none, linewidth = 0.5, xlims=[0, T_plot-1])
    Plots.savefig(fg, joinpath(PROJECT_ROOT, "output", "figures", "model_figures", "FigB14b_"*modl.name*"_IRFKa_decomp_rmtaureal_comp_phik1d8.pdf"))

    # Compute responses relative to hr_cf
    Ka_relresp_hr_cf_basephimresp = log.(airf_g_hr_cf_basephimresp_decomp_rmtaureal.Ka_col[1:T_plot]./aeq_hr_cf.Ka) ./ log.(airf_g_hr_cf_decomp_rmtaureal.Ka_col[1:T_plot]./aeq_hr_cf.Ka)
    Ka_relresp_base = log.(airf_g_base_decomp_rmtaureal.Ka_col[1:T_plot]./aeq_base.Ka) ./ log.(airf_g_hr_cf_decomp_rmtaureal.Ka_col[1:T_plot]./aeq_hr_cf.Ka)
end

# Recompute the relative role of the Q channel in explaining the (post-1990) gap, as reported in last Section of paper
if true
    # Load all the relevant agg responses
    # GE already loaded, noGE only here
    airf_g_base_hr_cf_basephimresp_noGE = load(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE_phik1d8.jld"), "airf_g");
    airf_g_base_hr_cf_basephimresp_noGE_decomp_rbtaureal = load(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE_decomp_rbtaureal_phik1d8.jld"), "airf_g");
    airf_g_base_hr_cf_basephimresp_noGE_decomp_rmtaureal = load(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE_decomp_rmtaureal_phik1d8.jld"), "airf_g");
    airf_g_base_hr_cf_basephimresp_noGE_decomp_Q = load(joinpath(savedsoln_dir, "irf_base_hr_cf_basephimresp_noGE_decomp_Q_phik1d8.jld"), "airf_g");

    # Compute full gap in K4
    GEnoGE_K4_gap = 100*log(airf_g_base.Ka_col[5] / aeq.Ka) - 100*log(airf_g_base_hr_cf_basephimresp_noGE.Ka_col[5] / aeq.Ka) 
    # And the parts explained by the channels
    GEnoGE_K4_rbtaureal_part = 100*log(airf_g_base_decomp_rbtaureal.Ka_col[5] / aeq.Ka) - 100*log(airf_g_base_hr_cf_basephimresp_noGE_decomp_rbtaureal.Ka_col[5] / aeq.Ka) 
    GEnoGE_K4_rmtaureal_part = 100*log(airf_g_base_decomp_rmtaureal.Ka_col[5] / aeq.Ka) - 100*log(airf_g_base_hr_cf_basephimresp_noGE_decomp_rmtaureal.Ka_col[5] / aeq.Ka) 
    GEnoGE_K4_Q_part = 100*log(airf_g_base_decomp_Q.Ka_col[5] / aeq.Ka) - 100*log(airf_g_base_hr_cf_basephimresp_noGE_decomp_Q.Ka_col[5] / aeq.Ka) 
    # And finally, the corresponding share
    GEnoGE_K4_rbtaureal_rmtaureal_share = (GEnoGE_K4_rbtaureal_part + GEnoGE_K4_rmtaureal_part) / GEnoGE_K4_gap
    GEnoGE_K4_Q_share = GEnoGE_K4_Q_part / GEnoGE_K4_gap

    open(joinpath(PROJECT_ROOT, "output", "other", "Text_AppxB11p2_GEPEshare.txt"), "w") do io
        println(io, "GE-PE gap share (in phi_k=1/8 model) explained by Q channel (phik=1/8): $(round(GEnoGE_K4_Q_share, digits=2)), or about $(round(Int, 5*GEnoGE_K4_Q_share))/5")
        println(io, "GE-PE gap share (in phi_k=1/8 model) explained by rb+rm real channels (phik=1/8): $(round(GEnoGE_K4_rbtaureal_rmtaureal_share, digits=2)), or about $(round(Int, 5*GEnoGE_K4_rbtaureal_rmtaureal_share))/5")
    end
end

