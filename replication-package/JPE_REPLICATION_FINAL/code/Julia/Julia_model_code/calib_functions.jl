# Procedures applied in computing calibration moments in model steady state, and additional descriptive model diagnostics, lifecycle behavior, and group-specific aggregated IRFs

# Function for computing pairwise correlation of (x,y), given grids and distribution L
function discr_corr(xgrid, ygrid, L)
    mean_x, mean_y    = L' * xgrid, L' * ygrid
    std_x, std_y      = sqrt(L' * ((xgrid.-mean_x).^2.0)), sqrt(L' * ((ygrid.-mean_y).^2.0))
    cov_x_y           = L' * ((xgrid.-mean_x) .* (ygrid.-mean_y))
    corr_x_y          = cov_x_y/(std_x * std_y)

    return corr_x_y
end
# Function for computing skewness of x, given grid and distribution L
function discr_skew(xgrid, L)
    mean_x            = L' * xgrid
    std_x             = sqrt(L' * ((xgrid.-mean_x).^2.0))
    skew_x            = L' * (((xgrid.-mean_x)/std_x).^3.0)

    return skew_x
end
# Function for computing quantiles of y, given grid and distribution
function discr_percs(ygrid, L, percs)
    isort_ygrid = sortperm(ygrid)
    ygrid_sort  = ygrid[isort_ygrid]
    L_sort      = L[isort_ygrid]
    # Create cumulative distribution
    L_ysort_cum = cumsum(L_sort)
    y_perc_vals = zeros(length(percs))
    for ip=1:length(percs)
        # Determine a specific percentile
        y_perc         = percs[ip]
        # The percentile value
        y_perc_i       = findfirst(L_ysort_cum .> y_perc)
        y_perc_val     = ygrid_sort[y_perc_i-1] + (ygrid_sort[y_perc_i]-ygrid_sort[y_perc_i-1]) * (y_perc-L_ysort_cum[y_perc_i-1])/(L_ysort_cum[y_perc_i]-L_ysort_cum[y_perc_i-1])
        y_perc_vals[ip]= y_perc_val
    end
    return y_perc_vals
end

# Function for generating unconditional distribution that is observed in a balanced panel of length "Tyears", when model is quarterly
function gen_Lbalpan_tastd(Linit, Q, Tyears, condl_ann_invk, condl_ann_sqinvk)
    # Compute time-average std of cross-sectional investment rates
    Nd      = size(Linit,1)
    Tquarts = 4*(Tyears-1) + 1
    L_col   = zeros(Nd, Tyears)
    L_col[:,1] = Linit
    Lold    = Linit
    for tt=2:Tquarts
        Lnew = Q' * Lold
        if mod(tt-1,4) == 0 # Save every 4th distribution
            L_col[:,Int((tt-1)/4+1)] = Lnew
        end
        Lold = Lnew
    end

    # Compute series of stds
    std_col = zeros(Tyears)
    for tt=1:Tyears
        avg_ann_invk_cur    = L_col[:,tt]' * condl_ann_invk
        std_col[tt]         = sqrt(L_col[:,tt]' * condl_ann_sqinvk - avg_ann_invk_cur^2)
    end

    Ltot    = sum(L_col,dims=2)
    Ltot    = Ltot/sum(Ltot)

    return vec(Ltot), mean(std_col)
end

# Write function that takes as an argument the steady state firm solution on the population grid (fsoln_f), a given initial distribution of incumbents (Lin), and the steady state transition matrix Q, and instead of simulating a distribution of firms, uses the discretized initial state of incumbents Lin and the transition matrix Q.
function calib_nosimcalc(fsoln_f, Lin, modl, sspace, opts)

    # Initialize steady state prices
    pQ_ss       = 1.0
    Rc_ss       = (1.0+modl.rb)/(modl.Pi*pQ_ss)
    gammatil_ss = modl.gamma/modl.Pi

    # Start as if solving for the steady state L
    sf      = sspace.sf
    Nd = size(sf,1)

    # sf-length grid for LEVEL of debt
    b_trgrid = sf[:,3].*sf[:,1] / Rc_ss

    # Can compute some aggregates of interest without simulation directly from the approximate stationary distribution
    M_agg  = Lin'*sf[:,2]
    K_agg  = Lin'*sf[:,1]
    B_agg  = Lin'*b_trgrid
    # Aggregate ratios
    MA_agg = M_agg/(K_agg+M_agg)
    BA_agg = B_agg/(K_agg+M_agg)

    # Initialize some vectors containing policies
    Ba_prob_vec      = reshape(fsoln_f.Ba_prob, Nd)
    kpA_vec, mpA_vec, bpA_vec = reshape(fsoln_f.kpA, Nd), reshape(fsoln_f.mpA, Nd), reshape(fsoln_f.bpA, Nd)
    kpN_vec, mpN_vec, bpN_vec = reshape(fsoln_f.kpN, Nd), reshape(fsoln_f.mpN, Nd), reshape(fsoln_f.bpN, Nd)

    # And the size of the issuances (NEGATIVE ENTRY is ISSUANCE)
    diss_size_vec   = kpA_vec .* bpA_vec / Rc_ss - (gammatil_ss * b_trgrid)
    # Redefine diss_size based on size of issuance (as ratio of assets, as in data)
    diss_vec = diss_size_vec ./ (sf[:,1] + sf[:,2]) .< -0.01

    # Multiply up the fraction of debt issuers, taking into account the fraction of firms adjusting in each state
    diss_frac_ss    = (Lin' * (Ba_prob_vec .* diss_vec))[1]

    # Compute the skewness of the k- and log(k)-distributions
    skew_logk       = discr_skew(log.(sf[:,1]), Lin)

    # Compute investment autocorrelations and standard deviations by simulation
    # Set seed
    Random.seed!(1111)
    if true
        Nfirms=50000;
        Tsim=9;
        zi_col, xi_cdf_col =  Array{Int64,2}(undef, Nfirms,Tsim), zeros(Nfirms, Tsim)
        mcz         = MarkovChain(sspace.P)
        # Draw initial distribution
        # Distribution from given Lin
        ii_samp = wsample(1:sspace.Nsf, Lin, Nfirms)
        se_init = sspace.sf[ii_samp, 1:3]
        xi_cdf_col = rand(Nfirms,Tsim)
        zi_col_init= repeat(1:sspace.nf[4], inner=sspace.nf[1]*sspace.nf[2]*sspace.nf[3])[ii_samp]
        # Simulate actual MChains
        for ii in 1:Nfirms
            zi_col[ii,:]   = simulate(mcz,Tsim,init=zi_col_init[ii])
        end
        # Simulate paths in SS
        sim_paths_SS  = sim_firm_SS(se_init, zi_col, xi_cdf_col, fsoln_f, true, modl, sspace, opts)
        # Compute paths of annual investment rates
        ann_inv_rat_col = (sim_paths_SS.k_path[:,(2+3):end] .- ((1.0-modl.delta)^4)*sim_paths_SS.k_path[:,1:(end-1-3)])./sim_paths_SS.k_path[:,1:(end-1-3)]

        # Compute averages, autocorrelations and standard deviations
        # Annual
        avg_ann_invk_sim        = mean(ann_inv_rat_col[:,1])
        std_ann_invk_sim        = std(ann_inv_rat_col[:,1])
        acor_ann_invkpinvk_sim  = cor(ann_inv_rat_col[:,5], ann_inv_rat_col[:,1])
    end

    # Compute the issuance of firms with capital below percentile "small_k_perc"
    # To determine "small" capital-bin cutoff, use the marginal distribution of incoming k among Lin
    # Need to compute marginal k-distribution here
    Lin_k = kron(ones(1,sspace.nf[4]*sspace.nf[3]*sspace.nf[2]), I(sspace.nf[1])) * Lin
    Lin_k_cum  = cumsum(Lin_k)
    # Compute "small" firm size cutoff
    small_k_perc       = 0.50
    small_ih           = findfirst(Lin_k_cum .> small_k_perc)
    small_k_val        = sspace.k_gridf[small_ih-1] + (sspace.k_gridf[small_ih]-sspace.k_gridf[small_ih-1]) * (small_k_perc-Lin_k_cum[small_ih-1])/(Lin_k_cum[small_ih]-Lin_k_cum[small_ih-1])

    # Select firms below the size cutoff
    Lin_sel_small = Lin .* (sf[:,1] .<= small_k_val)
    # Unconditional frequency of issuance of these small firms
    small_k_diss_frac = (Lin_sel_small' * (Ba_prob_vec.* diss_vec))[1] / sum(Lin_sel_small)

    # Repeat given assets-based size split
    asset_trgrid = sf[:,1] .+ sf[:,2]

    # Sort assets and corresponding stationary weights
    asset_sort_ind = sortperm(asset_trgrid)
    asset_sort_val = asset_trgrid[asset_sort_ind]
    Lin_asset_sort = Lin[asset_sort_ind]
    Lin_asset_cum  = cumsum(Lin_asset_sort)

    # Compute "small" asset cutoff
    small_a_perc = 0.50
    small_ih = findfirst(Lin_asset_cum .> small_a_perc)
    small_a_val = asset_sort_val[small_ih-1] + (asset_sort_val[small_ih] - asset_sort_val[small_ih-1]) * (small_a_perc - Lin_asset_cum[small_ih-1]) / (Lin_asset_cum[small_ih] - Lin_asset_cum[small_ih-1])

    # Select firms below the asset cutoff
    Lin_sel_small = Lin .* (asset_trgrid .<= small_a_val)

    # Unconditional frequency of issuance among these "small" firms
    small_a_diss_frac = (Lin_sel_small' * (Ba_prob_vec .* diss_vec))[1] / sum(Lin_sel_small)

    # Compute the average issuance cost-issuance ratio
    xi_bar_vec          = Ba_prob_vec * modl.xi_ub
    # xi_condexpn_vec     = sf[:,1] .* xi_condexpn(xi_bar_vec, modl.xi_ub, modl.xi_lb, modl.xi_dist) # NOTE: SCALES with k!
    xi_condexpn_vec     = xi_condexpn(xi_bar_vec, modl.xi_ub, modl.xi_lb, modl.xi_dist) # NOTE: NOT SCALED with k!
    # Conditional on ISSUANCE (diss_vec, diss_size_vec)
    diss_pos_ind        = diss_vec .> 0.0              # Select only states with issuance
    diss_pos_size_vec   = diss_size_vec[diss_pos_ind] # And record the sizes of these NEGATIVE issuances
    Ba_prob_diss_vec    = Ba_prob_vec[diss_pos_ind]            # Probability of issuance in each of these states
    L_diss           = (Lin[diss_pos_ind] .* Ba_prob_diss_vec) / sum(Lin[diss_pos_ind] .* Ba_prob_diss_vec)  # The distribution conditional on issuing across these states
    # Average issuance cost relative to issuance size vector
    xidb_condexpn_vec = -xi_condexpn_vec[diss_pos_ind] ./ diss_pos_size_vec
    Exibd_Ba          = L_diss' * xidb_condexpn_vec     # Mean cost/issuance ratio

    return [MA_agg, BA_agg, diss_frac_ss, Exibd_Ba, skew_logk, avg_ann_invk_sim, std_ann_invk_sim, acor_ann_invkpinvk_sim, small_a_diss_frac]
end

# Function to generate measure for firms in a given quarter-age interval, implementing Tan's (2020) algorithm.
function gen_L_ageint(a1, a2, Qend, Le, modl, sspace, opts)
    La_last    = sparse(Le) # Helper to save last cohort's mass, initialize as mass of entrants (age 0q)

    # Construct matrices to implement Tan (2020) algorithm
    Nd = size(sspace.sf,1)
    Gammamat  = spzeros(Nd,Nd)
    Nend      = sspace.Nk*sspace.Nm*sspace.Nb
    for iiz=1:sspace.Nz
        Gammamat[((iiz-1)*Nend+1):(iiz*Nend),((iiz-1)*Nend+1):(iiz*Nend)] = Qend[((iiz-1)*Nend+1):(iiz*Nend),:]
    end

    # Initialize La_last to the "cohort of age a1"
    for ii=1:a1
        Ltil    = Gammamat'*La_last
        Lamtil  = reshape(Ltil, Nend, sspace.Nz)
        La_last = (1.0-modl.eta)*reshape(Lamtil*sspace.P, Nd)
        # La_last = (1.0-modl.eta) * Q'*La_last
    end

    # Given this, sum up all "countable cohorts"
    L_sum  = copy(La_last)
    for ii = (a1+1):a2
        Ltil    = Gammamat'*La_last
        Lamtil  = reshape(Ltil, Nend, sspace.Nz)
        La_last = (1.0-modl.eta)*reshape(Lamtil*sspace.P, Nd)
        L_sum   = L_sum + La_last
    end

    return L_sum
end


# Function to characterize moments across firms' life-cycle. Currently simply computing entrants' size in terms of employment, relative to average firm's employment.
function calib_age_nosimcalc(fsoln_f, aeq, modl, sspace, opts)

    # Start as if solving for the steady state L
    sf      = sspace.sf
    Nd      = size(sf,1)

    # Construct measure of entrants
    fspaceergk  = fundef([:spli, sspace.k_gridf, 0, 1])
    fspaceergm  = fundef([:spli, sspace.m_gridf, 0, 1])
    fspaceergb  = fundef([:spli, sspace.b_gridf, 0, 1])
    # Generating the mass of firms entering
    QKe         = funbase(fspaceergk, ones(Nd)*modl.k_0)
    QMe         = funbase(fspaceergm, ones(Nd)*modl.m_0)
    QBe         = funbase(fspaceergb, ones(Nd)*modl.b_0)
    QZe         = ones(Nd) * transpose(modl.z_0_dist)

    # Multiply up into "full" transition matrix
    Qe = row_kron(QZe, row_kron(QBe, row_kron(QMe, QKe)))

    Le          = Qe' * (modl.eta * ones(Nd) / Nd)

    # Initialize labor policy vector
    n_vec = reshape(fsoln_f.n, Nd)

    # Compute employment-size of average entrant
    n_0     = (Le' * n_vec) / sum(Le)
    n_0_dN  = n_0/aeq.Na

    return [n_0_dN]
end

# Function to generate distribution of firms over lifecycle, CONDITIONAL ON SURVIVAL (i.e., measure 1 at any age), up to a given Tage, implementing Tan's (2020) algorithm.
function gen_L_lifecycle(Tage, Qend, Le_in, modl, sspace, opts)

    # Construct matrices to implement Tan (2020) algorithm
    Nd = size(sspace.sf,1)
    Gammamat  = spzeros(Nd,Nd)
    Nend      = sspace.Nk*sspace.Nm*sspace.Nb
    for iiz=1:sspace.Nz
        Gammamat[((iiz-1)*Nend+1):(iiz*Nend),((iiz-1)*Nend+1):(iiz*Nend)] = Qend[((iiz-1)*Nend+1):(iiz*Nend),:]
    end

    # Normalize distribution up to measure 1
    Le_loc = Le_in / sum(Le_in);

    # Construct collector of distributions
    L_age_col = sparse(zeros(Nd, Tage));

    # Initialize the first age distribution as the entrants'
    L_age_col[:,1] = Le_loc;
    # Initialize La_last to the "previous cohort"
    La_last = Le_loc;
    # Start loop
    for ii=2:Tage
        Ltil      = Gammamat'*La_last
        Lamtil    = reshape(Ltil, Nend, sspace.Nz)
        La_last   = reshape(Lamtil*sspace.P, Nd)
        L_age_col[:,ii] = La_last
    end

    return L_age_col
end
# Function to characterize average firm's life cycle, up to age "Tage" (quarterly)
function char_lifecycle(Tage, fsoln_f, aeq, modl, sspace, opts)

    # Initialize some steady state prices
    pQ_ss       = 1.0
    Rc_ss       = (1.0+modl.rb)/(modl.Pi*pQ_ss)

    # Start as if solving for the steady state L
    sf      = sspace.sf
    Nd      = size(sf,1)

    # Construct measure of entrants
    fspaceergk  = fundef([:spli, sspace.k_gridf, 0, 1])
    fspaceergm  = fundef([:spli, sspace.m_gridf, 0, 1])
    fspaceergb  = fundef([:spli, sspace.b_gridf, 0, 1])
    # Generating the mass of firms entering
    QKe         = funbase(fspaceergk, ones(Nd)*modl.k_0)
    QMe         = funbase(fspaceergm, ones(Nd)*modl.m_0)
    QBe         = funbase(fspaceergb, ones(Nd)*modl.b_0)
    QZe         = ones(Nd) * transpose(modl.z_0_dist)

    # Multiply up into "full" transition matrix
    Qe = row_kron(QZe, row_kron(QBe, row_kron(QMe, QKe)))

    Le          = Qe' * (modl.eta * ones(Nd) / Nd)

    # Compute age-conditional distribution collection
    L_age_col = gen_L_lifecycle(Tage, aeq.Qend, Le, modl, sspace, opts);

    # Plot age-specific mean characteristics
    # Create help vectors for simpler aggregation
    b_trgrid        = sf[:,3].*sf[:,1] / Rc_ss; # sf-length grid for LEVEL of debt
    lev_trgrid      = - b_trgrid ./ (sf[:,1] + sf[:,2]);
    cheat_trgrid    = sf[:,2] ./ (sf[:,1] + sf[:,2]);

    # Capital stock
    k_age_col       = (sf[:,1]' * L_age_col)';
    # Debt
    b_age_col     = (b_trgrid' * L_age_col)';
    # Leverage
    lev_age_col     = (lev_trgrid' * L_age_col)';
    # Liquidity ratio
    cheat_age_col   = (cheat_trgrid' * L_age_col)';
    # z draw
    z_age_col       = (sf[:,4]' * L_age_col)';
    # Dividends
    d_age_col       = (reshape(fsoln_f.Ba_prob .* fsoln_f.dA + (1.0 .- fsoln_f.Ba_prob) .* fsoln_f.dN, Nd)' * L_age_col)';

    # Plot
    manl_ftsize = 13
    fg = Plots.plot(1:Tage, [k_age_col, -b_age_col, lev_age_col, cheat_age_col, d_age_col, z_age_col], layout=(2,3), title=["Capital" "Debt" "Leverage ratio" "Liquidity ratio" "Dividends" "Productivity (\$z\$)" ], legend=:none, size=(800,460), linewidth=2, linecolor=:blue, xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize-1, legendfontsize=manl_ftsize-2, xticks=0:20:Tage, bottom_margin = 4Plots.mm)
    fg = Plots.plot!(subplot=4, xlabel="Age")
    fg = Plots.plot!(subplot=5, xlabel="Age")
    fg = Plots.plot!(subplot=6, xlabel="Age")
    Plots.savefig(joinpath(opts.project_root, "output", "figures", "model_figures", "FigB1_" * modl.name * "_avglifecycle.pdf"))

    # Also, plot the content for Footnote 87:
    diss_1q = L_age_col[:,1]' * reshape(fsoln_f.Ba_prob,Nd);
    diss_2q = L_age_col[:,2]' * reshape(fsoln_f.Ba_prob,Nd);

    # Print output (Footnote 46)
    open(joinpath(opts.project_root, "output", "other", "Footnote87_justification.txt"), "w") do io
        println(io, "In the quarter of birth, about $(round(Int,100*round(diss_1q,digits=1)))% of firms issue debt. In the second quarter of life, about $(round(Int,100*round(diss_2q,digits=1)))% do so.")
    end
end

# Function to characterize public vs private firm averages
function char_pubvspriv(Lin_pub, fsoln_f, aeq, modl, sspace, opts)

    # Start as if solving for the steady state L
    sf      = sspace.sf
    Nd      = size(sf,1)

    # Construct measure of entrants
    fspaceergk  = fundef([:spli, sspace.k_gridf, 0, 1])
    fspaceergm  = fundef([:spli, sspace.m_gridf, 0, 1])
    fspaceergb  = fundef([:spli, sspace.b_gridf, 0, 1])
    # Generating the mass of firms entering
    QKe         = funbase(fspaceergk, ones(Nd)*modl.k_0)
    QMe         = funbase(fspaceergm, ones(Nd)*modl.m_0)
    QBe         = funbase(fspaceergb, ones(Nd)*modl.b_0)
    QZe         = ones(Nd) * transpose(modl.z_0_dist)

    # Multiply up into "full" transition matrix
    Qe = row_kron(QZe, row_kron(QBe, row_kron(QMe, QKe)))
    Le          = Qe' * (modl.eta * ones(Nd) / Nd)

    Lin_priv    = gen_L_ageint(1, 20, aeq.Qend, Le, modl, sspace, opts); # Create conditional distribution of the private firms
    Lin_priv_mass = sum(Lin_priv);
    Lin_priv    = Lin_priv / Lin_priv_mass; # Normalize to unit mass

    # Compute public-private mean differences in selected characteristics
    # Create help vectors for simpler aggregation
    cheat_trgrid    = sf[:,2] ./ (sf[:,1] + sf[:,2]);

    # Liquidity ratio
    cheat_avg_pub     = cheat_trgrid' * Lin_pub;
    cheat_avg_priv    = cheat_trgrid' * Lin_priv;
    # Age 
    age_avg_pub     = 20.0 - 1.0 + 1.0/modl.eta;
    age_avg_priv    = (1.0/(1.0-(1.0-modl.eta)^(20-1)))*((1.0 - ((1.0-modl.eta)^(20-1))*(1.0+modl.eta*(20-1)) ))/modl.eta;
    # Aggregated investment
    invA_grid = reshape(fsoln_f.kpA, Nd) - sf[:,1];
    invN_grid = reshape(fsoln_f.kpN, Nd) - sf[:,1];
    inv_sum_pub = (1.0-Lin_priv_mass)*(reshape(fsoln_f.Ba_prob, Nd).*invA_grid + (1.0.-reshape(fsoln_f.Ba_prob, Nd)).*invN_grid )'* Lin_pub;
    inv_sum_priv = Lin_priv_mass*(reshape(fsoln_f.Ba_prob, Nd).*invA_grid + (1.0.-reshape(fsoln_f.Ba_prob, Nd)).*invN_grid )'* Lin_priv;

    # Print output
    open(joinpath(opts.project_root, "output", "tables", "_partof_TabB5_model.txt"), "w") do io
        println(io, rpad("Difference in age (pub-priv):", 32), round((age_avg_pub - age_avg_priv)/4, digits=1))
        println(io, rpad("I^pub/I share:", 32), round(inv_sum_pub / (inv_sum_pub + inv_sum_priv), digits=3))
        println(io, rpad("Difference in liq (pub-priv):", 32), round(cheat_avg_pub - cheat_avg_priv, digits=3))
    end
end

# Function for group-specific IRF diagnostics, computing steady-state and shocked aggregate paths for firms in selected ex-ante liquidity groups, conditional on their initial-state distribution. 
# Returns group-specific capital IRFs together with steady-state capital-share weights (used for Table 2).
function gen_eqIRF_groupspecs(Tgen, qL_cut, qH_cut, irfsoln, fsoln, aeq, modl, sspace, opts)
    # Globals
    sf      = sspace.sf
    Nd = size(sf,1)
    
    # Collections that will contain aggregate equilibrium solution objects along path
    # For aggregate, in both SS and IRF
    L_col_SS, L_col_IRF           = zeros(length(aeq.L),Tgen), zeros(length(aeq.L),Tgen);
    Ya_col_SS, Ya_col_IRF         = Array{Float64}(undef, Tgen), Array{Float64}(undef, Tgen);
    Ka_col_SS, Ka_col_IRF         = Array{Float64}(undef, Tgen), Array{Float64}(undef, Tgen);
    Ma_col_SS, Ma_col_IRF         = Array{Float64}(undef, Tgen), Array{Float64}(undef, Tgen);
    # And for "H" and "L" groups
    L_H_col_SS, L_L_col_SS           = zeros(length(aeq.L),Tgen), zeros(length(aeq.L),Tgen);
    Ya_H_col_SS, Ya_L_col_SS         = Array{Float64}(undef, Tgen),  Array{Float64}(undef, Tgen);
    Ka_H_col_SS, Ka_L_col_SS         = Array{Float64}(undef, Tgen),  Array{Float64}(undef, Tgen);
    Ma_H_col_SS, Ma_L_col_SS         = Array{Float64}(undef, Tgen),  Array{Float64}(undef, Tgen);
    L_H_col_IRF, L_L_col_IRF           = zeros(length(aeq.L),Tgen), zeros(length(aeq.L),Tgen);
    Ya_H_col_IRF, Ya_L_col_IRF         = Array{Float64}(undef, Tgen),  Array{Float64}(undef, Tgen);
    Ka_H_col_IRF, Ka_L_col_IRF         = Array{Float64}(undef, Tgen),  Array{Float64}(undef, Tgen);
    Ma_H_col_IRF, Ma_L_col_IRF         = Array{Float64}(undef, Tgen),  Array{Float64}(undef, Tgen);

    # Looping forward, solve for the implied distributions of (k,m,b,z), conditional on initial distributions
    # Initial distribution of incumbents is simply the steady state one -- because no shock (except a direct transfer) can affect the firms' states (k,m,b,z)
    L_col_SS[:,1], L_col_IRF[:,1]   = aeq.Lin, aeq.Lin;

    # Specify the indicator that picks the H/L groups
    # Split based on liquidity (m/a) ratio percentiles
    ma_grid         = sf[:,2] ./ (sf[:,1] + sf[:,2])
    ma_Hcutoff       = discr_percs(ma_grid, aeq.Lin, qH_cut)[1];
    ma_Lcutoff       = discr_percs(ma_grid, aeq.Lin, qL_cut)[1];
    H_ind_vec       = ma_grid .>= ma_Hcutoff; 
    L_ind_vec       = ma_grid .< ma_Lcutoff; 

    L_H_col_SS[:,1], L_H_col_IRF[:,1] = H_ind_vec .* aeq.Lin, H_ind_vec .* aeq.Lin;
    L_L_col_SS[:,1], L_L_col_IRF[:,1] = L_ind_vec .* aeq.Lin, L_ind_vec .* aeq.Lin;

    # Unpack the solution matrices for SS behavior into vector/matrix form, as necessary
    kpN_SS, kpA_SS = reshape(fsoln.kpN, Nd), reshape(fsoln.kpA, Nd);
    mpN_SS, mpA_SS = reshape(fsoln.mpN, Nd), reshape(fsoln.mpA, Nd);
    bpN_SS, bpA_SS = reshape(fsoln.bpN, Nd), reshape(fsoln.bpA, Nd);
    Ba_prob_SS  = reshape(fsoln.Ba_prob, Nd);

    # Use CompEcon tools to construct transition matrices on discretized space
    fspaceergk  = fundef([:spli, sspace.k_gridf, 0, 1])
    fspaceergm  = fundef([:spli, sspace.m_gridf, 0, 1])
    fspaceergb  = fundef([:spli, sspace.b_gridf, 0, 1])
    QK_N_SS, QK_A_SS  = funbase(fspaceergk, kpN_SS), funbase(fspaceergk, kpA_SS);
    QM_N_SS, QM_A_SS  = funbase(fspaceergm, mpN_SS), funbase(fspaceergm, mpA_SS);
    QB_N_SS, QB_A_SS  = funbase(fspaceergb, bpN_SS), funbase(fspaceergb, bpA_SS);

    # Construct product of QB*QM*QK only for endogenous choices
    Qend_N_SS = row_kron(QB_N_SS,row_kron(QM_N_SS,QK_N_SS))
    Qend_A_SS = row_kron(QB_A_SS,row_kron(QM_A_SS,QK_A_SS))
    # Given the adjustment probabilities "weigh up" into aggregate transition matrices of endogenous choices
    Qend_SS   = row_kron(reshape(1.0.-Ba_prob_SS,Nd,1), Qend_N_SS) + row_kron(sparse(reshape(Ba_prob_SS,Nd,1)), Qend_A_SS)
    # Make this "Qend" into a block matrix Gamma, analogously as in Tan (2020), but adjusting for my different ordering of states
    Gammamat_SS  = spzeros(Nd,Nd)
    Nend      = sspace.Nk*sspace.Nm*sspace.Nb
    for iiz=1:sspace.Nz
        Gammamat_SS[((iiz-1)*Nend+1):(iiz*Nend),((iiz-1)*Nend+1):(iiz*Nend)] = Qend_SS[((iiz-1)*Nend+1):(iiz*Nend),:]
    end

    # Start with time iterations forward
    for tt = 1:Tgen
        # Use the IRF solutions to construct transition matrix to next period
        # Unpack the solution matrices to vector form
        kpN_cur, kpA_cur = reshape(irfsoln.kpN_col[:,:,:,:,tt], Nd), reshape(irfsoln.kpA_col[:,:,:,:,tt], Nd)
        mpN_cur, mpA_cur = reshape(irfsoln.mpN_col[:,:,:,:,tt], Nd), reshape(irfsoln.mpA_col[:,:,:,:,tt], Nd)
        bpN_cur, bpA_cur = reshape(irfsoln.bpN_col[:,:,:,:,tt], Nd), reshape(irfsoln.bpA_col[:,:,:,:,tt], Nd)
        Ba_prob_cur  = reshape(irfsoln.Ba_prob_col[:,:,:,:,tt], Nd)
        # Use CompEcon tools to construct transition matrices on discretized space
        QK_N_cur, QK_A_cur  = funbase(fspaceergk, kpN_cur), funbase(fspaceergk, kpA_cur)
        QM_N_cur, QM_A_cur  = funbase(fspaceergm, mpN_cur), funbase(fspaceergm, mpA_cur)
        QB_N_cur, QB_A_cur  = funbase(fspaceergb, bpN_cur), funbase(fspaceergb, bpA_cur)

        # Construct product of QB*QM*QK only for endogenous choices!
        Qend_N_cur = row_kron(QB_N_cur,row_kron(QM_N_cur,QK_N_cur))
        Qend_A_cur = row_kron(QB_A_cur,row_kron(QM_A_cur,QK_A_cur))
        # Given the adjustment probabilities "weigh up" into aggregate transition matrices of endogenous choices
        Qend_cur   = row_kron(reshape(1.0.-Ba_prob_cur,Nd,1), Qend_N_cur) + row_kron(sparse(reshape(Ba_prob_cur,Nd,1)), Qend_A_cur)
        # Make this "Qend" into a block matrix Gamma, analogously as in Tan (2020), but adjusting for my different ordering of states
        Gammamat_cur  = spzeros(Nd,Nd)
        Nend      = sspace.Nk*sspace.Nm*sspace.Nb
        for iiz=1:sspace.Nz
            Gammamat_cur[((iiz-1)*Nend+1):(iiz*Nend),((iiz-1)*Nend+1):(iiz*Nend)] = Qend_cur[((iiz-1)*Nend+1):(iiz*Nend),:]
        end

        # Propagate from SS forward to construct incoming incumbents, with the "current t incumbents" propagated forward
        if tt<Tgen
            # For SS, aggregate
            Ltil_SS     = Gammamat_SS'*L_col_SS[:,tt]
            Lamtil_SS   = reshape(Ltil_SS, Nend, sspace.Nz)
            Lincumb_SS  = reshape(Lamtil_SS*sspace.P, Nd, 1)
            L_col_SS[:,tt+1] = reshape(Lincumb_SS, Nd)
            # For SS, H group
            Ltil_SS     = Gammamat_SS'*L_H_col_SS[:,tt]
            Lamtil_SS   = reshape(Ltil_SS, Nend, sspace.Nz)
            Lincumb_SS  = reshape(Lamtil_SS*sspace.P, Nd, 1)
            L_H_col_SS[:,tt+1] = reshape(Lincumb_SS, Nd)
            # For SS, L group
            Ltil_SS     = Gammamat_SS'*L_L_col_SS[:,tt]
            Lamtil_SS   = reshape(Ltil_SS, Nend, sspace.Nz)
            Lincumb_SS  = reshape(Lamtil_SS*sspace.P, Nd, 1)
            L_L_col_SS[:,tt+1] = reshape(Lincumb_SS, Nd)

            # For IRF
            Ltil_cur     = Gammamat_cur'*L_col_IRF[:,tt]
            Lamtil_cur   = reshape(Ltil_cur, Nend, sspace.Nz)
            Lincumb_cur  = reshape(Lamtil_cur*sspace.P, Nd, 1)
            L_col_IRF[:,tt+1] = reshape(Lincumb_cur, Nd)
            # For IRF, H group
            Ltil_cur     = Gammamat_cur'*L_H_col_IRF[:,tt]
            Lamtil_cur   = reshape(Ltil_cur, Nend, sspace.Nz)
            Lincumb_cur  = reshape(Lamtil_cur*sspace.P, Nd, 1)
            L_H_col_IRF[:,tt+1] = reshape(Lincumb_cur, Nd)
            # For IRF, L group
            Ltil_cur     = Gammamat_cur'*L_L_col_IRF[:,tt]
            Lamtil_cur   = reshape(Ltil_cur, Nend, sspace.Nz)
            Lincumb_cur  = reshape(Lamtil_cur*sspace.P, Nd, 1)
            L_L_col_IRF[:,tt+1] = reshape(Lincumb_cur, Nd)
        end

        # Computing the aggregates
        # For SS
        Ya_col_SS[tt], Ya_H_col_SS[tt], Ya_L_col_SS[tt] = L_col_SS[:,tt]' * reshape(fsoln.Y[:,:,:,:], Nd), L_H_col_SS[:,tt]' * reshape(fsoln.Y[:,:,:,:], Nd), L_L_col_SS[:,tt]' * reshape(fsoln.Y[:,:,:,:], Nd);
        Ka_col_SS[tt], Ka_H_col_SS[tt], Ka_L_col_SS[tt] = L_col_SS[:,tt]' * sf[:,1], L_H_col_SS[:,tt]' * sf[:,1], L_L_col_SS[:,tt]' * sf[:,1];
        Ma_col_SS[tt], Ma_H_col_SS[tt], Ma_L_col_SS[tt] = L_col_SS[:,tt]' * sf[:,2], L_H_col_SS[:,tt]' * sf[:,2], L_L_col_SS[:,tt]' * sf[:,2];
        # For IRF
        Ya_col_IRF[tt], Ya_H_col_IRF[tt], Ya_L_col_IRF[tt]  = L_col_IRF[:,tt]' * reshape(irfsoln.Y_col[:,:,:,:,tt], Nd),  L_H_col_IRF[:,tt]' * reshape(irfsoln.Y_col[:,:,:,:,tt], Nd),  L_L_col_IRF[:,tt]' * reshape(irfsoln.Y_col[:,:,:,:,tt], Nd)
        Ka_col_IRF[tt], Ka_H_col_IRF[tt], Ka_L_col_IRF[tt]  = L_col_IRF[:,tt]' * sf[:,1],  L_H_col_IRF[:,tt]' * sf[:,1],  L_L_col_IRF[:,tt]' * sf[:,1]; 
        Ma_col_IRF[tt], Ma_H_col_IRF[tt], Ma_L_col_IRF[tt]  = L_col_IRF[:,tt]' * sf[:,2],  L_H_col_IRF[:,tt]' * sf[:,2],  L_L_col_IRF[:,tt]' * sf[:,2]; 
    end

    # Compute impulse responses of groups' aggregates
    ldev_Ka_col, ldev_Ka_H_col, ldev_Ka_L_col = 100*log.(Ka_col_IRF./Ka_col_SS),  100*log.(Ka_H_col_IRF./Ka_H_col_SS),  100*log.(Ka_L_col_IRF./Ka_L_col_SS);
    # Compute shares of groups' capital in aggregate capital for decomposition
    Ka_H_SSshare_col, Ka_L_SSshare_col = Ka_H_col_SS./Ka_col_SS,  Ka_L_col_SS./Ka_col_SS;

    # Pack relevant moments up
    return [ldev_Ka_col ldev_Ka_H_col ldev_Ka_L_col], [ones(Tgen) Ka_H_SSshare_col Ka_L_SSshare_col]
end

# Function for generating a panel of firms' leverage ratios and their log(k) deviations from steady state, analogously as for baseline liq ratio regressions, conditional on a given IRF scenario, as embedded in "irfsoln"-argument (used for Figure B.5) 
function sim_IRFSS_respbinscat(Nfirms, T, Lin_arg, fsoln, irfsoln, printwarnings, modl, sspace, opts)

    # Set seed
    Random.seed!(999)
    # Initialize
    zi_col, xi_cdf_col =  Array{Int64,2}(undef, Nfirms,T), zeros(Nfirms, T)
    mcz         = MarkovChain(sspace.P)

    # Draw initial distribution
    # Distribution from given Lin
    ii_samp = wsample(1:sspace.Nsf, Lin_arg, Nfirms)
    se_init = sspace.sf[ii_samp, 1:3]
    xi_cdf_col = rand(Nfirms,T)
    zi_col_init= repeat(1:sspace.nf[4], inner=sspace.nf[1]*sspace.nf[2]*sspace.nf[3])[ii_samp]
    # Simulate MChains
    for ii in 1:Nfirms
        zi_col[ii,:]   = simulate(mcz,T,init=zi_col_init[ii])
    end

    # First, simulate paths in IRF and SS
    sim_paths_SS  = sim_firm_SS(se_init, zi_col, xi_cdf_col, fsoln, printwarnings, modl, sspace, opts)
    sim_paths_IRF = sim_firm_IRF(se_init, zi_col, xi_cdf_col, irfsoln, printwarnings, modl, sspace, opts)

    # Read T from the length of these paths
    T = size(zi_col,2)

    # For computing true b/k ratio at market debt values, compute the Rc_ss
    pQ_ss = 1.0
    Rc_ss = (1.0+modl.rb)/(modl.Pi*pQ_ss)

    # Compile dataframe to be returned
    data        = DataFrame(k=se_init[:,1], m=se_init[:,2], klev=-se_init[:,3]./Rc_ss)
    data.lev    = allowmissing((data.klev.*data.k) ./ (data.k + data.m))

    # Add time-series data for log(k) deviations, and drop outliers, as in empirics
    for tt=1:T
        str_temp         = "lk_dev_t$(tt)"
        data[!,str_temp] = allowmissing(100*log.(sim_paths_IRF.k_path[:,tt]./sim_paths_SS.k_path[:,tt]))
        q_l_temp, q_h_temp = quantile(data[!,str_temp],0.01), quantile(data[!,str_temp],0.99)
        ind_l_temp, ind_h_temp = data[!,str_temp] .<= q_l_temp, data[!,str_temp] .>= q_h_temp
        if tt>1
            data[!,str_temp][ind_l_temp] .= missing
            data[!,str_temp][ind_h_temp] .= missing
        end
    end

    return data
end