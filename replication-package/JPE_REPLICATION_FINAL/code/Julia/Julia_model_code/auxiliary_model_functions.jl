# Module containing all relevant functions for solving infinite-horizon firm problem, computing stationary distribution, aggregation, and all other procedures

module auxiliary_model_functions

using Plots
using Interpolations
import QuantEcon.rouwenhorst
import QuantEcon.MarkovChain, QuantEcon.simulate
using LinearAlgebra, SparseArrays
using CompEcon
using Roots
using Distributions
using DataFrames, GLM, CSV
using JLD

export manprintln, lognorm_discrshift, Options, StateSpace, Model, setup_MarkovZ, setup!, solve_V_EGM, solve_V0, solve_VN, solve_V2N, solve_V1N, solve_VA, solve_V3A, solve_V2A, solve_V1A, gen_ar1, solve_IRF, sim_firm_SS, sim_firm_IRF, sim_firm_IRF_dev, solve_Leq, gen_L_Tage, xi_cdfun, xi_condexpn, ymenufun, ymenufun_mat, solve_eqIRF_NKMg_xsh, solve_eqIRF_NKMg_xsh_rmsh, solve_eqIRF_decomp, sim_IRFSS_anl_rand, solve_Leq_keeppsi, solve_Leq_endp

# Extra commands for printing etc
function manprintln(s::String, filename::String)
    println(s)
    read_tmp = read(filename, String)
    write(filename, read_tmp * s * "\n")
end

# Function for discretizing logN(mu_zs, sig_zs) onto set of gridpoints given by "lz_grid". mu_zs and sig_zs are the unconditional mean and standard deviation of the log of the variable whose distribution is to be discretized
function lognorm_discrshift(mu_zs, sig_zs, lz_grid)
    # Compute midpoints
    lz_midp     = 0.5*(lz_grid[1:end-1]+lz_grid[2:end])
    cdf_zs_midp = cdf.(Normal(), (lz_midp.-mu_zs)/sig_zs)

    # Combine all together into vector of masses
    zs_dist     = vcat(cdf_zs_midp[1], diff(cdf_zs_midp), 1.0-cdf_zs_midp[end])

    return zs_dist
end

# Computation options
mutable struct Options

    # Tolerances, iterations
    Nbell::Integer      # Number of Bellman (Contraction) iterations
    tolV::Float64       # Tolerance on value functions
    phi_V::Array{Float64,1}      # Weight on previous value functions when updating guess; array with 4 entries, corresponding to [Vk,Vm,Vb,V]
    itermaxL::Integer   # Maximum iterations to find stationary dist L
    tolL::Float64       # Tolerance on L

    # Print / plot
    prnt::String        # Print out V-solution convergence and other textual outputs

    # Solving equilibrium
    tolp::Float64       # Tolerance on price
    phi_peq::Float64    # Weight on previous p when updating steady state p
    eqplot::String
    eqprint::String
    LoadV::String       # For new guess of p use old V as starting guess
    flag_Vconv::Integer

    # IRF solution
    tol_p_col::Float64      # Tolerance on paths for IRF solution
    itermax_p_col::Int64    # Max iterations of loop for IRF solution
    phi_p::Float64          # Weight on previous paths when updating whole path of IRFs
    phi_pQ_acl::Float64     # Acceleration weight on updating path for pQ relative to updating MU (if <1, DEceleration)

    # Related to Sobol calibration
    n_job::Integer          # The number of the job in a given "calibX_" run

    # Specify project_root directory
    project_root::String   # Top-level project directory
end

# Statespace parameters
mutable struct StateSpace
    n::Array{Integer,1}     # Number of nodes in each (k,m,b,z) dimension
    nf::Array{Integer,1}    # Number of points for (k,m,b,z) in histogram L
    naux::Array{Integer,1}  # Number of nodes in each auxiliary (a, atil, btil, mtil) dimension
    nlm::Array{Integer,1}   # Number of nodes in each lagrange multiplier (mu, chiS, chiB) dimension
    curv::Array{Float64,1}  # Curvature for (k,m,b) grids (1 is no curvature)
    curvf::Array{Float64,1}  # Curvature for fine (k,m,b) grids for histogram (1 is no curvature)
    curvaux::Array{Float64,1}  # Curvature for auxiliary (a, atil, btil, mtil) grids (1 is no curvature)
    curvlm::Array{Float64,1}  # Curvature for Lagrange multiplier (mu, chiS, chiB) grids (1 is no curvature)
    spliorder::Array{Integer,1} # Order of splines: [1,1,1,1] used in calling some CompEcon functions
    k_min::Float64           # Lower bound on k
    k_max::Float64           # Upper bound on k
    m_min::Float64           # Lower bound on m
    m_max::Float64           # Upper bound on m
    b_min::Float64           # Lower bound on (real-rate) "adjusted" leverage b
    b_max::Float64           # Upper bound on (real-rate) "adjusted" leverage b

    a_min::Float64           # Lower bound on liquid financial resources (per capital!): A problem
    a_max::Float64           # Upper bound on liquid financial resources (per capital!): A problem
    atil_min::Float64        # Lower bound on A's available resources (per capital!) after paying dividends
    atil_max::Float64        # Upper bound on A's available resources (per capital!) after paying dividends
    btil_min::Float64        # Lower bound on A's available resources (per capital!) after choosing m'
    btil_max::Float64        # Upper bound on A's available resources (per capital!) after choosing m'
    mtil_min::Float64        # Lower bound on N's available resources (per capital!) after paying dividends
    mtil_max::Float64        # Upper bound on N's available resources (per capital!) after paying dividends
    mu_min::Float64          # Lower bound on Lagrange multiplier on (m' ≥ 0) for N
    mu_max::Float64          # Upper bound on Lagrange multiplier on (m' ≥ 0) for N
    muk_min::Float64         # Lower bound on Lagrange multiplier on (b' ≥ -θ) for N
    muk_max::Float64         # Upper bound on Lagrange multiplier on (b' ≥ -θ) for N
    chiS_min::Float64        # Lower bound on Lagrange multiplier on (b' ≤ 0) for A
    chiS_max::Float64        # Upper bound on Lagrange multiplier on (b' ≤ 0) for A
    chiB_min::Float64        # Lower bound on Lagrange multiplier on (b' ≥ -θ) for A
    chiB_max::Float64        # Upper bound on Lagrange multiplier on (b' ≥ -θ) for A

    k_minf::Float64           # Lower bound on k in fine GRID for histogram
    k_maxf::Float64           # Upper bound on k in fine GRID for histogram
    m_minf::Float64           # Lower bound on m in fine GRID for histogram
    m_maxf::Float64           # Upper bound on m in fine GRID for histogram
    b_minf::Float64           # Lower bound on (real-rate) "adjusted" leverage in fine GRID for histogram
    b_maxf::Float64           # Upper bound on (real-rate) "adjusted" leverage in fine GRID for histogram

    # Added below
    k_gridf::Array{Float64,1}                # Fine grid for k
    m_gridf::Array{Float64,1}                # Fine grid for m
    b_gridf::Array{Float64,1}                # Fine grid for b
    z_gridf::Array{Float64,1}                # Fine grid for z
    sf::Array{Float64,2}                    # Fine composite grid (k,m,b,z)
    Nsf::Int64                              # Length of fine composite grid
    Psszf::Array{Float64,1}                 # Stationary, iterated z-transition matrix
    QZ::SparseMatrixCSC{Float64,Int64}      # Transition matrix for firm z states
    k_grid0::Array{Float64,1}                # To save initial grid for computing basis matrices b/c cubic spline extends grid
    k_grid::Array{Float64,1}
    m_grid0::Array{Float64,1}                # To save initial grid for computing basis matrices b/c cubic spline extends grid
    m_grid::Array{Float64,1}
    b_grid0::Array{Float64,1}
    b_grid::Array{Float64,1}
    z_grid0::Array{Float64,1}
    z_grid::Array{Float64,1}

    a_grid::Array{Float64,1}
    atil_grid::Array{Float64,1}
    btil_grid::Array{Float64,1}
    mtil_grid::Array{Float64,1}
    mu_grid::Array{Float64,1}
    muk_grid::Array{Float64,1}
    chiS_grid::Array{Float64,1}
    chiB_grid::Array{Float64,1}
    k_Egrid::Array{Float64,4}               # Full, extended matrix of k in KxMxBxZ-shape
    m_Egrid::Array{Float64,4}               # Full, extended matrix of m in KxMxBxZ-shape
    b_Egrid::Array{Float64,4}               # Full, extended matrix of b in KxMxBxZ-shape
    z_Egrid::Array{Float64,4}               # Full, extended matrix of z in KxMxBxZ-shape

    P::Array{Float64,2}                     # Transition matrix for z
    Pssz::Array{Float64,1}                  # Stationary transition matrix for z
    Nk::Int64                               # Length of k-grid
    Nm::Int64                               # Length of m-grid
    Nb::Int64                               # Length of b-grid
    Nz::Int64                               # Length of z-grid
    Na::Int64                               # Length of a-grid
    Natil::Int64                            # Length of atil-grid
    Nbtil::Int64                            # Length of btil-grid
    Nmtil::Int64                            # Length of mtil-grid
    Nmu::Int64                              # Length of mu-grid
    Nmuk::Int64                             # Length of muk-grid
    NchiS::Int64                            # Length of chiS-grid
    NchiB::Int64                            # Length of chiB-grid
    s::Array{Float64,2}                     # Composite grid (k,m,b,z) combination
    Ns::Int64                               # Length of composite grid
end

# Model parameters
mutable struct Model
    beta::Float64           # Discount factor
    psi::Float64            # Coefficient on disutility of labor
    varpi::Float64          # Coefficient of household's relative risk aversion
    rhoz::Float64           # Persistence of idiosyncratic TFP
    sige::Float64           # Std of iid shock to idiosycratic TFP
    A::Float64              # Aggregate TFP
    alpha::Float64          # Exponent on k in production function
    nu::Float64             # Exponent on n in production function
    delta::Float64          # Depreciation rate
    rm::Float64             # Net NOMINAL rate on liquid asset
    rf::Float64             # Net NOMINAL rate on illiquid asset
    bspread::Float64        # Spread of net NOMINAL borrowing rate over net NOMINAL rate on liquid asset
    rb::Float64             # Net NOMINAL borrowing rate -- just place-filler for easy reference
    Pi::Float64             # Steady state GROSS inflation rate
    theta::Float64          # Borrowing limit (on leverage)
    gamma::Float64          # "Fundamental" debt maturity
    kappa::Float64          # Convex capital adjustment cost parameter
    eta::Float64            # Probability of exogenous exit
    xi_lb::Float64          # Lower bound of xi in uniform distribution
    xi_ub::Float64          # Upper bound of xi in uniform distribution
    xi_dist::String         # String ("uniform", or "logN"), denoting distribution of debt issuance cost
    tau::Float64            # Corporate tax rate
    phi_w::Float64          # Coefficient on wage payment working capital constraint
    k_0::Float64            # Entrants' k
    m_0::Float64            # Entrants' m
    b_0::Float64            # Entrants' "adjusted" leverage
    z_0_dist::Array{Float64,1} # Entrants' distribution of productivity
    phi_k::Float64          # Capital price elasticity
    phi_pi::Float64         # Responsiveness in Taylor rule
    kappa_p::Float64        # Slope of NKPC
    phi_m::Float64          # SS exposure of nominal rm to nominal rf
    phi_mresp::Float64      # IRF exposure of nominal rm to nominal rf
    zh::Float64             # High added TFP level
    pzh::Float64            # Probability of transitioning to zh from any current TFP, if ==0, not added
    Vresult::Array{Float64,4} # Container to keep value function solution as initial guess -- in the shape of V0e matrix!
    Lresult::SparseVector{Float64,Int64} # Container to keep steady state distribution as initial guess
    name::String            # Model name for saving files
end

# Container for individual firm solution, on full 4-dimensional grid, given prices, contains BOTH extensive margin policies
mutable struct FirmSolution
    kpN::Array{Float64,4}
    mpN::Array{Float64,4}
    bpN::Array{Float64,4}
    dN::Array{Float64,4}
    kpA::Array{Float64,4}
    mpA::Array{Float64,4}
    bpA::Array{Float64,4}
    dA::Array{Float64,4}
    Ba_prob::Array{Float64,4}
    V0ek::Array{Float64,4}          
    V0em::Array{Float64,4}
    V0eb::Array{Float64,4}
    V0e::Array{Float64,4}
    Yn::Array{Float64,4}                # Real operational profits
    Y::Array{Float64,4}                 # Output
    n::Array{Float64,4}                 # Labor
    ACN::Array{Float64,4}               # Adjustment costs for non-issuer
    ACA::Array{Float64,4}               # Adjustment costs for issuer
end

# Firms' responses/simulations container
mutable struct FirmSimResult
    zi_col::Array{Float64,2}        # Container (NxT) of firm's faced paths of idiosyncratic TFP levels
    xi_cdf_col::Array{Float64,2}    # Container (NxT) of firm's faced paths of issuance cost CDF values
    k_path::Array{Float64,2}          # Container (NxT) of firm's chosen k paths (includes initial condition, as for m and b)
    m_path::Array{Float64,2}          # Container (NxT) of firm's chosen m paths
    b_path::Array{Float64,2}          # Container (NxT) of firm's chosen b paths
    d_path::Array{Float64,2}          # Container (NxT) of firm's chosen dividend paths
    Ba_prob_path::Array{Float64,2}    # Container (NxT) of firm's probabilities of debt adjustment
    Ba_path::Array{Float64,2}         # Container (NxT) of firm's indicator of debt adjustment
    Yn_path::Array{Float64,2}          # Container (NxT) of firm's chosen Yn paths
    Y_path::Array{Float64,2}          # Container (NxT) of firms' chosen Y paths
    n_path::Array{Float64,2}          # Container (NxT) of firms' chosen n paths
    AC_path::Array{Float64,2}          # Container (NxT) of firms' chosen adjustment cost paths
end

# Container for aggregate equilibrium objects, including distribution, transition matrix, and marginal distributions
mutable struct AggEqSolution
    p::Float64                  # Price inferred from household consumption
    L::SparseVector{Float64,Int64}  # Unconditional distribution across individual states
    Ya::Float64                 # Aggregate output
    Na::Float64                 # Aggregate labor
    Ca::Float64                 # Aggregate consumption
    Ka::Float64                 # Agg capital
    Ba::Float64                 # Agg firms' borrowing
    Ma::Float64                 # Agg firms' m holdings
    ACa::Float64                # Agg firms' k adjustment costs
    Q::SparseMatrixCSC{Float64,Int64}         # Transition matrix
    Qend::SparseMatrixCSC{Float64,Int64}      # Transition matrix for "endogenous states" only! (To implement Tan (2020) approach)
    Lin::SparseVector{Float64,Int64}       # Unconditional distribution of "incumbents"
    Lk::Array{Float64,1}        # Unconditional distribution on k
    Lm::Array{Float64,1}        # Unconditional distribution on m
    Lb::Array{Float64,1}        # Unconditional distribution on b
    Lz::Array{Float64,1}        # Unconditional distribution on z
end

# Container for firm's solution of transition path after MIT shock
mutable struct IRFSolution
    # For convenience, also keep all ingoing price paths
    MU_col::Array{Float64,1}        # Path of gross inverse markup
    w_col::Array{Float64,1}         # Path of real wage
    M_tp1_col::Array{Float64,1}     # Path of owner SDF, M_{t+1}
    pQ_col::Array{Float64,1}        # Path of capital price
    Pi_col::Array{Float64,1}        # Path of gross inflation
    rb_col::Array{Float64,1}        # Path of rb
    rm_col::Array{Float64,1}        # Path of rm
    Rc_col::Array{Float64,1}        # Path of Rc
    gammatil_col::Array{Float64,1}  # Path of gammatil
    # Actual solution:
    kpN_col::Array{Float64,5}       # Container of values for kpN policy on full (KxMxBxZ)xT-grid
    mpN_col::Array{Float64,5}        
    bpN_col::Array{Float64,5}
    dN_col::Array{Float64,5}
    kpA_col::Array{Float64,5}
    mpA_col::Array{Float64,5}
    bpA_col::Array{Float64,5}
    dA_col::Array{Float64,5}
    Ba_prob_col::Array{Float64,5}
    V0ek_col::Array{Float64,5}      # Container of values for V0ek_t on full (KxMxBxZ)xT-grid
    V0em_col::Array{Float64,5}      # Container of values for V0em_t on full (KxMxBxZ)xT-grid
    V0eb_col::Array{Float64,5}      # Container of values for V0eb_t on full (KxMxBxZ)xT-grid
    V0e_col::Array{Float64,5}       # Container of values for V0e_t on full (KxMxBxZ)xT-grid
    Yn_col::Array{Float64,5}        # Real operational profits
    Y_col::Array{Float64,5}         # Output
    n_col::Array{Float64,5}         # Labor
    ACN_col::Array{Float64,5}       # Adjustment costs for non-issuer
    ACA_col::Array{Float64,5}       # Adjustment costs for issuer
end

# Container for aggregate IRF equilibrium objects path, including distribution, transition matrix, and marginal distributions
mutable struct AggIRFSolution
    p_col::Array{Float64,1}     # Path for p
    rf_col::Array{Float64,1}    # Path for rf
    rm_col::Array{Float64,1}    # Path for rm
    rb_col::Array{Float64,1}    # Path for rb
    Pi_col::Array{Float64,1}    # Path for gross inflation
    MU_col::Array{Float64,1}   # Path for gross markup
    pQ_col::Array{Float64,1}    # Path of capital price
    L_col::Array{Float64,2}     # Path for unconditional distribution across individual states
    Ya_col::Array{Float64,1}  # Path for Aggregate output
    Na_col::Array{Float64,1}  # Path for Aggregate labor
    Ca_col::Array{Float64,1}  # Path for Aggregate consumption
    Ka_col::Array{Float64,1}  # Path for Aggregate capital
    Ba_col::Array{Float64,1}  # Path for Aggregate firms' borrowing
    Ma_col::Array{Float64,1}  # Path for Aggregate firms' m holdings
    ACa_col::Array{Float64,1}  # Path for Aggregate firms' K adjustment costs
    Lin_col::Array{Float64,2}       # Path for unconditional distribution across individual states for incumbents in a given state
    Lk_col::Array{Float64,2}        # Path for Unconditional distribution on k
    Lm_col::Array{Float64,2}        # Path for Unconditional distribution on m
    Lb_col::Array{Float64,2}        # Path for Unconditional distribution on b
    Lz_col::Array{Float64,2}        # Path for Unconditional distribution on z
end

# Set up the types
function Options(; Nbell::Integer=100, tolV::Float64=1e-5, phi_V::Array{Float64,1}=[0.0,0.0,0.0,0.0], itermaxL::Integer=500, tolL::Float64=1e-10, prnt::String="Y", tolp::Float64=1e-4, phi_peq::Float64=0.9, eqplot::String="N", eqprint::String="Y", LoadV::String="N", flag_Vconv::Integer=0, tol_p_col::Float64=1e-4, itermax_p_col::Int64=100, phi_p::Float64=0.9, phi_pQ_acl::Float64=1.0, n_job::Integer=999, project_root::String="")

    opts = Options(Nbell, tolV, phi_V, itermaxL, tolL, prnt, tolp, phi_peq, eqplot, eqprint, LoadV, flag_Vconv, tol_p_col, itermax_p_col, phi_p, phi_pQ_acl, n_job, project_root)

    return opts
end

function StateSpace(; n::Array{Int64,1}=[4,4,4,4], nf::Array{Int64,1}=[100,100,100,4], naux::Array{Int64,1}=[4,4,4,4], nlm::Array{Int64,1}=[4,4,4], curv::Array{Float64,1}=[1.0,1.0,1.0], curvf::Array{Float64,1}=[1.0,1.0,1.0], curvaux::Array{Float64,1}=[1.0,1.0,1.0,1.0], curvlm::Array{Float64,1}=[1.0,1.0,1.0,1.0], spliorder::Array{Int64,1}=[1,1,1,1], k_min::Float64=0.1, k_max::Float64=20.0, m_min::Float64=0.0, m_max::Float64=0.5, b_min::Float64=-0.5, b_max::Float64=0.0, a_min::Float64=0.1, a_max::Float64=20.0, atil_min::Float64=0.1, atil_max::Float64=20.0, btil_min::Float64=0.1, btil_max::Float64=20.0, mtil_min::Float64=0.1, mtil_max::Float64=20.0, mu_min::Float64=0.0, mu_max::Float64=20.0, muk_min::Float64=0.0, muk_max::Float64=20.0, chiS_min::Float64=0.0, chiS_max::Float64=20.0, chiB_min::Float64=0.0, chiB_max::Float64=20.0, k_minf::Float64=0.1, k_maxf::Float64=20.0, m_minf::Float64=0.0, m_maxf::Float64=0.5, b_minf::Float64=-0.5, b_maxf::Float64=0.0, k_gridf=Array{Float64}(undef, 0), m_gridf=Array{Float64}(undef, 0), b_gridf=Array{Float64}(undef, 0), z_gridf=Array{Float64}(undef, 0), sf=Array{Float64}(undef, 0,0), Nsf=0, Psszf=Array{Float64}(undef, 0),  QZ=spzeros(0,0), k_grid0=Array{Float64}(undef, 0), k_grid=Array{Float64}(undef, 0), m_grid0=Array{Float64}(undef, 0), m_grid=Array{Float64}(undef, 0), b_grid0=Array{Float64}(undef, 0), b_grid=Array{Float64}(undef, 0), z_grid0=Array{Float64}(undef, 0), z_grid=Array{Float64}(undef, 0), a_grid=Array{Float64}(undef, 0), atil_grid=Array{Float64}(undef, 0), btil_grid=Array{Float64}(undef, 0), mtil_grid=Array{Float64}(undef, 0), mu_grid=Array{Float64}(undef, 0), muk_grid=Array{Float64}(undef, 0), chiS_grid=Array{Float64}(undef, 0), chiB_grid=Array{Float64}(undef, 0), k_Egrid=Array{Float64}(undef, 0,0,0,0), m_Egrid=Array{Float64}(undef, 0,0,0,0), b_Egrid=Array{Float64}(undef, 0,0,0,0), z_Egrid=Array{Float64}(undef, 0,0,0,0), P=Array{Float64}(undef, 0,0), Pssz=Array{Float64}(undef, 0), Nk=4, Nm=4, Nb=4, Nz=4, Na=4, Natil=4, Nbtil=4, Nmtil=4, Nmu=4, Nmuk=4, NchiS=4, NchiB=4, s=Array{Float64}(undef, 0,0), Ns=0)

    sspace = StateSpace(n, nf, naux, nlm, curv, curvf, curvaux, curvlm, spliorder, k_min, k_max, m_min, m_max, b_min, b_max, a_min, a_max, atil_min, atil_max, btil_min, btil_max, mtil_min, mtil_max, mu_min, mu_max, muk_min, muk_max, chiS_min, chiS_max, chiB_min, chiB_max, k_minf, k_maxf, m_minf, m_maxf, b_minf, b_maxf, k_gridf, m_gridf, b_gridf, z_gridf, sf, Nsf, Psszf,  QZ, k_grid0, k_grid, m_grid0, m_grid, b_grid0, b_grid, z_grid0, z_grid, a_grid, atil_grid, btil_grid, mtil_grid, mu_grid, muk_grid, chiS_grid, chiB_grid, k_Egrid, m_Egrid, b_Egrid, z_Egrid, P, Pssz, Nk, Nm, Nb, Nz, Na, Natil, Nbtil, Nmtil, Nmu, Nmuk, NchiS, NchiB, s, Ns)

    return sspace
end

function Model(; beta::Float64=0.99, psi::Float64=2.4, varpi::Float64=1.0, rhoz::Float64=0.95, sige::Float64=0.01, A::Float64=1.0, alpha::Float64=0.256, nu::Float64=0.640, delta::Float64=0.069, rm::Float64=0.01, bspread::Float64=0.005, rb::Float64=0.015, Pi::Float64=1.0, theta::Float64=0.5, gamma::Float64=0.6, kappa::Float64=0.1, eta::Float64=0.1, xi_lb::Float64=0.0, xi_ub::Float64=0.0, xi_dist::String="uniform", tau::Float64=0.0, phi_w::Float64=0.0, k_0::Float64=1.0, m_0::Float64=0.0, b_0::Float64=0.0, z_0_dist=Array{Float64}(undef, 0), phi_k::Float64=0.0, phi_pi::Float64=1.5, kappa_p::Float64=0.1, phi_m::Float64=1.0, phi_mresp::Float64=1.0, zh::Float64=1.0, pzh::Float64=0.0, Vresult=Array{Float64}(undef, 0,0,0,0), Lresult=Array{Float64}(undef, 0), name::String="mod")

    modl = Model(beta, psi, varpi, rhoz, sige, A, alpha, nu, delta, phi_m*(Pi/beta - 1), Pi/beta - 1, bspread, Pi/beta - 1 + bspread, Pi, theta, gamma, kappa, eta, xi_lb, xi_ub, xi_dist, tau, phi_w, k_0, m_0, b_0, z_0_dist, phi_k, phi_pi, kappa_p, phi_m, phi_mresp, zh, pzh, Vresult, Lresult, name)

    return modl
end


# Setup Markov
function setup_MarkovZ(Nz,sige,rhoz,A)
    # Setting up Rouwenhorst approximation of AR1
    # Assuming that we are looking for a transition matrix for variable Z_t such that log(Z_t) follows an AR(1), with persistence "rhoz" and standard error of Normal shocks "sige", and E[Z_t]=A. "mu_uncond" refers to E[log(Z_t)], and "sig_uncond" refers to the std[log(Z_t)]
    # i.e. log Z_t = c_0 + rhoz * log Z_(t-1) + sige * eps_t
    c_0 = (1-rhoz)*log(A) - 0.5*(sige^2)/(1+rhoz)

    mchain = rouwenhorst(Nz, rhoz, sige, c_0)

    Pz = mchain.p
    zvec = exp.(mchain.state_values)
    Pssz = Pz^10000
    Pssz = Pssz[1,:]

    Pz = Pz./sum(Pz,dims=2)
    Pz[:,end]   = 1.0 .- sum(Pz[:,1:end-1],dims=2);  # Ensure sum to 1, up to machine epsilon.
    Pssz        = Pz^10000;
    Pssz        = Pssz[1,:];

    return Pz, zvec, Pssz
end

# Model setup
function setup!(modl, sspace, opts)
    # State space for idiosyncratic productivity
    Nz = sspace.n[4]
    P, z_grid, Pssz = setup_MarkovZ(Nz, modl.sige, modl.rhoz,1)

    # Now, if required, add extra zh state on top of the distribution
    if (modl.pzh > 0.0) & (modl.zh > z_grid[end])
        z_grid   = vcat(z_grid, modl.zh)
        P       = hcat((1.0-modl.pzh)*P, modl.pzh*ones(Nz))
        P       = vcat(P, zeros(Nz+1)')
        P[Nz+1,Nz+1] = 1.0

        # Update all references to Nz
        Nz          = Nz+1
        sspace.n[4] = Nz
        sspace.nf[4]= Nz
        Pssz        = ((1.0-modl.eta)*P + modl.eta*repeat(modl.z_0_dist',inner=[Nz,1]) )^1000
        Pssz        = Pssz[1,:]
    elseif (modl.pzh > 0.0) & (modl.zh < z_grid[1])
        z_grid   = vcat(modl.zh, z_grid)
        P       = hcat(modl.pzh*ones(Nz), (1.0-modl.pzh)*P)
        P       = vcat(modl.pzh*ones(Nz+1)', P)
        P[1,1]  = 1.0-modl.pzh*Nz

        # Update all references to Nz
        Nz          = Nz+1
        sspace.n[4] = Nz
        sspace.nf[4]= Nz
        Pssz        = ((1.0-modl.eta)*P + modl.eta*repeat(modl.z_0_dist',inner=[Nz,1]) )^1000
        Pssz        = Pssz[1,:]
    end

    z_grid0 = z_grid

    # State space for endogenous variables (k,m,b)
    Nk, Nm, Nb              = sspace.n[1:3]
    Na, Natil, Nbtil, Nmtil = sspace.naux
    Nmu, Nmuk, NchiS, NchiB = sspace.nlm
    sspace.Na, sspace.Natil, sspace.Nbtil, sspace.Nmtil = Na, Natil, Nbtil, Nmtil
    sspace.Nmu, sspace.Nmuk, sspace.NchiS, sspace.NchiB = Nmu, Nmuk, NchiS, NchiB

    curv        = sspace.curv
    curvf       = sspace.curvf
    curvaux     = sspace.curvaux
    curvlm      = sspace.curvlm

    k_grid      = range(sspace.k_min^curv[1], sspace.k_max^curv[1], length=Nk).^(1.0/curv[1])
    k_grid0     = k_grid     # Save for computing basis matrices (cubic spline extends grid); from old code
    m_grid      = range((sspace.m_min-sspace.m_min)^curv[2], (sspace.m_max-sspace.m_min)^curv[2], length=Nm).^(1.0/curv[2]) .+ sspace.m_min
    m_grid0     = m_grid     # Save for computing basis matrices (cubic spline extends grid)
    b_grid      = range((sspace.b_min-sspace.b_min)^curv[3], (sspace.b_max-sspace.b_min)^curv[3], length=Nb).^(1.0/curv[3]) .+ sspace.b_min # Because b-values are negative, shift them all to a positive range)
    b_grid0     = b_grid
    # Auxiliary grids
    sspace.a_grid      = range((sspace.a_min-sspace.a_min)^curvaux[1], (sspace.a_max-sspace.a_min)^curvaux[1], length=Na).^(1.0/curvaux[1]) .+ sspace.a_min
    sspace.atil_grid   = range((sspace.atil_min-sspace.atil_min)^curvaux[2], (sspace.atil_max-sspace.atil_min)^curvaux[2], length=Natil).^(1.0/curvaux[2]) .+ sspace.atil_min
    sspace.btil_grid   = range((sspace.btil_min-sspace.btil_min)^curvaux[3], (sspace.btil_max-sspace.btil_min)^curvaux[3], length=Nbtil).^(1.0/curvaux[3]) .+ sspace.btil_min
    sspace.mtil_grid   = range((sspace.mtil_min-sspace.mtil_min)^curvaux[4], (sspace.mtil_max-sspace.mtil_min)^curvaux[4], length=Nmtil).^(1.0/curvaux[4]) .+ sspace.mtil_min
    # Lagrange multiplier grids
    sspace.mu_grid     = range(sspace.mu_max^curvlm[1], sspace.mu_min^curvlm[1], length=Nmu).^(1.0/curvlm[1]) # FLIPPED ORDER
    sspace.muk_grid    = range(sspace.muk_max^curvlm[2], sspace.muk_min^curvlm[2], length=Nmuk).^(1.0/curvlm[2]) # FLIPPED ORDER
    sspace.chiS_grid   = range(sspace.chiS_min^curvlm[3], sspace.chiS_max^curvlm[3], length=NchiS).^(1.0/curvlm[3]) # NOT FLIPPED ORDER -- it's a SAVING constraint
    sspace.chiB_grid   = range(sspace.chiB_max^curvlm[4], sspace.chiB_min^curvlm[4], length=NchiB).^(1.0/curvlm[4]) # NOT FLIPPED ORDER

    # Function space and nodes (fspace adds knot if one was using cubic splines, not applicable in this case, given spliorder=[1,1,1,1])
    fspace      = fundef([:spli, k_grid, 0, sspace.spliorder[1]], [:spli, m_grid, 0, sspace.spliorder[2]], [:spli, b_grid, 0, sspace.spliorder[3]], [:spli, z_grid, 0, sspace.spliorder[4]])
    s, s_grid   = funnode(fspace)
    Ns          = size(s,1)

    # Reconstruct grids after fspace added points for the spline (two knot points for cubic spline); OLD CODE
    k_grid  = s_grid[1]
    m_grid  = s_grid[2]
    b_grid  = s_grid[3]
    z_grid  = s_grid[4]
    Nk      = size(k_grid,1)
    Nm      = size(m_grid,1)
    Nb      = size(b_grid,1)
    Nz      = size(z_grid,1)

    # Construct fine grid for histogram
    Nkf             = sspace.nf[1]
    Nmf             = sspace.nf[2]
    Nbf             = sspace.nf[3]
    k_gridf         = range(sspace.k_minf^curvf[1], sspace.k_maxf^curvf[1], length=Nkf).^(1.0/curvf[1])
    m_gridf         = range((sspace.m_minf-sspace.m_minf)^curvf[2], (sspace.m_maxf-sspace.m_minf)^curvf[2], length=Nmf).^(1.0/curvf[2]) .+ sspace.m_minf
    b_gridf         = range((sspace.b_minf-sspace.b_minf)^curvf[3], (sspace.b_maxf-sspace.b_minf)^curvf[3], length=Nbf).^(1.0/curvf[3]) .+ sspace.b_minf

    z_gridf         = z_grid
    Nzf             = length(z_gridf)
    sf              = gridmake(k_gridf,m_gridf,b_gridf,z_gridf)
    Nsf             = size(sf,1)

    sspace.k_gridf   = k_gridf
    sspace.m_gridf   = m_gridf
    sspace.b_gridf   = b_gridf
    sspace.z_gridf   = z_gridf
    sspace.sf       = sf
    sspace.Nsf      = Nsf

    # Compute QZ matrix for transition of stationary distribution
    sspace.QZ   = kron(P,ones(Nkf*Nmf*Nbf))

    # Declare additional global variables
    sspace.k_grid0   = k_grid0
    sspace.k_grid    = k_grid
    sspace.m_grid0   = m_grid0
    sspace.m_grid    = m_grid
    sspace.b_grid0   = b_grid0
    sspace.b_grid    = b_grid
    sspace.z_grid0   = z_grid0
    sspace.z_grid    = z_grid
    sspace.P        = P
    sspace.Pssz     = Pssz
    sspace.Nz       = Nz
    sspace.Nk       = Nk
    sspace.Nm       = Nm
    sspace.Nb       = Nb
    sspace.s        = s
    sspace.Ns       = Ns

    # Extended matrices of grids
    sspace.k_Egrid = repeat(reshape(k_grid, (Nk,1,1,1)), 1, Nm, Nb, Nz)
    sspace.m_Egrid = repeat(reshape(m_grid, (1,Nm,1,1)), Nk, 1, Nb, Nz)
    sspace.b_Egrid = repeat(reshape(b_grid, (1,1,Nb,1)), Nk, Nm, 1, Nz)
    sspace.z_Egrid = repeat(reshape(z_grid, (1,1,1,Nz)), Nk, Nm, Nb, 1)

    # Initialize entrants' z distribution -- if not specified, set to ergodic dist
    if isempty(modl.z_0_dist)
        modl.z_0_dist = sspace.Pssz
    end

    return modl, sspace
end

# Functions to solve firm's t-problem on full KxMxBxZ grid given continuation value function V0e_tp1

# Stationary model solution (absent prices, setting w_t=1)
# Set w_ss=1, and all other variables at SS levels
function solve_V_EGM(modl,sspace,opts; fsoln_in=Array{Float64}(undef, 0),  w_ss = 1.0)

    ### Initialization

    # Set up all steady state values of prices that firm takes as given
    MU_ss, M_ss, pQ_ss = 10.0/(10.0-1.0), modl.beta, 1.0
    Pi_ss, rb_ss, rm_ss = modl.Pi, modl.rb, modl.rm
    Rc_ss       = (1.0+modl.rb)/(modl.Pi*pQ_ss)
    gammatil_ss = modl.gamma/modl.Pi
    # The A multiplying net earnings
    A_val_ss = (1.0-modl.tau)*(1.0-modl.nu)*(MU_ss^(-1.0/(1.0-modl.nu)))*((modl.nu/w_ss)^(modl.nu/(1.0-modl.nu)))

    # If modl.cresult has old guess in it, use that, otherwise make "educated guess" (= exit value)
    V0ek_old, V0em_old, V0eb_old, V0e_old = Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz);
    if !(typeof(fsoln_in)==Array{Float64,1})
        V0ek_old, V0em_old, V0eb_old, V0e_old = fsoln_in.V0ek, fsoln_in.V0em, fsoln_in.V0eb, fsoln_in.V0e;
    else
        # Guesses for value function 
        V0ek_old = 0.99*(A_val_ss * (modl.alpha/(1.0-modl.nu)) * sspace.z_Egrid .* sspace.k_Egrid.^(modl.alpha/(1.0-modl.nu)-1.0) .+ (1.0-modl.delta) + ((1.0+(1.0-modl.tau)*rb_ss)/Pi_ss) .* sspace.b_Egrid);
        V0em_old = 0.99*exp.(-0.0*(sspace.k_Egrid.-sspace.k_max)).*repeat([(1.0+(1.0-modl.tau)*rm_ss)/Pi_ss], sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz);
        V0eb_old = 0.99*((1.0+(1.0-modl.tau)*rb_ss)/Pi_ss) .* sspace.k_Egrid;
        V0e_old  = 0.99*(A_val_ss * sspace.z_Egrid .* sspace.k_Egrid.^(modl.alpha/(1.0-modl.nu)) + (1.0-modl.delta)*sspace.k_Egrid + ((1.0+(1.0-modl.tau)*rb_ss)/Pi_ss) .* sspace.b_Egrid .* sspace.k_Egrid + ((1.0+(1.0-modl.tau)*rm_ss)/Pi_ss) .* sspace.m_Egrid);
    end

    # Containers for solutions of EXPECTED value function V0e
    V0ek_new, V0em_new, V0eb_new, V0e_new = Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz);
    # And also policies, so that can in the end immediately return function output
    kpN_cur, mpN_cur, bpN_cur, dN_cur, kpA_cur, mpA_cur, bpA_cur, dA_cur, Ba_prob_cur = Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz);
    # And policies for returning final output/labor etc results
    Yn_cur, Y_cur, n_cur, ACN_cur, ACA_cur = Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz);

    flag_Vconv, opts.flag_Vconv  = false, 0
    t_eqstart   = time_ns()
    txtout_path = joinpath(opts.project_root, "interim_output", "model_interim_output", "savedout", modl.name * "_output.txt")
    # EGM iteration
    if opts.prnt =="Y"; manprintln("EGM iterations for value function:", txtout_path); end

    for Viter = 1:opts.Nbell

        # Solve firm problem given continuation (expected) value function V0e
        fsoln_cur = solve_V0(V0ek_old, V0em_old, V0eb_old, V0e_old, MU_ss, w_ss, M_ss, pQ_ss, Pi_ss, rb_ss, rm_ss, Rc_ss, Rc_ss, gammatil_ss, modl, sspace, opts);
        # Unpack solution
        kpN_cur, mpN_cur, bpN_cur, dN_cur, kpA_cur, mpA_cur, bpA_cur, dA_cur, Ba_prob_cur, V0k_cur, V0m_cur, V0b_cur, V0_cur = fsoln_cur.kpN, fsoln_cur.mpN, fsoln_cur.bpN, fsoln_cur.dN, fsoln_cur.kpA, fsoln_cur.mpA, fsoln_cur.bpA, fsoln_cur.dA, fsoln_cur.Ba_prob, fsoln_cur.V0ek, fsoln_cur.V0em, fsoln_cur.V0eb, fsoln_cur.V0e;
        # Also the output etc results, although not needed
        Yn_cur, Y_cur, n_cur, ACN_cur, ACA_cur = fsoln_cur.Yn, fsoln_cur.Y, fsoln_cur.n, fsoln_cur.ACN, fsoln_cur.ACA

        # Integrate out z
        for iz=1:sspace.Nz
            V0ek_new[:,:,:,iz] = sum([sspace.P[iz,izp] * V0k_cur[:,:,:,izp] for izp=1:sspace.Nz])
            V0em_new[:,:,:,iz] = sum([sspace.P[iz,izp] * V0m_cur[:,:,:,izp] for izp=1:sspace.Nz])
            V0eb_new[:,:,:,iz] = sum([sspace.P[iz,izp] * V0b_cur[:,:,:,izp] for izp=1:sspace.Nz])
            V0e_new[:,:,:,iz]  = sum([sspace.P[iz,izp] * V0_cur[:,:,:,izp] for izp=1:sspace.Nz])
        end

        # Value function distances
        diff_V0ek = maximum(abs.(V0ek_old-V0ek_new))
        diff_V0em = maximum(abs.(V0em_old-V0em_new))
        diff_V0eb = maximum(abs.(V0eb_old-V0eb_new))
        diff_V0e  = maximum(abs.(V0e_old-V0e_new))

        # Check convergence
        if mod(Viter,5) == 0; manprintln("$Viter\t dV0ek = $diff_V0ek,\t dV0em = $diff_V0em,\t dV0eb = $diff_V0eb,\t dV0e = $diff_V0e, \t Time: $((time_ns()-t_eqstart)/1.0e9)", txtout_path); end
        if (diff_V0ek < opts.tolV) & (diff_V0em < opts.tolV) & (diff_V0eb < opts.tolV) & (diff_V0e < opts.tolV); flag_Vconv = true; opts.flag_Vconv = 1; end

        # Update value functions
        V0ek_old, V0em_old, V0eb_old, V0e_old = V0ek_old + (1.0-opts.phi_V[1])*(V0ek_new-V0ek_old), V0em_old + (1.0-opts.phi_V[2])*(V0em_new-V0em_old), V0eb_old + (1.0-opts.phi_V[3])*(V0eb_new-V0eb_old), V0e_old + (1.0-opts.phi_V[4])*(V0e_new-V0e_old)

        if flag_Vconv; manprintln("Bellman EGM iterations -> convergence in steps: $Viter, time: $((time_ns()-t_eqstart)/1.0e9).", txtout_path); break;  end
        if !flag_Vconv && Viter==opts.Nbell; manprintln("Finished $Viter Bellman EGM iterations without convergence, distances: $diff_V0ek, $diff_V0em, $diff_V0eb, $diff_V0e, time: $((time_ns()-t_eqstart)/1.0e9).", txtout_path); end
    end

    # Return the solution
    return FirmSolution(kpN_cur, mpN_cur, bpN_cur, dN_cur, kpA_cur, mpA_cur, bpA_cur, dA_cur, Ba_prob_cur, V0ek_old, V0em_old, V0eb_old, V0e_old, Yn_cur, Y_cur, n_cur, ACN_cur, ACA_cur)
end

###
# Functions for solving firm problem in any t
###

# Full firm problem
function solve_V0(V0ek_tp1, V0em_tp1, V0eb_tp1, V0e_tp1, MU_t, w_t, M_tp1, pQ_t, Pi_t, rb_t, rm_t, Rc_tm1, Rc_t, gammatil_t, modl, sspace, opts)

    # 1b - Solve problem conditional on adjusting
    kpA_cur, mpA_cur, bpA_cur, dA_cur, V1Ak_cur, V1Am_cur, V1Ab_cur, V1A_cur = solve_VA(V0ek_tp1, V0em_tp1, V0eb_tp1, V0e_tp1, MU_t, w_t, M_tp1, pQ_t, Pi_t, rb_t, rm_t, Rc_tm1, Rc_t, gammatil_t, modl, sspace, opts);

    # Do not solve non-adjuster problem when issuance costs are zero
    if (modl.xi_ub==0.0) & (modl.xi_dist=="uniform")
        # 1a - Solve problem conditional on not adjusting
        kpN_cur, mpN_cur, bpN_cur, dN_cur, V1Nk_cur, V1Nm_cur, V1Nb_cur, V1N_cur = zeros(size(kpA_cur)), zeros(size(kpA_cur)), zeros(size(kpA_cur)), zeros(size(kpA_cur)), zeros(size(kpA_cur)), zeros(size(kpA_cur)), zeros(size(kpA_cur)), zeros(size(kpA_cur));
        # Cutoff issuance cost
        xi_c_cur    = zeros(size(kpA_cur));
        # Probabilities of adjusting
        Ba_prob_cur = ones(size(kpA_cur));
    else
        # 1a - Solve problem conditional on not adjusting
        kpN_cur, mpN_cur, bpN_cur, dN_cur, V1Nk_cur, V1Nm_cur, V1Nb_cur, V1N_cur = solve_VN(V0ek_tp1, V0em_tp1, V0eb_tp1, V0e_tp1, MU_t, w_t, M_tp1, pQ_t, Pi_t, rb_t, rm_t, Rc_tm1, Rc_t, gammatil_t, modl, sspace, opts);

        # 2 - Compute issuance decisions

        # Cutoff issuance cost
        xi_c_cur    = max.((V1A_cur - V1N_cur), 0.0)
        # Probabilities of adjusting
        Ba_prob_cur = xi_cdfun(xi_c_cur, modl.xi_ub, modl.xi_lb, modl.xi_dist);
    end


    # Take expectations over exit shock and adjustment cost to get V0 on (m,k,b,z)-space, and the corresponding derivatives
    # Precompute the A multiplying net earnings
    # Value of exiting, and its derivatives:
    v_t = (1.0-modl.tau)*ymenufun_mat("yp", sspace.k_Egrid, sspace.m_Egrid/Pi_t, sspace.z_Egrid, MU_t, w_t, modl) + (1.0-(1.0-modl.tau)*modl.delta)*pQ_t*sspace.k_Egrid + ((1.0+(1.0-modl.tau)*rb_t)/Pi_t) * (sspace.k_Egrid .* sspace.b_Egrid)/Rc_tm1 + ((1.0+(1.0-modl.tau)*rm_t)/Pi_t) * sspace.m_Egrid
    vk_t = (1.0-modl.tau)*ymenufun_mat("ypk", sspace.k_Egrid, sspace.m_Egrid/Pi_t, sspace.z_Egrid, MU_t, w_t, modl) .+ (1.0-(1.0-modl.tau)*modl.delta)*pQ_t .+ ((1.0+(1.0-modl.tau)*rb_t)/Pi_t) * sspace.b_Egrid/Rc_tm1
    vm_t = ((1.0+(1.0-modl.tau)*rm_t)/Pi_t) * ones(size(vk_t)) .+ (1.0-modl.tau)*ymenufun_mat("ypm", sspace.k_Egrid, sspace.m_Egrid/Pi_t, sspace.z_Egrid, MU_t, w_t, modl)/Pi_t
    vb_t = ((1.0+(1.0-modl.tau)*rb_t)/Pi_t) * sspace.k_Egrid/Rc_tm1

    # Take expectations
    V0_cur  = modl.eta * v_t + (1.0-modl.eta) * ( Ba_prob_cur .* (V1A_cur - xi_condexpn(xi_c_cur, modl.xi_ub, modl.xi_lb)) + (ones(size(Ba_prob_cur)) - Ba_prob_cur).* V1N_cur )
    V0k_cur = modl.eta * vk_t + (1.0-modl.eta) * ( Ba_prob_cur .* V1Ak_cur + (ones(size(Ba_prob_cur)) - Ba_prob_cur).* V1Nk_cur )
    V0m_cur = modl.eta * vm_t + (1.0-modl.eta) * ( Ba_prob_cur .* V1Am_cur + (ones(size(Ba_prob_cur)) - Ba_prob_cur).* V1Nm_cur ) 
    V0b_cur = modl.eta * vb_t + (1.0-modl.eta) * ( Ba_prob_cur .* V1Ab_cur + (ones(size(Ba_prob_cur)) - Ba_prob_cur).* V1Nb_cur ) 

    # Compute the additional outcome policies, to return final solution
    Y_cur = ymenufun_mat("y", sspace.k_Egrid, sspace.m_Egrid/Pi_t, sspace.z_Egrid, MU_t, w_t, modl)
    Yn_cur = (1.0-modl.tau)*ymenufun_mat("yp", sspace.k_Egrid, sspace.m_Egrid/Pi_t, sspace.z_Egrid, MU_t, w_t, modl)
    n_cur = ymenufun_mat("n", sspace.k_Egrid, sspace.m_Egrid/Pi_t, sspace.z_Egrid, MU_t, w_t, modl)
    ACN_cur = pQ_t*(modl.kappa/2.0)*(kpN_cur./sspace.k_Egrid.-1.0).^2
    ACA_cur = pQ_t*(modl.kappa/2.0)*(kpA_cur./sspace.k_Egrid.-1.0).^2

    # Return the "full policies" and value functions V0 on the (k,m,b,z)-space
    return FirmSolution(kpN_cur, mpN_cur, bpN_cur, dN_cur, kpA_cur, mpA_cur, bpA_cur, dA_cur, Ba_prob_cur, V0k_cur, V0m_cur, V0b_cur, V0_cur, Yn_cur, Y_cur, n_cur, ACN_cur, ACA_cur)
end

# Solve problem conditional on N
function solve_VN(V0ek_tp1, V0em_tp1, V0eb_tp1, V0e_tp1, MU_t, w_t, M_tp1, pQ_t, Pi_t, rb_t, rm_t, Rc_tm1, Rc_t, gammatil_t, modl, sspace, opts)

    # 1ai   - Second stage of N
    kpN_cur, mpN_cur, bpN_cur, V2Nk_cur, V2Nmtil_cur, V2Nb_cur, V2N_cur = solve_V2N(V0ek_tp1, V0em_tp1, V0eb_tp1, V0e_tp1, MU_t, w_t, M_tp1, pQ_t, Pi_t, rb_t, rm_t, Rc_tm1, Rc_t, gammatil_t, modl, sspace, opts);

    # 1aii  - First stage of N
    mtilN_cur, dN_cur, V1Nk_cur, V1Nm_cur, V1Nb_cur, V1N_cur = solve_V1N(V2Nk_cur, V2Nmtil_cur, V2Nb_cur, V2N_cur, MU_t, w_t, M_tp1, pQ_t, Pi_t, rb_t, rm_t, Rc_tm1, Rc_t, gammatil_t, modl, sspace, opts);

    # 1aiii -- Use the above policies to create the overall policies conditional on N, on the (k,m,b,z)-space
    # First, map the "second stage" policies for k', m' and b' from the (k,mtil,b)-grid to the (k,m,b)-grid
    kpN2_itp = extrapolate(interpolate((sspace.k_grid, sspace.mtil_grid, sspace.b_grid, sspace.z_grid), kpN_cur, Gridded(Linear())), Line())
    mpN2_itp = extrapolate(interpolate((sspace.k_grid, sspace.mtil_grid, sspace.b_grid, sspace.z_grid), mpN_cur, Gridded(Linear())), Line())
    bpN2_itp = extrapolate(interpolate((sspace.k_grid, sspace.mtil_grid, sspace.b_grid, sspace.z_grid), bpN_cur, Gridded(Linear())), Line())
    kpN1_imp, mpN1_imp, bpN1_imp = Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz)
    for ik=1:sspace.Nk
        for im=1:sspace.Nm
            for ib=1:sspace.Nb
                for iz=1:sspace.Nz
                    kpN1_imp[ik,im,ib,iz], mpN1_imp[ik,im,ib,iz], bpN1_imp[ik,im,ib,iz] = kpN2_itp(sspace.k_grid[ik], mtilN_cur[ik,im,ib,iz], sspace.b_grid[ib], sspace.z_grid[iz]), mpN2_itp(sspace.k_grid[ik], mtilN_cur[ik,im,ib,iz], sspace.b_grid[ib], sspace.z_grid[iz]), bpN2_itp(sspace.k_grid[ik], mtilN_cur[ik,im,ib,iz], sspace.b_grid[ib], sspace.z_grid[iz])
                end
            end
        end
    end

    # Return the "full policies" and value functions V1 on the (k,m,b,z)-space
    return kpN1_imp, mpN1_imp, bpN1_imp, dN_cur, V1Nk_cur, V1Nm_cur, V1Nb_cur, V1N_cur
end

# Solve second stage of N
function solve_V2N(V0ek_tp1, V0em_tp1, V0eb_tp1, V0e_tp1, MU_t, w_t, M_tp1, pQ_t, Pi_t, rb_t, rm_t, Rc_tm1, Rc_t, gammatil_t, modl, sspace, opts)

    # Set output text printing path
    txtout_path = joinpath(opts.project_root, "interim_output", "model_interim_output", "savedout", modl.name * "_output.txt")

    #### N - 2nd stage
    # For each b', solve the 2-dim EGM problem over (k,m)

    # Create collectors of overall, final policies for this stage
    kpN_sol_fin, mpN_sol_fin, bpN_sol_fin = Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb, sspace.Nz);
    V2Nk_sol_fin, V2Nmtil_sol_fin, V2Nb_sol_fin, V2N_sol_fin = Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb, sspace.Nz);

    for iz = 1:sspace.Nz # Iterating over all z separately
        # For each given z, construct temporary collectors of policies (only over KxMxB-dimensions) -- UNCONSTRAINED k'
        # In order to do the inversion from b' to b in the end, create collectors of policies conditional on bp
        kpN_sol_bp, mpN_sol_bp, lambda2N_sol_bp = Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb);
        # And for the value functions also
        V2Nk_sol_bp, V2Nmtil_sol_bp, V2Nb_sol_bp, V2N_sol_bp = Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb);

        # For each b'
        for ibp = 1:sspace.Nb
            # N2-Step 1: For each mp, "solve out"  the k-dimension
            # Initialize relevant (intermediate) Step 1 "collectors" over the (k,m')-space, as functions of (k,m')
            # NOTE: Don't include b'-dimension because these will be used only "within" this iteration
            kpN_sol_k_temp, lambda2N_sol_k_temp = Array{Float64}(undef, sspace.Nk, sspace.Nm), Array{Float64}(undef, sspace.Nk, sspace.Nm);
            # Also, we need to "map back" ("interpolate onto today's state") the continuation value fn derivative wrt b, and the value function value
            V0eb_sol_k_temp, V0e_sol_k_temp = Array{Float64}(undef, sspace.Nk, sspace.Nm), Array{Float64}(undef, sspace.Nk, sspace.Nm);
            V0ebdk_sol_k_temp = Array{Float64}(undef, sspace.Nk, sspace.Nm);

            # For each m' being UC, solve for k
            for imp = 1:sspace.Nm
                # Solve FOC for k vector (over k')
                kN_uc_val = sspace.k_grid ./ max.(1e-8, 1.0 .+ (1.0/modl.kappa)*( M_tp1*(V0ek_tp1[:,imp,ibp,iz] -V0eb_tp1[:,imp,ibp,iz]*sspace.b_grid[ibp]./sspace.k_grid) ./ (pQ_t*M_tp1*V0em_tp1[:,imp,ibp,iz]) .- 1.0)  )

                # Create interpolants for kN_uc and lambda2N_uc
                # If endogenous grid not ordered, deal with the implied non-convexity and find the global optimum in the spirit of Fella (2014)
                # Deal with non-concavity
                if !issorted(kN_uc_val)
                    # Find the "flipping points"
                    kN_uc_diff = kN_uc_val[2:end] - kN_uc_val[1:(end-1)]
                    i_flip_set = findall(i->(i<0.0), kN_uc_diff)
                    kN_uc_fmax = maximum(kN_uc_val[i_flip_set]) # The "upper bound" on the non-concavity region, in k-space
                    kN_uc_fmin = minimum(kN_uc_val[i_flip_set.+1]) # The "lower bound" on the non-concavity region, in k-space

                    # Find the "boundaries" of the non-concavity region, in the index-space
                    # In case the "lowest flipping point" is itself the LOWEST POINT of kN_uc_val, then i_LB=0
                    if any(kN_uc_val .< kN_uc_fmin)
                        i_LB = maximum(findall(i->(i<kN_uc_fmin), kN_uc_val))
                    else
                        i_LB = 0
                    end
                    # In case the "highest flipping point" is itself the HIGHEST POINT of kN_uc_val, then i_UB=Nb+1
                    if any(kN_uc_val .> kN_uc_fmax)
                        i_UB = minimum(findall(i->(i>kN_uc_fmax), kN_uc_val))
                    else
                        i_UB = length(kN_uc_val) + 1
                    end

                    # Create collector of indices which to not drop
                    ind_keep_col = ones(Bool,length(kN_uc_val))
                    # Now, consider each index inbetween i_LB and i_UB individually, in order to establish whether the found point is actually a global optimum
                    for ii=(i_LB+1):(i_UB-1)
                        # Implied incoming states (k,mtil) at ii
                        kN_cur_val = kN_uc_val[ii]
                        # Invert the budget constraint for mtil
                        mtilN_cur_val = (-(1.0-(1.0-modl.tau)*modl.delta)*pQ_t*kN_cur_val - sspace.m_grid[1] + sspace.m_grid[imp] + pQ_t*sspace.k_grid[ii] + pQ_t*(modl.kappa/2.0)*kN_cur_val*(sspace.k_grid[ii]/kN_cur_val - 1.0)^2 - pQ_t*kN_cur_val*(-(sspace.b_grid[ibp]/modl.theta)*sspace.k_grid[ii]/kN_cur_val + (modl.kappa/2.0)*(-(sspace.b_grid[ibp]/modl.theta)*sspace.k_grid[ii]/kN_cur_val - 1.0).^2) ) / (pQ_t*kN_cur_val)

                        # Create collector for candidate values
                        V_vals = -999.9*ones(size(sspace.k_grid))
                        # Evaluate the value function at the indices in the non-concave region
                        for ic=(i_LB+1):(i_UB-1)
                            # Implied choices (kp, mp) at candidate ic
                            kp_cur_val = sspace.k_grid[ic]
                            # Solve the budget constraint for mp
                            mp_cur_val = (1.0-(1.0-modl.tau)*modl.delta)*pQ_t*kN_cur_val + sspace.m_grid[1] + mtilN_cur_val*pQ_t*kN_cur_val - pQ_t*kp_cur_val - pQ_t*(modl.kappa/2.0)*kN_cur_val*(kp_cur_val/kN_cur_val - 1.0)^2 + pQ_t*kN_cur_val*(-(sspace.b_grid[ibp]/modl.theta)*kp_cur_val/kN_cur_val + (modl.kappa/2.0)*(-(sspace.b_grid[ibp]/modl.theta)*kp_cur_val/kN_cur_val - 1.0).^2)

                            # Now, evaluate the value function at the implied (kp, bp)
                            # If the implied mp_cur_val is below the lower bound on m, then cannot update V_vals, leaving it at the -999
                            if mp_cur_val >= sspace.m_grid[1]
                                V0e_itp_temp      = extrapolate(interpolate((sspace.k_grid, sspace.m_grid), V0e_tp1[:,:,ibp,iz], Gridded(Linear())), NaN)
                                # Evaluate the continuation value function
                                V_vals[ic] = V0e_itp_temp(kp_cur_val, mp_cur_val)
                            end
                        end
                        # Find the optimum index
                        ic_max = findmax(V_vals)[2]
                        # If this is not equal to ii, add ii to be dropped
                        if ic_max != ii
                            ind_keep_col[ii]=0
                        end
                    end

                    # Having established which elements of the k'- and k-vectors to keep, construct them here to use for interpolation
                    # In order to make indexing of all the relevant continuation vectors easier, first sort kN_uc_val, and then drop the indices that are meant to be dropped
                    i_kN_sort      = sortperm(kN_uc_val)
                    ind_keep_col_sort = ind_keep_col[i_kN_sort]
                    i_kN_NC_val    = i_kN_sort[ind_keep_col_sort] # Use this "indexer" to now construct the correctly ordered "kN_val" vector

                    # Create interpolants
                    kpN_uc_itp      = extrapolate(interpolate((kN_uc_val[i_kN_NC_val], ), sspace.k_grid[i_kN_NC_val], Gridded(Linear())), Line())
                    lambda2N_uc_itp = extrapolate(interpolate((kN_uc_val[i_kN_NC_val], ), M_tp1*V0em_tp1[i_kN_NC_val,imp,ibp,iz], Gridded(Linear())), Line())
                    # And the continuation value function
                    V0eb_uc_itp     = extrapolate(interpolate((kN_uc_val[i_kN_NC_val], ), V0eb_tp1[i_kN_NC_val,imp,ibp,iz], Gridded(Linear())), Line())
                    V0e_uc_itp      = extrapolate(interpolate((kN_uc_val[i_kN_NC_val], ), V0e_tp1[i_kN_NC_val,imp,ibp,iz], Gridded(Linear())), Line())
                    V0ebdk_uc_itp   = extrapolate(interpolate((kN_uc_val[i_kN_NC_val], ), V0eb_tp1[i_kN_NC_val,imp,ibp,iz]./sspace.k_grid[i_kN_NC_val], Gridded(Linear())), Line())
                else
                    # Create interpolants -- no concavity issues
                    kpN_uc_itp      = extrapolate(interpolate((kN_uc_val, ), sspace.k_grid, Gridded(Linear())), Line())
                    lambda2N_uc_itp = extrapolate(interpolate((kN_uc_val, ), M_tp1*V0em_tp1[:,imp,ibp,iz], Gridded(Linear())), Line())
                    # And the continuation value function
                    V0eb_uc_itp     = extrapolate(interpolate((kN_uc_val, ), V0eb_tp1[:,imp,ibp,iz], Gridded(Linear())), Line())
                    V0e_uc_itp      = extrapolate(interpolate((kN_uc_val, ), V0e_tp1[:,imp,ibp,iz], Gridded(Linear())), Line())
                    V0ebdk_uc_itp   = extrapolate(interpolate((kN_uc_val, ), V0eb_tp1[:,imp,ibp,iz]./sspace.k_grid, Gridded(Linear())), Line())
                end

                # Evaluate interpolants
                kpN_uc_imp, lambda2N_uc_imp = kpN_uc_itp.(sspace.k_grid), lambda2N_uc_itp.(sspace.k_grid)
                V0eb_uc_imp, V0e_uc_imp  = V0eb_uc_itp.(sspace.k_grid), V0e_uc_itp.(sspace.k_grid)
                V0ebdk_uc_imp = V0ebdk_uc_itp.(sspace.k_grid)

                # Save in the corresponding matrices
                kpN_sol_k_temp[:,imp], lambda2N_sol_k_temp[:,imp] = kpN_uc_imp, lambda2N_uc_imp
                V0eb_sol_k_temp[:,imp], V0e_sol_k_temp[:,imp] = V0eb_uc_imp, V0e_uc_imp
                V0ebdk_sol_k_temp[:,imp] = V0ebdk_uc_imp
            end


            # Now, for each k, solve out the m-dimension
            for ik = 1:sspace.Nk
                # First, over the unconstrained m', invert the budget constraint for mtil
                mtilN_uc_val = (-(1.0-(1.0-modl.tau)*modl.delta)*pQ_t*sspace.k_grid[ik] .- sspace.m_grid[1] .+ sspace.m_grid .+ pQ_t*kpN_sol_k_temp[ik,:] + pQ_t*(modl.kappa/2.0)*sspace.k_grid[ik].*(kpN_sol_k_temp[ik,:]./sspace.k_grid[ik] .- 1.0).^2 - pQ_t*sspace.k_grid[ik].*(-(sspace.b_grid[ibp]/modl.theta)*kpN_sol_k_temp[ik,:]./sspace.k_grid[ik] + (modl.kappa/2.0)*(-(sspace.b_grid[ibp]/modl.theta)*kpN_sol_k_temp[ik,:]./sspace.k_grid[ik] .- 1.0).^2) ) ./ (pQ_t*sspace.k_grid[ik])
                if ibp==1
                    mtilN_uc_val[1] = copy(sspace.mtil_grid[1])
                end

                # On the mtil points "spanned" by mtilN_uc_val, do the usual inversion, but NOT extrapolating below
                # Create interpolants for all objects, conditional on UC and C
                # If endogenous grid not ordered, deal with the implied non-convexity and find the global optimum in the spirit of Fella (2014)
                if !issorted(mtilN_uc_val)

                    # Find the "flipping points"
                    mtilN_uc_diff = mtilN_uc_val[2:end] - mtilN_uc_val[1:(end-1)]
                    i_flip_set = findall(i->(i<0.0), mtilN_uc_diff)
                    mtilN_uc_fmax = maximum(mtilN_uc_val[i_flip_set]) # The "upper bound" on the non-concavity region, in mtil-space
                    mtilN_uc_fmin = minimum(mtilN_uc_val[i_flip_set.+1]) # The "lower bound" on the non-concavity region, in mtil-space
                    # Find the "boundaries" of the non-concavity region, in the index-space
                    # In case the "lowest flipping point" is itself the LOWEST POINT of mtilN_uc_val, then i_LB=0
                    if any(mtilN_uc_val .< mtilN_uc_fmin)
                        i_LB = maximum(findall(i->(i<mtilN_uc_fmin), mtilN_uc_val))
                    else
                        i_LB = 0
                    end
                    # In case the "highest flipping point" is itself the HIGHEST POINT of mtilN_uc_val, then i_UB=Nb+1
                    if any(mtilN_uc_val .> mtilN_uc_fmax)
                        i_UB = minimum(findall(i->(i>mtilN_uc_fmax), mtilN_uc_val))
                    else
                        i_UB = length(mtilN_uc_val) + 1
                    end

                    # Create collector of indices which to not drop
                    ind_keep_col = ones(Bool,length(mtilN_uc_val))
                    # Now, consider each index inbetween i_LB and i_UB individually, in order to establish whether the found point is actually a global optimum
                    for ii=(i_LB+1):(i_UB-1)
                        # Implied incoming states (k,mtil) at ii
                        mtilN_cur_val = mtilN_uc_val[ii]
                        kN_cur_val   = sspace.k_grid[ik]
                        # Create collector for candidate values
                        V_vals = -999.9*ones(size(sspace.m_grid))
                        # Evaluate the value function at the indices in the non-concave region
                        for ic=(i_LB+1):(i_UB-1)
                            # Implied choices (kp, mp) at candidate ic
                            mp_cur_val = sspace.m_grid[ic]
                            # If the implied mp_cur_val is below the lower bound on m, then cannot update V_vals, leaving it at the -999
                            if mp_cur_val >= sspace.m_grid[1]
                                # Solve the quadratic budget constraint for kp
                                a_kpdkq_temp = 0.5*modl.kappa*(1.0-(-sspace.b_grid[ibp]/modl.theta)^2)
                                b_kpdkq_temp = (1.0-modl.kappa)*(1.0+sspace.b_grid[ibp]/modl.theta)
                                c_kpdkq_temp = -((1.0-(1.0-modl.tau)*modl.delta) + mtilN_cur_val - (mp_cur_val-sspace.m_grid[1])/(pQ_t*kN_cur_val))
                                # Consider candidate mp only if it implies a solution for kp, otherwise leave value to -999
                                if (b_kpdkq_temp.^2 - 4*a_kpdkq_temp*c_kpdkq_temp) >= 0.0
                                    kpdk_temp_val = (- b_kpdkq_temp .+ sqrt.(b_kpdkq_temp.^2 .- 4*a_kpdkq_temp*c_kpdkq_temp))/(2.0*a_kpdkq_temp)
                                    # The implied kp
                                    kpN_val     = kpdk_temp_val*kN_cur_val

                                    # Now, evaluate the value function at the implied (kp, mp)
                                    V0e_itp_temp      = extrapolate(interpolate((sspace.k_grid, sspace.m_grid), V0e_tp1[:,:,ibp,iz], Gridded(Linear())), NaN)
                                    # Evaluate the continuation value function
                                    V_vals[ic] = V0e_itp_temp(kpN_val, mp_cur_val)
                                end
                            end
                        end
                        # Find the optimum index
                        ic_max = findmax(V_vals)[2]
                        # If this is not equal to ii, add ii to be dropped
                        if ic_max != ii
                            ind_keep_col[ii]=0
                        end
                    end

                    # Having established which elements of the m'- and mtil-vectors to keep, construct them here to use for interpolation
                    # In order to make indexing of all the relevant continuation vectors easier, first sort mtilN_uc_val, and then drop the indices that are meant to be dropped
                    i_mtilN_sort      = sortperm(mtilN_uc_val)
                    ind_keep_col_sort = ind_keep_col[i_mtilN_sort]
                    i_mtilN_NC_val    = i_mtilN_sort[ind_keep_col_sort] # Use this "indexer" to now construct the correctly ordered "btilA_val" vector

                    # Create interpolants
                    mpN_uc_itp      = extrapolate(interpolate((mtilN_uc_val[i_mtilN_NC_val], ), sspace.m_grid[i_mtilN_NC_val], Gridded(Linear())), Line())
                    kpN_uc_itp      = extrapolate(interpolate((mtilN_uc_val[i_mtilN_NC_val], ), kpN_sol_k_temp[ik,i_mtilN_NC_val], Gridded(Linear())), Line())
                    lambda2N_uc_itp = extrapolate(interpolate((mtilN_uc_val[i_mtilN_NC_val], ), lambda2N_sol_k_temp[ik,i_mtilN_NC_val], Gridded(Linear())), Line())
                    # And the continuation value function
                    V0eb_uc_itp     = extrapolate(interpolate((mtilN_uc_val[i_mtilN_NC_val], ), V0eb_sol_k_temp[ik,i_mtilN_NC_val], Gridded(Linear())), Line())
                    V0e_uc_itp      = extrapolate(interpolate((mtilN_uc_val[i_mtilN_NC_val], ), V0e_sol_k_temp[ik,i_mtilN_NC_val], Gridded(Linear())), Line())
                    V0ebdk_uc_itp   = extrapolate(interpolate((mtilN_uc_val[i_mtilN_NC_val], ), V0ebdk_sol_k_temp[ik,i_mtilN_NC_val], Gridded(Linear())), Line())
                else
                    # Create interpolants -- no concavity issues
                    mpN_uc_itp      = extrapolate(interpolate((mtilN_uc_val, ), sspace.m_grid, Gridded(Linear())), Line())
                    kpN_uc_itp      = extrapolate(interpolate((mtilN_uc_val, ), kpN_sol_k_temp[ik,:], Gridded(Linear())), Line())
                    lambda2N_uc_itp = extrapolate(interpolate((mtilN_uc_val, ), lambda2N_sol_k_temp[ik,:], Gridded(Linear())), Line())
                    # And the continuation value function
                    V0eb_uc_itp     = extrapolate(interpolate((mtilN_uc_val, ), V0eb_sol_k_temp[ik,:], Gridded(Linear())), Line())
                    V0e_uc_itp      = extrapolate(interpolate((mtilN_uc_val, ), V0e_sol_k_temp[ik,:], Gridded(Linear())), Line())
                    V0ebdk_uc_itp   = extrapolate(interpolate((mtilN_uc_val, ), V0ebdk_sol_k_temp[ik,:], Gridded(Linear())), Line())
                end

                # And evaluate these on the mtil_grid
                mpN_uc_imp, lambda2N_uc_imp, kpN_uc_imp = mpN_uc_itp.(sspace.mtil_grid), lambda2N_uc_itp.(sspace.mtil_grid), kpN_uc_itp.(sspace.mtil_grid)
                V0eb_uc_imp, V0e_uc_imp     = V0eb_uc_itp.(sspace.mtil_grid), V0e_uc_itp.(sspace.mtil_grid)
                V0ebdk_uc_imp   = V0ebdk_uc_itp.(sspace.mtil_grid)
                # Allow for extrapolation in mtil space above the upper bound, but not below
                low_ind_uc = sspace.mtil_grid .< mtilN_uc_val[1]
                if any(low_ind_uc)
                    kpN_uc_imp[low_ind_uc] .= NaN
                    mpN_uc_imp[low_ind_uc] .= NaN
                    lambda2N_uc_imp[low_ind_uc] .= NaN
                    V0eb_uc_imp[low_ind_uc] .= NaN
                    V0e_uc_imp[low_ind_uc]  .= NaN
                    V0ebdk_uc_imp[low_ind_uc] .= NaN
                end


                # Only if ibp>1 AND the bottom end of mtil "not spanned", solve the CONSTRAINED m' solution. By construction, if ibp=1, then the full mtil_grid must be spanned by UC and this is not necessary
                if (ibp >1) & (sspace.mtil_grid[1] < mtilN_uc_val[1])
                    # Now, solve conditional on k, over the mtil grid, the implied values by m' ≧ m_1 being BINDING, but reading multipliers DIRECTLY from FOCs
                    mpN_c_imp = repeat([sspace.m_grid[1]], sspace.Nmtil)
                    # For k', first solve the budget constraint for the implied (k'/k), since at m'=m_1, it is HOD1 in k and independent of Q
                    a_kq_temp = 0.5*modl.kappa*(1.0-(-sspace.b_grid[ibp]/modl.theta)^2)
                    b_kq_temp = (1.0-modl.kappa)*(1.0+sspace.b_grid[ibp]/modl.theta)
                    c_kq_temp = -(1.0-(1.0-modl.tau)*modl.delta) .- sspace.mtil_grid
                    kpdk_c_val = (- b_kq_temp .+ sqrt.(b_kq_temp.^2 .- 4*a_kq_temp*c_kq_temp))/(2.0*a_kq_temp)

                    # The implied kp
                    kpN_c_imp = kpdk_c_val * sspace.k_grid[ik]
                    # To bound away from dealing with values potentially diverging to infinity, impose small kpN_c_imp[1]=1e-6
                    kpN_c_imp[1] = 1e-6

                    # Given these kp and mp policies, can back out the required multipliers etc, conditional on being CONSTRAINED
                    # For this, need to construct interpolants of continuation value functions over the k'-grid, conditional on m'=m_1 and b'=b[ibp]
                    V0ek_c_itp     = extrapolate(interpolate((sspace.k_grid, ), V0ek_tp1[:,1,ibp,iz], Gridded(Linear())), Line())
                    V0em_c_itp     = extrapolate(interpolate((sspace.k_grid, ), V0em_tp1[:,1,ibp,iz], Gridded(Linear())), Line())
                    V0eb_c_itp     = extrapolate(interpolate((sspace.k_grid, ), V0eb_tp1[:,1,ibp,iz], Gridded(Linear())), Line())
                    V0e_c_itp      = extrapolate(interpolate((sspace.k_grid, ), V0e_tp1[:,1,ibp,iz], Gridded(Linear())), Line())
                    # The object to interpolate is V0ebdk_c_itp=V0eb/k'
                    V0ebdk_c_itp   = extrapolate(interpolate((sspace.k_grid, ), V0eb_tp1[:,1,ibp,iz]./sspace.k_grid, Gridded(Linear())), Line())

                    # And evaluate these at the implied (k',m') policies
                    V0ek_c_imp, V0em_c_imp, V0eb_c_imp, V0e_c_imp = V0ek_c_itp.(kpN_c_imp), V0em_c_itp.(kpN_c_imp), V0eb_c_itp.(kpN_c_imp), V0e_c_itp.(kpN_c_imp)
                    V0ebdk_c_imp = V0ebdk_c_itp.(kpN_c_imp)

                    # Lambda from k-FOC
                    lambda2N_c_imp = M_tp1*( V0ek_c_imp - V0ebdk_c_imp*sspace.b_grid[ibp] ) ./ (pQ_t*( 1.0.+ modl.kappa*(kpN_c_imp./sspace.k_grid[ik] .- 1.0) ))
                end

                # Combine constrained and unconstrained mp and lambda solutions
                if (ibp >1) & (sspace.mtil_grid[1] < mtilN_uc_val[1])
                    kpN_imp, mpN_imp, lambda2N_imp, V0eb2N_imp, V0e2N_imp = copy(kpN_uc_imp), copy(mpN_uc_imp), copy(lambda2N_uc_imp), copy(V0eb_uc_imp), copy(V0e_uc_imp)
                    V0ebdk2N_imp = copy(V0ebdk_uc_imp)
                    # Replace C policies wherever UC was missing
                    ic_ind = isnan.(mpN_uc_imp)
                    kpN_imp[ic_ind] = kpN_c_imp[ic_ind]
                    mpN_imp[ic_ind] = mpN_c_imp[ic_ind]
                    lambda2N_imp[ic_ind] = lambda2N_c_imp[ic_ind]
                    V0eb2N_imp[ic_ind]   = V0eb_c_imp[ic_ind]
                    V0e2N_imp[ic_ind]    = V0e_c_imp[ic_ind]
                    V0ebdk2N_imp[ic_ind] = V0ebdk_c_imp[ic_ind]
                else
                    kpN_imp, mpN_imp, lambda2N_imp, V0eb2N_imp, V0e2N_imp = kpN_uc_imp, mpN_uc_imp, lambda2N_uc_imp, V0eb_uc_imp, V0e_uc_imp
                    V0ebdk2N_imp = V0ebdk_uc_imp
                end

                # Save these into the "final" collectors, still CONDITIONAL on b'
                kpN_sol_bp[ik,:,ibp], mpN_sol_bp[ik,:,ibp], lambda2N_sol_bp[ik,:,ibp] = kpN_imp, mpN_imp, lambda2N_imp

                # And for the value functions as well, using envelope conditions
                # Note that the grid-point of b'=b_grid_set[ibp] equals, in math, gamma*b*(Rc_t/Rc_tm1)*k/k', so if we need its partial derivative wrt k, i.e. gamma*b*(Rc_t/Rc_tm1)/k', we can just compute it as: sspace.b_grid[ibp] / sspace.k_grid[ik]
                V2Nk_sol_bp[ik,:,ibp]    = lambda2N_imp .* ((1.0.-(1.0-modl.tau)*modl.delta)*pQ_t .+ sspace.mtil_grid*pQ_t .+ pQ_t*((-sspace.b_grid[ibp]/modl.theta)*(kpN_imp./sspace.k_grid[ik]) .+ (modl.kappa/2.0)*((-sspace.b_grid[ibp]/modl.theta)*(kpN_imp./sspace.k_grid[ik]).-1.0).^2 ) .+ pQ_t*(modl.kappa/2.0)*( (kpN_imp ./ sspace.k_grid[ik]).^2 .- 1.0) ) .+ M_tp1*V0eb2N_imp.*sspace.b_grid[ibp]./sspace.k_grid[ik]
                V2Nmtil_sol_bp[ik,:,ibp] = lambda2N_imp .* sspace.k_grid[ik] * pQ_t
                V2Nb_sol_bp[ik,:,ibp]    = M_tp1*V0ebdk2N_imp*gammatil_t*(Rc_t/Rc_tm1) .* sspace.k_grid[ik] .+ lambda2N_imp .* pQ_t .* (-gammatil_t/modl.theta)*(Rc_t/Rc_tm1) .* (1.0 .+ modl.kappa*((-sspace.b_grid[ibp]/modl.theta)*(kpN_imp./sspace.k_grid[ik]) .- 1.0 )) * sspace.k_grid[ik]
                V2N_sol_bp[ik,:,ibp]     = M_tp1*V0e2N_imp
            end
        end

        # Repeat the above iterations "conditional on muk", instead of bp
        # For each given z, construct temporary collectors of policies (only over KxMxmuk-dimensions) -- CONSTRAINED k'
        # In order to do the inversion from muk to b in the end, create collectors of policies conditional on muk
        kpN_sol_muk, mpN_sol_muk, lambda2N_sol_muk = Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nmuk), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nmuk), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nmuk);
        # And for the value functions also
        V2Nk_sol_muk, V2Nmtil_sol_muk, V2Nb_sol_muk, V2N_sol_muk = Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nmuk), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nmuk), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nmuk), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nmuk);

        # For each muk
        for imuk = 1:sspace.Nmuk
            # N2-Step 1: For each mp, "solve out"  the k-dimension
            # Initialize relevant (intermediate) Step 1 "collectors" over the (k,m')-space, as functions of (k,m')
            # NOTE: Don't include b'-dimension because these will be used only "within" this iteration
            kpN_sol_k_temp, lambda2N_sol_k_temp = Array{Float64}(undef, sspace.Nk, sspace.Nm), Array{Float64}(undef, sspace.Nk, sspace.Nm)
            # Also, we need to "map back" ("interpolate onto today's state") the continuation value fn derivative wrt b, and the value function value
            V0eb_sol_k_temp, V0e_sol_k_temp = Array{Float64}(undef, sspace.Nk, sspace.Nm), Array{Float64}(undef, sspace.Nk, sspace.Nm)

            # For each m' being UC, solve for k
            for imp = 1:sspace.Nm
                # Solve FOC for k vector (over k')
                kN_uc_val = sspace.k_grid ./ max.(1e-8, 1.0 .+ (1.0/modl.kappa)*(sspace.muk_grid[imuk] .+ M_tp1*(V0ek_tp1[:,imp,1,iz] - V0eb_tp1[:,imp,1,iz]*sspace.b_grid[1]./sspace.k_grid) ./ (pQ_t*M_tp1*V0em_tp1[:,imp,1,iz]) .- 1.0)  )

                # Create interpolants for kpN_uc and lambda2N_uc
                # If endogenous grid not ordered, deal with the implied non-convexity and find the global optimum in the spirit of Fella (2014)
                if !issorted(kN_uc_val)
                    # Find the "flipping points"
                    kN_uc_diff = kN_uc_val[2:end] - kN_uc_val[1:(end-1)]
                    i_flip_set = findall(i->(i<0.0), kN_uc_diff)
                    kN_uc_fmax = maximum(kN_uc_val[i_flip_set]) # The "upper bound" on the non-concavity region, in k-space
                    kN_uc_fmin = minimum(kN_uc_val[i_flip_set.+1]) # The "lower bound" on the non-concavity region, in k-space

                    # Find the "boundaries" of the non-concavity region, in the index-space
                    # In case the "lowest flipping point" is itself the LOWEST POINT of kN_uc_val, then i_LB=0
                    if any(kN_uc_val .< kN_uc_fmin)
                        i_LB = maximum(findall(i->(i<kN_uc_fmin), kN_uc_val))
                    else
                        i_LB = 0
                    end
                    # In case the "highest flipping point" is itself the HIGHEST POINT of kN_uc_val, then i_UB=Nb+1
                    if any(kN_uc_val .> kN_uc_fmax)
                        i_UB = minimum(findall(i->(i>kN_uc_fmax), kN_uc_val))
                    else
                        i_UB = length(kN_uc_val) + 1
                    end

                    # Create collector of indices which to not drop
                    ind_keep_col = ones(Bool,length(kN_uc_val))
                    # Now, consider each index inbetween i_LB and i_UB individually, in order to establish whether the found point is actually a global optimum
                    for ii=(i_LB+1):(i_UB-1)
                        # Implied incoming states (k,mtil) at ii
                        kN_cur_val = kN_uc_val[ii]
                        # Invert the budget constraint for mtil
                        mtilN_cur_val = (-(1.0-(1.0-modl.tau)*modl.delta)*pQ_t*kN_cur_val - sspace.m_grid[1] + sspace.m_grid[imp] + pQ_t*sspace.k_grid[ii] + pQ_t*(modl.kappa/2.0)*kN_cur_val*(sspace.k_grid[ii]/kN_cur_val - 1.0)^2 - pQ_t*kN_cur_val*(-(sspace.b_grid[1]/modl.theta)*sspace.k_grid[ii]/kN_cur_val + (modl.kappa/2.0)*(-(sspace.b_grid[1]/modl.theta)*sspace.k_grid[ii]/kN_cur_val - 1.0).^2) ) / (pQ_t*kN_cur_val)

                        # Create collector for candidate values
                        V_vals = -999.9*ones(size(sspace.k_grid))
                        # Evaluate the value function at the indices in the non-concave region
                        for ic=(i_LB+1):(i_UB-1)
                            # Implied choices (kp, mp) at candidate ic
                            kp_cur_val = sspace.k_grid[ic]
                            # Solve the budget constraint for mp
                            mp_cur_val = (1.0-(1.0-modl.tau)*modl.delta)*pQ_t*kN_cur_val + sspace.m_grid[1] + mtilN_cur_val*pQ_t*kN_cur_val - pQ_t*kp_cur_val - pQ_t*(modl.kappa/2.0)*kN_cur_val*(kp_cur_val/kN_cur_val - 1.0)^2 + pQ_t*kN_cur_val*(-(sspace.b_grid[1]/modl.theta)*kp_cur_val/kN_cur_val + (modl.kappa/2.0)*(-(sspace.b_grid[1]/modl.theta)*kp_cur_val/kN_cur_val - 1.0).^2)

                            # Now, evaluate the value function at the implied (kp, bp)
                            # If the implied mp_cur_val is below the lower bound on m, then cannot update V_vals, leaving it at the -999
                            if mp_cur_val >= sspace.m_grid[1]
                                V0e_itp_temp      = extrapolate(interpolate((sspace.k_grid, sspace.m_grid), V0e_tp1[:,:,1,iz], Gridded(Linear())), NaN)
                                # Evaluate the continuation value function
                                V_vals[ic] = V0e_itp_temp(kp_cur_val, mp_cur_val)
                            end
                        end
                        # Find the optimum index
                        ic_max = findmax(V_vals)[2]
                        # If this is not equal to ii, add ii to be dropped
                        if ic_max != ii
                            ind_keep_col[ii]=0
                        end
                    end

                    # Having established which elements of the k'- and k-vectors to keep, construct them here to use for interpolation
                    # In order to make indexing of all the relevant continuation vectors easier, first sort kN_uc_val, and then drop the indices that are meant to be dropped
                    i_kN_sort      = sortperm(kN_uc_val)
                    ind_keep_col_sort = ind_keep_col[i_kN_sort]
                    i_kN_NC_val    = i_kN_sort[ind_keep_col_sort] # Use this "indexer" to now construct the correctly ordered "kN_val" vector

                    # Create interpolants
                    kpN_uc_itp      = extrapolate(interpolate((kN_uc_val[i_kN_NC_val], ), sspace.k_grid[i_kN_NC_val], Gridded(Linear())), Line())
                    lambda2N_uc_itp = extrapolate(interpolate((kN_uc_val[i_kN_NC_val], ), M_tp1*V0em_tp1[i_kN_NC_val,imp,1,iz], Gridded(Linear())), Line())
                    # And the continuation value function
                    V0eb_uc_itp     = extrapolate(interpolate((kN_uc_val[i_kN_NC_val], ), V0eb_tp1[i_kN_NC_val,imp,1,iz], Gridded(Linear())), Line())
                    V0e_uc_itp      = extrapolate(interpolate((kN_uc_val[i_kN_NC_val], ), V0e_tp1[i_kN_NC_val,imp,1,iz], Gridded(Linear())), Line())
                else
                    # Create interpolants
                    kpN_uc_itp      = extrapolate(interpolate((kN_uc_val, ), sspace.k_grid, Gridded(Linear())), Line())
                    lambda2N_uc_itp = extrapolate(interpolate((kN_uc_val, ), M_tp1*V0em_tp1[:,imp,1,iz], Gridded(Linear())), Line())
                    # And the continuation value function
                    V0eb_uc_itp     = extrapolate(interpolate((kN_uc_val, ), V0eb_tp1[:,imp,1,iz], Gridded(Linear())), Line())
                    V0e_uc_itp      = extrapolate(interpolate((kN_uc_val, ), V0e_tp1[:,imp,1,iz], Gridded(Linear())), Line())
                end

                kpN_uc_imp, lambda2N_uc_imp = kpN_uc_itp.(sspace.k_grid), lambda2N_uc_itp.(sspace.k_grid)
                V0eb_uc_imp, V0e_uc_imp  = V0eb_uc_itp.(sspace.k_grid), V0e_uc_itp.(sspace.k_grid)

                # Save in the corresponding matrices
                kpN_sol_k_temp[:,imp], lambda2N_sol_k_temp[:,imp] = kpN_uc_imp, lambda2N_uc_imp
                V0eb_sol_k_temp[:,imp], V0e_sol_k_temp[:,imp] = V0eb_uc_imp, V0e_uc_imp
            end

            # Now, for each k, solve out the m-dimension
            for ik = 1:sspace.Nk
                # First, over the unconstrained m', invert the budget constraint for mtil
                mtilN_uc_val = (-(1.0-(1.0-modl.tau)*modl.delta)*pQ_t*sspace.k_grid[ik] .- sspace.m_grid[1] .+ sspace.m_grid .+ pQ_t*kpN_sol_k_temp[ik,:] + pQ_t*(modl.kappa/2.0)*sspace.k_grid[ik].*(kpN_sol_k_temp[ik,:]./sspace.k_grid[ik] .- 1.0).^2 - pQ_t*sspace.k_grid[ik].*(-(sspace.b_grid[1]/modl.theta)*kpN_sol_k_temp[ik,:]./sspace.k_grid[ik] + (modl.kappa/2.0)*(-(sspace.b_grid[1]/modl.theta)*kpN_sol_k_temp[ik,:]./sspace.k_grid[ik] .- 1.0).^2) ) ./ (pQ_t * sspace.k_grid[ik])
                # Because bp=b_grid[1]=-theta, then by construction, at the lowest m', must have mtil at lowest mtil
                mtilN_uc_val[1] = copy(sspace.mtil_grid[1])

                # Create interpolants for all objects, conditional on UC
                # If endogenous grid not ordered, deal with the implied non-convexity and find the global optimum in the spirit of Fella (2014)
                if !issorted(mtilN_uc_val)

                    # Find the "flipping points"
                    mtilN_uc_diff = mtilN_uc_val[2:end] - mtilN_uc_val[1:(end-1)]
                    i_flip_set = findall(i->(i<0.0), mtilN_uc_diff)
                    mtilN_uc_fmax = maximum(mtilN_uc_val[i_flip_set]) # The "upper bound" on the non-concavity region, in mtil-space
                    mtilN_uc_fmin = minimum(mtilN_uc_val[i_flip_set.+1]) # The "lower bound" on the non-concavity region, in mtil-space
                    # Find the "boundaries" of the non-concavity region, in the index-space
                    # In case the "lowest flipping point" is itself the LOWEST POINT of mtilN_uc_val, then i_LB=0
                    if any(mtilN_uc_val .< mtilN_uc_fmin)
                        i_LB = maximum(findall(i->(i<mtilN_uc_fmin), mtilN_uc_val))
                    else
                        i_LB = 0
                    end
                    # In case the "highest flipping point" is itself the HIGHEST POINT of mtilN_uc_val, then i_UB=Nb+1
                    if any(mtilN_uc_val .> mtilN_uc_fmax)
                        i_UB = minimum(findall(i->(i>mtilN_uc_fmax), mtilN_uc_val))
                    else
                        i_UB = length(mtilN_uc_val) + 1
                    end

                    # Create collector of indices which to not drop
                    ind_keep_col = ones(Bool,length(mtilN_uc_val))
                    # Now, consider each index inbetween i_LB and i_UB individually, in order to establish whether the found point is actually a global optimum
                    for ii=(i_LB+1):(i_UB-1)
                        # Implied incoming states (k,mtil) at ii
                        mtilN_cur_val = mtilN_uc_val[ii]
                        kN_cur_val   = sspace.k_grid[ik]
                        # Create collector for candidate values
                        V_vals = -999.9*ones(size(sspace.m_grid))
                        # Evaluate the value function at the indices in the non-concave region
                        for ic=(i_LB+1):(i_UB-1)
                            # Implied choices (kp, mp) at candidate ic
                            mp_cur_val = sspace.m_grid[ic]
                            # If the implied mp_cur_val is below the lower bound on m, then cannot update V_vals, leaving it at the -999
                            if mp_cur_val >= sspace.m_grid[1]
                                # Solve the quadratic budget constraint for kp
                                a_kpdkq_temp = 0.5*modl.kappa*(1.0-(-sspace.b_grid[1]/modl.theta)^2)
                                b_kpdkq_temp = (1.0-modl.kappa)*(1.0+sspace.b_grid[1]/modl.theta)
                                c_kpdkq_temp = -((1.0-(1.0-modl.tau)*modl.delta) + mtilN_cur_val - (mp_cur_val-sspace.m_grid[1])/(pQ_t*kN_cur_val))
                                # Consider candidate mp only if it implies a solution for kp, otherwise leave value to -999
                                if (b_kpdkq_temp.^2 - 4*a_kpdkq_temp*c_kpdkq_temp) >= 0.0
                                    kpdk_temp_val = (- b_kpdkq_temp .+ sqrt.(b_kpdkq_temp.^2 .- 4*a_kpdkq_temp*c_kpdkq_temp))/(2.0*a_kpdkq_temp)
                                    # The implied kp
                                    kpN_val     = kpdk_temp_val*kN_cur_val

                                    # Now, evaluate the value function at the implied (kp, mp)
                                    V0e_itp_temp      = extrapolate(interpolate((sspace.k_grid, sspace.m_grid), V0e_tp1[:,:,1,iz], Gridded(Linear())), NaN)
                                    # Evaluate the continuation value function
                                    V_vals[ic] = V0e_itp_temp(kpN_val, mp_cur_val)
                                end
                            end
                        end
                        # Find the optimum index
                        ic_max = findmax(V_vals)[2]
                        # If this is not equal to ii, add ii to be dropped
                        if ic_max != ii
                            ind_keep_col[ii]=0
                        end
                    end

                    # Having established which elements of the m'- and mtil-vectors to keep, construct them here to use for interpolation
                    # In order to make indexing of all the relevant continuation vectors easier, first sort mtilN_uc_val, and then drop the indices that are meant to be dropped
                    i_mtilN_sort      = sortperm(mtilN_uc_val)
                    ind_keep_col_sort = ind_keep_col[i_mtilN_sort]
                    i_mtilN_NC_val    = i_mtilN_sort[ind_keep_col_sort] # Use this "indexer" to now construct the correctly ordered "btilA_val" vector

                    # Create interpolants
                    mpN_uc_itp      = extrapolate(interpolate((mtilN_uc_val[i_mtilN_NC_val], ), sspace.m_grid[i_mtilN_NC_val], Gridded(Linear())), Line())
                    kpN_uc_itp      = extrapolate(interpolate((mtilN_uc_val[i_mtilN_NC_val], ), kpN_sol_k_temp[ik,i_mtilN_NC_val], Gridded(Linear())), Line())
                    lambda2N_uc_itp = extrapolate(interpolate((mtilN_uc_val[i_mtilN_NC_val], ), lambda2N_sol_k_temp[ik,i_mtilN_NC_val], Gridded(Linear())), Line())
                    # And the continuation value function
                    V0eb_uc_itp     = extrapolate(interpolate((mtilN_uc_val[i_mtilN_NC_val], ), V0eb_sol_k_temp[ik,i_mtilN_NC_val], Gridded(Linear())), Line())
                    V0e_uc_itp      = extrapolate(interpolate((mtilN_uc_val[i_mtilN_NC_val], ), V0e_sol_k_temp[ik,i_mtilN_NC_val], Gridded(Linear())), Line())
                else
                    # Create interpolants
                    mpN_uc_itp      = extrapolate(interpolate((mtilN_uc_val, ), sspace.m_grid, Gridded(Linear())), Line())
                    kpN_uc_itp      = extrapolate(interpolate((mtilN_uc_val, ), kpN_sol_k_temp[ik,:], Gridded(Linear())), Line())
                    lambda2N_uc_itp = extrapolate(interpolate((mtilN_uc_val, ), lambda2N_sol_k_temp[ik,:], Gridded(Linear())), Line())
                    # And the continuation value function
                    V0eb_uc_itp     = extrapolate(interpolate((mtilN_uc_val, ), V0eb_sol_k_temp[ik,:], Gridded(Linear())), Line())
                    V0e_uc_itp      = extrapolate(interpolate((mtilN_uc_val, ), V0e_sol_k_temp[ik,:], Gridded(Linear())), Line())
                end

                # And evaluate these on the mtil_grid
                mpN_uc_imp, lambda2N_uc_imp, kpN_uc_imp = mpN_uc_itp.(sspace.mtil_grid), lambda2N_uc_itp.(sspace.mtil_grid), kpN_uc_itp.(sspace.mtil_grid)
                V0eb_uc_imp, V0e_uc_imp     = V0eb_uc_itp.(sspace.mtil_grid), V0e_uc_itp.(sspace.mtil_grid)

                # N2-Step 1c: by construction, the UC solution spans whole mtil_grid!
                kpN_imp, mpN_imp, lambda2N_imp, V0eb2N_imp, V0e2N_imp  = kpN_uc_imp, mpN_uc_imp, lambda2N_uc_imp, V0eb_uc_imp, V0e_uc_imp

                # Save these into the "final" collectors, now CONDITIONAL on muk
                kpN_sol_muk[ik,:,imuk], mpN_sol_muk[ik,:,imuk], lambda2N_sol_muk[ik,:,imuk] = kpN_imp, mpN_imp, lambda2N_imp

                # And for the value functions as well, using envelope conditions
                # Note that the grid-point of b'=b_grid_set[ibp] equals, in math, gamma*b*(Rc_t/Rc_tm1)*k/k', so if we need its partial derivative wrt k, i.e. gamma*b*(Rc_t/Rc_tm1)/k', we can just compute it as: sspace.b_grid[ibp] / sspace.k_grid[ik]
                V2Nk_sol_muk[ik,:,imuk]    = lambda2N_imp .* ((1.0.-(1.0-modl.tau)*modl.delta)*pQ_t .+ sspace.mtil_grid*pQ_t .+ pQ_t*((-sspace.b_grid[1]/modl.theta)*(kpN_imp./sspace.k_grid[ik]) .+ (modl.kappa/2.0)*((-sspace.b_grid[1]/modl.theta)*(kpN_imp./sspace.k_grid[ik]).-1.0).^2 ) .+ pQ_t*(modl.kappa/2.0)*( (kpN_imp ./ sspace.k_grid[ik]).^2 .- 1.0) ) .+ M_tp1*V0eb2N_imp.*sspace.b_grid[1]./sspace.k_grid[ik] .+ sspace.muk_grid[imuk]*sspace.b_grid[1]*(kpN_imp./sspace.k_grid[ik])/modl.theta
                V2Nmtil_sol_muk[ik,:,imuk] = lambda2N_imp .* sspace.k_grid[ik] * pQ_t
                V2Nb_sol_muk[ik,:,imuk]    = M_tp1*V0eb2N_imp*gammatil_t*(Rc_t/Rc_tm1) .* (sspace.k_grid[ik]./kpN_imp) .+ lambda2N_imp .* pQ_t .* (-gammatil_t/modl.theta)*(Rc_t/Rc_tm1) .* (1.0 .+ modl.kappa*((-sspace.b_grid[1]/modl.theta)*(kpN_imp./sspace.k_grid[ik]) .- 1.0 )) * sspace.k_grid[ik] .+ sspace.muk_grid[imuk]*gammatil_t*(Rc_t/Rc_tm1)*sspace.k_grid[ik]/modl.theta
                V2N_sol_muk[ik,:,imuk]     = M_tp1*V0e2N_imp
            end
        end

        # Finally, given the solutions conditional on b' and conditional on muk, "solve out" the b-dimension
        # Create the collectors for the final solutions, still conditional on z
        kpN_sol_z, mpN_sol_z, bpN_sol_z, lambda2N_sol_z = Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb);
        # And for the value functions also
        V2Nk_sol_z, V2Nmtil_sol_z, V2Nb_sol_z, V2N_sol_z = Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb), Array{Float64}(undef, sspace.Nk, sspace.Nmtil, sspace.Nb);
        # Define gamma_ul for brevity
        gamma_ul_t = -(gammatil_t/modl.theta)*(Rc_t/Rc_tm1)

        # Iterating over the current (k,mtil), "solve out" b
        for iik=1:sspace.Nk
            for iimtil=1:sspace.Nmtil
                # "Iterating over" b', infer the current implied b_UC
                b_UC_val = sspace.b_grid .* kpN_sol_bp[iik,iimtil,:] .* (Rc_tm1 / Rc_t) ./ (gammatil_t * sspace.k_grid[iik])
                # Specify the 0 as the last b, by definition of multiplying by sspace.b_grid[end]
                b_UC_val[end] = 0.0
                if b_UC_val[1] <= sspace.b_grid[1]
                    # No need to make the "inversion" conditional on muk>0, because the b-grid is already covered
                    b_C_val  = repeat([b_UC_val[1]],sspace.Nmuk)  - (sspace.Nmuk:-1:1)*1e-12
                    f_bCind  = 1
                else
                    # "Iterating over" muk, infer the current implied b_C
                    b_C_val  = sspace.b_grid[1] .* kpN_sol_muk[iik,iimtil,:] .* (Rc_tm1 / Rc_t) ./ (gammatil_t * sspace.k_grid[iik])
                    # To avoid issues implied by the extrapolation in the k-dimension associated with states mpN_sol_muk[iik,iimtil,:]<m_grid[1] outside the incoming m-space and thus not applicable in practice, override the b_C_val values with the b value corresponding to the highest muk (the highest implied k' choice)
                    b_C_val[mpN_sol_muk[iik,iimtil,:] .< sspace.m_grid[1]] .= b_C_val[1]
                    # Also, some extremely high muk values may require that the implied choice of k is such that k is actually GROWING (and the implied b values are far below the minimal feasible b-point)
                    # Such values may lead to some extreme behavior in the k' policy, mainly due to a "near constrained" m' policy, and thus induce close to flat k' policies, with b-grids that have potential nonmonotonicities. Leave these out.
                    f_gblim  = findfirst(b_C_val .> sspace.b_grid[1])
                    if f_gblim==1
                        f_bCind = 1
                    else
                        f_bCind = f_gblim-1
                    end
                end

                # Unconstrained b'
                # If endogenous grid not ordered, deal with the implied non-convexity and find the global optimum in the spirit of Fella (2014)
                if !issorted(b_UC_val)
                    # Find the "flipping points"
                    b_UC_diff = b_UC_val[2:end] - b_UC_val[1:(end-1)]
                    i_flip_set = findall(i->(i<0.0), b_UC_diff)
                    b_UC_fmax = maximum(b_UC_val[i_flip_set]) # The "upper bound" on the non-concavity region, in b-space
                    b_UC_fmin = minimum(b_UC_val[i_flip_set.+1]) # The "lower bound" on the non-concavity region, in b-space

                    # Find the "boundaries" of the non-concavity region, in the index-space
                    # In case the "lowest flipping point" is itself the LOWEST POINT of b_UC_val, then i_LB=0
                    if any(b_UC_val .< b_UC_fmin)
                        i_LB = maximum(findall(i->(i<b_UC_fmin), b_UC_val))
                    else
                        i_LB = 0
                    end
                    # In case the "highest flipping point" is itself the HIGHEST POINT of b_UC_val, then i_UB=Nb+1
                    if any(b_UC_val .> b_UC_fmax)
                        i_UB = minimum(findall(i->(i>b_UC_fmax), b_UC_val))
                    else
                        i_UB = length(b_UC_val) + 1
                    end

                    # Create collector of indices which to not drop
                    ind_keep_col = ones(Bool,length(b_UC_val))
                    # Now, consider each index inbetween i_LB and i_UB individually, in order to establish whether the found point is actually a global optimum
                    for ii=(i_LB+1):(i_UB-1)
                        # Implied incoming states (k,mtil,b)
                        b_cur_val    = b_UC_val[ii]
                        k_cur_val    = sspace.k_grid[iik]
                        mtil_cur_val = sspace.mtil_grid[iimtil]

                        # Create collector for candidate values
                        V_vals = -999.9*ones(size(sspace.b_grid))
                        # Evaluate the value function at the indices in the non-concave region
                        for ic=(i_LB+1):(i_UB-1)
                            # Implied choice of bp at candidate ic
                            bp_cur_val = sspace.b_grid[ic]
                            # Use the relation between b' and k'/k and b' to infer the implied k'/k and k'
                            kpdk_cur_val = gammatil_t*(Rc_t/Rc_tm1)*(b_cur_val/bp_cur_val)
                            kp_cur_val = kpdk_cur_val*k_cur_val
                            # Use the budget constraint to finally infer the implied mp value
                            mp_cur_val = (1.0-(1.0-modl.tau)*modl.delta)*pQ_t*k_cur_val + sspace.m_grid[1] + mtil_cur_val*pQ_t*k_cur_val - pQ_t*kp_cur_val - pQ_t*(modl.kappa/2.0)*k_cur_val*(kp_cur_val/k_cur_val - 1.0)^2 + pQ_t*k_cur_val*(gamma_ul_t*b_cur_val + (modl.kappa/2.0)*(gamma_ul_t*b_cur_val - 1.0).^2)

                            # Now, evaluate the value function at the implied (kp, mp, bp)
                            # If the implied mp_cur_val is below the lower bound on m, then cannot update V_vals, leaving it at the -999
                            if mp_cur_val >= sspace.m_grid[1]
                                V0e_itp_temp      = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid), V0e_tp1[:,:,:,iz], Gridded(Linear())), Line())
                                # Evaluate the continuation value function
                                V_vals[ic] = V0e_itp_temp(kp_cur_val, mp_cur_val, bp_cur_val)
                            end
                        end
                        # Find the optimum index
                        ic_max = findmax(V_vals)[2]
                        # If this is not equal to ii, add ii to be dropped
                        if ic_max != ii
                            ind_keep_col[ii]=0
                        end
                    end

                    # Having established which elements of the b'- and b-vectors to keep, construct them here to use for interpolation
                    # In order to make indexing of all the relevant continuation vectors easier, first sort b_UC_val, and then drop the indices that are meant to be dropped
                    i_b_sort      = sortperm(b_UC_val)
                    ind_keep_col_sort = ind_keep_col[i_b_sort]
                    i_b_NC_val    = i_b_sort[ind_keep_col_sort] # Use this "indexer" to now construct the correctly ordered "b_UC_val" vector

                    # Create interpolants for bpN and all the functions that were solved above conditional on bp
                    bpN_UC_itp = extrapolate(interpolate((b_UC_val[i_b_NC_val], ), sspace.b_grid[i_b_NC_val], Gridded(Linear())), NaN)
                    kpN_UC_itp = extrapolate(interpolate((b_UC_val[i_b_NC_val], ), kpN_sol_bp[iik,iimtil,i_b_NC_val], Gridded(Linear())), NaN)
                    mpN_UC_itp = extrapolate(interpolate((b_UC_val[i_b_NC_val], ), mpN_sol_bp[iik,iimtil,i_b_NC_val], Gridded(Linear())), NaN)
                    lambda2N_UC_itp = extrapolate(interpolate((b_UC_val[i_b_NC_val], ), lambda2N_sol_bp[iik,iimtil,i_b_NC_val], Gridded(Linear())), NaN)
                    V2Nk_UC_itp    = extrapolate(interpolate((b_UC_val[i_b_NC_val], ), V2Nk_sol_bp[iik,iimtil,i_b_NC_val], Gridded(Linear())), NaN)
                    V2Nmtil_UC_itp = extrapolate(interpolate((b_UC_val[i_b_NC_val], ), V2Nmtil_sol_bp[iik,iimtil,i_b_NC_val], Gridded(Linear())), NaN)
                    V2Nb_UC_itp    = extrapolate(interpolate((b_UC_val[i_b_NC_val], ), V2Nb_sol_bp[iik,iimtil,i_b_NC_val], Gridded(Linear())), NaN)
                    V2N_UC_itp     = extrapolate(interpolate((b_UC_val[i_b_NC_val], ), V2N_sol_bp[iik,iimtil,i_b_NC_val], Gridded(Linear())), NaN)
                else
                    # Create interpolants for bpN and all the functions that were solved above conditional on bp
                    bpN_UC_itp = extrapolate(interpolate((b_UC_val, ), sspace.b_grid, Gridded(Linear())), NaN)
                    kpN_UC_itp = extrapolate(interpolate((b_UC_val, ), kpN_sol_bp[iik,iimtil,:], Gridded(Linear())), NaN)
                    mpN_UC_itp = extrapolate(interpolate((b_UC_val, ), mpN_sol_bp[iik,iimtil,:], Gridded(Linear())), NaN)
                    lambda2N_UC_itp = extrapolate(interpolate((b_UC_val, ), lambda2N_sol_bp[iik,iimtil,:], Gridded(Linear())), NaN)
                    V2Nk_UC_itp    = extrapolate(interpolate((b_UC_val, ), V2Nk_sol_bp[iik,iimtil,:], Gridded(Linear())), NaN)
                    V2Nmtil_UC_itp = extrapolate(interpolate((b_UC_val, ), V2Nmtil_sol_bp[iik,iimtil,:], Gridded(Linear())), NaN)
                    V2Nb_UC_itp    = extrapolate(interpolate((b_UC_val, ), V2Nb_sol_bp[iik,iimtil,:], Gridded(Linear())), NaN)
                    V2N_UC_itp     = extrapolate(interpolate((b_UC_val, ), V2N_sol_bp[iik,iimtil,:], Gridded(Linear())), NaN)
                end

                # Constrained b'
                # ERROR if endogenous grid not ordered
                if !issorted(b_C_val[f_bCind:end]); manprintln("ERROR: in sV2N, b_C_val is not ordered (ik=$(iik),iimtil=$(iimtil),iz=$(iz)).", txtout_path); end
                bpN_C_itp = extrapolate(interpolate((b_C_val[f_bCind:end], ), repeat([sspace.b_grid[1]], length(b_C_val[f_bCind:end])), Gridded(Linear())), NaN)
                kpN_C_itp = extrapolate(interpolate((b_C_val[f_bCind:end], ), kpN_sol_muk[iik,iimtil,f_bCind:end], Gridded(Linear())), NaN)
                mpN_C_itp = extrapolate(interpolate((b_C_val[f_bCind:end], ), mpN_sol_muk[iik,iimtil,f_bCind:end], Gridded(Linear())), NaN)
                lambda2N_C_itp = extrapolate(interpolate((b_C_val[f_bCind:end], ), lambda2N_sol_muk[iik,iimtil,f_bCind:end], Gridded(Linear())), NaN)
                V2Nk_C_itp    = extrapolate(interpolate((b_C_val[f_bCind:end], ), V2Nk_sol_muk[iik,iimtil,f_bCind:end], Gridded(Linear())), NaN)
                V2Nmtil_C_itp = extrapolate(interpolate((b_C_val[f_bCind:end], ), V2Nmtil_sol_muk[iik,iimtil,f_bCind:end], Gridded(Linear())), NaN)
                V2Nb_C_itp    = extrapolate(interpolate((b_C_val[f_bCind:end], ), V2Nb_sol_muk[iik,iimtil,f_bCind:end], Gridded(Linear())), NaN)
                V2N_C_itp     = extrapolate(interpolate((b_C_val[f_bCind:end], ), V2N_sol_muk[iik,iimtil,f_bCind:end], Gridded(Linear())), NaN)

                # And evaluate these on the b_grid
                kpN_UC_imp, kpN_C_imp   = kpN_UC_itp.(sspace.b_grid), kpN_C_itp.(sspace.b_grid)
                mpN_UC_imp, mpN_C_imp   = mpN_UC_itp.(sspace.b_grid), mpN_C_itp.(sspace.b_grid)
                bpN_UC_imp, bpN_C_imp   = bpN_UC_itp.(sspace.b_grid), bpN_C_itp.(sspace.b_grid)
                V2Nk_UC_imp, V2Nk_C_imp   = V2Nk_UC_itp.(sspace.b_grid), V2Nk_C_itp.(sspace.b_grid)
                V2Nmtil_UC_imp, V2Nmtil_C_imp   = V2Nmtil_UC_itp.(sspace.b_grid), V2Nmtil_C_itp.(sspace.b_grid)
                V2Nb_UC_imp, V2Nb_C_imp   = V2Nb_UC_itp.(sspace.b_grid), V2Nb_C_itp.(sspace.b_grid)
                V2N_UC_imp, V2N_C_imp     = V2N_UC_itp.(sspace.b_grid), V2N_C_itp.(sspace.b_grid)

                # N2-Step 1c: combine constrained and unconstrained solutions
                kpN_UC_imp, kpN_C_imp = replace(kpN_UC_imp, NaN=> - Inf), replace(kpN_C_imp, NaN=> - Inf)
                mpN_UC_imp, mpN_C_imp = replace(mpN_UC_imp, NaN=> - Inf), replace(mpN_C_imp, NaN=> - Inf)
                bpN_UC_imp, bpN_C_imp = replace(bpN_UC_imp, NaN=> - Inf), replace(bpN_C_imp, NaN=> - Inf)
                V2Nk_UC_imp, V2Nk_C_imp = replace(V2Nk_UC_imp, NaN=> - Inf), replace(V2Nk_C_imp, NaN=> - Inf)
                V2Nmtil_UC_imp, V2Nmtil_C_imp = replace(V2Nmtil_UC_imp, NaN=> - Inf), replace(V2Nmtil_C_imp, NaN=> - Inf)
                V2Nb_UC_imp, V2Nb_C_imp = replace(V2Nb_UC_imp, NaN=> - Inf), replace(V2Nb_C_imp, NaN=> - Inf)
                V2N_UC_imp, V2N_C_imp   = replace(V2N_UC_imp, NaN=> - Inf), replace(V2N_C_imp, NaN=> - Inf)

                # Directly save in the collectors
                kpN_sol_z[iik,iimtil,:]    = max.(kpN_UC_imp, kpN_C_imp)
                mpN_sol_z[iik,iimtil,:]    = max.(mpN_UC_imp, mpN_C_imp)
                bpN_sol_z[iik,iimtil,:]    = max.(bpN_UC_imp, bpN_C_imp)
                V2Nk_sol_z[iik,iimtil,:]    = max.(V2Nk_UC_imp, V2Nk_C_imp)
                V2Nmtil_sol_z[iik,iimtil,:] = max.(V2Nmtil_UC_imp, V2Nmtil_C_imp)
                V2Nb_sol_z[iik,iimtil,:]    = max.(V2Nb_UC_imp, V2Nb_C_imp)
                V2N_sol_z[iik,iimtil,:]     = max.(V2N_UC_imp, V2N_C_imp)
                    
            end
        end

        # And finally, save the solution into the final collectors
        kpN_sol_fin[:,:,:,iz] = kpN_sol_z
        mpN_sol_fin[:,:,:,iz] = mpN_sol_z
        bpN_sol_fin[:,:,:,iz] = bpN_sol_z
        V2Nk_sol_fin[:,:,:,iz]      = V2Nk_sol_z
        V2Nmtil_sol_fin[:,:,:,iz]   = V2Nmtil_sol_z
        V2Nb_sol_fin[:,:,:,iz]      = V2Nb_sol_z
        V2N_sol_fin[:,:,:,iz]       = V2N_sol_z
    end

    # Return the policy and and value functions
    return kpN_sol_fin, mpN_sol_fin, bpN_sol_fin, V2Nk_sol_fin, V2Nmtil_sol_fin, V2Nb_sol_fin, V2N_sol_fin
end


# Solve first stage of N
function solve_V1N(V2Nk_in, V2Nmtil_in, V2Nb_in, V2N_in, MU_t, w_t, M_tp1, pQ_t, Pi_t, rb_t, rm_t, Rc_tm1, Rc_t, gammatil_t, modl, sspace, opts)

    # Set output text printing path
    txtout_path = joinpath(opts.project_root, "interim_output", "model_interim_output", "savedout", modl.name * "_output.txt")

    #### N - 1st stage

    # Precompute the gamma_ul_t
    gamma_ul_t = -(gammatil_t/modl.theta)*(Rc_t/Rc_tm1)

    # Before iterations create collectors for final policies for this stage
    mtilN_sol_fin, dN_sol_fin = Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz)
    V1Nk_sol_fin, V1Nm_sol_fin, V1Nb_sol_fin, V1N_sol_fin = Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz)
    V1Nb_imp_s1, V1Nb_imp_s2, V1Nb_imp_s3 = Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz)

    # For each (b,k), solve the "beginning of period" problem over m
    for iz=1:sspace.Nz
        for ik=1:sspace.Nk
            for ib=1:sspace.Nb

                # Using interpolation of V2Nmtil, compute the values of lambda1N implied by d=0 to determine which current m states are constrained
                lambda1N_mtilitp= extrapolate(interpolate((sspace.mtil_grid, ), V2Nmtil_in[ik,:,ib,iz]/(pQ_t*sspace.k_grid[ik]), Gridded(Linear())), Flat())
                # NOTE: FLAT extrapolation -- REASON: to not make inference about a potentially "unconstrained state" (e.g. if mtil=(R*m+zf(k))/k maps into a point outside mtil grid that might imply being unconstrained due to linear extrapolation, yet no state in sspace.mtil_grid might itself be unconstrained -- so it's safer to assume ALL states in sspace.mtil_grid are still constrained.)
                mtil_dzero = (((1.0+(1.0-modl.tau)*rm_t)/Pi_t)*sspace.m_grid .+ ((1.0+(1.0-modl.tau)*rb_t)/Pi_t - gammatil_t)*sspace.b_grid[ib]*sspace.k_grid[ik]/Rc_tm1 .- sspace.m_grid[1] .- pQ_t*(gamma_ul_t*sspace.b_grid[ib] + (modl.kappa/2.0)*(gamma_ul_t*sspace.b_grid[ib]-1.0)^2 )*sspace.k_grid[ik] .+ (1.0-modl.tau) * ymenufun_mat("yp", repeat([sspace.k_grid[ik]],sspace.Nm), sspace.m_grid/Pi_t, repeat([sspace.z_grid[iz]],sspace.Nm), MU_t, w_t, modl))/(pQ_t * sspace.k_grid[ik])
                lambda1N_mimp   = lambda1N_mtilitp.(mtil_dzero)
                # Wherever this is less than 1: the first-step solution is unconstrained!
                # Generate indicator for constrained states in m-space
                dc_ind      = lambda1N_mimp .> 1.0

                # Find the cutoff mtil ("mtil_crit") based on V2Nmtil_in on sspace.mtil_grid
                V2Nm_in_constr_ind = V2Nmtil_in[ik,:,ib,iz]/(pQ_t*sspace.k_grid[ik]) .> 1.0

                if all(V2Nm_in_constr_ind)
                    # If all mtils on grid are constrained, don't "extrapolate" unconstrainedness (as discussed above) and just impose that mtil_crit corresponds to the highest point on m_grid, i.e. (R*m_max + zf(k))/k
                    mtil_crit = (((1.0+(1.0-modl.tau)*rm_t)/Pi_t)*sspace.m_grid[end] .+ ((1.0+(1.0-modl.tau)*rb_t)/Pi_t-gammatil_t)*sspace.b_grid[ib]*sspace.k_grid[ik]/Rc_tm1 .- sspace.m_grid[1] .- pQ_t*(gamma_ul_t*sspace.b_grid[ib] + (modl.kappa/2.0)*(gamma_ul_t*sspace.b_grid[ib]-1.0)^2 )*sspace.k_grid[ik] .+ (1.0-modl.tau) * ymenufun_mat("yp", repeat([sspace.k_grid[ik]],sspace.Nm), repeat([sspace.m_grid[end]],sspace.Nm)/Pi_t , repeat([sspace.z_grid[iz]],sspace.Nm), MU_t, w_t, modl))/(pQ_t * sspace.k_grid[ik])
                else
                    fuc_i = findfirst(.!V2Nm_in_constr_ind)
                    if fuc_i == 1
                        # If the first mtil gridpoint is already unconstrained, set it as the "critical level"
                        mtil_crit = sspace.mtil_grid[1]
                    else
                        # Otherwise interpolate mtil_crit linearly based on the two adjacent mtil grid points; note that being unconstrained means "V2Nmtil = Q*k"
                        mtil_crit = sspace.mtil_grid[fuc_i-1] + (sspace.mtil_grid[fuc_i]-sspace.mtil_grid[fuc_i-1])*(pQ_t*sspace.k_grid[ik]-V2Nmtil_in[ik,fuc_i-1,ib,iz])/(V2Nmtil_in[ik,fuc_i,ib,iz]-V2Nmtil_in[ik,fuc_i-1,ib,iz])
                    end
                end

                # Compute implied dividends and mtil, given the m-grid
                d_imp    = 0.0 .+ (1.0.-dc_ind) .* (mtil_dzero .- mtil_crit)*sspace.k_grid[ik]*pQ_t
                mtil_imp = dc_ind .* mtil_dzero .+ (1.0.-dc_ind) .* mtil_crit


                # Before continuing, check if V2Nmtil_in[ik,:,iz]/Qk has multiple crossings of 1. If yes, must search for global optimum
                n_cross1 = sum(abs.(V2Nm_in_constr_ind[2:end]-V2Nm_in_constr_ind[1:(end-1)]))

                if (n_cross1 >= 2) & true

                    # Find the highest crossing
                    ic_last = findlast(abs.(V2Nm_in_constr_ind[2:end]-V2Nm_in_constr_ind[1:(end-1)]).>0)
                    # Interpolate for the exact value of mtil at this crossing
                    mtil_lastcross = sspace.mtil_grid[ic_last] + (sspace.mtil_grid[ic_last+1]-sspace.mtil_grid[ic_last])*(pQ_t*sspace.k_grid[ik]-V2Nmtil_in[ik,ic_last,ib,iz])/(V2Nmtil_in[ik,ic_last+1,ib,iz]-V2Nmtil_in[ik,ic_last,ib,iz])
                    # And find the lowest crossing
                    ic_first = findfirst(abs.(V2Nm_in_constr_ind[2:end]-V2Nm_in_constr_ind[1:(end-1)]).>0)
                    mtil_firstcross = sspace.mtil_grid[ic_first] + (sspace.mtil_grid[ic_first+1]-sspace.mtil_grid[ic_first])*(pQ_t*sspace.k_grid[ik]-V2Nmtil_in[ik,ic_first,ib,iz])/(V2Nmtil_in[ik,ic_first+1,ib,iz]-V2Nmtil_in[ik,ic_first,ib,iz])

                    # Now, do global maximization, conditional on incoming m value, but impose that for all values above the last crossing, the solution must be the same (because in that region the marginal benefit is strictly below one throughout)
                    # And for all values below the first crossing, the solution must be constrained
                    # Find the first m value that implies an mtil just above the last mtil cutoff
                    if any(mtil_dzero .> mtil_lastcross)
                        im_first = findfirst(mtil_dzero .> mtil_lastcross)
                    else
                        im_first = sspace.Nm
                    end
                    # Find the last m value that implies an mtil just below the first mtil cutoff
                    if any(mtil_dzero .< mtil_firstcross)
                        im_last = findlast(mtil_dzero .< mtil_firstcross)
                    else
                        im_last = 1
                    end
                    # Create interpolant of continuation V2A
                    V2N_itp     = extrapolate(interpolate((sspace.mtil_grid, ), V2N_in[ik,:,ib,iz], Gridded(Linear())), NaN)

                    for im=im_last:im_first
                        # Set up the objective function
                        V1N_obj_temp(x) = - x*pQ_t*sspace.k_grid[ik] + ((1.0+(1.0-modl.tau)*rm_t)/Pi_t)*sspace.m_grid[im] .+ ((1.0+(1.0-modl.tau)*rb_t)/Pi_t - gammatil_t)*sspace.b_grid[ib]*sspace.k_grid[ik]/Rc_tm1 .- sspace.m_grid[1] .- pQ_t*(gamma_ul_t*sspace.b_grid[ib] + (modl.kappa/2.0)*(gamma_ul_t*sspace.b_grid[ib]-1.0)^2 )*sspace.k_grid[ik] .+ (1.0-modl.tau) * ymenufun("yp", sspace.k_grid[ik], sspace.m_grid[im]/Pi_t, sspace.z_grid[iz], MU_t, w_t, modl) + V2N_itp(x)

                        # Manually search on mtil grid -- note that mtil_dzero[im] exactly implies the upper bound on mtil that corresponds to d=0 for the given m
                        mtil_grid_temp = Array(range(sspace.mtil_grid[1], min(mtil_dzero[im], mtil_lastcross), step=1e-2))
                        # Make sure to include the endpoint in the grid
                        append!(mtil_grid_temp, min(mtil_dzero[im], mtil_lastcross))
                        V1N_obj_vals = V1N_obj_temp.(mtil_grid_temp)
                        i_max = argmax(V1N_obj_vals) # Index of optimum

                        # Impose values in collectors
                        mtil_imp[im] = mtil_grid_temp[i_max]
                        d_imp[im]    = - mtil_imp[im]*pQ_t*sspace.k_grid[ik] + ((1.0+(1.0-modl.tau)*rm_t)/Pi_t)*sspace.m_grid[im] .+ ((1.0+(1.0-modl.tau)*rb_t)/Pi_t - gammatil_t)*sspace.b_grid[ib]*sspace.k_grid[ik]/Rc_tm1 .- sspace.m_grid[1] .- pQ_t*(gamma_ul_t*sspace.b_grid[ib] + (modl.kappa/2.0)*(gamma_ul_t*sspace.b_grid[ib]-1.0)^2 )*sspace.k_grid[ik] .+ (1.0-modl.tau) * ymenufun("yp", sspace.k_grid[ik], sspace.m_grid[im]/Pi_t, sspace.z_grid[iz], MU_t, w_t, modl)
                    end

                    # For all the values below im_first, impose the constrained values
                    mtil_imp[1:im_last] .= mtil_dzero[1:im_last]
                    d_imp[1:im_last]    .= - mtil_imp[1:im_last]*pQ_t*sspace.k_grid[ik] .+ ((1.0+(1.0-modl.tau)*rm_t)/Pi_t)*sspace.m_grid[1:im_last] .+ ((1.0+(1.0-modl.tau)*rb_t)/Pi_t - gammatil_t)*sspace.b_grid[ib]*sspace.k_grid[ik]/Rc_tm1 .- sspace.m_grid[1] .- pQ_t*(gamma_ul_t*sspace.b_grid[ib] + (modl.kappa/2.0)*(gamma_ul_t*sspace.b_grid[ib]-1.0)^2 )*sspace.k_grid[ik] .+ (1.0-modl.tau) * ymenufun_mat("yp", repeat([sspace.k_grid[ik]],length(mtil_imp[1:im_last])), sspace.m_grid[1:im_last]/Pi_t , repeat([sspace.z_grid[iz]],length(mtil_imp[1:im_last])), MU_t, w_t, modl)

                    # For all the values above im_first, impose the same values
                    mtil_imp[im_first:end] .= copy(mtil_imp[im_first])
                    d_imp[im_first:end]    .= - mtil_imp[im_first:end]*pQ_t*sspace.k_grid[ik] .+ ((1.0+(1.0-modl.tau)*rm_t)/Pi_t)*sspace.m_grid[im_first:end] .+ ((1.0+(1.0-modl.tau)*rb_t)/Pi_t - gammatil_t)*sspace.b_grid[ib]*sspace.k_grid[ik]/Rc_tm1 .- sspace.m_grid[1] .- pQ_t*(gamma_ul_t*sspace.b_grid[ib] + (modl.kappa/2.0)*(gamma_ul_t*sspace.b_grid[ib]-1.0)^2 )*sspace.k_grid[ik] .+ (1.0-modl.tau) * ymenufun_mat("yp", repeat([sspace.k_grid[ik]],length(mtil_imp[im_first:end])), sspace.m_grid[im_first:end]/Pi_t , repeat([sspace.z_grid[iz]],length(mtil_imp[im_first:end])), MU_t, w_t, modl)
                end

                # And use the mtil_imp policy to infer all value function derivatives, given the m-grid using interpolation
                V2Nk_itp    = extrapolate(interpolate((sspace.mtil_grid, ), V2Nk_in[ik,:,ib,iz], Gridded(Linear())), Line())
                V2Nmtil_itp = extrapolate(interpolate((sspace.mtil_grid, ), V2Nmtil_in[ik,:,ib,iz], Gridded(Linear())), Line())
                V2Nb_itp    = extrapolate(interpolate((sspace.mtil_grid, ), V2Nb_in[ik,:,ib,iz], Gridded(Linear())), Line())
                V2N_itp     = extrapolate(interpolate((sspace.mtil_grid, ), V2N_in[ik,:,ib,iz], Gridded(Linear())), Line())
                # And evaluate at the solved mtil policy
                V2Nk_imp    = V2Nk_itp.(mtil_imp)
                V2Nmtil_imp = V2Nmtil_itp.(mtil_imp)
                V2Nb_imp    = V2Nb_itp.(mtil_imp)
                V2N_imp     = V2N_itp.(mtil_imp)

                # Impose that lambda must be bounded below by 1.0!
                lambda_LB   = 1.0

                # And map these into V1 and its derivatives based on the envelope conditions. NOTE: lambda1N=V2Nmtil/k
                V1Nk_imp    = V2Nk_imp .+ max.(V2Nmtil_imp/(sspace.k_grid[ik]pQ_t),lambda_LB) .* ((1.0-modl.tau)*ymenufun_mat("ypk", repeat([sspace.k_grid[ik]],sspace.Nm), sspace.m_grid/Pi_t, repeat([sspace.z_grid[iz]],sspace.Nm), MU_t, w_t, modl) .- mtil_imp*pQ_t .+ ((1.0+(1.0-modl.tau)*rb_t)/Pi_t - gammatil_t)*sspace.b_grid[ib]/Rc_tm1 .- pQ_t*(gamma_ul_t*sspace.b_grid[ib] + (modl.kappa/2.0)*(gamma_ul_t*sspace.b_grid[ib]-1.0)^2 ))
                V1Nm_imp    = max.(V2Nmtil_imp/(sspace.k_grid[ik]*pQ_t), lambda_LB) .* ((1.0+(1.0-modl.tau)*rm_t)/Pi_t .+ (1.0-modl.tau)*ymenufun_mat("ypm", repeat([sspace.k_grid[ik]],sspace.Nm), sspace.m_grid/Pi_t, repeat([sspace.z_grid[iz]],sspace.Nm), MU_t, w_t, modl)/Pi_t)
                V1Nb_imp    = V2Nb_imp + max.(V2Nmtil_imp/(sspace.k_grid[ik]*pQ_t), lambda_LB) .* ( ((1.0+(1.0-modl.tau)*rb_t)/Pi_t - gammatil_t)/Rc_tm1 .- pQ_t*gamma_ul_t*(1.0 + modl.kappa*(gamma_ul_t*sspace.b_grid[ib]-1.0)) ) * sspace.k_grid[ik]
                V1N_imp     = d_imp + V2N_imp

                V1Nb_imp_s1[ik,:,ib,iz]    = V2Nb_imp
                V1Nb_imp_s2[ik,:,ib,iz]    = (V2Nmtil_imp/(sspace.k_grid[ik]*pQ_t))
                V1Nb_imp_s3[ik,:,ib,iz]    .= ( ((1.0+(1.0-modl.tau)*rb_t)/Pi_t - gammatil_t)/Rc_tm1 .- pQ_t*gamma_ul_t*(1.0 + modl.kappa*(gamma_ul_t*sspace.b_grid[ib]-1.0)) ) * sspace.k_grid[ik]


                # Save these into the collectors
                mtilN_sol_fin[ik,:,ib,iz], dN_sol_fin[ik,:,ib,iz]  = mtil_imp, d_imp
                V1Nk_sol_fin[ik,:,ib,iz], V1Nm_sol_fin[ik,:,ib,iz], V1Nb_sol_fin[ik,:,ib,iz], V1N_sol_fin[ik,:,ib,iz] = V1Nk_imp, V1Nm_imp, V1Nb_imp, V1N_imp
            end
        end
    end

    # Return the policy and and value functions
    return mtilN_sol_fin, dN_sol_fin, V1Nk_sol_fin, V1Nm_sol_fin, V1Nb_sol_fin, V1N_sol_fin
end

# Solve problem conditional on A
function solve_VA(V0ek_tp1, V0em_tp1, V0eb_tp1, V0e_tp1, MU_t, w_t, M_tp1, pQ_t, Pi_t, rb_t, rm_t, Rc_tm1, Rc_t, gammatil_t, modl, sspace, opts)

    # 1bi   - Third stage of A
    kpA_cur, bpA_cur, V3Ak_cur, V3Am_cur, V3Abtil_cur, V3A_cur = solve_V3A(V0ek_tp1, V0em_tp1, V0eb_tp1, V0e_tp1, MU_t, w_t, M_tp1, pQ_t, Pi_t, rb_t, rm_t, Rc_tm1, Rc_t, gammatil_t, modl, sspace, opts);

    # 1bii  - Second stage of A
    mpA_cur, btilA_cur, V2Ak_cur, V2Aatil_cur, V2A_cur = solve_V2A(V3Ak_cur, V3Am_cur, V3Abtil_cur, V3A_cur, MU_t, w_t, M_tp1, pQ_t, Pi_t, rb_t, rm_t, Rc_tm1, Rc_t, gammatil_t, modl, sspace, opts);

    # 1biii - First stage of A
    atilA_cur, dA_cur, V1Ak_cur, V1Aa_cur, V1A_cur = solve_V1A(V2Ak_cur, V2Aatil_cur, V2A_cur, MU_t, w_t, M_tp1, pQ_t, Pi_t, rb_t, rm_t, Rc_tm1, Rc_t, gammatil_t, modl, sspace, opts);

    # 2aiv -- Use the above policies to create the overall policies conditional on A, mapping everything into the "fundamental (k,m,b,z)-space"

    # First, map the "third stage" policies for k' and  b' from the (k,m',btil,z)-grid to the (k,atil,z)-grid
    kpA3_itp = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.btil_grid, sspace.z_grid), kpA_cur, Gridded(Linear())), Line())
    bpA3_itp = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.btil_grid, sspace.z_grid), bpA_cur, Gridded(Linear())), Line())
    # NOTE: Because the "second stage" m' policies can be extreme, can have NaN returned, but unless any prior stages imply stepping into that state, this does not affect anything nor generate NaN in the full policy of interest
    kpA2_imp, bpA2_imp = Array{Float64}(undef, sspace.Nk, sspace.Natil, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Natil, sspace.Nz)
    for iz=1:sspace.Nz
        for ik=1:sspace.Nk
            for iatil=1:sspace.Natil
                kpA2_imp[ik,iatil,iz], bpA2_imp[ik,iatil,iz] = kpA3_itp(sspace.k_grid[ik], mpA_cur[ik,iatil,iz], btilA_cur[ik,iatil,iz], sspace.z_grid[iz]), bpA3_itp(sspace.k_grid[ik], mpA_cur[ik,iatil,iz], btilA_cur[ik,iatil,iz], sspace.z_grid[iz])
            end
        end
    end
    # Second, create the interpolants of the "second stage" policies for k', m', and b' to map them from the (k,atil,z)-grid to the (k,a,z)-grid
    kpA2_itp = extrapolate(interpolate((sspace.k_grid, sspace.atil_grid, sspace.z_grid), kpA2_imp, Gridded(Linear())), NaN)
    bpA2_itp = extrapolate(interpolate((sspace.k_grid, sspace.atil_grid, sspace.z_grid), bpA2_imp, Gridded(Linear())), NaN)
    mpA2_itp = extrapolate(interpolate((sspace.k_grid, sspace.atil_grid, sspace.z_grid), mpA_cur, Gridded(Linear())), NaN)
    kpA1_imp, mpA1_imp, bpA1_imp = Array{Float64}(undef, sspace.Nk, sspace.Na, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Na, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Na, sspace.Nz)
    for iz=1:sspace.Nz
        for ik=1:sspace.Nk
            for ia=1:sspace.Na
                kpA1_imp[ik,ia,iz], mpA1_imp[ik,ia,iz], bpA1_imp[ik,ia,iz] = kpA2_itp(sspace.k_grid[ik], atilA_cur[ik,ia,iz], sspace.z_grid[iz]), mpA2_itp(sspace.k_grid[ik], atilA_cur[ik,ia,iz], sspace.z_grid[iz]), bpA2_itp(sspace.k_grid[ik], atilA_cur[ik,ia,iz], sspace.z_grid[iz])
            end
        end
    end

    # Create all interpolants on (k,a); including for the implied value functions
    kpA1_itp = extrapolate(interpolate((sspace.k_grid, sspace.a_grid, sspace.z_grid), kpA1_imp, Gridded(Linear())), Flat())
    bpA1_itp = extrapolate(interpolate((sspace.k_grid, sspace.a_grid, sspace.z_grid), bpA1_imp, Gridded(Linear())), Flat())
    mpA1_itp = extrapolate(interpolate((sspace.k_grid, sspace.a_grid, sspace.z_grid), mpA1_imp, Gridded(Linear())), Flat())
    dA1_itp  = extrapolate(interpolate((sspace.k_grid, sspace.a_grid, sspace.z_grid), dA_cur, Gridded(Linear())), Flat())
    V1Ak_itp = extrapolate(interpolate((sspace.k_grid, sspace.a_grid, sspace.z_grid), V1Ak_cur, Gridded(Linear())), Flat())
    V1Aa_itp = extrapolate(interpolate((sspace.k_grid, sspace.a_grid, sspace.z_grid), V1Aa_cur, Gridded(Linear())), Flat())
    V1A_itp  = extrapolate(interpolate((sspace.k_grid, sspace.a_grid, sspace.z_grid), V1A_cur, Gridded(Linear())), Flat())

    # Finally, create all the interpolants on (k,m,b)
    kpA1f_imp, mpA1f_imp, bpA1f_imp, dA1f_imp = Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz)
    V1Akf_imp, V1Amf_imp, V1Abf_imp, V1Af_imp = Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nb, sspace.Nz)
    for iz=1:sspace.Nz
        for ik=1:sspace.Nk
            for imm=1:sspace.Nm
                for ib=1:sspace.Nb
                    a_imp_val_temp = ((1.0+(1.0-modl.tau)*rm_t)/Pi_t)*sspace.m_grid[imm]/sspace.k_grid[ik] + ((1.0+(1.0-modl.tau)*rb_t)/Pi_t)*sspace.b_grid[ib]/Rc_tm1 + (1.0-modl.tau)*ymenufun("yp", sspace.k_grid[ik], sspace.m_grid[imm]/Pi_t, sspace.z_grid[iz], MU_t, w_t, modl)/sspace.k_grid[ik]
                    kpA1f_imp[ik,imm,ib,iz] = kpA1_itp(sspace.k_grid[ik], a_imp_val_temp, sspace.z_grid[iz])
                    mpA1f_imp[ik,imm,ib,iz] = mpA1_itp(sspace.k_grid[ik], a_imp_val_temp, sspace.z_grid[iz])
                    bpA1f_imp[ik,imm,ib,iz] = bpA1_itp(sspace.k_grid[ik], a_imp_val_temp, sspace.z_grid[iz])
                    dA1f_imp[ik,imm,ib,iz]  = dA1_itp(sspace.k_grid[ik], a_imp_val_temp, sspace.z_grid[iz])
                    V1Akf_imp[ik,imm,ib,iz] = V1Ak_itp(sspace.k_grid[ik], a_imp_val_temp, sspace.z_grid[iz]) + V1Aa_itp(sspace.k_grid[ik], a_imp_val_temp, sspace.z_grid[iz]) * ( -((1.0+(1.0-modl.tau)*rm_t)/Pi_t)*sspace.m_grid[imm]/(sspace.k_grid[ik]^2) + (1.0-modl.tau)*ymenufun("ypk", sspace.k_grid[ik], sspace.m_grid[imm]/Pi_t, sspace.z_grid[iz], MU_t, w_t, modl)/sspace.k_grid[ik] - (1.0-modl.tau)*ymenufun("yp", sspace.k_grid[ik], sspace.m_grid[imm]/Pi_t, sspace.z_grid[iz], MU_t, w_t, modl)/(sspace.k_grid[ik]^2) )
                    V1Amf_imp[ik,imm,ib,iz] = V1Aa_itp(sspace.k_grid[ik], a_imp_val_temp, sspace.z_grid[iz]) * (((1.0+(1.0-modl.tau)*rm_t)/Pi_t)/sspace.k_grid[ik] + (1.0-modl.tau)*ymenufun("ypm", sspace.k_grid[ik], sspace.m_grid[imm]/Pi_t, sspace.z_grid[iz], MU_t, w_t, modl)/(Pi_t*sspace.k_grid[ik]) )
                    V1Abf_imp[ik,imm,ib,iz] = V1Aa_itp(sspace.k_grid[ik], a_imp_val_temp, sspace.z_grid[iz]) * ((1.0+(1.0-modl.tau)*rb_t)/Pi_t)/Rc_tm1
                    V1Af_imp[ik,imm,ib,iz]  = V1A_itp(sspace.k_grid[ik], a_imp_val_temp, sspace.z_grid[iz])
                end
            end
        end
    end

    # Return the "full policies" and value functions V1 on the (k,m,b,z)-space
    return kpA1f_imp, mpA1f_imp, bpA1f_imp, dA1f_imp, V1Akf_imp, V1Amf_imp, V1Abf_imp, V1Af_imp
end

# Solve third stage of A
function solve_V3A(V0ek_tp1, V0em_tp1, V0eb_tp1, V0e_tp1, MU_t, w_t, M_tp1, pQ_t, Pi_t, rb_t, rm_t, Rc_tm1, Rc_t, gammatil_t, modl, sspace, opts)

    txtout_path = joinpath(opts.project_root, "interim_output", "model_interim_output", "savedout", modl.name * "_output.txt")

    #### A - 3rd stage
    # For each m', solve the 2-dim EGM problem over (k,btil)

    # Create collectors of overall, final policies for this stage
    kpA_sol_fin, bpA_sol_fin = Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nbtil, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nbtil, sspace.Nz);
    V3Ak_sol_fin, V3Am_sol_fin, V3Abtil_sol_fin, V3A_sol_fin = Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nbtil, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nbtil, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nbtil, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Nm, sspace.Nbtil, sspace.Nz);

    for iz = 1:sspace.Nz
        # For each m'
        for imp = 1:sspace.Nm
            # A3-Step 1: For each kp, "solve out"  the b-dimension
            # Initialize relevant (intermediate) Step 1 "collectors" over the (k,b̃)-space, as functions of (k',b̃)
            # NOTE: Don't include (m',z)-dimension because these will be used only "within" this (m',z) iteration
            bpA_sol_btil_temp, lambda3A_sol_btil_temp = Array{Float64}(undef, sspace.Nk, sspace.Nbtil), Array{Float64}(undef, sspace.Nk, sspace.Nbtil);
            # Also, need to "map back" ("interpolate onto today's state") the continuation value fn derivative wrt m, and the value function value
            V0em_sol_btil_temp, V0e_sol_btil_temp = Array{Float64}(undef, sspace.Nk, sspace.Nbtil), Array{Float64}(undef, sspace.Nk, sspace.Nbtil);

            # For each k'
            for ikp = 1:sspace.Nk
                # A3-Step 1a: unconstrained b'
                # Solve FOC for k vector (over b')
                kA_uc_val = sspace.k_grid[ikp] ./ (1.0 .+ (1.0/modl.kappa)*( ( M_tp1*V0ek_tp1[ikp,imp,:,iz] ./ (M_tp1*V0eb_tp1[ikp,imp,:,iz]/sspace.k_grid[ikp]) - sspace.b_grid)/(Rc_t*pQ_t) .- 1.0))
                # And the BC for implied btil vector (over b')
                btilA_uc_val = ( -(1.0-(1.0-modl.tau)*modl.delta)*pQ_t*kA_uc_val .+ (pQ_t .+ sspace.b_grid/Rc_t)*sspace.k_grid[ikp] + pQ_t*(modl.kappa/2.0)*kA_uc_val.*(sspace.k_grid[ikp]./kA_uc_val .- 1.0).^2 ) ./ kA_uc_val

                # Create interpolants for bpA_uc and lambda3A_uc
                # If endogenous grid not ordered, deal with the implied non-convexity and find the global optimum in the spirit of Fella (2014)
                if !issorted(btilA_uc_val)
                    # Find the "flipping points"
                    btilA_uc_diff = btilA_uc_val[2:end] - btilA_uc_val[1:(end-1)]
                    i_flip_set = findall(i->(i<0.0), btilA_uc_diff)
                    btilA_uc_fmax = maximum(btilA_uc_val[i_flip_set]) # The "upper bound" on the non-concavity region, in btil-space
                    btilA_uc_fmin = minimum(btilA_uc_val[i_flip_set.+1]) # The "lower bound" on the non-concavity region, in btil-space

                    # Find the "boundaries" of the non-concavity region, in the index-space
                    # In case the "lowest flipping point" is itself the LOWEST POINT of btilA_uc_val, then i_LB=0
                    if any(btilA_uc_val .< btilA_uc_fmin)
                        i_LB = maximum(findall(i->(i<btilA_uc_fmin), btilA_uc_val))
                    else
                        i_LB = 0
                    end
                    # In case the "highest flipping point" is itself the HIGHEST POINT of btilA_uc_val, then i_UB=Nb+1
                    if any(btilA_uc_val .> btilA_uc_fmax)
                        i_UB = minimum(findall(i->(i>btilA_uc_fmax), btilA_uc_val))
                    else
                        i_UB = length(btilA_uc_val) + 1
                    end

                    # Create collector of indices which to not drop
                    ind_keep_col = ones(Bool,length(btilA_uc_val))
                    # Now, consider each index inbetween i_LB and i_UB individually, in order to establish whether the found point is actually a global optimum
                    for ii=(i_LB+1):(i_UB-1)
                        # Implied incoming states (k,btil) at ii
                        btilA_cur_val = btilA_uc_val[ii]
                        kA_cur_val   = sspace.k_grid[ikp] ./ (1.0 .+ (1.0/modl.kappa)*( ( M_tp1*V0ek_tp1[ikp,imp,ii,iz] ./ (M_tp1*V0eb_tp1[ikp,imp,ii,iz]/sspace.k_grid[ikp]) - sspace.b_grid[ii])/(Rc_t*pQ_t) .- 1.0))
                        # Create collector for candidate values
                        V_vals = -999.9*ones(size(sspace.b_grid))
                        # Evaluate the value function at the indices in the non-concave region
                        for ic=(i_LB+1):(i_UB-1)
                            # Implied choices (kp, bp) at candidate ic
                            bp_cur_val = sspace.b_grid[ic]
                            # Solve the quadratic budget constraint for kp
                            a_kpq_temp = pQ_t*(modl.kappa/2.0)/kA_cur_val
                            b_kpq_temp = (1.0-modl.kappa)*pQ_t + bp_cur_val/Rc_t
                            c_kpq_temp = (pQ_t*modl.kappa/2.0 - (1.0-(1.0-modl.tau)*modl.delta)*pQ_t - btilA_cur_val)*kA_cur_val
                            # Consider candidate bp only if it implies a solution for kp, otherwise leave value to -999
                            if (b_kpq_temp.^2 - 4*a_kpq_temp*c_kpq_temp) >= 0.0
                                kpA_val = (- b_kpq_temp + sqrt.(b_kpq_temp.^2 - 4*a_kpq_temp*c_kpq_temp))/(2.0*a_kpq_temp)

                                # Now, evaluate the value function at the implied (kp, bp)
                                V0e_itp_temp      = extrapolate(interpolate((sspace.k_grid, sspace.b_grid), V0e_tp1[:,imp,:,iz], Gridded(Linear())), NaN)
                                # Evaluate the continuation value function
                                V_vals[ic] = V0e_itp_temp(kpA_val, bp_cur_val)
                            end
                        end
                        # Find the optimum index
                        ic_max = findmax(V_vals)[2]
                        # If this is not equal to ii, add ii to be dropped
                        if ic_max != ii
                            ind_keep_col[ii]=0
                        end
                    end

                    # Having established which elements of the b'- and b-vectors to keep, construct them here to use for interpolation
                    # In order to make indexing of all the relevant continuation vectors easier, first sort btilA_uc_val, and then drop the indices that are meant to be dropped
                    i_btilA_sort      = sortperm(btilA_uc_val)
                    ind_keep_col_sort = ind_keep_col[i_btilA_sort]
                    i_btilA_NC_val    = i_btilA_sort[ind_keep_col_sort] # Use this "indexer" to now construct the correctly ordered "btilA_val" vector

                    # Create interpolants
                    bpA_uc_itp      = extrapolate(interpolate((btilA_uc_val[i_btilA_NC_val], ), sspace.b_grid[i_btilA_NC_val], Gridded(Linear())), NaN)
                    lambda3A_uc_itp = extrapolate(interpolate((btilA_uc_val[i_btilA_NC_val], ), Rc_t*M_tp1*V0eb_tp1[ikp,imp,i_btilA_NC_val,iz]/sspace.k_grid[ikp], Gridded(Linear())), NaN)
                    # And the continuation value function
                    V0em_uc_itp     = extrapolate(interpolate((btilA_uc_val[i_btilA_NC_val], ), V0em_tp1[ikp,imp,i_btilA_NC_val,iz], Gridded(Linear())), NaN)
                    V0e_uc_itp      = extrapolate(interpolate((btilA_uc_val[i_btilA_NC_val], ), V0e_tp1[ikp,imp,i_btilA_NC_val,iz], Gridded(Linear())), NaN)
                else
                    # Create interpolants
                    bpA_uc_itp      = extrapolate(interpolate((btilA_uc_val, ), sspace.b_grid, Gridded(Linear())), NaN)
                    lambda3A_uc_itp = extrapolate(interpolate((btilA_uc_val, ), Rc_t*M_tp1*V0eb_tp1[ikp,imp,:,iz]/sspace.k_grid[ikp], Gridded(Linear())), NaN)
                    # And the continuation value function
                    V0em_uc_itp     = extrapolate(interpolate((btilA_uc_val, ), V0em_tp1[ikp,imp,:,iz], Gridded(Linear())), NaN)
                    V0e_uc_itp      = extrapolate(interpolate((btilA_uc_val, ), V0e_tp1[ikp,imp,:,iz], Gridded(Linear())), NaN)
                end

                # And evaluate these on the btil_grid
                bpA_uc_imp, lambda3A_uc_imp = bpA_uc_itp.(sspace.btil_grid), lambda3A_uc_itp.(sspace.btil_grid)
                V0em_uc_imp, V0e_uc_imp     = V0em_uc_itp.(sspace.btil_grid), V0e_uc_itp.(sspace.btil_grid)

                # A3-Step 1b: borrowing-constrained b'
                # Solve FOC for k vector (over chiB)
                kA_Bc_val = sspace.k_grid[ikp] ./ (1.0 .+ (1.0/modl.kappa)*( ( M_tp1*V0ek_tp1[ikp,imp,1,iz] ./ ((M_tp1*V0eb_tp1[ikp,imp,1,iz] .+ sspace.chiB_grid.*sspace.k_grid[ikp])/sspace.k_grid[ikp]) .- sspace.b_grid[1])/(Rc_t*pQ_t) .- 1.0))
                # And the BC for implied btil vector (over chiB)
                btilA_Bc_val = ( -(1.0-(1.0-modl.tau)*modl.delta)*pQ_t*kA_Bc_val .+ (pQ_t .+ sspace.b_grid[1]/Rc_t)*sspace.k_grid[ikp] + pQ_t*(modl.kappa/2.0)*kA_Bc_val.*(sspace.k_grid[ikp]./kA_Bc_val .- 1.0).^2 ) ./ kA_Bc_val

                # Create interpolants for bpA_Bc and lambda3A_Bc
                # ERROR if endogenous grid not ordered
                if !issorted(btilA_Bc_val); manprintln("ERROR: in sV3A, btilA_Bc_val is not ordered (ikp=$(ikp),imp=$(imp),iz=$(iz)).", txtout_path); end

                bpA_Bc_itp      = extrapolate(interpolate((btilA_Bc_val, ), repeat([sspace.b_grid[1]], sspace.NchiB), Gridded(Linear())), NaN)
                lambda3A_Bc_itp = extrapolate(interpolate((btilA_Bc_val, ), Rc_t*(M_tp1*V0eb_tp1[ikp,imp,1,iz] .+ sspace.chiB_grid.*sspace.k_grid[ikp])/sspace.k_grid[ikp], Gridded(Linear())), NaN)
                # And the continuation value function
                V0em_Bc_itp     = extrapolate(interpolate((btilA_Bc_val, ), repeat([V0em_tp1[ikp,imp,1,iz]], sspace.NchiB), Gridded(Linear())), NaN)
                V0e_Bc_itp      = extrapolate(interpolate((btilA_Bc_val, ), repeat([V0e_tp1[ikp,imp,1,iz]], sspace.NchiB), Gridded(Linear())), NaN)
                # And evaluate these on the btil_grid
                bpA_Bc_imp, lambda3A_Bc_imp = bpA_Bc_itp.(sspace.btil_grid), lambda3A_Bc_itp.(sspace.btil_grid)
                V0em_Bc_imp, V0e_Bc_imp     = V0em_Bc_itp.(sspace.btil_grid), V0e_Bc_itp.(sspace.btil_grid)

                # A3-Step 1c: saving-constrained b'
                # Check manually if chiS upper bound high enough
                kA_Sc_chiSmax = sspace.k_grid[ikp] ./ (1.0 .+ (1.0/modl.kappa)*( ( M_tp1*V0ek_tp1[ikp,imp,sspace.Nb,iz] ./ ((M_tp1*V0eb_tp1[ikp,imp,sspace.Nb,iz] .- sspace.chiS_grid[end].*sspace.k_grid[ikp])/sspace.k_grid[ikp]) .- sspace.b_grid[sspace.Nb])/(Rc_t*pQ_t) .- 1.0))
                btilA_Sc_chiSmax = ( -(1.0-(1.0-modl.tau)*modl.delta)*pQ_t*kA_Sc_chiSmax .+ (pQ_t .+ sspace.b_grid[sspace.Nb]/Rc_t)*sspace.k_grid[ikp] + pQ_t*(modl.kappa/2.0)*kA_Sc_chiSmax.*(sspace.k_grid[ikp]./kA_Sc_chiSmax .- 1.0).^2 ) ./ kA_Sc_chiSmax

                if btilA_Sc_chiSmax >= sspace.btil_grid[end]
                    # If high enough, settle on the exogenously set grid
                    chiS_grid_used = copy(sspace.chiS_grid)
                else
                    # Keep increasing chiS_max until it's high enough
                    chiS_max_prime = copy(sspace.chiS_grid[end])
                    while (btilA_Sc_chiSmax < sspace.btil_grid[end]) & (kA_Sc_chiSmax > 0.1) # Make sure that the implied capital does not collapse very small
                        chiS_max_prime = chiS_max_prime + 0.1
                        kA_Sc_chiSmax = sspace.k_grid[ikp] ./ (1.0 .+ (1.0/modl.kappa)*( ( M_tp1*V0ek_tp1[ikp,imp,sspace.Nb,iz] ./ ((M_tp1*V0eb_tp1[ikp,imp,sspace.Nb,iz] .- chiS_max_prime.*sspace.k_grid[ikp])/sspace.k_grid[ikp]) .- sspace.b_grid[sspace.Nb])/(Rc_t*pQ_t) .- 1.0))
                        btilA_Sc_chiSmax = ( -(1.0-(1.0-modl.tau)*modl.delta)*pQ_t*kA_Sc_chiSmax .+ (pQ_t .+ sspace.b_grid[sspace.Nb]/Rc_t)*sspace.k_grid[ikp] + pQ_t*(modl.kappa/2.0)*kA_Sc_chiSmax.*(sspace.k_grid[ikp]./kA_Sc_chiSmax .- 1.0).^2 ) ./ kA_Sc_chiSmax
                    end
                    # Construct a new, temporary grid over chiS to correspond to this max value
                    chiS_grid_used   = range(sspace.chiS_min^sspace.curvlm[3], chiS_max_prime^sspace.curvlm[3], length=sspace.NchiS).^(1.0/sspace.curvlm[3])
                end

                # Solve FOC for k vector (over chiS)
                kA_Sc_val = sspace.k_grid[ikp] ./ (1.0 .+ (1.0/modl.kappa)*( ( M_tp1*V0ek_tp1[ikp,imp,sspace.Nb,iz] ./ ((M_tp1*V0eb_tp1[ikp,imp,sspace.Nb,iz] .- chiS_grid_used.*sspace.k_grid[ikp])/sspace.k_grid[ikp]) .- sspace.b_grid[sspace.Nb])/(Rc_t*pQ_t) .- 1.0))
                # And the BC for implied btil vector (over chiS)
                btilA_Sc_val = ( -(1.0-(1.0-modl.tau)*modl.delta)*pQ_t*kA_Sc_val .+ (pQ_t .+ sspace.b_grid[sspace.Nb]/Rc_t)*sspace.k_grid[ikp] + pQ_t*(modl.kappa/2.0)*kA_Sc_val.*(sspace.k_grid[ikp]./kA_Sc_val .- 1.0).^2 ) ./ kA_Sc_val

                # Create interpolants for bpA_Sc and lambda3A_Sc
                # ERROR if endogenous grid not ordered
                if !issorted(btilA_Sc_val); manprintln("ERROR: in sV3A, btilA_Sc_val is not ordered (ikp=$(ikp),imp=$(imp),iz=$(iz)).", txtout_path); end

                bpA_Sc_itp      = extrapolate(interpolate((btilA_Sc_val, ), repeat([sspace.b_grid[sspace.Nb]], sspace.NchiS), Gridded(Linear())), NaN)
                lambda3A_Sc_itp = extrapolate(interpolate((btilA_Sc_val, ), Rc_t*(M_tp1*V0eb_tp1[ikp,imp,sspace.Nb,iz] .- chiS_grid_used.*sspace.k_grid[ikp])/sspace.k_grid[ikp], Gridded(Linear())), NaN)
                # And the continuation value function
                V0em_Sc_itp     = extrapolate(interpolate((btilA_Sc_val, ), repeat([V0em_tp1[ikp,imp,sspace.Nb,iz]], sspace.NchiS), Gridded(Linear())), NaN)
                V0e_Sc_itp      = extrapolate(interpolate((btilA_Sc_val, ), repeat([V0e_tp1[ikp,imp,sspace.Nb,iz]], sspace.NchiS), Gridded(Linear())), NaN)
                # And evaluate these on the btil_grid
                bpA_Sc_imp, lambda3A_Sc_imp = bpA_Sc_itp.(sspace.btil_grid), lambda3A_Sc_itp.(sspace.btil_grid)
                V0em_Sc_imp, V0e_Sc_imp     = V0em_Sc_itp.(sspace.btil_grid), V0e_Sc_itp.(sspace.btil_grid)


                # A3-Step 1d: combine constrained and unconstrained bp and lambda solutions
                bpA_uc_imp, bpA_Bc_imp, bpA_Sc_imp = replace(bpA_uc_imp, NaN=> - 999), replace(bpA_Bc_imp, NaN=> - 999), replace(bpA_Sc_imp, NaN=> - 999)
                lambda3A_uc_imp, lambda3A_Bc_imp, lambda3A_Sc_imp = replace(lambda3A_uc_imp, NaN=> - 999), replace(lambda3A_Bc_imp, NaN=> - 999), replace(lambda3A_Sc_imp, NaN=> - 999)
                V0em_uc_imp, V0em_Bc_imp, V0em_Sc_imp = replace(V0em_uc_imp, NaN=> - 999), replace(V0em_Bc_imp, NaN=> - 999), replace(V0em_Sc_imp, NaN=> - 999)
                V0e_uc_imp, V0e_Bc_imp, V0e_Sc_imp = replace(V0e_uc_imp, NaN=> - 999), replace(V0e_Bc_imp, NaN=> - 999), replace(V0e_Sc_imp, NaN=> - 999)
                bpA_imp  = max.(bpA_uc_imp, bpA_Bc_imp, bpA_Sc_imp)
                lambda3A_imp  = max.(lambda3A_uc_imp, lambda3A_Bc_imp, lambda3A_Sc_imp)
                V0em_imp  = max.(V0em_uc_imp, V0em_Bc_imp, V0em_Sc_imp)
                V0e_imp   = max.(V0e_uc_imp, V0e_Bc_imp, V0e_Sc_imp)

                # Save these into the "intermediate" collectors
                bpA_sol_btil_temp[ikp,:], lambda3A_sol_btil_temp[ikp,:] = bpA_imp, lambda3A_imp
                V0em_sol_btil_temp[ikp,:], V0e_sol_btil_temp[ikp,:] = V0em_imp, V0e_imp
            end


            # A3-Step 2: For each btil, "solve out" the k-dimension
            # For each btil
            for ibtil = 1:sspace.Nbtil
                # Solve budget constraint for k vector (over k'), quadratic solution:
                # For simplicity, separately write out separately the coefs of the quadratic eqn for k
                a_kq_temp = (1.0-(1.0-modl.tau)*modl.delta)*pQ_t + sspace.btil_grid[ibtil] - pQ_t*modl.kappa/2.0
                b_kq_temp = sspace.k_grid .* (-bpA_sol_btil_temp[:,ibtil]/Rc_t .- pQ_t*(1.0-modl.kappa))
                c_kq_temp = -pQ_t*(modl.kappa/2.0)*(sspace.k_grid.^2)
                kA_val = (- b_kq_temp + sqrt.(b_kq_temp.^2 - 4*a_kq_temp*c_kq_temp))/(2.0*a_kq_temp)

                # Create interpolants for kp, and also bp and lambda3A and the continuation V0em and V0e on this endogenous grid for k
                # If endogenous grid not ordered, deal with the implied non-convexity and find the global optimum in the spirit of Fella (2014)
                if !issorted(kA_val)

                    # Find the "flipping points"
                    kA_diff = kA_val[2:end] - kA_val[1:(end-1)]
                    i_flip_set = findall(i->(i<0.0), kA_diff)
                    kA_fmax = maximum(kA_val[i_flip_set]) # The "upper bound" on the non-concavity region, in k-space
                    kA_fmin = minimum(kA_val[i_flip_set.+1]) # The "lower bound" on the non-concavity region, in k-space
                    # Find the "boundaries" of the non-concavity region, in the index-space
                    # In case the "lowest flipping point" is itself the LOWEST POINT of kA_val, then i_LB=0
                    if any(kA_val .< kA_fmin)
                        i_LB = maximum(findall(i->(i<kA_fmin), kA_val))
                    else
                        i_LB = 0
                    end
                    # In case the "highest flipping point" is itself the HIGHEST POINT of kA_val, then i_UB=Nb+1
                    if any(kA_val .> kA_fmax)
                        i_UB = minimum(findall(i->(i>kA_fmax), kA_val))
                    else
                        i_UB = length(kA_val) + 1
                    end

                    # Create collector of indices which to not drop
                    ind_keep_col = ones(Bool,length(kA_val))
                    # Now, consider each index inbetween i_LB and i_UB individually, in order to establish whether the found point is actually a global optimum
                    for ii=(i_LB+1):(i_UB-1)
                        # Implied incoming states (k,mtil) at ii
                        kA_cur_val = kA_val[ii]
                        btil_cur_val   = sspace.btil_grid[ibtil]
                        # Create collector for candidate values
                        V_vals = -999.9*ones(size(sspace.k_grid))
                        # Evaluate the value function at the indices in the non-concave region
                        for ic=(i_LB+1):(i_UB-1)
                            # Implied choices (kp, mp) at candidate ic
                            kp_cur_val = sspace.k_grid[ic]
                            # Compute the implied bp from the budget constraint
                            bp_cur_val = Rc_t*( (1.0-(1.0-modl.tau)*modl.delta)*pQ_t*kA_cur_val - pQ_t*kp_cur_val + btil_cur_val*kA_cur_val - 0.5*modl.kappa*kA_cur_val*((kp_cur_val/kA_cur_val-1.0)^2) )/kp_cur_val - 1e-8

                            # If the implied bp_cur_val is above or below the upper and lower bound on b, then cannot update V_vals, leaving it at the -999
                            if (bp_cur_val <= sspace.b_grid[end]) & (bp_cur_val >= sspace.b_grid[1])
                                # Evaluate the value function at the implied (kp, bp)
                                V0e_itp_temp      = extrapolate(interpolate((sspace.k_grid, sspace.b_grid), V0e_tp1[:,imp,:,iz], Gridded(Linear())), NaN)
                                # Evaluate the continuation value function
                                V_vals[ic] = V0e_itp_temp(kp_cur_val, bp_cur_val)
                            end
                        end
                        # Find the optimum index
                        ic_max = findmax(V_vals)[2]
                        # If this is not equal to ii, add ii to be dropped
                        if ic_max != ii
                            ind_keep_col[ii]=0
                        end
                    end
                    # Having established which elements of the k'- and k-vectors to keep, construct them here to use for interpolation
                    # In order to make indexing of all the relevant continuation vectors easier, first sort kA_val, and then drop the indices that are meant to be dropped
                    i_kA_sort      = sortperm(kA_val)
                    ind_keep_col_sort = ind_keep_col[i_kA_sort]
                    i_kA_NC_val    = i_kA_sort[ind_keep_col_sort] # Use this "indexer" to now construct the correctly ordered "btilA_val" vector

                    # Create interpolants -- f_052026
                    kpA_itp = extrapolate(interpolate((kA_val[i_kA_NC_val], ), sspace.k_grid[i_kA_NC_val], Gridded(Linear())), Line())
                    bpA_itp = extrapolate(interpolate((kA_val[i_kA_NC_val], ), bpA_sol_btil_temp[i_kA_NC_val,ibtil], Gridded(Linear())), Line())
                    lambda3A_itp = extrapolate(interpolate((kA_val[i_kA_NC_val], ), lambda3A_sol_btil_temp[i_kA_NC_val,ibtil], Gridded(Linear())), Line())
                    V0em3A_itp = extrapolate(interpolate((kA_val[i_kA_NC_val], ), V0em_sol_btil_temp[i_kA_NC_val,ibtil], Gridded(Linear())), Line())
                    V0e3A_itp  = extrapolate(interpolate((kA_val[i_kA_NC_val], ), V0e_sol_btil_temp[i_kA_NC_val,ibtil], Gridded(Linear())), Line())
                else
                    # Create interpolants
                    kpA_itp = extrapolate(interpolate((kA_val, ), sspace.k_grid, Gridded(Linear())), Line())
                    bpA_itp = extrapolate(interpolate((kA_val, ), bpA_sol_btil_temp[:,ibtil], Gridded(Linear())), Line())
                    lambda3A_itp = extrapolate(interpolate((kA_val, ), lambda3A_sol_btil_temp[:,ibtil], Gridded(Linear())), Line())
                    V0em3A_itp = extrapolate(interpolate((kA_val, ), V0em_sol_btil_temp[:,ibtil], Gridded(Linear())), Line())
                    V0e3A_itp  = extrapolate(interpolate((kA_val, ), V0e_sol_btil_temp[:,ibtil], Gridded(Linear())), Line())
                end

                # Finally, evaluate these on the k-grid
                kpA_imp, bpA_imp, lambda3A_imp   = kpA_itp.(sspace.k_grid), bpA_itp.(sspace.k_grid), lambda3A_itp.(sspace.k_grid)
                V0em3A_imp, V0e3A_imp       = V0em3A_itp.(sspace.k_grid), V0e3A_itp.(sspace.k_grid)

                # Save these into the "final" collectors
                kpA_sol_fin[:,imp,ibtil,iz], bpA_sol_fin[:,imp,ibtil,iz] = kpA_imp, bpA_imp
                # And for the value functions as well, using envelope conditions
                V3Ak_sol_fin[:,imp,ibtil,iz] = lambda3A_imp .* ((1.0.-(1.0-modl.tau)*modl.delta)*pQ_t .+ sspace.btil_grid[ibtil] .+ (modl.kappa/2.0)*( (kpA_imp ./ sspace.k_grid).^2  .- 1.0) )
                V3Am_sol_fin[:,imp,ibtil,iz]   = M_tp1*V0em3A_imp
                V3Abtil_sol_fin[:,imp,ibtil,iz]= lambda3A_imp .* sspace.k_grid
                V3A_sol_fin[:,imp,ibtil,iz]    = M_tp1*V0e3A_imp
            end
        end
    end
    
    # Return the policy and and value functions
    return kpA_sol_fin, bpA_sol_fin, V3Ak_sol_fin, V3Am_sol_fin, V3Abtil_sol_fin, V3A_sol_fin
end

# Solve second stage of A
function solve_V2A(V3Ak_in, V3Am_in, V3Abtil_in, V3A_in, MU_t, w_t, M_tp1, pQ_t, Pi_t, rb_t, rm_t, Rc_tm1, Rc_t, gammatil_t, modl, sspace, opts)

    txtout_path = joinpath(opts.project_root, "interim_output", "model_interim_output", "savedout", modl.name * "_output.txt")

    #### A - 2nd stage

    # For each (k,atil), solve the "middle-of-period" problem, choosing (btil,m') "by hand"
    # Create collectors of overall, final policies for this stage
    mpA_sol_fin, btilA_sol_fin = Array{Float64}(undef, sspace.Nk, sspace.Natil, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Natil, sspace.Nz)
    V2Ak_sol_fin, V2Aatil_sol_fin, V2A_sol_fin = Array{Float64}(undef, sspace.Nk, sspace.Natil, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Natil, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Natil, sspace.Nz)

    # For each (k,atil,iz)
    for iz=1:sspace.Nz
        for ik=1:sspace.Nk
            for iatil=1:sspace.Natil
                # Create interpolants for the continuation value function derivatives V3Am, V3Abtil/k, and V3A, conditional on k (and z); and for completeness, also V3Ak
                V3Ak_itp_k_temp     = extrapolate(interpolate((sspace.m_grid, sspace.btil_grid), V3Ak_in[ik,:,:,iz], Gridded(Linear())), Line())
                V3Am_itp_k_temp     = extrapolate(interpolate((sspace.m_grid, sspace.btil_grid), V3Am_in[ik,:,:,iz], Gridded(Linear())), Line())
                V3Abtil_itp_k_temp  = extrapolate(interpolate((sspace.m_grid, sspace.btil_grid), V3Abtil_in[ik,:,:,iz]/sspace.k_grid[ik], Gridded(Linear())), Line())
                V3A_itp_k_temp      = extrapolate(interpolate((sspace.m_grid, sspace.btil_grid), V3A_in[ik,:,:,iz], Gridded(Linear())), Line())

                # Compute the btil implied by mp=m_min
                btil_atilimp_mp0 = sspace.atil_grid[iatil]  - sspace.m_grid[1] / sspace.k_grid[ik]
                # Compute the value of the V3Am/V3Abtil ratio at m'=m_min
                V3AmDbtil_mp0 = V3Am_itp_k_temp(sspace.m_grid[1], btil_atilimp_mp0) / V3Abtil_itp_k_temp(sspace.m_grid[1], btil_atilimp_mp0)

                # Maximal value allowed for m' by the grids for m(max) and b_til(min)
                mp_max = (sspace.atil_grid[iatil]-sspace.btil_grid[1])*sspace.k_grid[ik]

                # Define function of ratio over m', minus 1
                function V3AmDbtil_fun(mp_val)
                    btil_imp_val = sspace.atil_grid[iatil] - mp_val / sspace.k_grid[ik]
                    return V3Am_itp_k_temp(mp_val, btil_imp_val) / V3Abtil_itp_k_temp(mp_val, btil_imp_val) - 1.0
                end
                # Evaluate to check whether it crosses zero more than once
                mgrid_check = sspace.m_grid[sspace.m_grid.<mp_max]
                append!(mgrid_check, mp_max)
                ratio_check = V3AmDbtil_fun.(mgrid_check)
                ratio_vs_0  = ratio_check .> 0.0
                ind_cross   = abs.(ratio_vs_0[2:end]-ratio_vs_0[1:(end-1)])
                n_cross0 = sum(ind_cross)

                if (n_cross0 >= 2) 
                    # Create global optimization objective
                    V3A_obj_temp(x) = V3A_itp_k_temp(x, sspace.atil_grid[iatil] - x / sspace.k_grid[ik])

                    # Apply global optimizer over "non-concave crossing region"
                    # Find the highest crossing
                    ic_last = findlast(ind_cross.>0)
                    # Interpolate for the exact value of mp at this crossing
                    mp_lastcross = mgrid_check[ic_last] + (mgrid_check[ic_last+1]-mgrid_check[ic_last])*(0.0-ratio_check[ic_last])/(ratio_check[ic_last+1]-ratio_check[ic_last])
                    # And find the lowest crossing
                    ic_first = findfirst(ind_cross.>0)
                    mp_firstcross = mgrid_check[ic_first] + (mgrid_check[ic_first+1]-mgrid_check[ic_first])*(0.0-ratio_check[ic_first])/(ratio_check[ic_first+1]-ratio_check[ic_first])
                    # Search for optimum between the mp values that lie between the m-grid values bracketing these crossings
                    if any(sspace.m_grid .< mp_firstcross)
                        im_low = findlast(sspace.m_grid .< mp_firstcross)
                    else
                        im_low = 1
                    end
                    m_low  = sspace.m_grid[im_low]
                    if any(sspace.m_grid .> mp_lastcross)
                        im_high= findfirst(sspace.m_grid .> mp_lastcross)
                    else
                        im_high=sspace.Nm
                    end
                    m_high  = sspace.m_grid[im_high]

                    # Manually search on m grid
                    mp_grid_temp = Array(range(m_low, m_high, step=1e-2))
                    # Make sure to include the endpoint in the grid
                    append!(mp_grid_temp, m_high)
                    V3A_obj_vals = V3A_obj_temp.(mp_grid_temp)
                    i_max = argmax(V3A_obj_vals) # Index of optimum

                    # Impose value in collector
                    mpA_sol_val = mp_grid_temp[i_max]
                elseif V3AmDbtil_mp0 <= 1.0
                    # If the ratio is < 1 already at m'=m_min, the solution is m'=m_min (if the effective ratio is truly downward-sloping)
                    # Else, find the point on the m space where the ratio equals 1
                    mpA_sol_val = sspace.m_grid[1]
                else

                    # If value at maximal m' is still not <= 1, then impose that the choice was at the lower bound of the btil_grid, and mp potentially outside the m_grid
                    if V3AmDbtil_fun(mp_max) > 0.0
                        mpA_sol_val = (sspace.atil_grid[iatil]-sspace.btil_grid[1])*sspace.k_grid[ik]
                    else
                        mpA_sol_val = find_zero(V3AmDbtil_fun, (sspace.m_grid[1],mp_max), Bisection())
                    end
                end

                # Compute all the relevant values implied by the m' solution
                btilA_sol_val = sspace.atil_grid[iatil] - mpA_sol_val / sspace.k_grid[ik]

                # And save in the collectors
                mpA_sol_fin[ik,iatil,iz], btilA_sol_fin[ik,iatil,iz] = mpA_sol_val, btilA_sol_val

                # Value function
                lambda2A_val = V3Abtil_itp_k_temp(mpA_sol_val, btilA_sol_val)
                V2Ak_sol_fin[ik,iatil,iz]       = V3Ak_itp_k_temp(mpA_sol_val, btilA_sol_val) + lambda2A_val * (sspace.atil_grid[iatil] - btilA_sol_val)
                V2Aatil_sol_fin[ik,iatil,iz]    = lambda2A_val * sspace.k_grid[ik]
                V2A_sol_fin[ik,iatil,iz]        = V3A_itp_k_temp(mpA_sol_val, btilA_sol_val)
            end
        end
    end

    return mpA_sol_fin, btilA_sol_fin, V2Ak_sol_fin, V2Aatil_sol_fin, V2A_sol_fin
end

# Solve first stage of A
function solve_V1A(V2Ak_in, V2Aatil_in, V2A_in, MU_t, w_t, M_tp1, pQ_t, Pi_t, rb_t, rm_t, Rc_tm1, Rc_t, gammatil_t, modl, sspace, opts)

    txtout_path = joinpath(opts.project_root, "interim_output", "model_interim_output", "savedout", modl.name * "_output.txt")

    #### A - 1st stage

    # Before iterations create collectors for final policies for this stage
    atilA_sol_fin, dA_sol_fin = Array{Float64}(undef, sspace.Nk, sspace.Na, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Na, sspace.Nz)
    V1Ak_sol_fin, V1Aa_sol_fin, V1A_sol_fin = Array{Float64}(undef, sspace.Nk, sspace.Na, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Na, sspace.Nz), Array{Float64}(undef, sspace.Nk, sspace.Na, sspace.Nz)

    # For each k (and z), solve the "beginning of period" problem over a
    for iz=1:sspace.Nz
        for ik=1:sspace.Nk
            # Using interpolation of V2Aatil, compute the values of lambda1A implied by d=0 to determine which current a states are constrained
            lambda1A_atilitp= extrapolate(interpolate((sspace.atil_grid, ), V2Aatil_in[ik,:,iz]/sspace.k_grid[ik], Gridded(Linear())), Flat())
            # NOTE: FLAT extrapolation -- REASON: to not make inference about a potentially "unconstrained state" (e.g. if atil=a maps into a point outside atil grid that might imply being unconstrained due to linear extrapolation, yet no state in atil_grid might itself be unconstrained -- so it's safer to assume ALL states in a_grid are still constrained)
            lambda1A_aimp   = lambda1A_atilitp.(sspace.a_grid)

            # Wherever this is less than 1: the first-step solution is unconstrained
            # Generate indicator for constrained states in m-space
            dc_ind          = lambda1A_aimp .> 1.0
            # Find the cutoff mtil ("mtil_crit") based on V2Aatil_in on atil_grid
            V2Aatil_in_constr_ind = V2Aatil_in[ik,:,iz]/sspace.k_grid[ik] .> 1.0

            if all(V2Aatil_in_constr_ind)
                # If all atils on grid are constrained, don't "extrapolate" unconstrainedness (as discussed above) and just impose that atil_crit corresponds to the highest point on a_grid, i.e. a_max
                atil_crit = sspace.a_grid[end]
            else
                fuc_i = findfirst(.!V2Aatil_in_constr_ind)
                if fuc_i == 1
                    # If the first atil gridpoint is already unconstrained, set it as the "critical level"
                    atil_crit = sspace.atil_grid[1]
                else
                    # Otherwise interpolate atil_crit linearly based on the two adjacent atil grid points; note that being unconstrained means "V2Aatil = k"
                    atil_crit = sspace.atil_grid[fuc_i-1] + (sspace.atil_grid[fuc_i]-sspace.atil_grid[fuc_i-1])*(sspace.k_grid[ik]-V2Aatil_in[ik,fuc_i-1,iz])/(V2Aatil_in[ik,fuc_i,iz]-V2Aatil_in[ik,fuc_i-1,iz])
                end
            end

            # Compute implied dividends and atil, given the a-grid
            d_imp    = 0.0 .+ (1.0.-dc_ind) .* (sspace.a_grid .- atil_crit) * sspace.k_grid[ik]
            atil_imp = dc_ind .* sspace.a_grid .+ (1.0.-dc_ind) .* atil_crit

            # Before continuing, check if V2Aatil_in[ik,:,iz]/k has multiple crossings of 1. If yes, must solve for global solution
            n_cross1 = sum(abs.(V2Aatil_in_constr_ind[2:end]-V2Aatil_in_constr_ind[1:(end-1)]))
            if (n_cross1 >= 2) & true

                # Find the highest crossing
                ic_last = findlast(abs.(V2Aatil_in_constr_ind[2:end]-V2Aatil_in_constr_ind[1:(end-1)]).>0)
                # Interpolate for the exact value of atil at this crossing
                atil_lastcross = sspace.atil_grid[ic_last] + (sspace.atil_grid[ic_last+1]-sspace.atil_grid[ic_last])*(sspace.k_grid[ik]-V2Aatil_in[ik,ic_last,iz])/(V2Aatil_in[ik,ic_last+1,iz]-V2Aatil_in[ik,ic_last,iz])

                # Now, do global maximization, conditional on incoming a value, but impose that for all values above the last crossing, the solution must be the same (because in that region the marginal benefit is strictly below one throughout)
                # Find the first a value above the last atil cutoff
                ia_first = findfirst(sspace.a_grid .> atil_lastcross)
                # Create interpolant of continuation V2A
                V2A_itp     = extrapolate(interpolate((sspace.atil_grid, ), V2A_in[ik,:,iz], Gridded(Linear())), NaN)
                for ia=1:ia_first
                    # Set up the objective function
                    V1A_obj_temp(x) = (sspace.a_grid[ia]-x)*sspace.k_grid[ik] + V2A_itp(x)
                    # Manually search on atil grid
                    atil_grid_temp = Array(range(sspace.atil_grid[1], min(sspace.a_grid[ia], atil_lastcross), step=5.0*1e-3))
                    # Make sure to include the endpoint in the grid
                    append!(atil_grid_temp, min(sspace.a_grid[ia], atil_lastcross))
                    V1A_obj_vals = V1A_obj_temp.(atil_grid_temp)
                    i_max = argmax(V1A_obj_vals) # Index of optimum

                    # Impose values in collectors
                    atil_imp[ia] = atil_grid_temp[i_max]
                    d_imp[ia]    = (sspace.a_grid[ia]-atil_grid_temp[i_max])*sspace.k_grid[ik]
                end
                # For all the values above ia_first, impose the same values
                atil_imp[ia_first:end] .= copy(atil_imp[ia_first])
                d_imp[ia_first:end]    .= (sspace.a_grid[ia_first:end] - atil_imp[ia_first:end])*sspace.k_grid[ik]
            end

            # And use the atil_imp policy to infer all value function derivatives, given the a-grid using interpolation
            V2Ak_itp    = extrapolate(interpolate((sspace.atil_grid, ), V2Ak_in[ik,:,iz], Gridded(Linear())), NaN)
            V2Aatil_itp = extrapolate(interpolate((sspace.atil_grid, ), V2Aatil_in[ik,:,iz], Gridded(Linear())), NaN)
            V2A_itp     = extrapolate(interpolate((sspace.atil_grid, ), V2A_in[ik,:,iz], Gridded(Linear())), NaN)
            # And evaluate at the solved atil policy
            V2Ak_imp    = V2Ak_itp.(atil_imp)
            V2Aatil_imp = V2Aatil_itp.(atil_imp)
            V2A_imp     = V2A_itp.(atil_imp)

            # And map these into V1 and its derivatives based on the envelope conditions; lambda1A = V2Aatil/k
            V1Ak_imp    = V2Ak_imp .+ (V2Aatil_imp./sspace.k_grid[ik]).*(sspace.a_grid - atil_imp)
            V1Aa_imp    = V2Aatil_imp
            V1A_imp     = d_imp + V2A_imp

            # Save these into the collectors
            atilA_sol_fin[ik,:,iz], dA_sol_fin[ik,:,iz]  = atil_imp, d_imp
            V1Ak_sol_fin[ik,:,iz], V1Aa_sol_fin[ik,:,iz], V1A_sol_fin[ik,:,iz] = V1Ak_imp, V1Aa_imp, V1A_imp
        end
    end

    # Return the policy and and value functions
    return atilA_sol_fin, dA_sol_fin, V1Ak_sol_fin, V1Aa_sol_fin, V1A_sol_fin
end

###
# Functions for solving firm problem along IRF
###

"Generic function to generate path for any AR1 with steady state value meanv, shock Ts periods after beginning. Argument Tm the period after which (but not including it), the variable is back to meanv."
function gen_ar1(T::Int64, Ts::Int64, Tm::Int64, meanv::Float64, rho::Float64, epss::Float64)
    if Tm > T
        error("Tm must be smaller or equal than T!")
    end

    epss_col     = zeros(T)
    epss_col[Ts] = epss
    ar1_col      = meanv*ones(T)
    ar1_col[1]   = meanv + epss_col[1]
    for t in 2:Tm
        ar1_col[t] = (1.0-rho)*meanv + rho*ar1_col[t-1] + epss_col[t]
    end

    return ar1_col
end

# Solve the firm's problem along a transition path, given a collection of price-paths faced.
# Read T from the length of the entered rm, rb paths.
function solve_IRF(fsoln, MU_col, w_col, M_tp1_col, pQ_col, Pi_col, rb_col, rm_col, Rc_col, gammatil_col, modl, sspace, opts)

    # Set output text printing path
    txtout_path = joinpath(opts.project_root, "interim_output", "model_interim_output", "savedout", modl.name * "_output.txt")

    # Initialize; NOTE: Must ensure that all series have this length
    T               = length(rm_col)
    # Need to compute the Rc_ss, i.e. the expectation of (1+rb)/Pi*pQ in SS
    Rc_ss   = (1.0+modl.rb)/(modl.Pi*1.0)

    # Collectors that will contain solution objects along path
    kpN_col, mpN_col, bpN_col, dN_col, kpA_col, mpA_col, bpA_col, dA_col, Ba_prob_col, V0ek_col, V0em_col, V0eb_col, V0e_col, Yn_col, Y_col, n_col, ACN_col, ACA_col = Array{Float64}(undef,sspace.Nk,sspace.Nm,sspace.Nb,sspace.Nz,T), Array{Float64}(undef,sspace.Nk,sspace.Nm,sspace.Nb,sspace.Nz,T), Array{Float64}(undef,sspace.Nk,sspace.Nm,sspace.Nb,sspace.Nz,T), Array{Float64}(undef,sspace.Nk,sspace.Nm,sspace.Nb,sspace.Nz,T), Array{Float64}(undef,sspace.Nk,sspace.Nm,sspace.Nb,sspace.Nz,T), Array{Float64}(undef,sspace.Nk,sspace.Nm,sspace.Nb,sspace.Nz,T), Array{Float64}(undef,sspace.Nk,sspace.Nm,sspace.Nb,sspace.Nz,T), Array{Float64}(undef,sspace.Nk,sspace.Nm,sspace.Nb,sspace.Nz,T), Array{Float64}(undef,sspace.Nk,sspace.Nm,sspace.Nb,sspace.Nz,T), Array{Float64}(undef,sspace.Nk,sspace.Nm,sspace.Nb,sspace.Nz,T), Array{Float64}(undef,sspace.Nk,sspace.Nm,sspace.Nb,sspace.Nz,T), Array{Float64}(undef,sspace.Nk,sspace.Nm,sspace.Nb,sspace.Nz,T), Array{Float64}(undef,sspace.Nk,sspace.Nm,sspace.Nb,sspace.Nz,T), Array{Float64}(undef,sspace.Nk,sspace.Nm,sspace.Nb,sspace.Nz,T), Array{Float64}(undef,sspace.Nk,sspace.Nm,sspace.Nb,sspace.Nz,T), Array{Float64}(undef,sspace.Nk,sspace.Nm,sspace.Nb,sspace.Nz,T), Array{Float64}(undef,sspace.Nk,sspace.Nm,sspace.Nb,sspace.Nz,T), Array{Float64}(undef,sspace.Nk,sspace.Nm,sspace.Nb,sspace.Nz,T)

    t_start = time_ns()
    # Loop backward, solving the firm's problem
    manprintln("Computing firm IRF solution for T=$(T).", txtout_path)
    for tt = T:-1:1
        manprintln("Period $(tt).", txtout_path)

        # Solve firm problem given continuation (expected) value function V0e
        if tt==T
            fsoln_cur = solve_V0(fsoln.V0ek, fsoln.V0em, fsoln.V0eb, fsoln.V0e, MU_col[tt], w_col[tt], M_tp1_col[tt], pQ_col[tt], Pi_col[tt], rb_col[tt], rm_col[tt], Rc_col[tt-1], Rc_col[tt], gammatil_col[tt], modl, sspace, opts)
        elseif tt==1
            fsoln_cur = solve_V0(V0ek_col[:,:,:,:,tt+1], V0em_col[:,:,:,:,tt+1], V0eb_col[:,:,:,:,tt+1], V0e_col[:,:,:,:,tt+1], MU_col[tt], w_col[tt], M_tp1_col[tt], pQ_col[tt], Pi_col[tt], rb_col[tt], rm_col[tt], Rc_ss, Rc_col[tt], gammatil_col[tt], modl, sspace, opts)
        else
            fsoln_cur = solve_V0(V0ek_col[:,:,:,:,tt+1], V0em_col[:,:,:,:,tt+1], V0eb_col[:,:,:,:,tt+1], V0e_col[:,:,:,:,tt+1], MU_col[tt], w_col[tt], M_tp1_col[tt], pQ_col[tt], Pi_col[tt], rb_col[tt], rm_col[tt], Rc_col[tt-1], Rc_col[tt], gammatil_col[tt], modl, sspace, opts)
        end

        # Unpack policies solution
        kpN_col[:,:,:,:,tt], mpN_col[:,:,:,:,tt], bpN_col[:,:,:,:,tt], dN_col[:,:,:,:,tt], kpA_col[:,:,:,:,tt], mpA_col[:,:,:,:,tt], bpA_col[:,:,:,:,tt], dA_col[:,:,:,:,tt], Ba_prob_col[:,:,:,:,tt] = fsoln_cur.kpN, fsoln_cur.mpN, fsoln_cur.bpN, fsoln_cur.dN, fsoln_cur.kpA, fsoln_cur.mpA, fsoln_cur.bpA, fsoln_cur.dA, fsoln_cur.Ba_prob
        # Also the output etc results, although not needed
        Yn_col[:,:,:,:,tt], Y_col[:,:,:,:,tt], n_col[:,:,:,:,tt], ACN_col[:,:,:,:,tt], ACA_col[:,:,:,:,tt] = fsoln_cur.Yn, fsoln_cur.Y, fsoln_cur.n, fsoln_cur.ACN, fsoln_cur.ACA

        # Finally, integrate out z in the value functions
        V0k_cur, V0m_cur, V0b_cur, V0_cur = fsoln_cur.V0ek, fsoln_cur.V0em, fsoln_cur.V0eb, fsoln_cur.V0e
        for iz=1:sspace.Nz
            V0ek_col[:,:,:,iz,tt] = sum([sspace.P[iz,izp] * V0k_cur[:,:,:,izp] for izp=1:sspace.Nz])
            V0em_col[:,:,:,iz,tt] = sum([sspace.P[iz,izp] * V0m_cur[:,:,:,izp] for izp=1:sspace.Nz])
            V0eb_col[:,:,:,iz,tt] = sum([sspace.P[iz,izp] * V0b_cur[:,:,:,izp] for izp=1:sspace.Nz])
            V0e_col[:,:,:,iz,tt]  = sum([sspace.P[iz,izp] * V0_cur[:,:,:,izp] for izp=1:sspace.Nz])
        end
    end
    manprintln("Computation of firm IRF solution complete in time: $((time_ns()-t_start)/1.0e9).", txtout_path)

    # Return the solution
    return IRFSolution(MU_col, w_col, M_tp1_col, pQ_col, Pi_col, rb_col, rm_col, Rc_col, gammatil_col, kpN_col, mpN_col, bpN_col, dN_col, kpA_col, mpA_col, bpA_col, dA_col, Ba_prob_col, V0ek_col, V0em_col, V0eb_col, V0e_col, Yn_col, Y_col, n_col, ACN_col, ACA_col)
end





###
# Functions for firm-level simulations

# STEADY STATE FUNCTION
# Simulate N firms, taking as given a CONSTANT set of policy functions on (k,m,b,z)
# Conditional on initial (k,m,b), contained in Nx3 array se_init, and given a PATH of idiosyncratic TFP levels indices (z_i), contained in the T-length array zi_col
# Also, must enter path of issuance cost CDF realizations xi_cdf_col (NxT) -- so if a given xi_cdf > Ba_prob, then it means that the issuance cost is from "higher up" in the xi distribution than the cutoff xi_c, and the firm does not issue.
function sim_firm_SS(se_init, zi_col, xi_cdf_col, fsoln, printwarnings, modl, sspace, opts)

    # Set output text printing path
    txtout_path = joinpath(opts.project_root, "interim_output", "model_interim_output", "savedout", modl.name * "_output.txt")

    if size(se_init,1) != size(zi_col,1)
        error("Number of initial s's must be equal to paths of z!")
    end
    if size(se_init,1) != size(xi_cdf_col,1)
        error("Number of initial s's must be equal to paths of xi_cdf!")
    end
    if size(zi_col,2) != size(xi_cdf_col,2)
        error("Lengths of paths for z and xi_cdf must agree!")
    end

    T   = size(zi_col,2)
    Nfi = size(se_init,1)

    # Unpack policy functions
    kpN_pol, mpN_pol, bpN_pol, dN_pol, kpA_pol, mpA_pol, bpA_pol, dA_pol, Ba_prob_pol = fsoln.kpN, fsoln.mpN, fsoln.bpN, fsoln.dN, fsoln.kpA, fsoln.mpA, fsoln.bpA, fsoln.dA, fsoln.Ba_prob
    Yn_pol, Y_pol, n_pol, ACN_pol, ACA_pol = fsoln.Yn, fsoln.Y, fsoln.n, fsoln.ACN, fsoln.ACA

    # Create interpolants of the policy functions. For general robustness, allow for linear extrapolation of the policies, but whenever the code resorts to extrapolation, print an explicit warning below. In practice, this never happens in the quantitative exercises of the final paper. (See separate comment for bp policy below.)
    kpN_itp = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), kpN_pol, Gridded(Linear())), Line())
    mpN_itp = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), mpN_pol, Gridded(Linear())), Line())
    bpN_itp = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), bpN_pol, Gridded(Linear())), Line())
    dN_itp  = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), dN_pol, Gridded(Linear())), Line())
    kpA_itp = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), kpA_pol, Gridded(Linear())), Line())
    mpA_itp = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), mpA_pol, Gridded(Linear())), Line())
    bpA_itp = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), bpA_pol, Gridded(Linear())), Line())
    dA_itp  = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), dA_pol, Gridded(Linear())), Line())
    Ba_prob_itp  = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), Ba_prob_pol, Gridded(Linear())), Line())

    Yn_itp  = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), Yn_pol, Gridded(Linear())), Line())
    Y_itp   = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), Y_pol, Gridded(Linear())), Line())
    n_itp   = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), n_pol, Gridded(Linear())), Line())
    ACN_itp = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), ACN_pol, Gridded(Linear())), Line())
    ACA_itp = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), ACA_pol, Gridded(Linear())), Line())

    # Create collectors of outcomes
    k_col, m_col, b_col, d_col, Ba_prob_col, Ba_col = Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T)
    Yn_col, Y_col, n_col, AC_col = Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T)

    # Iterate over the Nfi firms
    for ii = 1:Nfi
        k_col[ii,1], m_col[ii,1], b_col[ii,1] = se_init[ii,:]

        # Simulate
        for tt=1:T
            # Determine issuance
            Ba_prob_col[ii,tt] = Ba_prob_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]])
            Ba_col[ii,tt]      = Float64(xi_cdf_col[ii,tt] < Ba_prob_col[ii,tt])
            d_col[ii,tt]       = Ba_col[ii,tt]*dA_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]]) + (1.0-Ba_col[ii,tt])*dN_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]])

            Yn_col[ii,tt] = Yn_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]])
            Y_col[ii,tt]  = Y_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]])
            n_col[ii,tt]  = n_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]])
            AC_col[ii,tt] = Ba_col[ii,tt]*ACA_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]]) + (1.0-Ba_col[ii,tt])*ACN_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]])
            if tt<T
                k_col[ii,tt+1]  = Ba_col[ii,tt]*kpA_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]]) + (1.0-Ba_col[ii,tt])*kpN_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]])
                m_col[ii,tt+1]  = Ba_col[ii,tt]*mpA_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]]) + (1.0-Ba_col[ii,tt])*mpN_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]])
                b_col[ii,tt+1]  = Ba_col[ii,tt]*bpA_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]]) + (1.0-Ba_col[ii,tt])*bpN_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]])

                # Check whether firm steps out of grid, so that extrapolation is used, and write a warning
                # k
                if k_col[ii,tt+1] > sspace.k_grid[end]
                    if printwarnings; manprintln("WARNING: Firm ii=$(ii) chooses kp=$(round(k_col[ii,tt+1],digits=4)) > k_max=$(round(sspace.k_grid[end],digits=4)): extrapolating policies.", txtout_path); end
                elseif k_col[ii,tt+1] < sspace.k_grid[1]
                    if printwarnings; manprintln("WARNING: Firm ii=$(ii) chooses kp=$(round(k_col[ii,tt+1],digits=4)) < k_min=$(round(sspace.k_grid[1],digits=4)): extrapolating policies.", txtout_path); end
                end
                # m
                if m_col[ii,tt+1] > sspace.m_grid[end]
                    if printwarnings; manprintln("WARNING: Firm ii=$(ii) chooses mp=$(round(m_col[ii,tt+1],digits=4)) > m_max=$(round(sspace.m_grid[end],digits=4)): extrapolating policies.", txtout_path); end
                elseif m_col[ii,tt+1] < sspace.m_grid[1]
                    if printwarnings; manprintln("WARNING: Firm ii=$(ii) chooses mp=$(round(m_col[ii,tt+1],digits=4)) < m_min=$(round(sspace.m_grid[1],digits=4)): extrapolating policies.", txtout_path); end
                end
                # b
                if b_col[ii,tt+1] > sspace.b_grid[end]
                    if printwarnings; manprintln("WARNING: Firm ii=$(ii) chooses bp=$(round(b_col[ii,tt+1],digits=4)) > b_max=$(round(sspace.b_grid[end],digits=4)): extrapolating policies.", txtout_path); end
                elseif b_col[ii,tt+1] < sspace.b_grid[1]
                    # Because the fsoln object may include bp policies below the lowest point of the b-space up to machine epsilon (1e-16), ensure that the actual firm's choice b is set to the end-point, so in this case no actual extrapolation of firm behavior outside the state space occurs. 
                    b_col[ii,tt+1] = sspace.b_grid[1]
                end
                
            end

        end
    end

    # Return paths for outcomes
    return FirmSimResult(zi_col, xi_cdf_col, k_col, m_col, b_col, d_col, Ba_prob_col, Ba_col, Yn_col, Y_col, n_col, AC_col)
end

# Analogously, simulate firms taking as given the IRF set of policy functions on (k,m,b,z), contained in irfsoln
# Note that the length of the simulated path is the length implied by zi_col. If it is longer than the collection of policies in irfsoln, then give error
function sim_firm_IRF(se_init, zi_col, xi_cdf_col, irfsoln, printwarnings, modl, sspace, opts)

    # Set output text printing path
    txtout_path = joinpath(opts.project_root, "interim_output", "model_interim_output", "savedout", modl.name * "_output.txt")

    if size(se_init,1) != size(zi_col,1)
        error("Number of initial s's must be equal to paths of z!")
    end
    if size(se_init,1) != size(xi_cdf_col,1)
        error("Number of initial s's must be equal to paths of xi_cdf!")
    end
    if size(zi_col,2) != size(xi_cdf_col,2)
        error("Lengths of paths for z and xi_cdf must agree!")
    end

    if size(irfsoln.kpN_col,5) < size(zi_col,2)
        error("IRF solution policies t too short for firm-level z collection!")
    end

    T   = size(zi_col,2)
    Nfi = size(se_init,1)

    # Unpack policy functions
    kpN_pol, mpN_pol, bpN_pol, dN_pol, kpA_pol, mpA_pol, bpA_pol, dA_pol, Ba_prob_pol = irfsoln.kpN_col, irfsoln.mpN_col, irfsoln.bpN_col, irfsoln.dN_col, irfsoln.kpA_col, irfsoln.mpA_col, irfsoln.bpA_col, irfsoln.dA_col, irfsoln.Ba_prob_col
    Yn_pol, Y_pol, n_pol, ACN_pol, ACA_pol = irfsoln.Yn_col, irfsoln.Y_col, irfsoln.n_col, irfsoln.ACN_col, irfsoln.ACA_col

    # Create collectors of outcomes
    k_col, m_col, b_col, d_col, Ba_prob_col, Ba_col = Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T)
    Yn_col, Y_col, n_col, AC_col = Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T), Array{Float64}(undef, Nfi, T)

    # Initialize the Nfi firms
    for ii = 1:Nfi
        k_col[ii,1], m_col[ii,1], b_col[ii,1] = se_init[ii,:]
    end

    # Simulate over each t, with t running in the "outer loop"
    for tt=1:T
        # Create interpolants of the policy functions. For general robustness, allow for linear extrapolation of the policies, but whenever the code resorts to extrapolation, print an explicit warning below. In practice, this never happens in the quantitative exercises of the final paper. (See separate comment for bp policy below.)
        kpN_itp = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), kpN_pol[:,:,:,:,tt], Gridded(Linear())), Line())
        mpN_itp = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), mpN_pol[:,:,:,:,tt], Gridded(Linear())), Line())
        bpN_itp = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), bpN_pol[:,:,:,:,tt], Gridded(Linear())), Line())
        dN_itp  = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), dN_pol[:,:,:,:,tt], Gridded(Linear())), Line())
        kpA_itp = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), kpA_pol[:,:,:,:,tt], Gridded(Linear())), Line())
        mpA_itp = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), mpA_pol[:,:,:,:,tt], Gridded(Linear())), Line())
        bpA_itp = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), bpA_pol[:,:,:,:,tt], Gridded(Linear())), Line())
        dA_itp  = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), dA_pol[:,:,:,:,tt], Gridded(Linear())), Line())
        Ba_prob_itp  = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), Ba_prob_pol[:,:,:,:,tt], Gridded(Linear())), Line())

        Yn_itp  = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), Yn_pol[:,:,:,:,tt], Gridded(Linear())), Line())
        Y_itp   = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), Y_pol[:,:,:,:,tt], Gridded(Linear())), Line())
        n_itp   = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), n_pol[:,:,:,:,tt], Gridded(Linear())), Line())
        ACN_itp = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), ACN_pol[:,:,:,:,tt], Gridded(Linear())), Line())
        ACA_itp = extrapolate(interpolate((sspace.k_grid, sspace.m_grid, sspace.b_grid, sspace.z_grid), ACA_pol[:,:,:,:,tt], Gridded(Linear())), Line())

        # Inside, iterate over each firm
        for ii=1:Nfi
            # Determine issuance
            Ba_prob_col[ii,tt] = Ba_prob_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]])
            Ba_col[ii,tt]      = Float64(xi_cdf_col[ii,tt] < Ba_prob_col[ii,tt])
            d_col[ii,tt]       = Ba_col[ii,tt]*dA_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]]) + (1.0-Ba_col[ii,tt])*dN_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]])

            Yn_col[ii,tt] = Yn_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]])
            Y_col[ii,tt]  = Y_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]])
            n_col[ii,tt]  = n_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]])
            AC_col[ii,tt] = Ba_col[ii,tt]*ACA_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]]) + (1.0-Ba_col[ii,tt])*ACN_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]])
            if tt<T
                k_col[ii,tt+1]  = Ba_col[ii,tt]*kpA_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]]) + (1.0-Ba_col[ii,tt])*kpN_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]])
                m_col[ii,tt+1]  = Ba_col[ii,tt]*mpA_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]]) + (1.0-Ba_col[ii,tt])*mpN_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]])
                b_col[ii,tt+1]  = Ba_col[ii,tt]*bpA_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]]) + (1.0-Ba_col[ii,tt])*bpN_itp(k_col[ii,tt], m_col[ii,tt], b_col[ii,tt], sspace.z_grid[zi_col[ii,tt]])

                # Check whether firm steps out of grid, so that extrapolation is used, and write a warning
                # k
                if k_col[ii,tt+1] > sspace.k_grid[end]
                    if printwarnings; manprintln("WARNING: Firm ii=$(ii) chooses kp=$(round(k_col[ii,tt+1],digits=4)) > k_max=$(round(sspace.k_grid[end],digits=4)): extrapolating policies.", txtout_path); end
                elseif k_col[ii,tt+1] < sspace.k_grid[1]
                    if printwarnings; manprintln("WARNING: Firm ii=$(ii) chooses kp=$(round(k_col[ii,tt+1],digits=4)) < k_min=$(round(sspace.k_grid[1],digits=4)): extrapolating policies.", txtout_path); end
                end
                # m
                if m_col[ii,tt+1] > sspace.m_grid[end]
                    if printwarnings; manprintln("WARNING: Firm ii=$(ii) chooses mp=$(round(m_col[ii,tt+1],digits=4)) > m_max=$(round(sspace.m_grid[end],digits=4)): extrapolating policies.", txtout_path); end
                elseif m_col[ii,tt+1] < sspace.m_grid[1]
                    if printwarnings; manprintln("WARNING: Firm ii=$(ii) chooses mp=$(round(m_col[ii,tt+1],digits=4)) < m_min=$(round(sspace.m_grid[1],digits=4)): extrapolating policies.", txtout_path); end
                end
                # b
                if b_col[ii,tt+1] > sspace.b_grid[end]
                    if printwarnings; manprintln("WARNING: Firm ii=$(ii) chooses bp=$(round(b_col[ii,tt+1],digits=4)) > b_max=$(round(sspace.b_grid[end],digits=4)): extrapolating policies.", txtout_path); end
                elseif b_col[ii,tt+1] < sspace.b_grid[1]
                    # Because the irfsoln object may include bp policies below the lowest point of the b-space up to machine epsilon (1e-16), ensure that the actual firm's choice b is set to the end-point, so in this case no actual extrapolation of firm behavior outside the state space occurs. 
                    b_col[ii,tt+1] = sspace.b_grid[1]
                end
                    
            end
        end
    end

    # Return paths for outcomes
    return FirmSimResult(zi_col, xi_cdf_col, k_col, m_col, b_col, d_col, Ba_prob_col, Ba_col, Yn_col, Y_col, n_col, AC_col)
end

# Function that computes firm IRF paths in terms of all variables' deviations from the steady state, by calling sim_firm_SS and sim_firm_IRF and taking their ratios
function sim_firm_IRF_dev(se_init, zi_col, xi_cdf_col, fsoln, irfsoln, printwarnings, modl, sspace, opts)
    # Compute the firm IRF paths
    firm_irf_paths = sim_firm_IRF(se_init, zi_col, xi_cdf_col, irfsoln, printwarnings, modl, sspace, opts)
    # And the corresponding paths in SS
    # As steady state p use the LAST p_col entry
    firm_ss_paths  = sim_firm_SS(se_init, zi_col, xi_cdf_col, fsoln, printwarnings, modl, sspace, opts)

    # Compute return values
    zi_col      = firm_irf_paths.zi_col
    k_col       = firm_irf_paths.k_path ./ firm_ss_paths.k_path
    m_col       = firm_irf_paths.m_path - firm_ss_paths.m_path
    b_col       = firm_irf_paths.b_path - firm_ss_paths.b_path
    d_col       = firm_irf_paths.d_path - firm_ss_paths.d_path
    Ba_prob_col = firm_irf_paths.Ba_prob_path - firm_ss_paths.Ba_prob_path
    Ba_col      = firm_irf_paths.Ba_path - firm_ss_paths.Ba_path
    Yn_col      = firm_irf_paths.Yn_path ./ firm_ss_paths.Yn_path
    Y_col       = firm_irf_paths.Y_path ./ firm_ss_paths.Y_path
    n_col       = firm_irf_paths.n_path ./ firm_ss_paths.n_path
    AC_col      = firm_irf_paths.AC_path ./ firm_ss_paths.AC_path

    # Pack up the results
    return FirmSimResult(zi_col, xi_cdf_col, k_col, m_col, b_col, d_col, Ba_prob_col, Ba_col, Yn_col, Y_col, n_col, AC_col)
end


# Function for simulating a random sample in SS and IRF, given solutions fed in, and conducting study of IRF responses, conditional on firm financial conditions.
# Take Nfirms, T, and initial distribution Lin_arg where to draw from as arguments, and generate random sample
function sim_IRFSS_anl_rand(Nfirms, T, T_long, Lin_arg, fsoln, irfsoln, ind_plot, printwarnings, str_end, modl, sspace, opts)

    # Initialize
    zi_col, xi_cdf_col =  Array{Int64,2}(undef, Nfirms,T), zeros(Nfirms, T)
    zi_col_long, xi_cdf_col_long =  Array{Int64,2}(undef, Nfirms,T_long), zeros(Nfirms, T_long)
    mcz         = MarkovChain(sspace.P)

    # Draw initial distribution
    # Distribution from given Lin
    ii_samp = wsample(1:sspace.Nsf, Lin_arg, Nfirms)
    se_init = sspace.sf[ii_samp, 1:3]
    xi_cdf_col = rand(Nfirms,T)
    xi_cdf_col_long = rand(Nfirms,T_long)
    zi_col_init= repeat(1:sspace.nf[4], inner=sspace.nf[1]*sspace.nf[2]*sspace.nf[3])[ii_samp]
    # Simulate MChains
    for ii in 1:Nfirms
        zi_col[ii,:]   = simulate(mcz,T,init=zi_col_init[ii])
        zi_col_long[ii,:]   = simulate(mcz,T_long,init=zi_col_init[ii])
    end

    # Given this sample, run the "analysis" function
    sim_IRFSS_anl(se_init, zi_col, xi_cdf_col, zi_col_long, xi_cdf_col_long, fsoln, irfsoln, ind_plot, printwarnings, str_end, modl, sspace, opts)
end
# Function for taking a sample in SS and IRF, simulating, given solutions fed in, and conducting study of IRF responses, conditional on firm financial conditions.
# Take initial conditions and shocks as arguments, and use these to back out N and T
function sim_IRFSS_anl(se_init, zi_col, xi_cdf_col, zi_col_long, xi_cdf_col_long, fsoln, irfsoln, ind_plot, printwarnings, str_end, modl, sspace, opts)

    # First, simulate paths in IRF and SS
    sim_paths_SS  = sim_firm_SS(se_init, zi_col, xi_cdf_col, fsoln, printwarnings, modl, sspace, opts)
    sim_paths_SS_long  = sim_firm_SS(se_init, zi_col_long, xi_cdf_col_long, fsoln, printwarnings, modl, sspace, opts)
    sim_paths_IRF = sim_firm_IRF(se_init, zi_col, xi_cdf_col, irfsoln, printwarnings, modl, sspace, opts)

    # Read T from the length of these paths
    T = size(zi_col,2)

    # For inferring levels of debt b, compute the Rc_ss
    pQ_ss = 1.0
    Rc_ss = (1.0+modl.rb)/(modl.Pi*pQ_ss)

    # Run regressions
    # Compile dataframe
    data        = DataFrame(k=se_init[:,1], m=se_init[:,2])
    data.cheat  = allowmissing(data.m ./ (data.k + data.m))
    # Drop cheat outliers, as in empirics
    data.cheat[data.cheat.>=quantile(data.cheat,0.99)] .= missing
    # Demeaned cheat
    data.cheat_dm = data.cheat - mean(sim_paths_SS_long.m_path ./ (sim_paths_SS_long.m_path + sim_paths_SS_long.k_path), dims=2)[:]

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

    # Run regressions
    # Explanatory variable selector
    x_col_names = ["cheat","cheat_dm"]
    nx = length(x_col_names)
    gamma_col, gamma_ci_col         = zeros(T,nx), zeros(T,nx,2)
    gamma_clk_col, gamma_ci_clk_col = zeros(T,nx), zeros(T,nx,2)
    gammacontr_clk_col, gammacontr_ci_clk_col = zeros(T,nx), zeros(T,nx,2)

    # Construct scaler for "per 25bp annualized rb shock"
    # Create rf path (from rm) to scale the responses to "per 25bp annualized rf shock"
    rf_col_resp_here = (irfsoln.rm_col.-modl.rm)./modl.phi_mresp
    reg_rf_scaler = 0.0025/(4*rf_col_resp_here[2])
    # Manual y limits for lk response
    man_ylims_k_emp = [-6.1,2.1]

    # Iterate over explanatory variable of interest
    for (ix, x_str_temp) in enumerate(x_col_names)
        for tt=1:T
            # Outcome selector
            y_str_temp       = "lk_dev_t$(tt)"
            reg_mod = lm(term(y_str_temp) ~ term(x_str_temp), data)
            gamma_col[tt,ix] = reg_rf_scaler*coef(reg_mod)[2]
            gamma_ci_col[tt,ix,:] = [reg_rf_scaler*coeftable(reg_mod).cols[5][2], reg_rf_scaler*coeftable(reg_mod).cols[6][2]]
        end

        # Plot cheat alongside data estimates
        if ind_plot & (x_str_temp=="cheat") & (str_end in ["_base_pub", "_phiw0_pub", "_rbtaureal_pub", "_rsPiQ_pub", "_phik1d8_pub"])
            manl_ftsize = 13
            df_emp_betas= DataFrame(CSV.File(joinpath(opts.project_root, "interim_output", "Fig6_slevsbetas_vec_q_vals(lk_stock)(cheat_rat)_IV_FE.csv")))
            df_emp_ci   = DataFrame(CSV.File(joinpath(opts.project_root, "interim_output", "Fig6_slevsbetas_vec_q_cis(lk_stock)(cheat_rat)_IV_FE.csv")))
            emp_betas   = Array(df_emp_betas.V1)
            emp_ci      = zeros(size(emp_betas,1),2)
            emp_ci[:,1], emp_ci[:,2] = df_emp_ci.V1, df_emp_ci.V2
            # Plot
            T_emp_plot = min(length(gamma_col[2:end,ix]), length(emp_betas))
            if str_end == "_phik1d8_pub"
                Plots.plot(0:(T_emp_plot-1), -gamma_col[2:end,ix], xlabel="Quarters (h)", ylabel="Percent", ylims=man_ylims_k_emp, linecolor=:red, linestyle=:dot, linewidth=3.0, xticks=0:2:T_emp_plot, label="Model", legend=:bottomright, xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-1)
            else
                Plots.plot(0:(T_emp_plot-1), -gamma_col[2:end,ix], xlabel="Quarters (h)", ylabel="Percent", ylims=man_ylims_k_emp, linecolor=:red, linestyle=:dot, linewidth=3.0, xticks=0:2:T_emp_plot, label="Model", legend=:bottomleft, xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-1)
            end
            Plots.plot!(0:(T_emp_plot-1), emp_betas[1:T_emp_plot], linecolor=:blue, linewidth=2.0, label="Data")
            Plots.plot!(0:(T_emp_plot-1), emp_ci[1:T_emp_plot,:], linestyle=[:dash :dash], linecolor=:blue, linewidth=2.0, label=:none)
            Plots.plot!(0:(T_emp_plot-1), zeros(T_emp_plot), linestyle=:dash, linecolor=:black, linewidth=0.5, label=:none)

            if (str_end=="_base_pub")
                # Save figure
                savefig(joinpath(opts.project_root, "output", "figures", "model_figures", "Fig6_anl_regcfs_wemp_"*x_str_temp*str_end*".pdf"))
                # Export the coefficients into a file
                save(joinpath(opts.project_root, "interim_output", "model_interim_output", "anl_regcfs_vals_temp"*x_str_temp*str_end*".jld"),  "mod_coefs", -gamma_col[2:end,ix])
            elseif (str_end=="_phiw0_pub")
                # Save figure
                savefig(joinpath(opts.project_root, "output", "figures", "model_figures", "FigB11a_anl_regcfs_wemp_"*x_str_temp*str_end*".pdf"))
            elseif (str_end=="_phik1d8_pub")
                # Save figure
                savefig(joinpath(opts.project_root, "output", "figures", "model_figures", "FigB13a_anl_regcfs_wemp_"*x_str_temp*str_end*".pdf"))
            elseif (str_end in ["_rbtaureal_pub", "_rsPiQ_pub"])
                # Export the coefficients into a file
                save(joinpath(opts.project_root, "interim_output", "model_interim_output", "anl_regcfs_vals_temp"*x_str_temp*str_end*".jld"),  "mod_coefs", -gamma_col[2:end,ix])
            end

            # Also, add figure on within-firm demeaned cheat_rat
            if (str_end=="_base_pub")
                # Load data
                df_emp_betas_dm= DataFrame(CSV.File(joinpath(opts.project_root, "interim_output", "FigB2_slevsbetas_vec_q_vals(lk_stock)(cheat_rat_dm)_IV_FE.csv")))
                df_emp_ci_dm   = DataFrame(CSV.File(joinpath(opts.project_root, "interim_output", "FigB2_slevsbetas_vec_q_cis(lk_stock)(cheat_rat_dm)_IV_FE.csv")))
                emp_betas_dm   = Array(df_emp_betas_dm.V1)
                emp_ci_dm      = zeros(size(emp_betas_dm,1),2)
                emp_ci_dm[:,1], emp_ci_dm[:,2] = df_emp_ci_dm.V1, df_emp_ci_dm.V2
                # Plot separately, with only _dm
                Plots.plot(0:(T_emp_plot-1), -gamma_col[2:end,ix], xlabel="Quarters (h)", ylabel="Percent", ylims=man_ylims_k_emp, linecolor=:red, linestyle=:dot, linewidth=3.0, xticks=0:2:T_emp_plot, label="Model", legend=:bottomleft, xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-1)
                Plots.plot!(0:(T_emp_plot-1), emp_betas_dm[1:T_emp_plot], linecolor=:blue, linewidth=2.0, label="Data")
                Plots.plot!(0:(T_emp_plot-1), emp_ci_dm[1:T_emp_plot,:], linestyle=[:dash :dash], linecolor=:blue, linewidth=2.0, label=:none)
                Plots.plot!(0:(T_emp_plot-1), zeros(T_emp_plot), linestyle=:dash, linecolor=:black, linewidth=0.5, label=:none)
                savefig(joinpath(opts.project_root, "output", "figures", "model_figures", "FigB2_anl_regcfs_wemp_dm_"*x_str_temp*str_end*".pdf"))
            end
        end
    end

    # Generate figure for comparing non-demeaned vs demeaned regressions (Figure B.4)
    if ind_plot & (str_end=="_base_pub")
        manl_ftsize = 13
        T_emp_plot = 19
        Plots.plot(0:(T_emp_plot-1), -gamma_col[2:end,1], xlabel="Quarters (h)", ylabel="Percent", ylims=(-2.5, 0.2), linecolor=:red, linestyle=:dot, linewidth=2.0, xticks=0:2:T_emp_plot, label="No demeaning", legend=:bottomright, xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-1)
        Plots.plot!(0:(T_emp_plot-1), -gamma_col[2:end,2], linecolor=:green, linewidth=2.0, label="Demeaning")
        Plots.plot!(0:(T_emp_plot-1), zeros(T_emp_plot), linestyle=:dash, linecolor=:black, linewidth=0.5, label=:none)
        savefig(joinpath(opts.project_root, "output", "figures", "model_figures", "FigB4_anl_regcfs_cheatdmcomp"*str_end*".pdf"))
    end

    # Repeat above for (debt issuance)/assets as the outcome variable
    gamma_col, gamma_ci_col         = zeros(T-1,nx), zeros(T-1,nx,2)
    gamma_clk_col, gamma_ci_clk_col = zeros(T-1,nx), zeros(T-1,nx,2)
    gammacontr_clk_col, gammacontr_ci_clk_col = zeros(T-1,nx), zeros(T-1,nx,2)
    man_ylims_k_emp = [-1.4,0.8]

    # Add time-series data
    for tt=1:(T-1)
        str_temp         = "lk_dev_t$(tt)"
        data[!,str_temp] = allowmissing(100*log.(sim_paths_IRF.k_path[:,tt]./sim_paths_SS.k_path[:,tt]))
        q_l_temp, q_h_temp = quantile(data[!,str_temp],0.01), quantile(data[!,str_temp],0.99)
        # Overwrite the outcome variable to be (debt issuance)/assets ratio
        # Compute debt issuance behavior
        # Setup of RC and gamma in SS
        gammatil_ss = modl.gamma/modl.Pi
        # Steady state issuances
        b_tr_paths_SS = sim_paths_SS.k_path.*sim_paths_SS.b_path/ Rc_ss # Level real debt paths
        b_iss_paths_SS = max.(0.0, -(b_tr_paths_SS[:,(2:end)] - gammatil_ss * b_tr_paths_SS[:,1:(end-1)]) .* sim_paths_SS.Ba_path[:,1:(end-1)]) # Real issuances
        b_iss_at_rat_paths_SS = b_iss_paths_SS ./ (sim_paths_SS.k_path[:,1:(end-1)] + sim_paths_SS.m_path[:,1:(end-1)]) # Real issuance/assets ratio
        # Impulse response path issuances
        b_tr_paths_IRF = sim_paths_IRF.k_path.*sim_paths_IRF.b_path ./ repeat(vcat(Rc_ss, irfsoln.Rc_col[1:(T-1)])', size(sim_paths_SS.k_path,1)) # Level real debt paths
        b_iss_paths_IRF = max.(0.0, -(b_tr_paths_IRF[:,(2:end)] - repeat(irfsoln.gammatil_col[1:T-1]', size(sim_paths_SS.k_path,1)) .* b_tr_paths_IRF[:,1:(end-1)]) .* sim_paths_IRF.Ba_path[:,1:(end-1)]) # Real issuances
        b_iss_at_rat_paths_IRF = b_iss_paths_IRF ./ (sim_paths_IRF.k_path[:,1:(end-1)] + sim_paths_IRF.m_path[:,1:(end-1)]) # Real issuance/assets ratio
        data[!,str_temp] = allowmissing(100*(b_iss_at_rat_paths_IRF[:,tt] - b_iss_at_rat_paths_SS[:,tt]))
        q_l_temp, q_h_temp = quantile(skipmissing(filter(!isnan,data[!,str_temp])),0.01), quantile(skipmissing(filter(!isnan,data[!,str_temp])),0.99)
        ind_l_temp, ind_h_temp = data[!,str_temp] .<= q_l_temp, data[!,str_temp] .>= q_h_temp
        if tt>1
            data[!,str_temp][ind_l_temp] .= missing
            data[!,str_temp][ind_h_temp] .= missing
        end
    end

    # Run base regression
    for (ix, x_str_temp) in enumerate(x_col_names[[1]])
        for tt=1:(T-1)
            # Outcome selector
            y_str_temp       = "lk_dev_t$(tt)"
            reg_mod = lm(term(y_str_temp) ~ term(x_str_temp), data)
            gamma_col[tt,ix] = reg_rf_scaler*coef(reg_mod)[2]
            gamma_ci_col[tt,ix,:] = [reg_rf_scaler*coeftable(reg_mod).cols[5][2], reg_rf_scaler*coeftable(reg_mod).cols[6][2]]
        end

        # Plot coefficient alongside data estimates
        if ind_plot & (x_str_temp=="cheat") & (str_end=="_base_pub")
            manl_ftsize = 13
            df_emp_betas= DataFrame(CSV.File(joinpath(opts.project_root, "interim_output", "FigB3a_slevsbetas_vec_q_vals(dltisqat_rat)(cheat_rat)_IV_FE.csv")))
            df_emp_ci   = DataFrame(CSV.File(joinpath(opts.project_root, "interim_output", "FigB3a_slevsbetas_vec_q_cis(dltisqat_rat)(cheat_rat)_IV_FE.csv")))
            emp_betas   = Array(df_emp_betas.V1)
            emp_ci      = zeros(size(emp_betas,1),2)
            emp_ci[:,1], emp_ci[:,2] = df_emp_ci.V1, df_emp_ci.V2
            # Plot
            T_emp_plot = min(length(gamma_col[1:end,ix]), length(emp_betas))
            Plots.plot(0:(T_emp_plot-1), -gamma_col[1:T_emp_plot,ix], xlabel="Quarters (h)", ylabel="Percent", ylims=[-0.90, 0.35], linecolor=:red, linestyle=:dot, linewidth=3.0, xticks=0:2:T_emp_plot, label="Model", legend=:bottomright, xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-1)
            Plots.plot!(0:(T_emp_plot-1), emp_betas[1:T_emp_plot], linecolor=:blue, linewidth=2.0, label="Data")
            Plots.plot!(0:(T_emp_plot-1), emp_ci[1:T_emp_plot,:], linestyle=[:dash :dash], linecolor=:blue, linewidth=2.0, label=:none)
            Plots.plot!(0:(T_emp_plot-1), zeros(T_emp_plot), linestyle=:dash, linecolor=:black, linewidth=0.5, label=:none)
            savefig(joinpath(opts.project_root, "output", "figures", "model_figures", "FigB3a_anl_regcfs_dis_wemp_"*x_str_temp*str_end*".pdf"))
        end
    end


    ###
    # Repeat above for interest rate on long-term debt as the outcome variable
    ###
    gamma_col, gamma_ci_col         = zeros(T-1,nx), zeros(T-1,nx,2)
    gamma_clk_col, gamma_ci_clk_col = zeros(T-1,nx), zeros(T-1,nx,2)
    gammacontr_clk_col, gammacontr_ci_clk_col = zeros(T-1,nx), zeros(T-1,nx,2)
    man_ylims_k_emp = [-1.4,0.8]
    # Construct paths of implicit interest rates
    YTM_paths_SS     = modl.rb * ones(size(sim_paths_SS.k_path))
    YTM_paths_IRF    = zeros(size(sim_paths_SS.k_path))
    # Compute the path of the bond price (with implicit coupon of 1)
    T_soln = length(irfsoln.rb_col)
    qbar_T = 1 / (1 + modl.rb - modl.gamma)
    qbar_path = zeros(T_soln)
    qbar_path[end] = (1 + modl.gamma*qbar_T)/(1 + modl.rb)
    for tt=(T_soln-1):-1:1
        qbar_path[tt] = (1 + modl.gamma*qbar_path[tt+1])/(1 + irfsoln.rb_col[tt+1])
    end
    YTM_path = 1.0 ./ qbar_path .+ modl.gamma .- 1.0
    # Iterate forward on firms' debt adjustment decisions and infer the next period's implicit interest rate based on whether have adjusted or not
    YTM_paths_IRF[:,1] .= modl.rb
    for tt=2:T
        YTM_paths_IRF[:,tt] = YTM_paths_IRF[:,tt-1]
        YTM_paths_IRF[sim_paths_IRF.Ba_path[:,tt-1].==1,tt] .= YTM_path[tt-1]
    end

    # Add time-series data
    for tt=1:(T-1)
        str_temp         = "lk_dev_t$(tt)"
        # Override YTM as outcome variable of interest
        data[!,str_temp] = allowmissing(40000*(YTM_paths_IRF[:,tt] - YTM_paths_SS[:,tt]))
    end

    # Run base regression
    for (ix, x_str_temp) in enumerate(x_col_names[[1]])
        for tt=1:(T-1)
            # Outcome selector
            y_str_temp       = "lk_dev_t$(tt)"
            reg_mod = lm(term(y_str_temp) ~ term(x_str_temp), data)
            gamma_col[tt,ix] = reg_rf_scaler*coef(reg_mod)[2]
            gamma_ci_col[tt,ix,:] = [reg_rf_scaler*coeftable(reg_mod).cols[5][2], reg_rf_scaler*coeftable(reg_mod).cols[6][2]]
        end

        # Plot coefficient alongside data estimates
        if ind_plot & (x_str_temp=="cheat") & (str_end=="_base_pub")
            manl_ftsize = 13
            # Load data
            df_emp_betas= DataFrame(CSV.File(joinpath(opts.project_root, "interim_output", "FigB3b_slevsbetas_vec_q_vals(xintqdt_rat)(cheat_rat)_IV_FE.csv")))
            df_emp_ci   = DataFrame(CSV.File(joinpath(opts.project_root, "interim_output", "FigB3b_slevsbetas_vec_q_cis(xintqdt_rat)(cheat_rat)_IV_FE.csv")))
            emp_betas   = Array(df_emp_betas.V1)
            emp_ci      = zeros(size(emp_betas,1),2)
            emp_ci[:,1], emp_ci[:,2] = df_emp_ci.V1, df_emp_ci.V2
            # Plot
            T_emp_plot = min(length(gamma_col[1:end,ix]), length(emp_betas))
            Plots.plot(0:(T_emp_plot-1), -gamma_col[1:T_emp_plot,ix], xlabel="Quarters (h)", ylabel="Basis points", ylims=[-45.00, 95.00], linecolor=:red, linestyle=:dot, linewidth=3.0, xticks=0:2:T_emp_plot, label="Model", legend=:bottomleft, xtickfontsize=manl_ftsize-1, ytickfontsize=manl_ftsize-1, yguidefontsize=manl_ftsize, xguidefontsize=manl_ftsize, legendfontsize=manl_ftsize-1)
            Plots.plot!(0:(T_emp_plot-1), emp_betas[1:T_emp_plot], linecolor=:blue, linewidth=2.0, label="Data")
            Plots.plot!(0:(T_emp_plot-1), emp_ci[1:T_emp_plot,:], linestyle=[:dash :dash], linecolor=:blue, linewidth=2.0, label=:none)
            Plots.plot!(0:(T_emp_plot-1), zeros(T_emp_plot), linestyle=:dash, linecolor=:black, linewidth=0.5, label=:none)
            savefig(joinpath(opts.project_root, "output", "figures", "model_figures", "FigB3b_anl_regcfs_xint_wemp_"*x_str_temp*str_end*".pdf"))
        end
    end

end


#####
# Functions for aggregation

# Solve for stationary distribution AND aggregates; NOTE: adjusts the modl.psi to the value that is consistent with the normalized steady state wage w_ss, given as argument
# Implementing Tan's (2020) algorithm.
function solve_Leq(fsoln, modl, sspace, opts; w_ss = 1.0)

    # Set output text printing path
    txtout_path = joinpath(opts.project_root, "interim_output", "model_interim_output", "savedout", modl.name * "_output.txt")

    manprintln("Solving for stationary distribution.", txtout_path)
    sf      = sspace.sf
    Nd = size(sf,1)

    # Set up all steady state values of prices that firm takes as given
    MU_ss, M_ss, pQ_ss = 10.0/(10.0-1.0), modl.beta, 1.0
    Pi_ss, rb_ss, rm_ss = modl.Pi, modl.rb, modl.rm
    Rc_ss       = (1.0+modl.rb)/(modl.Pi*pQ_ss)
    gammatil_ss = modl.gamma/modl.Pi

    # Solve firm's problem to be used for the Young (2010) histogram approach to approximating population
    fsoln_f = solve_V0(fsoln.V0ek, fsoln.V0em, fsoln.V0eb, fsoln.V0e, MU_ss, w_ss, M_ss, pQ_ss, Pi_ss, rb_ss, rm_ss, Rc_ss, Rc_ss, gammatil_ss, modl, sspace, opts)

    # Compute stationary distribution
    # Separate "transition matrices" for non-adjusters and adjusters, denoted with "_N" and "_A"
    # Unpack the solution matrices to vector form
    kpN, kpA = reshape(fsoln_f.kpN, Nd), reshape(fsoln_f.kpA, Nd)
    mpN, mpA = reshape(fsoln_f.mpN, Nd), reshape(fsoln_f.mpA, Nd)
    bpN, bpA = reshape(fsoln_f.bpN, Nd), reshape(fsoln_f.bpA, Nd)
    Ba_prob  = reshape(fsoln_f.Ba_prob, Nd)
    Y  = reshape(fsoln_f.Y, Nd)
    n  = reshape(fsoln_f.n, Nd)
    ACN, ACA  = reshape(fsoln_f.ACN, Nd),  reshape(fsoln_f.ACA, Nd)
    # If any extreme states map into NaN for policy objects; warn if so, then replace with -1, for general robustness
    if any(isnan.(kpN)) || any(isnan.(kpA)) || any(isnan.(mpN)) || any(isnan.(mpA)) || any(isnan.(bpN)) || any(isnan.(bpA)) || any(isnan.(Ba_prob))
        manprintln("WARNING: NaN values detected in policy objects before transition-matrix construction; replacing them with -1.0.", txtout_path)
        kpN, kpA = replace(kpN, NaN => -1), replace(kpA, NaN => -1)
        mpN, mpA = replace(mpN, NaN => -1), replace(mpA, NaN => -1)
        bpN, bpA = replace(bpN, NaN => -1), replace(bpA, NaN => -1)
        Ba_prob  = replace(Ba_prob, NaN => -1)
    end

    # Use CompEcon tools to construct transition matrices on discretized space
    fspaceergk  = fundef([:spli, sspace.k_gridf, 0, 1])
    fspaceergm  = fundef([:spli, sspace.m_gridf, 0, 1])
    fspaceergb  = fundef([:spli, sspace.b_gridf, 0, 1])
    QK_N, QK_A  = funbase(fspaceergk, kpN), funbase(fspaceergk, kpA)
    QM_N, QM_A  = funbase(fspaceergm, mpN), funbase(fspaceergm, mpA)
    QB_N, QB_A  = funbase(fspaceergb, bpN), funbase(fspaceergb, bpA)
    QZ          = sspace.QZ

    # Construct standard Q for reference
    Q_N = row_kron(QZ, row_kron(QB_N,row_kron(QM_N,QK_N)))
    Q_A = row_kron(QZ, row_kron(QB_A,row_kron(QM_A,QK_A)))
    Q       = row_kron(reshape(1.0.-Ba_prob,Nd,1), Q_N) + row_kron(sparse(reshape(Ba_prob,Nd,1)), Q_A)

    # Construct product of QB*QM*QK only for endogenous choices!
    Qend_N = row_kron(QB_N,row_kron(QM_N,QK_N))
    Qend_A = row_kron(QB_A,row_kron(QM_A,QK_A))

    # Given the adjustment probabilities "weigh up" into aggregate transition matrices of endogenous choices
    Qend      = row_kron(reshape(1.0.-Ba_prob,Nd,1), Qend_N) + row_kron(sparse(reshape(Ba_prob,Nd,1)), Qend_A)
    # Make this "Qend" into a block matrix Gamma, analogously as in Tan (2020), but adjusting for my different ordering of states
    Gammamat  = spzeros(Nd,Nd)
    Nend      = sspace.Nk*sspace.Nm*sspace.Nb
    for iiz=1:sspace.Nz
        Gammamat[((iiz-1)*Nend+1):(iiz*Nend),((iiz-1)*Nend+1):(iiz*Nend)] = Qend[((iiz-1)*Nend+1):(iiz*Nend),:]
    end

    # Generating the mass of firms entering -- to save on computations, could also do inside "setup!"
    QKe         = funbase(fspaceergk, ones(Nd)*modl.k_0)
    QMe         = funbase(fspaceergm, ones(Nd)*modl.m_0)
    QBe         = funbase(fspaceergb, ones(Nd)*modl.b_0)
    QZe         = ones(Nd) * transpose(modl.z_0_dist)

    # Multiply up into "full" transition matrix
    Qe = row_kron(QZe, row_kron(QBe, row_kron(QMe, QKe)))

    Le          = Qe' * (modl.eta * ones(Nd) / Nd)

    # Initialize "Lbar" ("decision-makers") coming in from previous period (and surviving)
    L           = sparse(copy(Le))
    L           = (1.0-modl.eta)*sparse(L/sum(L))

    # If there exists a saved Lbar, then use it
    if !isempty(modl.Lresult)
        L = modl.Lresult
    end

    for itL = 1:opts.itermaxL
        # Construct incoming incumbents: Q'*L
        Ltil    = Gammamat'*L
        Lamtil  = reshape(Ltil, Nend, sspace.Nz)
        Lincumb = reshape(Lamtil*sspace.P, Nd, 1)
        Lnew    = reshape((1.0-modl.eta)*(Lincumb + Le), Nd)
        dL      = norm(Lnew-L)/norm(L)
        if dL < opts.tolL; manprintln("Convergence of distribution in $itL iterations.", txtout_path); break; end
        if mod(itL,10) == 0
            if opts.prnt == "Y"
                manprintln("iter: \t $itL, dL: \t $dL", txtout_path)
            end
        end
        L       = Lnew
    end
    # The distribution of incumbents
    Lin      = L / (1.0-modl.eta)

    # Create stationary distribution (marginals)
    Lk = kron(ones(1,sspace.nf[4]*sspace.nf[3]*sspace.nf[2]), I(sspace.nf[1])) * Lin
    Lm = kron(ones(1,sspace.nf[4]*sspace.nf[3]), kron(I(sspace.nf[2]),ones(1,sspace.nf[1]))) * Lin
    Lb = kron(ones(1,sspace.nf[4]), kron(I(sspace.nf[3]),ones(1,sspace.nf[1]*sspace.nf[2]))) * Lin
    Lz = kron(I(sspace.nf[4]), ones(1,sspace.nf[1]*sspace.nf[2]*sspace.nf[3])) * Lin

    # Entrants m_0 holdings and market debt b_0 LEVELS:
    m_0     = modl.m_0
    b_0     = modl.b_0 * modl.k_0 * Rc_ss
    # Given converged distribution, compute aggregates
    Ya      = (Lin' * Y)[1]
    Na      = (Lin' * n)[1]
    Ka      = (L' * ( (1.0.-Ba_prob) .* kpN + Ba_prob .* kpA) )[1] + modl.eta * modl.k_0 # Decision-makers in L + entrants # Identical if did: Lin' * sf[:,1]
    bpN_lev, bpA_lev = bpN .* kpN * Rc_ss, bpA .* kpA * Rc_ss
    Ba      = (L' * ( (1.0.-Ba_prob) .* bpN_lev + Ba_prob .* bpA_lev) )[1] + modl.eta * b_0 # Decision-makers in L + entrants
    Ma      = (L' * ( (1.0.-Ba_prob) .* mpN + Ba_prob .* mpA )  )[1] + modl.eta * m_0
    ACa     = (L' * ( (1.0.-Ba_prob) .* ACN + Ba_prob .* ACA )  )[1]
    Ca      = Ya - modl.delta * Ka - ACa
    p_imp   = Ca^(-modl.varpi)

    # Set psi in model so that it aligns with the SS wage imposed
    modl.psi = w_ss * p_imp

    aeq = AggEqSolution(p_imp, L, Ya, Na, Ca, Ka, Ba, Ma, ACa, Q, Qend, Lin, Lk, Lm, Lb, Lz)

    return aeq
end


# Solve for endogenous p (and w) that clear markets, starting from p_guess
function solve_Leq_endp(p_guess, modl,sspace,opts; fsoln_in=Array{Float64}(undef, 0))

    # Set output text printing path
    txtout_path = joinpath(opts.project_root, "interim_output", "model_interim_output", "savedout", modl.name * "_output.txt")

    # Initialize p_cur as p_guess, and the implied w_cur
    p_cur = p_guess;

    flag_pconv  = false

    if opts.prnt =="Y"; manprintln("Iterations for solving GE p:", txtout_path); end
    t_pstart   = time_ns()
    fsoln_cur, aeq_cur = [], [];

    # Start iterations
    for piter=1:opts.itermax_p_col
        # Infer wage from p guess
        w_cur = modl.psi / p_cur;
        # Given current guesses, solve for the SS policy function
        fsoln_cur = solve_V_EGM(modl, sspace, opts, fsoln_in=fsoln_in, w_ss=w_cur);
        # Given this solution solve for the aggregates and the implied p
        aeq_cur = solve_Leq_keeppsi(p_cur, fsoln_cur, modl, sspace, opts);
        p_out = deepcopy(aeq_cur.p);
        # Replace the incoming p_cur in aeq
        aeq_cur.p = p_cur;

        # p distances
        diff_p = abs(p_out-p_cur)/p_cur
        # Check convergence
        if mod(piter,1) == 0; manprintln("$piter\t dp = $diff_p, \t p_in=$p_cur, \t p_out=$(p_out), \t Time: $((time_ns()-t_pstart)/1.0e9)", txtout_path); end
        if (diff_p < opts.tolp); flag_pconv = true; end

        if flag_pconv; manprintln("p iterations complete in steps: $piter, time: $((time_ns()-t_pstart)/1.0e9).", txtout_path); break;  end
        if !flag_pconv && piter==opts.itermax_p_col; manprintln("Finished $piter p iterations without convergence, distance: $diff_p, time: $((time_ns()-t_pstart)/1.0e9).", txtout_path); end

        # Update p guess
        p_cur = p_cur + (1.0-opts.phi_peq)*(p_out-p_cur)
        # Update fsoln guess
        fsoln_in = fsoln_cur;
    end

    return fsoln_cur, aeq_cur
end

# Solve for stationary distribution AND aggregates, conditional on marginal utility p_in, and return the p_imp that markets imply as part of solution.
function solve_Leq_keeppsi(p_in, fsoln, modl, sspace, opts)

    # Set output text printing path
    txtout_path = joinpath(opts.project_root, "interim_output", "model_interim_output", "savedout", modl.name * "_output.txt")

    manprintln("Solving for stationary distribution.", txtout_path)
    sf      = sspace.sf
    Nd = size(sf,1)

    # Infer w_in from p_in given psi
    w_in = modl.psi / p_in

    # Set up all steady state values of prices that firm takes as given
    MU_ss, M_ss, pQ_ss = 10.0/(10.0-1.0), modl.beta, 1.0
    Pi_ss, rb_ss, rm_ss = modl.Pi, modl.rb, modl.rm
    Rc_ss       = (1.0+modl.rb)/(modl.Pi*pQ_ss)
    gammatil_ss = modl.gamma/modl.Pi

    # Solution on fine grid equals solution on "general grid", for now
    fsoln_f = solve_V0(fsoln.V0ek, fsoln.V0em, fsoln.V0eb, fsoln.V0e, MU_ss, w_in, M_ss, pQ_ss, Pi_ss, rb_ss, rm_ss, Rc_ss, Rc_ss, gammatil_ss, modl, sspace, opts)

    # Compute stationary distribution
    # Separate "transition matrices" for non-adjusters and adjusters, denoted with "_N" and "_A"
    # Unpack the solution matrices to vector form
    kpN, kpA = reshape(fsoln_f.kpN, Nd), reshape(fsoln_f.kpA, Nd)
    mpN, mpA = reshape(fsoln_f.mpN, Nd), reshape(fsoln_f.mpA, Nd)
    bpN, bpA = reshape(fsoln_f.bpN, Nd), reshape(fsoln_f.bpA, Nd)
    Ba_prob  = reshape(fsoln_f.Ba_prob, Nd)
    Y  = reshape(fsoln_f.Y, Nd)
    n  = reshape(fsoln_f.n, Nd)
    ACN, ACA  = reshape(fsoln_f.ACN, Nd),  reshape(fsoln_f.ACA, Nd)
    # If any extreme states map into NaN for policy objects; warn if so, then replace with -1, for general robustness
    if any(isnan.(kpN)) || any(isnan.(kpA)) || any(isnan.(mpN)) || any(isnan.(mpA)) || any(isnan.(bpN)) || any(isnan.(bpA)) || any(isnan.(Ba_prob))
        manprintln("WARNING: NaN values detected in policy objects before transition-matrix construction; replacing them with -1.0.", txtout_path)
        kpN, kpA = replace(kpN, NaN => -1), replace(kpA, NaN => -1)
        mpN, mpA = replace(mpN, NaN => -1), replace(mpA, NaN => -1)
        bpN, bpA = replace(bpN, NaN => -1), replace(bpA, NaN => -1)
        Ba_prob  = replace(Ba_prob, NaN => -1)
    end

    # Use CompEcon tools to construct transition matrices on discretized space
    fspaceergk  = fundef([:spli, sspace.k_gridf, 0, 1])
    fspaceergm  = fundef([:spli, sspace.m_gridf, 0, 1])
    fspaceergb  = fundef([:spli, sspace.b_gridf, 0, 1])
    QK_N, QK_A  = funbase(fspaceergk, kpN), funbase(fspaceergk, kpA)
    QM_N, QM_A  = funbase(fspaceergm, mpN), funbase(fspaceergm, mpA)
    QB_N, QB_A  = funbase(fspaceergb, bpN), funbase(fspaceergb, bpA)
    QZ          = sspace.QZ

    # Construct standard Q for reference
    Q_N = row_kron(QZ, row_kron(QB_N,row_kron(QM_N,QK_N)))
    Q_A = row_kron(QZ, row_kron(QB_A,row_kron(QM_A,QK_A)))
    Q       = row_kron(reshape(1.0.-Ba_prob,Nd,1), Q_N) + row_kron(sparse(reshape(Ba_prob,Nd,1)), Q_A)

    # Construct product of QB*QM*QK only for endogenous choices!
    Qend_N = row_kron(QB_N,row_kron(QM_N,QK_N))
    Qend_A = row_kron(QB_A,row_kron(QM_A,QK_A))

    # Given the adjustment probabilities "weigh up" into aggregate transition matrices of endogenous choices
    Qend      = row_kron(reshape(1.0.-Ba_prob,Nd,1), Qend_N) + row_kron(sparse(reshape(Ba_prob,Nd,1)), Qend_A)
    # Make this "Qend" into a block matrix Gamma, analogously as in Tan (2020), but adjusting for my different ordering of states
    Gammamat  = spzeros(Nd,Nd)
    Nend      = sspace.Nk*sspace.Nm*sspace.Nb
    for iiz=1:sspace.Nz
        Gammamat[((iiz-1)*Nend+1):(iiz*Nend),((iiz-1)*Nend+1):(iiz*Nend)] = Qend[((iiz-1)*Nend+1):(iiz*Nend),:]
    end

    # Generating the mass of firms entering
    QKe         = funbase(fspaceergk, ones(Nd)*modl.k_0)
    QMe         = funbase(fspaceergm, ones(Nd)*modl.m_0)
    QBe         = funbase(fspaceergb, ones(Nd)*modl.b_0)
    QZe         = ones(Nd) * transpose(modl.z_0_dist)

    # Multiply up into "full" transition matrix
    Qe = row_kron(QZe, row_kron(QBe, row_kron(QMe, QKe)))

    Le          = Qe' * (modl.eta * ones(Nd) / Nd)

    # Initialize "Lbar" ("decision-makers") coming in from previous period (and surviving)
    L           = sparse(copy(Le))
    L           = (1.0-modl.eta)*sparse(L/sum(L))

    # If there exists a saved Lbar, then use it
    if !isempty(modl.Lresult)
        L = modl.Lresult
    end

    for itL = 1:opts.itermaxL
        # Construct incoming incumbents: Q'*L
        Ltil    = Gammamat'*L
        Lamtil  = reshape(Ltil, Nend, sspace.Nz)
        Lincumb = reshape(Lamtil*sspace.P, Nd, 1)
        Lnew    = reshape((1.0-modl.eta)*(Lincumb + Le), Nd)
        dL      = norm(Lnew-L)/norm(L)
        if dL < opts.tolL; manprintln("Convergence of distribution in $itL iterations.", txtout_path); break; end
        if mod(itL,10) == 0
            if opts.prnt == "Y"
                manprintln("iter: \t $itL, dL: \t $dL", txtout_path)
            end
        end
        L       = Lnew
    end
    # The distribution of incumbents
    Lin      = L / (1.0-modl.eta)

    # Create stationary distribution (marginals)
    Lk = kron(ones(1,sspace.nf[4]*sspace.nf[3]*sspace.nf[2]), I(sspace.nf[1])) * Lin
    Lm = kron(ones(1,sspace.nf[4]*sspace.nf[3]), kron(I(sspace.nf[2]),ones(1,sspace.nf[1]))) * Lin
    Lb = kron(ones(1,sspace.nf[4]), kron(I(sspace.nf[3]),ones(1,sspace.nf[1]*sspace.nf[2]))) * Lin
    Lz = kron(I(sspace.nf[4]), ones(1,sspace.nf[1]*sspace.nf[2]*sspace.nf[3])) * Lin

    # Entrants m_0 holdings and market debt b_0 LEVELS:
    m_0     = modl.m_0
    b_0     = modl.b_0 * modl.k_0 * Rc_ss
    # Given converged distribution, compute aggregates
    Ya      = (Lin' * Y)[1]
    Na      = (Lin' * n)[1]
    Ka      = (L' * ( (1.0.-Ba_prob) .* kpN + Ba_prob .* kpA) )[1] + modl.eta * modl.k_0 # Decision-makers in L + entrants # Identical if did: Lin' * sf[:,1]
    bpN_lev, bpA_lev = bpN .* kpN * Rc_ss, bpA .* kpA * Rc_ss
    Ba      = (L' * ( (1.0.-Ba_prob) .* bpN_lev + Ba_prob .* bpA_lev) )[1] + modl.eta * b_0 # Decision-makers in L + entrants
    Ma      = (L' * ( (1.0.-Ba_prob) .* mpN + Ba_prob .* mpA )  )[1] + modl.eta * m_0
    ACa     = (L' * ( (1.0.-Ba_prob) .* ACN + Ba_prob .* ACA )  )[1]
    Ca      = Ya - modl.delta * Ka - ACa
    p_imp   = Ca^(-modl.varpi)

    # Return the implied p_imp
    aeq = AggEqSolution(p_imp, L, Ya, Na, Ca, Ka, Ba, Ma, ACa, Q, Qend, Lin, Lk, Lm, Lb, Lz)

    return aeq
end


# Function that uses an initial set of incumbents Lin, and the transmission matrix Q to return a distribution L of incumbents who are at least T quarters old
# Implementing Tan's (2020) algorithm.
function gen_L_Tage(T, Lin, Qend, modl, sspace, opts)
    L_cur = Lin
    L_new = copy(Lin)

    # Construct matrices to implement Tan (2020) algorithm
    Nd = size(sspace.sf,1)
    Gammamat  = spzeros(Nd,Nd)
    Nend      = sspace.Nk*sspace.Nm*sspace.Nb
    for iiz=1:sspace.Nz
        Gammamat[((iiz-1)*Nend+1):(iiz*Nend),((iiz-1)*Nend+1):(iiz*Nend)] = Qend[((iiz-1)*Nend+1):(iiz*Nend),:]
    end

    for tt=1:T
        # Construct incoming incumbents: Q'*L
        Ltil    = Gammamat'*L_cur
        Lamtil  = reshape(Ltil, Nend, sspace.Nz)
        L_new   = reshape(Lamtil*sspace.P, Nd)
        L_cur = copy(L_new)
    end

    return L_new
end


####
# Aggregate impulse responses
####

"Solve NK agg economy's IRF, with endogenous inflation, to monetary shock path zeta_col, an (rb-rm) spread shock path spread_col, and transfer shock x_shock_dK, by iterating on inverse markup and capital price (pQ) path. Read T from the entered zeta_col path. The first argument constitutes the measure of the transfer shock, defined as a fraction of the steady state capital."
function solve_eqIRF_NKMg_xsh(x_shock_dK, zeta_col, spread_col, fsoln, aeq, modl, sspace, opts; MU_col_0=Array{Float64}(undef, 0), phi_p_col=Array{Float64}(undef, 0), pQ_col_0=Array{Float64}(undef, 0),  w_ss = 1.0, rfixtaylor_ind=0)

    # Set output text printing path
    txtout_path = joinpath(opts.project_root, "interim_output", "model_interim_output", "savedout", modl.name * "_output.txt")

    # Globals
    sf      = sspace.sf
    Nd = size(sf,1)

    # Set up all steady state values of prices that firm takes as given
    MU_ss, M_ss, pQ_ss = 10.0/(10.0-1.0), modl.beta, 1.0
    Pi_ss, rb_ss, rm_ss = modl.Pi, modl.rb, modl.rm
    rf_ss       = modl.rf
    Rc_ss       = (1.0+modl.rb)/(modl.Pi*pQ_ss)
    gammatil_ss = modl.gamma/modl.Pi

    # To be used in transmitting distributions forward
    fspaceergk  = fundef([:spli, sspace.k_gridf, 0, 1])
    fspaceergm  = fundef([:spli, sspace.m_gridf, 0, 1])
    fspaceergb  = fundef([:spli, sspace.b_gridf, 0, 1])
    QZ          = sspace.QZ

    # Matrices for generating the mass of firms entering
    QKe         = funbase(fspaceergk, ones(Nd)*modl.k_0)
    QMe         = funbase(fspaceergm, ones(Nd)*modl.m_0)
    QBe         = funbase(fspaceergb, ones(Nd)*modl.b_0)
    QZe         = ones(Nd) * transpose(modl.z_0_dist) 

    # Multiply up into "full" transition matrix
    Qe = row_kron(QZe, row_kron(QBe, row_kron(QMe, QKe)))
    # And create entrants' vector
    Le          = Qe' * (modl.eta * ones(Nd) / Nd)

    # Initialize
    T               = length(zeta_col) - 1
    # Guess for price path -- if no other guess is fed as input, guess flat path at SS values
    # Note that I will also include the last entry of p_T as p[T+1]
    if isempty(MU_col_0)
        MU_col_0         = ones(T+1)*MU_ss
    else
        if length(MU_col_0) != length(zeta_col)
            error("MU_col guess must have same length as interest shock series!")
        end
    end
    if length(zeta_col) != length(spread_col)
        error("zeta series must have same length as spread series!")
    end
    # Same for capital price pQ
    if isempty(pQ_col_0)
        pQ_col_0         = ones(T+1)
    else
        if length(pQ_col_0) != length(zeta_col)
            error("pQ_col guess must have same length as interest shock series!")
        end
    end
    # The collection of phi_p-s to use
    if isempty(phi_p_col)
        phi_p_col         = opts.phi_p * ones(opts.itermax_p_col)
    else
        if length(phi_p_col) < opts.itermax_p_col
            error("The collection of phi_p's must have length at least as large as itermax_p_col!")
        end
    end
    # Collections that will contain aggregate equilibrium solution objects along path
    L_col           = zeros(length(aeq.L),T)
    Ya_col          = Array{Float64}(undef, T)
    Na_col          = Array{Float64}(undef, T)
    Ca_col          = Array{Float64}(undef, T)
    Ka_col          = Array{Float64}(undef, T)
    Ba_col          = Array{Float64}(undef, T)
    Ma_col          = Array{Float64}(undef, T)
    ACa_col         = Array{Float64}(undef, T)
    Lin_col         = zeros(length(aeq.L),T+1)
    Lk_col          = Array{Float64}(undef, length(aeq.Lk),T)
    Lm_col          = Array{Float64}(undef, length(aeq.Lm),T)
    Lb_col          = Array{Float64}(undef, length(aeq.Lb),T)
    Lz_col          = Array{Float64}(undef, length(aeq.Lz),T)

    # Price paths
    p_col_cur       = Array{Float64}(undef, T+1)   # The path of p known to hold in equilibrium
    p_col_upd       = Array{Float64}(undef, T+1)   # The implied p path returned by solving at any given iteration
    p_col_cur[T+1] = aeq.p # In period T, back in steady state
    # Current imu set to initial guess
    MU_col_cur   = MU_col_0
    # Same for pQ_col
    pQ_col_cur  = pQ_col_0
    # And create empty updated pQ path
    pQ_col_upd  = similar(pQ_col_cur)
    # Also create empty container for individual firm solution
    irfsoln     = []

    # Create empty Pi path, to be computed in each iteration below, and initialize end value
    Pi_col_cur  = similar(pQ_col_cur)
    Pi_col_cur[end] = Pi_ss
    # And similarly, let us also set up the "_cur" values of rb_col and rm_col, computed in each iteration below
    rf_col_cur  = similar(pQ_col_cur)
    rm_col_cur  = similar(pQ_col_cur)
    rb_col_cur  = similar(pQ_col_cur)
    rf_col_cur[1] = rf_ss

    manprintln("Solving aggregate IRF of length T=$T.", txtout_path)

    t_start = time_ns()
    # Start algorithm, outside loop over the relevant aggregate objects and price paths
    for ii = 1:opts.itermax_p_col
        # 0. Before solving firm problem, iterate on NKM eqm conditions to infer the aggregate paths consistent with MU_col guess and aggregate equilibrium conditions
        for tt=T:-1:1
            # NKPC implies t inflation
            Pi_col_cur[tt] = modl.Pi * exp(-modl.kappa_p*log(MU_col_cur[tt]/MU_ss) + M_ss*log(Pi_col_cur[tt+1]/Pi_ss))
            # Taylor rule implies t+1 nominal rf rate
            rf_col_cur[tt+1] = (1.0+rf_ss)*((Pi_col_cur[tt]/Pi_ss)^modl.phi_pi)*exp(zeta_col[tt]) - 1.0
            # Override Taylor with fixing real rate?
            if rfixtaylor_ind == 1
                rf_col_cur[tt+1] = (1.0+rf_ss)*(Pi_col_cur[tt+1]/Pi_ss)*exp(zeta_col[tt]) - 1.0
            end
            # The Euler equation implies a marginal utility of consumption (p) in t
            p_col_cur[tt] = modl.beta * ((1.0+rf_col_cur[tt+1])/Pi_col_cur[tt+1]) * p_col_cur[tt+1]
        end
        # Finally, the implied rb path is simply the rf path + spread shock
        rb_col_cur[2:end] = rf_col_cur[2:end] + spread_col[1:end-1]
        # And the implied rm path is simply the phi_mresp*(rf path - rf_ss)
        rm_col_cur = rm_ss .+ modl.phi_mresp*(rf_col_cur .- rf_ss)

        # Next, solve for the implied path of bond prices etc.
        q_ss        = 1.0/(1.0+rb_ss-modl.gamma)
        q_col       = zeros(T+1)
        q_col[T+1]  = q_ss
        # Iterate for q path
        for tt=T:-1:1
            q_col[tt] = (modl.gamma*q_col[tt+1]+1.0) / (1.0+rb_col_cur[tt+1])
        end
        # Given the q path, update rb at impact
        rb_col_cur[1] = rb_ss + modl.gamma*(q_col[1]/q_ss-1.0)
        # Given the q path, compute gammatil, not including T value at the end
        gammatil_col    = zeros(T)
        gammatil_col[1] = modl.gamma * (q_col[1]/q_ss)/Pi_col_cur[1] # Initial value needs q_ss
        for tt=2:T
            gammatil_col[tt] = modl.gamma * (q_col[tt]/q_col[tt-1]) / Pi_col_cur[tt]
        end
        # Compute the Rc path
        Rc_col      = zeros(T)
        for tt=1:T
            Rc_col[tt] = (1.0+rb_col_cur[tt+1])/(Pi_col_cur[tt+1]*pQ_col_cur[tt+1])
        end
        # Compute the real SDF M_{t+1} path. NOTE: This refers to M_{t+1} while all other _col variables refer to _{t} at the same time!
        M_tp1_col      = zeros(T)
        for tt=1:T
            M_tp1_col[tt] = Pi_col_cur[tt+1] / (1.0+rf_col_cur[tt+1])
        end
        # Infer the wage path as well from p_col_cur
        w_col_cur     = zeros(T+1)
        for tt=1:(T+1)
            w_col_cur[tt] = modl.psi / p_col_cur[tt]
        end

        # 1. Solve the firm problem given the current price paths
        # NOTE: The solve_IRF(...) function needs all ingoing paths to have the same length! (The firm's problem in the implied "last period" will also be solved, taking into account the t+1 value functions coming from what has been solved in fsoln.)
        irfsoln = solve_IRF(fsoln, MU_col_cur[1:T], w_col_cur[1:T], M_tp1_col, pQ_col_cur[1:T], Pi_col_cur[1:T], rb_col_cur[1:T], rm_col_cur[1:T], Rc_col, gammatil_col, modl, sspace, opts)

        # 2. Looping forward, solve for the implied distributions of (k,m,b,z)
        # Initial distribution of incumbents is simply the steady state one -- because no shock (except a direct transfer) can affect the firms' states (k,m,b,z)
        Lin_col[:,1]    = aeq.Lin
        # Do correct adjustment for incoming distribution, shocking m holdings "coming in from SS" (if there is an x_shock_dK, proportional to aggregate capital)
        if true
            # Solve problem in SS
            fsoln_f_ss = solve_V0(fsoln.V0ek, fsoln.V0em, fsoln.V0eb, fsoln.V0e, MU_ss, w_ss, M_ss, pQ_ss, Pi_ss, rb_ss, rm_ss, Rc_ss, Rc_ss, gammatil_ss, modl, sspace, opts)
            # Unpack the solution matrices to vector form
            kpN_ss, kpA_ss = reshape(fsoln_f_ss.kpN, Nd), reshape(fsoln_f_ss.kpA, Nd)
            mpN_ss, mpA_ss = reshape(fsoln_f_ss.mpN, Nd) .+ aeq.Ka*x_shock_dK, reshape(fsoln_f_ss.mpA, Nd) .+ aeq.Ka*x_shock_dK
            bpN_ss, bpA_ss = reshape(fsoln_f_ss.bpN, Nd), reshape(fsoln_f_ss.bpA, Nd)
            Ba_prob_ss  = reshape(fsoln_f_ss.Ba_prob, Nd)
            # To be used in transmitting distributions forward
            fspaceergk  = fundef([:spli, sspace.k_gridf, 0, 1])
            fspaceergm  = fundef([:spli, sspace.m_gridf, 0, 1])
            fspaceergb  = fundef([:spli, sspace.b_gridf, 0, 1])
            QK_N_ss, QK_A_ss  = funbase(fspaceergk, kpN_ss), funbase(fspaceergk, kpA_ss)
            QM_N_ss, QM_A_ss  = funbase(fspaceergm, mpN_ss), funbase(fspaceergm, mpA_ss)
            QB_N_ss, QB_A_ss  = funbase(fspaceergb, bpN_ss), funbase(fspaceergb, bpA_ss)

            # Construct product of QB*QM*QK only for endogenous choices!
            Qend_N_ss = row_kron(QB_N_ss,row_kron(QM_N_ss,QK_N_ss))
            Qend_A_ss = row_kron(QB_A_ss,row_kron(QM_A_ss,QK_A_ss))
            # Given the adjustment probabilities "weigh up" into aggregate transition matrices of endogenous choices
            Qend_ss   = row_kron(reshape(1.0.-Ba_prob_ss,Nd,1), Qend_N_ss) + row_kron(sparse(reshape(Ba_prob_ss,Nd,1)), Qend_A_ss)
            # Make this "Qend" into a block matrix Gamma, analogously as in Tan (2020), but adjusting for my different ordering of states
            Gammamat_ss  = spzeros(Nd,Nd)
            Nend      = sspace.Nk*sspace.Nm*sspace.Nb
            for iiz=1:sspace.Nz
                Gammamat_ss[((iiz-1)*Nend+1):(iiz*Nend),((iiz-1)*Nend+1):(iiz*Nend)] = Qend_ss[((iiz-1)*Nend+1):(iiz*Nend),:]
            end

            # Propagate from SS forward to construct incoming incumbents, with the "SS incumbents" propagated forward
            Ltil_temp    = Gammamat_ss'*aeq.Lin
            Lamtil_temp  = reshape(Ltil_temp, Nend, sspace.Nz)
            Lincumb_temp = reshape(Lamtil_temp*sspace.P, Nd, 1)
            Lin_col[:,1] = reshape((1.0-modl.eta)*Lincumb_temp + Le, Nd)
        end

        # Start with time iterations forward
        manprintln("Solving aggregate IRF forwards:", txtout_path)
        for tt = 1:T
            manprintln("Period t=$tt.", txtout_path)

            # Firms making decisions going forward from tt (the surviving incumbents):
            L_col[:,tt]     = (1.0-modl.eta)*Lin_col[:,tt]

            # Use the IRF solutions to construct transition matrix to next period
            # Unpack the solution matrices to vector form
            kpN_cur, kpA_cur = reshape(irfsoln.kpN_col[:,:,:,:,tt], Nd), reshape(irfsoln.kpA_col[:,:,:,:,tt], Nd)
            mpN_cur, mpA_cur = reshape(irfsoln.mpN_col[:,:,:,:,tt], Nd), reshape(irfsoln.mpA_col[:,:,:,:,tt], Nd)
            bpN_cur, bpA_cur = reshape(irfsoln.bpN_col[:,:,:,:,tt], Nd), reshape(irfsoln.bpA_col[:,:,:,:,tt], Nd)
            Ba_prob_cur  = reshape(irfsoln.Ba_prob_col[:,:,:,:,tt], Nd)
            fspaceergk  = fundef([:spli, sspace.k_gridf, 0, 1])
            fspaceergm  = fundef([:spli, sspace.m_gridf, 0, 1])
            fspaceergb  = fundef([:spli, sspace.b_gridf, 0, 1])
            QK_N_cur, QK_A_cur  = funbase(fspaceergk, kpN_cur), funbase(fspaceergk, kpA_cur)
            QM_N_cur, QM_A_cur  = funbase(fspaceergm, mpN_cur), funbase(fspaceergm, mpA_cur)
            QB_N_cur, QB_A_cur  = funbase(fspaceergb, bpN_cur), funbase(fspaceergb, bpA_cur)

            # Construct product of QB*QM*QK only for endogenous choices
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
            Ltil_cur     = Gammamat_cur'*Lin_col[:,tt]
            Lamtil_cur   = reshape(Ltil_cur, Nend, sspace.Nz)
            Lincumb_cur  = reshape(Lamtil_cur*sspace.P, Nd, 1)
            Lin_col[:,tt+1] = reshape((1.0-modl.eta)*Lincumb_cur + Le, Nd)

            # Create the implied marginals
            Lk_col[:,tt] = kron(ones(1,sspace.nf[4]*sspace.nf[3]*sspace.nf[2]), I(sspace.nf[1])) * Lin_col[:,tt]
            Lm_col[:,tt] = kron(ones(1,sspace.nf[4]*sspace.nf[3]), kron(I(sspace.nf[2]),ones(1,sspace.nf[1]))) * Lin_col[:,tt]
            Lb_col[:,tt] = kron(ones(1,sspace.nf[4]), kron(I(sspace.nf[3]),ones(1,sspace.nf[1]*sspace.nf[2]))) * Lin_col[:,tt]
            Lz_col[:,tt] = kron(I(sspace.nf[4]), ones(1,sspace.nf[1]*sspace.nf[2]*sspace.nf[3])) * Lin_col[:,tt]

            # Computing the aggregates
            Ya_col[tt]  = Lin_col[:,tt]' * reshape(irfsoln.Y_col[:,:,:,:,tt], Nd)
            Na_col[tt]  = Lin_col[:,tt]' * reshape(irfsoln.n_col[:,:,:,:,tt], Nd)
            Ka_col[tt]  = Lin_col[:,tt]' * sf[:,1] # Incoming capital
            Kp_cur      = L_col[:,tt]' * ( (1.0.-Ba_prob_cur) .* kpN_cur + Ba_prob_cur .* kpA_cur    ) + modl.eta * modl.k_0
            bpN_lev_cur, bpA_lev_cur = bpN_cur .* kpN_cur * Rc_col[tt], bpA_cur .* kpA_cur * Rc_col[tt]
            Ba_col[tt]  = L_col[:,tt]' * ( (1.0.-Ba_prob_cur) .* bpN_lev_cur + Ba_prob_cur .* bpA_lev_cur ) + modl.eta * modl.k_0 * modl.b_0
            Ma_col[tt]  = L_col[:,tt]' * ( (1.0.-Ba_prob_cur) .* mpN_cur + Ba_prob_cur .* mpA_cur    ) + modl.eta * modl.m_0
            ACa_col[tt] = L_col[:,tt]' * ( (1.0.-Ba_prob_cur).*reshape(irfsoln.ACN_col[:,:,:,:,tt], Nd)  + Ba_prob_cur .*reshape(irfsoln.ACA_col[:,:,:,:,tt], Nd)     )

            # Compute implied I/K aggregate adjustment costs:
            IdK_cost_cur = (((1.0-modl.phi_k)/(modl.delta^modl.phi_k))*(Kp_cur/Ka_col[tt]-(1.0-modl.delta) + modl.delta*modl.phi_k/(1.0-modl.phi_k)))^(1.0/(1.0-modl.phi_k))

            # Implied aggregate consumption by market clearing, and the outcoming price paths
            Ca_col[tt]    = Ya_col[tt] - IdK_cost_cur * Ka_col[tt] - ACa_col[tt]
            p_col_upd[tt]  = Ca_col[tt]^(-modl.varpi)
            pQ_col_upd[tt] = (IdK_cost_cur/modl.delta)^modl.phi_k
        end

        # Impose last entries of outcoming price paths to SS
        p_col_upd[end]      = aeq.p
        pQ_col_upd[end]     = 1.0

        # Infer the "updated wage path"
        w_col_upd           = modl.psi ./ p_col_upd
        # Given the updated outcoming wage path (relative to incoming wage path), update the markup path
        MU_col_upd          = MU_col_cur .* (w_col_upd./w_col_cur)

        # Check distance in MU vector, implicitly checking whether (w_col_upd, w_col_cur) are close, equivalently, whether (p_col_upd, p_col_cur) are close and the goods-market clearing consumption path is consistent with the initially assumed price paths
        dMU_col      = maximum(abs.(log.(MU_col_upd) - log.(MU_col_cur)))
        if opts.prnt == "Y"
            manprintln("$ii. dMU_col: \t $(dMU_col) t: \t $((time_ns()-t_start)/1.0e9)", txtout_path)
        end
        # Check distance in pQ vector
        dpQ_col     = maximum(abs.(log.(pQ_col_upd) - log.(pQ_col_cur)))
        if opts.prnt == "Y"
            manprintln("$ii. dpQ_col: \t $(dpQ_col) t: \t $((time_ns()-t_start)/1.0e9)", txtout_path)
        end

        # Check if converged
        if (dMU_col < opts.tol_p_col) & (dpQ_col < opts.tol_p_col); break; end
        # Otherwise, repeat above with updated weighted price
        phi_p_cur = phi_p_col[ii]
        # And compute updating weight for pQ
        phi_pQ_cur = 1.0-opts.phi_pQ_acl + opts.phi_pQ_acl*phi_p_cur

        # Update the relevant guesses
        MU_col_cur  = MU_col_cur .* ((w_col_upd./w_col_cur).^(1.0-phi_p_cur))
        pQ_col_cur  = exp.(phi_pQ_cur * log.(pQ_col_cur) + (1.0-phi_pQ_cur) * log.(pQ_col_upd))

    end

    # Pack everything up
    airf = AggIRFSolution(p_col_cur, rf_col_cur, rm_col_cur, rb_col_cur, Pi_col_cur, MU_col_cur, pQ_col_cur, L_col, Ya_col, Na_col, Ca_col, Ka_col, Ba_col, Ma_col, ACa_col, Lin_col, Lk_col, Lm_col, Lb_col, Lz_col)

    return airf, irfsoln
end

"Solve NK agg economy's IRF, with endogenous inflation, to monetary shock path zeta_col, an (rb-rm) spread shock path spread_col, an rm path shock rmsh_col, and transfer shock x_shock_dK, by iterating on inverse markup path. Read T from the entered zeta_col path. The first argument constitutes the measure of the transfer shock, defined as a fraction of the steady state capital."
function solve_eqIRF_NKMg_xsh_rmsh(x_shock_dK, zeta_col, spread_col, rmsh_col, fsoln, aeq, modl, sspace, opts; MU_col_0=Array{Float64}(undef, 0), phi_p_col=Array{Float64}(undef, 0), pQ_col_0=Array{Float64}(undef, 0),  w_ss = 1.0, rfixtaylor_ind=0)

    # Set output text printing path
    txtout_path = joinpath(opts.project_root, "interim_output", "model_interim_output", "savedout", modl.name * "_output.txt")

    # Globals
    sf      = sspace.sf
    Nd = size(sf,1)

    # Set up all steady state values of prices that firm takes as given
    MU_ss, M_ss, pQ_ss = 10.0/(10.0-1.0), modl.beta, 1.0
    Pi_ss, rb_ss, rm_ss = modl.Pi, modl.rb, modl.rm
    rf_ss       = modl.rf
    Rc_ss       = (1.0+modl.rb)/(modl.Pi*pQ_ss)
    gammatil_ss = modl.gamma/modl.Pi

    # To be used in transmitting distributions forward
    fspaceergk  = fundef([:spli, sspace.k_gridf, 0, 1])
    fspaceergm  = fundef([:spli, sspace.m_gridf, 0, 1])
    fspaceergb  = fundef([:spli, sspace.b_gridf, 0, 1])
    QZ          = sspace.QZ

    # Matrices for generating the mass of firms entering
    QKe         = funbase(fspaceergk, ones(Nd)*modl.k_0)
    QMe         = funbase(fspaceergm, ones(Nd)*modl.m_0)
    QBe         = funbase(fspaceergb, ones(Nd)*modl.b_0)
    QZe         = ones(Nd) * transpose(modl.z_0_dist) 

    # Multiply up into "full" transition matrix
    Qe = row_kron(QZe, row_kron(QBe, row_kron(QMe, QKe)))
    # And create entrants' vector
    Le          = Qe' * (modl.eta * ones(Nd) / Nd)

    # Initialize
    T               = length(zeta_col) - 1
    # Guess for price path -- if no other guess is fed as input, guess flat path at SS values
    # Note that I will also include the last entry of p_T as p[T+1]
    if isempty(MU_col_0)
        MU_col_0         = ones(T+1)*MU_ss
    else
        if length(MU_col_0) != length(zeta_col)
            error("MU_col guess must have same length as interest shock series!")
        end
    end
    if length(zeta_col) != length(spread_col)
        error("zeta series must have same length as spread series!")
    end
    # Same for capital price pQ
    if isempty(pQ_col_0)
        pQ_col_0         = ones(T+1)
    else
        if length(pQ_col_0) != length(zeta_col)
            error("pQ_col guess must have same length as interest shock series!")
        end
    end
    # The collection of phi_p-s to use
    if isempty(phi_p_col)
        phi_p_col         = opts.phi_p * ones(opts.itermax_p_col)
    else
        if length(phi_p_col) < opts.itermax_p_col
            error("The collection of phi_p's must have length at least as large as itermax_p_col!")
        end
    end
    # Collections that will contain aggregate equilibrium solution objects along path
    L_col           = zeros(length(aeq.L),T)
    Ya_col          = Array{Float64}(undef, T)
    Na_col          = Array{Float64}(undef, T)
    Ca_col          = Array{Float64}(undef, T)
    Ka_col          = Array{Float64}(undef, T)
    Ba_col          = Array{Float64}(undef, T)
    Ma_col          = Array{Float64}(undef, T)
    ACa_col         = Array{Float64}(undef, T)
    Lin_col         = zeros(length(aeq.L),T+1)
    Lk_col          = Array{Float64}(undef, length(aeq.Lk),T)
    Lm_col          = Array{Float64}(undef, length(aeq.Lm),T)
    Lb_col          = Array{Float64}(undef, length(aeq.Lb),T)
    Lz_col          = Array{Float64}(undef, length(aeq.Lz),T)

    # Price paths
    p_col_cur       = Array{Float64}(undef, T+1)   # The path of p known to hold in equilibrium
    p_col_upd       = Array{Float64}(undef, T+1)   # The implied p path returned by solving at any given iteration
    p_col_cur[T+1] = aeq.p # In period T, back in steady state
    # Current imu set to initial guess
    MU_col_cur   = MU_col_0
    # Same for pQ_col
    pQ_col_cur  = pQ_col_0
    # And create empty updated pQ path
    pQ_col_upd  = similar(pQ_col_cur)
    # Also create empty container for individual firm solution
    irfsoln     = []

    # Create empty Pi path, to be computed in each iteration below, and initialize end value
    Pi_col_cur  = similar(pQ_col_cur)
    Pi_col_cur[end] = Pi_ss
    # And similarly, let us also set up the "_cur" values of rb_col and rm_col, computed in each iteration below
    rf_col_cur  = similar(pQ_col_cur)
    rm_col_cur  = similar(pQ_col_cur)
    rb_col_cur  = similar(pQ_col_cur)
    rf_col_cur[1] = rf_ss

    manprintln("Solving aggregate IRF of length T=$T.", txtout_path)

    t_start = time_ns()
    # Start algorithm, outside loop over the relevant aggregate objects and price paths
    for ii = 1:opts.itermax_p_col
        # 0. Before solving firm problem, iterate on NKM eqm conditions to infer the aggregate paths consistent with MU_col guess and aggregate equilibrium conditions
        for tt=T:-1:1
            # NKPC implies t inflation
            Pi_col_cur[tt] = modl.Pi * exp(-modl.kappa_p*log(MU_col_cur[tt]/MU_ss) + M_ss*log(Pi_col_cur[tt+1]/Pi_ss))
            # Taylor rule implies t+1 nominal rf rate
            rf_col_cur[tt+1] = (1.0+rf_ss)*((Pi_col_cur[tt]/Pi_ss)^modl.phi_pi)*exp(zeta_col[tt]) - 1.0
            # Override Taylor with fixing real rate?
            if rfixtaylor_ind == 1
                rf_col_cur[tt+1] = (1.0+rf_ss)*(Pi_col_cur[tt+1]/Pi_ss)*exp(zeta_col[tt]) - 1.0
            end
            # The Euler equation implies a marginal utility of consumption (p) in t
            p_col_cur[tt] = modl.beta * ((1.0+rf_col_cur[tt+1])/Pi_col_cur[tt+1]) * p_col_cur[tt+1]
        end
        # Finally, the implied rb path is simply the rf path + spread shock
        rb_col_cur[2:end] = rf_col_cur[2:end] + spread_col[1:end-1]
        # And the implied rm path is simply the phi_mresp*(rf path - rf_ss) + rmsh
        rm_col_cur = rm_ss .+ modl.phi_mresp*(rf_col_cur .- rf_ss) + vcat(0, rmsh_col[1:end-1])

        # Next, solve for the implied path of bond prices etc.
        q_ss        = 1.0/(1.0+rb_ss-modl.gamma)
        q_col       = zeros(T+1)
        q_col[T+1]  = q_ss
        # Iterate for q path
        for tt=T:-1:1
            q_col[tt] = (modl.gamma*q_col[tt+1]+1.0) / (1.0+rb_col_cur[tt+1])
        end
        # Given the q path, update rb at impact
        rb_col_cur[1] = rb_ss + modl.gamma*(q_col[1]/q_ss-1.0)
        # Given the q path, compute gammatil, not including T value at the end
        gammatil_col    = zeros(T)
        gammatil_col[1] = modl.gamma * (q_col[1]/q_ss)/Pi_col_cur[1] # Initial value needs q_ss
        for tt=2:T
            gammatil_col[tt] = modl.gamma * (q_col[tt]/q_col[tt-1]) / Pi_col_cur[tt]
        end
        # Compute the Rc path
        Rc_col      = zeros(T)
        for tt=1:T
            Rc_col[tt] = (1.0+rb_col_cur[tt+1])/(Pi_col_cur[tt+1]*pQ_col_cur[tt+1])
        end
        # Compute the real SDF M_{t+1} path. NOTE: This refers to M_{t+1} while all other _col variables refer to _{t} at the same time!!
        M_tp1_col      = zeros(T)
        for tt=1:T
            M_tp1_col[tt] = Pi_col_cur[tt+1] / (1.0+rf_col_cur[tt+1])
        end
        # Infer the wage path as well from p_col_cur
        w_col_cur     = zeros(T+1)
        for tt=1:(T+1)
            w_col_cur[tt] = modl.psi / p_col_cur[tt]
        end

        # 1. Solve the firm problem given the current price paths
        # NOTE: The solve_IRF(...) function needs all ingoing paths to have the same length! (The firm's problem in the implied "last period" will also be solved, taking into account the t+1 value functions coming from what has been solved in fsoln.)
        irfsoln = solve_IRF(fsoln, MU_col_cur[1:T], w_col_cur[1:T], M_tp1_col, pQ_col_cur[1:T], Pi_col_cur[1:T], rb_col_cur[1:T], rm_col_cur[1:T], Rc_col, gammatil_col, modl, sspace, opts)

        # 2. Looping forward, solve for the implied distributions of (k,m,b,z)
        # Initial distribution of incumbents is simply the steady state one -- because no shock (except a direct transfer) can affect the firms' states (k,m,b,z)
        Lin_col[:,1]    = aeq.Lin
        # Do correct adjustment for incoming distribution, shocking m holdings "coming in from SS"
        if true
            # Solve problem in SS
            fsoln_f_ss = solve_V0(fsoln.V0ek, fsoln.V0em, fsoln.V0eb, fsoln.V0e, MU_ss, w_ss, M_ss, pQ_ss, Pi_ss, rb_ss, rm_ss, Rc_ss, Rc_ss, gammatil_ss, modl, sspace, opts)
            # Unpack the solution matrices to vector form
            kpN_ss, kpA_ss = reshape(fsoln_f_ss.kpN, Nd), reshape(fsoln_f_ss.kpA, Nd)
            mpN_ss, mpA_ss = reshape(fsoln_f_ss.mpN, Nd) .+ aeq.Ka*x_shock_dK, reshape(fsoln_f_ss.mpA, Nd) .+ aeq.Ka*x_shock_dK
            bpN_ss, bpA_ss = reshape(fsoln_f_ss.bpN, Nd), reshape(fsoln_f_ss.bpA, Nd)
            Ba_prob_ss  = reshape(fsoln_f_ss.Ba_prob, Nd)
            # To be used in transmitting distributions forward
            fspaceergk  = fundef([:spli, sspace.k_gridf, 0, 1])
            fspaceergm  = fundef([:spli, sspace.m_gridf, 0, 1])
            fspaceergb  = fundef([:spli, sspace.b_gridf, 0, 1])
            QK_N_ss, QK_A_ss  = funbase(fspaceergk, kpN_ss), funbase(fspaceergk, kpA_ss)
            QM_N_ss, QM_A_ss  = funbase(fspaceergm, mpN_ss), funbase(fspaceergm, mpA_ss)
            QB_N_ss, QB_A_ss  = funbase(fspaceergb, bpN_ss), funbase(fspaceergb, bpA_ss)
            # QZ          = sspace.QZ

            # Construct product of QB*QM*QK only for endogenous choices!
            Qend_N_ss = row_kron(QB_N_ss,row_kron(QM_N_ss,QK_N_ss))
            Qend_A_ss = row_kron(QB_A_ss,row_kron(QM_A_ss,QK_A_ss))
            # Given the adjustment probabilities "weigh up" into aggregate transition matrices of endogenous choices
            Qend_ss   = row_kron(reshape(1.0.-Ba_prob_ss,Nd,1), Qend_N_ss) + row_kron(sparse(reshape(Ba_prob_ss,Nd,1)), Qend_A_ss)
            # Make this "Qend" into a block matrix Gamma, analogously as in Tan (2020), but adjusting for my different ordering of states
            Gammamat_ss  = spzeros(Nd,Nd)
            Nend      = sspace.Nk*sspace.Nm*sspace.Nb
            for iiz=1:sspace.Nz
                Gammamat_ss[((iiz-1)*Nend+1):(iiz*Nend),((iiz-1)*Nend+1):(iiz*Nend)] = Qend_ss[((iiz-1)*Nend+1):(iiz*Nend),:]
            end

            # Propagate from SS forward to construct incoming incumbents, with the "SS incumbents" propagated forward
            Ltil_temp    = Gammamat_ss'*aeq.Lin
            Lamtil_temp  = reshape(Ltil_temp, Nend, sspace.Nz)
            Lincumb_temp = reshape(Lamtil_temp*sspace.P, Nd, 1)
            Lin_col[:,1] = reshape((1.0-modl.eta)*Lincumb_temp + Le, Nd)
        end

        # Start with time iterations forward
        manprintln("Solving aggregate IRF forwards:", txtout_path)
        for tt = 1:T
            manprintln("Period t=$tt.", txtout_path)

            # Firms making decisions going forward from tt (the surviving incumbents):
            L_col[:,tt]     = (1.0-modl.eta)*Lin_col[:,tt]

            # Use the IRF solutions to construct transition matrix to next period
            # Unpack the solution matrices to vector form
            kpN_cur, kpA_cur = reshape(irfsoln.kpN_col[:,:,:,:,tt], Nd), reshape(irfsoln.kpA_col[:,:,:,:,tt], Nd)
            mpN_cur, mpA_cur = reshape(irfsoln.mpN_col[:,:,:,:,tt], Nd), reshape(irfsoln.mpA_col[:,:,:,:,tt], Nd)
            bpN_cur, bpA_cur = reshape(irfsoln.bpN_col[:,:,:,:,tt], Nd), reshape(irfsoln.bpA_col[:,:,:,:,tt], Nd)
            Ba_prob_cur  = reshape(irfsoln.Ba_prob_col[:,:,:,:,tt], Nd)
            # To be used in transmitting distributions forward
            fspaceergk  = fundef([:spli, sspace.k_gridf, 0, 1])
            fspaceergm  = fundef([:spli, sspace.m_gridf, 0, 1])
            fspaceergb  = fundef([:spli, sspace.b_gridf, 0, 1])
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
            Ltil_cur     = Gammamat_cur'*Lin_col[:,tt]
            Lamtil_cur   = reshape(Ltil_cur, Nend, sspace.Nz)
            Lincumb_cur  = reshape(Lamtil_cur*sspace.P, Nd, 1)
            Lin_col[:,tt+1] = reshape((1.0-modl.eta)*Lincumb_cur + Le, Nd)

            # Create the implied marginals
            Lk_col[:,tt] = kron(ones(1,sspace.nf[4]*sspace.nf[3]*sspace.nf[2]), I(sspace.nf[1])) * Lin_col[:,tt]
            Lm_col[:,tt] = kron(ones(1,sspace.nf[4]*sspace.nf[3]), kron(I(sspace.nf[2]),ones(1,sspace.nf[1]))) * Lin_col[:,tt]
            Lb_col[:,tt] = kron(ones(1,sspace.nf[4]), kron(I(sspace.nf[3]),ones(1,sspace.nf[1]*sspace.nf[2]))) * Lin_col[:,tt]
            Lz_col[:,tt] = kron(I(sspace.nf[4]), ones(1,sspace.nf[1]*sspace.nf[2]*sspace.nf[3])) * Lin_col[:,tt]

            # Computing the aggregates
            Ya_col[tt]  = Lin_col[:,tt]' * reshape(irfsoln.Y_col[:,:,:,:,tt], Nd)
            Na_col[tt]  = Lin_col[:,tt]' * reshape(irfsoln.n_col[:,:,:,:,tt], Nd)
            Ka_col[tt]  = Lin_col[:,tt]' * sf[:,1] # Incoming capital
            Kp_cur      = L_col[:,tt]' * ( (1.0.-Ba_prob_cur) .* kpN_cur + Ba_prob_cur .* kpA_cur    ) + modl.eta * modl.k_0
            bpN_lev_cur, bpA_lev_cur = bpN_cur .* kpN_cur * Rc_col[tt], bpA_cur .* kpA_cur * Rc_col[tt]
            Ba_col[tt]  = L_col[:,tt]' * ( (1.0.-Ba_prob_cur) .* bpN_lev_cur + Ba_prob_cur .* bpA_lev_cur ) + modl.eta * modl.k_0 * modl.b_0
            Ma_col[tt]  = L_col[:,tt]' * ( (1.0.-Ba_prob_cur) .* mpN_cur + Ba_prob_cur .* mpA_cur    ) + modl.eta * modl.m_0
            ACa_col[tt] = L_col[:,tt]' * ( (1.0.-Ba_prob_cur).*reshape(irfsoln.ACN_col[:,:,:,:,tt], Nd)  + Ba_prob_cur .*reshape(irfsoln.ACA_col[:,:,:,:,tt], Nd)     )

            # Compute implied I/K aggregate adjustment costs:
            IdK_cost_cur = (((1.0-modl.phi_k)/(modl.delta^modl.phi_k))*(Kp_cur/Ka_col[tt]-(1.0-modl.delta) + modl.delta*modl.phi_k/(1.0-modl.phi_k)))^(1.0/(1.0-modl.phi_k))

            # Implied aggregate consumption by market clearing, and the outcoming price paths
            Ca_col[tt]    = Ya_col[tt] - IdK_cost_cur * Ka_col[tt] - ACa_col[tt]
            p_col_upd[tt]  = Ca_col[tt]^(-modl.varpi)
            pQ_col_upd[tt] = (IdK_cost_cur/modl.delta)^modl.phi_k
        end

        # Impose last entries of outcoming price paths to SS
        p_col_upd[end]      = aeq.p
        pQ_col_upd[end]     = 1.0

        # Infer the "updated wage path"
        w_col_upd           = modl.psi ./ p_col_upd
        # Given the updated outcoming wage path (relative to incoming wage path), update the markup path
        MU_col_upd          = MU_col_cur .* (w_col_upd./w_col_cur)

        # Check distance in MU vector, implicitly checking whether (w_col_upd, w_col_cur) are close, equivalently, whether (p_col_upd, p_col_cur) are close and the goods-market clearing consumption path is consistent with the initially assumed price paths
        dMU_col      = maximum(abs.(log.(MU_col_upd) - log.(MU_col_cur)))
        if opts.prnt == "Y"
            #@printf "%d. dp_col:\t %.6f \t t: %.4f \n" ii dp_col ((time_ns()-t_start)/1.0e9)
            manprintln("$ii. dMU_col: \t $(dMU_col) t: \t $((time_ns()-t_start)/1.0e9)", txtout_path)
        end

        # Check distance in pQ vector -- COPY EXACTLY as for MU
        dpQ_col     = maximum(abs.(log.(pQ_col_upd) - log.(pQ_col_cur)))
        if opts.prnt == "Y"
            #@printf "%d. dp_col:\t %.6f \t t: %.4f \n" ii dp_col ((time_ns()-t_start)/1.0e9)
            manprintln("$ii. dpQ_col: \t $(dpQ_col) t: \t $((time_ns()-t_start)/1.0e9)", txtout_path)
        end

        # Check if converged
        if (dMU_col < opts.tol_p_col) & (dpQ_col < opts.tol_p_col); break; end
        # Otherwise, repeat above with updated weighted price
        phi_p_cur = phi_p_col[ii]
        # And compute updating weight for pQ
        phi_pQ_cur = 1.0-opts.phi_pQ_acl + opts.phi_pQ_acl*phi_p_cur

        # Update the relevant guesses
        MU_col_cur  = MU_col_cur .* ((w_col_upd./w_col_cur).^(1.0-phi_p_cur))
        pQ_col_cur  = exp.(phi_pQ_cur * log.(pQ_col_cur) + (1.0-phi_pQ_cur) * log.(pQ_col_upd))

    end

    # Pack everything up
    airf = AggIRFSolution(p_col_cur, rf_col_cur, rm_col_cur, rb_col_cur, Pi_col_cur, MU_col_cur, pQ_col_cur, L_col, Ya_col, Na_col, Ca_col, Ka_col, Ba_col, Ma_col, ACa_col, Lin_col, Lk_col, Lm_col, Lb_col, Lz_col)

    return airf, irfsoln
end



"Decompose the responses to a given agg economy's IRF, by feeding in the AggIRFSolution struct (airf_in), and solving the firm's problem given certain price paths coming from it, while other price paths are kept at steady state values. Admissible values for channel_selector = (Q,rbtaureal,rmtaureal,rsPiQ,M,MU)"
function solve_eqIRF_decomp(channel_selector, airf_in, fsoln, aeq, modl, sspace, opts)

    # Set output text printing path
    txtout_path = joinpath(opts.project_root, "interim_output", "model_interim_output", "savedout", modl.name * "_output.txt")

    # Globals
    sf      = sspace.sf
    Nd = size(sf,1)

    # Set up all steady state values of prices that firm takes as given
    MU_ss, M_ss, pQ_ss = 10.0/(10.0-1.0), modl.beta, 1.0
    Pi_ss, rb_ss, rm_ss = modl.Pi, modl.rb, modl.rm
    rf_ss       = modl.rf
    Rc_ss       = (1.0+modl.rb)/(modl.Pi*pQ_ss)
    gammatil_ss = modl.gamma/modl.Pi

    # To be used in transmitting distributions forward
    fspaceergk  = fundef([:spli, sspace.k_gridf, 0, 1])
    fspaceergm  = fundef([:spli, sspace.m_gridf, 0, 1])
    fspaceergb  = fundef([:spli, sspace.b_gridf, 0, 1])

    # Matrices for generating the mass of firms entering -- to save on computations, could also do inside "setup!"
    QKe         = funbase(fspaceergk, ones(Nd)*modl.k_0)
    QMe         = funbase(fspaceergm, ones(Nd)*modl.m_0)
    QBe         = funbase(fspaceergb, ones(Nd)*modl.b_0)
    QZe         = ones(Nd) * transpose(modl.z_0_dist) 

    # Multiply up into "full" transition matrix
    Qe = row_kron(QZe, row_kron(QBe, row_kron(QMe, QKe)))
    # And create entrants' vector
    Le          = Qe' * (modl.eta * ones(Nd) / Nd)

    # Initialize
    T               = length(airf_in.rb_col) - 1
    # Initialize "steady state paths" for all price paths -- useful to refer to later when solving individual problem
    MU_col_ss         = ones(T+1) * MU_ss
    pQ_col_ss         = ones(T+1)
    w_col_ss          = ones(T+1) * modl.psi / aeq.p
    M_tp1_col_ss      = ones(T) * M_ss
    Pi_col_ss         = ones(T+1) * Pi_ss
    rb_col_ss         = ones(T+1) * rb_ss
    rm_col_ss         = ones(T+1) * rm_ss
    Rc_col_ss         = ones(T) * Rc_ss
    gammatil_col_ss   = ones(T) * gammatil_ss
    # Not used for any calculations, but just create as placekeepers:
    p_col_ss          = ones(T+1) * aeq.p
    rf_col_ss         = ones(T+1) * modl.rf
    # And initialize collectors for "_cur" paths actually applied
    MU_col_cur, pQ_col_cur, w_col_cur, M_tp1_col_cur, Pi_col_cur, rb_col_cur, rm_col_cur, Rc_col_cur, gammatil_col_cur = Array{Float64}(undef, T+1), Array{Float64}(undef, T+1), Array{Float64}(undef, T+1), Array{Float64}(undef, T), Array{Float64}(undef, T+1), Array{Float64}(undef, T+1), Array{Float64}(undef, T+1), Array{Float64}(undef, T), Array{Float64}(undef, T)
    p_col_cur, rf_col_cur = Array{Float64}(undef, T+1), Array{Float64}(undef, T+1)

    if channel_selector == "Q"
        # Only channel is capital price Q
        MU_col_cur, w_col_cur, M_tp1_col_cur, Pi_col_cur, rb_col_cur, rm_col_cur, gammatil_col_cur = MU_col_ss, w_col_ss, M_tp1_col_ss, Pi_col_ss, rb_col_ss, rm_col_ss, gammatil_col_ss
        p_col_cur  = p_col_ss
        rf_col_cur = rf_col_ss

        # Insert correct values for pQ and Rc
        pQ_col_cur = airf_in.pQ_col
        # Compute the Rc path
        for tt=1:T
            Rc_col_cur[tt] = (1.0+rb_col_cur[tt+1])/(Pi_col_cur[tt+1]*pQ_col_cur[tt+1])
        end
    elseif channel_selector == "M"
        # Only channel is SDF M
        MU_col_cur, w_col_cur, pQ_col_cur, Pi_col_cur, rb_col_cur, rm_col_cur, gammatil_col_cur = MU_col_ss, w_col_ss, pQ_col_ss, Pi_col_ss, rb_col_ss, rm_col_ss, gammatil_col_ss
        p_col_cur  = p_col_ss
        rf_col_cur = rf_col_ss

        # Compute correct path for M_tp1, as in baseline GE methods
        M_tp1_col_cur = zeros(T)
        for tt=1:T
            M_tp1_col_cur[tt] = airf_in.Pi_col[tt+1] / (1.0+airf_in.rf_col[tt+1])
        end

        # Compute the Rc path
        for tt=1:T
            Rc_col_cur[tt] = (1.0+rb_col_cur[tt+1])/(Pi_col_cur[tt+1]*pQ_col_cur[tt+1])
        end
    elseif channel_selector == "rsPiQ"
        # Interest rates + inflation + capital price channel: rm, rb, rf, Pi, Q
        MU_col_cur, w_col_cur, M_tp1_col_cur = MU_col_ss, w_col_ss, M_tp1_col_ss
        p_col_cur  = p_col_ss

        # Insert correct values for rm, rb, rf
        rm_col_cur = airf_in.rm_col
        rb_col_cur = airf_in.rb_col
        rf_col_cur = airf_in.rf_col
        # Insert correct values for Pi
        Pi_col_cur = airf_in.Pi_col
        # Insert correct values for pQ
        pQ_col_cur = airf_in.pQ_col

        # Solve for the implied path of bond prices etc.
        q_ss        = 1.0/(1.0+rb_ss-modl.gamma)
        q_col       = zeros(T+1)
        q_col[T+1]  = q_ss
        # Iterate for q path
        for tt=T:-1:1
            q_col[tt] = (modl.gamma*q_col[tt+1]+1.0) / (1.0+rb_col_cur[tt+1])
        end
        # Given the q path, compute gammatil, not including T value at the end
        gammatil_col_cur[1] = modl.gamma * (q_col[1]/q_ss)/Pi_col_cur[1] # Initial value needs q_ss
        for tt=2:T
            gammatil_col_cur[tt] = modl.gamma * (q_col[tt]/q_col[tt-1]) / Pi_col_cur[tt]
        end
        # Compute the Rc path
        for tt=1:T
            Rc_col_cur[tt] = (1.0+rb_col_cur[tt+1])/(Pi_col_cur[tt+1]*pQ_col_cur[tt+1])
        end
    elseif channel_selector == "rbtaureal"
        # Interest rates + inflation channel: rm, rb, rf + Pi
        MU_col_cur, pQ_col_cur, w_col_cur, M_tp1_col_cur = MU_col_ss, pQ_col_ss, w_col_ss, M_tp1_col_ss
        p_col_cur  = p_col_ss

        # Insert correct values for the rate of interest, inferring a nominal rb rate path that, given a flat Pi path is consistent with the REAL rate on borrowing in the fed in paths
        rb_col_cur = (Pi_ss*(1.0 .+ (1.0-modl.tau)*airf_in.rb_col)./airf_in.Pi_col .- 1.0)/(1.0-modl.tau)

        # And keep other nominal rates and the inflation rate constant at steady state
        rm_col_cur = rm_col_ss
        rf_col_cur = rf_col_ss
        # Insert correct values for Pi
        Pi_col_cur = Pi_col_ss

        # Solve for the implied path of bond prices etc.
        q_ss        = 1.0/(1.0+rb_ss-modl.gamma)
        q_col       = zeros(T+1)
        q_col[T+1]  = q_ss
        # Iterate for q path
        for tt=T:-1:1
            q_col[tt] = (modl.gamma*q_col[tt+1]+1.0) / (1.0+rb_col_cur[tt+1])
        end
        # Given the q path, compute gammatil, not including T value at the end
        gammatil_col_cur[1] = modl.gamma * (q_col[1]/q_ss)/Pi_col_cur[1] # Initial value needs q_ss
        for tt=2:T
            gammatil_col_cur[tt] = modl.gamma * (q_col[tt]/q_col[tt-1]) / Pi_col_cur[tt]
        end
        # Compute the Rc path
        for tt=1:T
            Rc_col_cur[tt] = (1.0+rb_col_cur[tt+1])/(Pi_col_cur[tt+1]*pQ_col_cur[tt+1])
        end
    elseif channel_selector == "rmtaureal"
        # Interest rates + inflation channel: rm, rb, rf + Pi
        MU_col_cur, pQ_col_cur, w_col_cur, M_tp1_col_cur = MU_col_ss, pQ_col_ss, w_col_ss, M_tp1_col_ss
        p_col_cur  = p_col_ss

        # Insert correct values for the rate of interest, inferring a nominal rb rate path that, given a flat Pi path is consistent with the REAL rate on borrowing in the fed in paths
        rm_col_cur = (Pi_ss*(1.0 .+ (1.0-modl.tau)*airf_in.rm_col)./airf_in.Pi_col .- 1.0)/(1.0-modl.tau)

        # And keep other nominal rates and the inflation rate constant at steady state
        rb_col_cur = rb_col_ss
        rf_col_cur = rf_col_ss
        # Insert correct values for Pi
        Pi_col_cur = Pi_col_ss

        # Solve for the implied path of bond prices etc.
        q_ss        = 1.0/(1.0+rb_ss-modl.gamma)
        q_col       = zeros(T+1)
        q_col[T+1]  = q_ss
        # Iterate for q path
        for tt=T:-1:1
            q_col[tt] = (modl.gamma*q_col[tt+1]+1.0) / (1.0+rb_col_cur[tt+1])
        end
        # Given the q path, compute gammatil, not including T value at the end
        gammatil_col_cur[1] = modl.gamma * (q_col[1]/q_ss)/Pi_col_cur[1] # Initial value needs q_ss
        for tt=2:T
            gammatil_col_cur[tt] = modl.gamma * (q_col[tt]/q_col[tt-1]) / Pi_col_cur[tt]
        end
        # Compute the Rc path
        for tt=1:T
            Rc_col_cur[tt] = (1.0+rb_col_cur[tt+1])/(Pi_col_cur[tt+1]*pQ_col_cur[tt+1])
        end
    elseif channel_selector == "MU"
        # Only channel is markup MU
        pQ_col_cur, w_col_cur, M_tp1_col_cur, Pi_col_cur, rb_col_cur, rm_col_cur, gammatil_col_cur = pQ_col_ss, w_col_ss, M_tp1_col_ss, Pi_col_ss, rb_col_ss, rm_col_ss, gammatil_col_ss
        p_col_cur  = p_col_ss
        rf_col_cur = rf_col_ss

        # Insert correct values for MU and Rc
        MU_col_cur = airf_in.MU_col
        # Compute the Rc path
        for tt=1:T
            Rc_col_cur[tt] = (1.0+rb_col_cur[tt+1])/(Pi_col_cur[tt+1]*pQ_col_cur[tt+1])
        end
    else
        error("Unsupported channel_selector: $channel_selector")
    end

    manprintln("Solving aggregate IRF for decomposition of channel: "*channel_selector*".", txtout_path)

    # Solve the firm's IRF policy
    irfsoln = solve_IRF(fsoln, MU_col_cur[1:T], w_col_cur[1:T], M_tp1_col_cur, pQ_col_cur[1:T], Pi_col_cur[1:T], rb_col_cur[1:T], rm_col_cur[1:T], Rc_col_cur, gammatil_col_cur, modl, sspace, opts)

    # Collections that will contain aggregate equilibrium solution objects along path
    L_col           = zeros(length(aeq.L),T)
    Ya_col          = Array{Float64}(undef, T)
    Na_col          = Array{Float64}(undef, T)
    Ca_col          = Array{Float64}(undef, T)
    Ka_col          = Array{Float64}(undef, T)
    Ba_col          = Array{Float64}(undef, T)
    Ma_col          = Array{Float64}(undef, T)
    ACa_col         = Array{Float64}(undef, T)
    Lin_col         = zeros(length(aeq.L),T+1)
    Lk_col          = Array{Float64}(undef, length(aeq.Lk),T)
    Lm_col          = Array{Float64}(undef, length(aeq.Lm),T)
    Lb_col          = Array{Float64}(undef, length(aeq.Lb),T)
    Lz_col          = Array{Float64}(undef, length(aeq.Lz),T)

    # 2. Looping forward, solve for the implied distributions of (k,m,b,z)
    # Initial distribution of incumbents is simply the steady state one -- because no shock (except a direct transfer) can affect the firms' states (k,m,b,z)
    Lin_col[:,1]    = aeq.Lin

    # Start with time iterations forward
    manprintln("Solving aggregate IRF forwards:", txtout_path)
    for tt = 1:T
        manprintln("Period t=$tt.", txtout_path)

        # Firms making decisions going forward from tt (the surviving incumbents):
        L_col[:,tt]     = (1.0-modl.eta)*Lin_col[:,tt]

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
        Ltil_cur     = Gammamat_cur'*Lin_col[:,tt]
        Lamtil_cur   = reshape(Ltil_cur, Nend, sspace.Nz)
        Lincumb_cur  = reshape(Lamtil_cur*sspace.P, Nd, 1)
        Lin_col[:,tt+1] = reshape((1.0-modl.eta)*Lincumb_cur + Le, Nd)

        # Create the implied marginals
        Lk_col[:,tt] = kron(ones(1,sspace.nf[4]*sspace.nf[3]*sspace.nf[2]), I(sspace.nf[1])) * Lin_col[:,tt]
        Lm_col[:,tt] = kron(ones(1,sspace.nf[4]*sspace.nf[3]), kron(I(sspace.nf[2]),ones(1,sspace.nf[1]))) * Lin_col[:,tt]
        Lb_col[:,tt] = kron(ones(1,sspace.nf[4]), kron(I(sspace.nf[3]),ones(1,sspace.nf[1]*sspace.nf[2]))) * Lin_col[:,tt]
        Lz_col[:,tt] = kron(I(sspace.nf[4]), ones(1,sspace.nf[1]*sspace.nf[2]*sspace.nf[3])) * Lin_col[:,tt]

        # Computing the aggregates
        Ya_col[tt]  = Lin_col[:,tt]' * reshape(irfsoln.Y_col[:,:,:,:,tt], Nd)
        Na_col[tt]  = Lin_col[:,tt]' * reshape(irfsoln.n_col[:,:,:,:,tt], Nd)
        Ka_col[tt]  = Lin_col[:,tt]' * sf[:,1] # Incoming capital
        Kp_cur      = L_col[:,tt]' * ( (1.0.-Ba_prob_cur) .* kpN_cur + Ba_prob_cur .* kpA_cur    ) + modl.eta * modl.k_0
        bpN_lev_cur, bpA_lev_cur = bpN_cur .* kpN_cur * Rc_col_cur[tt], bpA_cur .* kpA_cur * Rc_col_cur[tt]
        Ba_col[tt]  = L_col[:,tt]' * ( (1.0.-Ba_prob_cur) .* bpN_lev_cur + Ba_prob_cur .* bpA_lev_cur ) + modl.eta * modl.k_0 * modl.b_0
        Ma_col[tt]  = L_col[:,tt]' * ( (1.0.-Ba_prob_cur) .* mpN_cur + Ba_prob_cur .* mpA_cur    ) + modl.eta * modl.m_0
        ACa_col[tt] = L_col[:,tt]' * ( (1.0.-Ba_prob_cur).*reshape(irfsoln.ACN_col[:,:,:,:,tt], Nd)  + Ba_prob_cur .*reshape(irfsoln.ACA_col[:,:,:,:,tt], Nd)     )

        # Compute implied I/K aggregate adjustment costs:
        IdK_cost_cur = (((1.0-modl.phi_k)/(modl.delta^modl.phi_k))*(Kp_cur/Ka_col[tt]-(1.0-modl.delta) + modl.delta*modl.phi_k/(1.0-modl.phi_k)))^(1.0/(1.0-modl.phi_k))

        # Implied aggregate consumption by market clearing, and the outcoming price paths
        Ca_col[tt]    = Ya_col[tt] - IdK_cost_cur * Ka_col[tt] - ACa_col[tt]
    end

    # Pack everything up
    airf = AggIRFSolution(p_col_cur, rf_col_cur, rm_col_cur, rb_col_cur, Pi_col_cur, MU_col_cur, pQ_col_cur, L_col, Ya_col, Na_col, Ca_col, Ka_col, Ba_col, Ma_col, ACa_col, Lin_col, Lk_col, Lm_col, Lb_col, Lz_col)

    return airf, irfsoln
end

### Additional functions

# Define functions related to adjustment cost
# "dist" refers to either "uniform" or "logN"
# If "uniform": "xi_ub" and "xi_lb" refer to the respective "bounds" of the U distribution
# If "logN": "xi_ub" refers to the MEAN and xi_lb to the standard deviation of the lognormal distribution
function xi_cdfun(x, xi_ub, xi_lb)
    # Allow for no adjustment costs
    if xi_ub==0.0
        return ones(size(x))
    end

    if true
        # Firstly, simply apply the cdf function to x, and then set negative numbers to zero, and numbers above 1 to 1
        probs = (x .- xi_lb*ones(size(x)))./(xi_ub*ones(size(x)) - xi_lb*ones(size(x)))
        probs = min.(max.(zeros(size(x)), probs), ones(size(x)))
    else
        # Repeat instead for log Normal (ln(xi_ub), lnsig=1)
        mu_here     = log(xi_ub)
        lnsig_here  = 1.0
        probs = cdf.(LogNormal(mu_here, lnsig_here), x)
    end

    return probs
end
function xi_cdfun(x, xi_ub, xi_lb, dist)
    # Allow for no adjustment costs
    if xi_ub==0.0
        return ones(size(x))
    end

    if dist=="uniform"
        # Firstly, simply apply the cdf function to x, and then set negative numbers to zero, and numbers above 1 to 1
        probs = (x .- xi_lb*ones(size(x)))./(xi_ub*ones(size(x)) - xi_lb*ones(size(x)))
        probs = min.(max.(zeros(size(x)), probs), ones(size(x)))
    elseif dist=="logN"
        # Repeat instead for log Normal; notation: log(X) ~ N(mu_here, sig_here)
        mu_here     = log(xi_ub)
        lnsig_here  = xi_lb
        probs = cdf.(LogNormal(mu_here, lnsig_here), x)
    end

    return probs
end

function xi_condexpn(x, xi_ub, xi_lb)

    if true
        # Firstly, simply apply the expectation function to x, and then set "low" realizations to zero, and "high" realizations to unconditional mean
        expns = 0.5 * (x .+ xi_lb*ones(size(x)))
        expns = min.(max.(zeros(size(x)), expns), 0.5*ones(size(x))*(xi_lb + xi_ub))
    else
        mu_here     = log(xi_ub)
        lnsig_here  = 1.0
        x = max.(x, zeros(size(x)))
        expns = exp(mu_here + (lnsig_here^2)/2) * cdf.(Normal(), (log.(x) - ones(size(x))*(mu_here + lnsig_here^2))/lnsig_here) ./ cdf.(Normal(), (log.(x) - ones(size(x))*mu_here)/lnsig_here)
        # Set Nan to 0
        ind_expns_nan       = isnan.(expns)
        num_isnan           = sum(ind_expns_nan)
        expns[ind_expns_nan]= zeros(num_isnan)
    end

    return expns
end
function xi_condexpn(x, xi_ub, xi_lb, dist)
    if dist=="uniform"
        # Firstly, simply apply the expectation function to x, and then set "low" realizations to zero, and "high" realizations to unconditional mean
        expns = 0.5 * (x .+ xi_lb*ones(size(x)))
        expns = min.(max.(zeros(size(x)), expns), 0.5*ones(size(x))*(xi_lb + xi_ub))
    elseif dist=="logN"
        mu_here     = log(xi_ub)
        lnsig_here  = xi_lb

        # Deal separately with scalar case
        if typeof(x)==Float64
            if x <= 0.0
                expns = 0.0
            else
                expns = exp(mu_here + (lnsig_here^2)/2) * cdf.(Normal(), (log.(x) .- (mu_here + lnsig_here^2))/lnsig_here) ./ cdf.(Normal(), (log.(x) .- mu_here)/lnsig_here)
            end
        else
            x = max.(x, zeros(size(x)))
            expns = exp(mu_here + (lnsig_here^2)/2) * cdf.(Normal(), (log.(x) .- (mu_here + lnsig_here^2))/lnsig_here) ./ cdf.(Normal(), (log.(x) .- mu_here)/lnsig_here)
            # Set Nan to 0
            ind_expns_nan       = isnan.(expns)
            num_isnan           = sum(ind_expns_nan)
            expns[ind_expns_nan]= zeros(num_isnan)
        end
    end

    return expns
end



# Menu function for returning outcomes related to production, as requested by "flag", which could equal one of ("n", "y", "yp", "ypk", "ypm")
# For the "delayed revenue" working capital constraint on wages
function ymenufun(flag, k_in, m_in_true, z_in, MU_t, w_t, modl)
    # Make copy of incoming m
    m_in = copy(m_in_true)

    # First, to determine whether n choice is constrained or not, compute unconstrained n_star
    n_star = ((modl.nu/(MU_t*w_t))^(1.0/(1.0-modl.nu))) * z_in * k_in^(modl.alpha/(1.0-modl.nu))
    # Compute the implied m_bar
    m_bar  = w_t*n_star - (1.0-modl.phi_w)*(1.0/MU_t)*(z_in^(1.0-modl.nu))*(k_in^modl.alpha)*(n_star^modl.nu)

    # Is n unconstrained?
    ind_nuc = m_in >= m_bar

    # Because will need n solution anywhere below, compute it here
    if ind_nuc
        n_val =  n_star
    else
        n_0m  = (((1.0-modl.phi_w)/(w_t*MU_t))^(1.0/(1.0-modl.nu))) * z_in * k_in^(modl.alpha/(1.0-modl.nu))
        if m_in==0.0
            n_val = n_0m
        else
            # Define the deviation from working capital constraint
            f(x)  = w_t*x - m_in - (1.0-modl.phi_w)*(1.0/MU_t)*(z_in^(1.0-modl.nu))*(k_in^modl.alpha)*(x^modl.nu)
            n_val = find_zero(f, (n_0m, n_star), Bisection())
        end
    end

    # Return the requested outcome
    if flag=="n"
        return n_val
    elseif flag=="y"
        return (z_in^(1.0-modl.nu)) * (k_in^(modl.alpha)) * (n_val^modl.nu)
    elseif flag=="yp"
        if ind_nuc
            return (1.0-modl.nu) * (MU_t^(-1.0/(1.0-modl.nu))) * ((modl.nu/w_t)^(modl.nu/(1.0-modl.nu))) * z_in * k_in^(modl.alpha/(1.0-modl.nu))
        else
            return (MU_t^(-1)) * (z_in^(1.0-modl.nu)) * (k_in^(modl.alpha)) * (n_val^modl.nu) - w_t*n_val
        end
    elseif flag=="ypk"
        if ind_nuc
            return (modl.alpha/(1.0-modl.nu)) * (1.0-modl.nu) * (MU_t^(-1.0/(1.0-modl.nu))) * ((modl.nu/w_t)^(modl.nu/(1.0-modl.nu))) * z_in * k_in^(modl.alpha/(1.0-modl.nu)-1.0)
        else
            dndk = modl.alpha*(1.0-modl.phi_w)*(1.0/MU_t)*(z_in^(1.0-modl.nu))*(k_in^(modl.alpha-1))*(n_val^modl.nu)/(w_t - modl.nu*(1.0-modl.phi_w)*(1.0/MU_t)*(z_in^(1.0-modl.nu))*(k_in^modl.alpha)*(n_val^(modl.nu-1)))

            return modl.alpha*(1.0/MU_t)*(z_in^(1.0-modl.nu))*(k_in^(modl.alpha-1))*(n_val^modl.nu) + dndk*(modl.nu*(1.0/MU_t)*(z_in^(1.0-modl.nu))*(k_in^modl.alpha)*(n_val^(modl.nu-1)) - w_t)
        end
    elseif flag=="ypm"
        if ind_nuc
            return 0.0
        else
            dndm = 1.0/(w_t - modl.nu*(1.0-modl.phi_w)*(1.0/MU_t)*(z_in^(1.0-modl.nu))*(k_in^modl.alpha)*(n_val^(modl.nu-1)))

            return dndm*(modl.nu*(1.0/MU_t)*(z_in^(1.0-modl.nu))*(k_in^modl.alpha)*(n_val^(modl.nu-1)) - w_t)
        end
    end
end
# Repeat in full matrix form, where k_in, m_in, and z_in must be "conformable arrays"
function ymenufun_mat(flag, k_in_mat, m_in_mat_true, z_in_mat, MU_t, w_t, modl)
    # Make copy of incoming m
    m_in_mat = copy(m_in_mat_true)

    # First, to determine whether n choice is constrained or not, compute unconstrained n_star
    n_star_mat = ((modl.nu/(MU_t*w_t))^(1.0/(1.0-modl.nu))) * z_in_mat .* k_in_mat.^(modl.alpha/(1.0-modl.nu))

    # Because will need n solution anywhere below, compute it here
    # Initialize as n_star
    n_val_mat = copy(n_star_mat)
    # And override state-by-state, if threat of being constrained
    if modl.phi_w > 1.0-modl.nu
        # Compute the implied m_bar
        m_bar_mat  = w_t*n_star_mat - (1.0-modl.phi_w)*(1.0/MU_t)*(z_in_mat.^(1.0-modl.nu)).*(k_in_mat.^modl.alpha).*(n_star_mat.^modl.nu)
        # Is n unconstrained?
        ind_nuc_mat = m_in_mat .>= m_bar_mat

        # Identify constrained states indices
        indices_c_mat = findall(.!ind_nuc_mat)
        for ii in 1:size(indices_c_mat,1)
            ind_cur  = indices_c_mat[ii]
            k_in_cur = k_in_mat[ind_cur]
            m_in_cur = m_in_mat[ind_cur]
            z_in_cur = z_in_mat[ind_cur]

            n_0m  = (((1.0-modl.phi_w)/(w_t*MU_t))^(1.0/(1.0-modl.nu))) * z_in_cur * k_in_cur^(modl.alpha/(1.0-modl.nu))

            if m_in_cur==0.0
                n_val_mat[ind_cur] = n_0m
            else
                # Define the deviation from working capital constraint
                f(x)  = w_t*x - m_in_cur - (1.0-modl.phi_w)*(1.0/MU_t)*(z_in_cur^(1.0-modl.nu))*(k_in_cur^modl.alpha)*(x^modl.nu)
                n_val = find_zero(f, (n_0m, n_star_mat[ind_cur]), Bisection())
                n_val_mat[ind_cur] = n_val 
            end
        end
    end

    # Return the requested outcome
    if flag=="n"
        return n_val_mat

    elseif flag=="y"
        return (z_in_mat.^(1.0-modl.nu)) .* (k_in_mat.^modl.alpha) .* (n_val_mat.^modl.nu)

    elseif flag=="yp"
        # Initialize as unconstrained
        yp_mat = (1.0-modl.nu) * (MU_t^(-1.0/(1.0-modl.nu))) * ((modl.nu/w_t)^(modl.nu/(1.0-modl.nu))) * z_in_mat .* k_in_mat.^(modl.alpha/(1.0-modl.nu))

        # And override state-by-state, if threat of being constrained
        if modl.phi_w > 1.0-modl.nu
            # Identify constrained states indices
            for ii in 1:size(indices_c_mat,1)
                ind_cur  = indices_c_mat[ii]
                k_in_cur = k_in_mat[ind_cur]
                m_in_cur = m_in_mat[ind_cur]
                z_in_cur = z_in_mat[ind_cur]
                n_val_cur = n_val_mat[ind_cur]

                yp_mat[ind_cur] =  (MU_t^(-1)) * (z_in_cur^(1.0-modl.nu)) * (k_in_cur^(modl.alpha)) * (n_val_cur^modl.nu) - w_t*n_val_cur
            end
        end

        return yp_mat
    elseif flag=="ypk"
        # Initialize as unconstrained
        ypk_mat = (modl.alpha/(1.0-modl.nu)) * (1.0-modl.nu) * (MU_t^(-1.0/(1.0-modl.nu))) * ((modl.nu/w_t)^(modl.nu/(1.0-modl.nu))) * z_in_mat .* k_in_mat.^(modl.alpha/(1.0-modl.nu)-1.0)

        # And override state-by-state, if threat of being constrained
        if modl.phi_w > 1.0-modl.nu
            # Identify constrained states indices
            for ii in 1:size(indices_c_mat,1)
                ind_cur  = indices_c_mat[ii]
                k_in_cur = k_in_mat[ind_cur]
                m_in_cur = m_in_mat[ind_cur]
                z_in_cur = z_in_mat[ind_cur]
                n_val_cur = n_val_mat[ind_cur]

                dndk_cur = modl.alpha*(1.0-modl.phi_w)*(1.0/MU_t)*(z_in_cur^(1.0-modl.nu))*(k_in_cur^(modl.alpha-1))*(n_val_cur^modl.nu)/(w_t - modl.nu*(1.0-modl.phi_w)*(1.0/MU_t)*(z_in_cur^(1.0-modl.nu))*(k_in_cur^modl.alpha)*(n_val_cur^(modl.nu-1)))

                ypk_mat[ind_cur] = modl.alpha*(1.0/MU_t)*(z_in_cur^(1.0-modl.nu))*(k_in_cur^(modl.alpha-1))*(n_val_cur^modl.nu) + dndk_cur*(modl.nu*(1.0/MU_t)*(z_in_cur^(1.0-modl.nu))*(k_in_cur^modl.alpha)*(n_val_cur^(modl.nu-1)) - w_t)
            end
        end

        return ypk_mat
    elseif flag=="ypm"
        # Initialize as unconstrained
        ypm_mat = zeros(size(n_star_mat))

        # And override state-by-state, if threat of being constrained
        if modl.phi_w > 1.0-modl.nu
            # Identify constrained states indices
            for ii in 1:size(indices_c_mat,1)
                ind_cur  = indices_c_mat[ii]
                k_in_cur = k_in_mat[ind_cur]
                m_in_cur = m_in_mat[ind_cur]
                z_in_cur = z_in_mat[ind_cur]
                n_val_cur = n_val_mat[ind_cur]

                dndm_cur = 1.0/(w_t - modl.nu*(1.0-modl.phi_w)*(1.0/MU_t)*(z_in_cur^(1.0-modl.nu))*(k_in_cur^modl.alpha)*(n_val_cur^(modl.nu-1)))

                ypm_mat[ind_cur] = dndm_cur*(modl.nu*(1.0/MU_t)*(z_in_cur^(1.0-modl.nu))*(k_in_cur^modl.alpha)*(n_val_cur^(modl.nu-1)) - w_t)
            end
        end

        return ypm_mat
    end
end



end # End module