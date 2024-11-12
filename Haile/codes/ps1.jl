using Random
using Distributions
using LinearAlgebra
using NLsolve
using CSV
using GLM
using DelimitedFiles
using DataFrames
using Econometrics
using RegressionTables
using Plots
Random.seed!(1);

# set home directory
HOME_DIR = "C:/Users/guilh/Dropbox/ECON 600 - Industrial Organization I/First Part/Problem Sets/"

# problem parameters
N_draws = 500;
T = 600;
J = 4;

# functions 
### function for derivatives and market shares
function derivatives_sh(p, x, beta_1, beta_2, beta_3, alpha, N_draws, T)

    # delta
    delta_jt = alpha .* p + beta_1 .* x[:, 3] + x[:, 6] # column 3: x_jt; column 6: ξ_jt

    # loop over i
    mu_ijt = zeros((N_draws, T*J))
    s_ijt = zeros((N_draws, T, J))
    D = zeros((J, J, T))
    G = zeros((J, J, T))
    
    for i in 1:N_draws
        mu_ijt[i, :, :] = beta_2[i] .* x[:, 4] + beta_3[i] .* x[:, 5] 
        
        # reshape data (each line is a market, each column is a product)
        mu_ijt_reshaped = reshape(mu_ijt[i, :, :], (T, J))
        delta_jt_reshaped = reshape(delta_jt, (T, J))

        num_ijt = exp.(delta_jt_reshaped + mu_ijt_reshaped)
        denom_ijt = 1 .+ sum(num_ijt, dims=2)

        s_ijt[i, :, :] = num_ijt ./ repeat(denom_ijt, outer = [1, 4]) 

        # elasticities
        for t in 1:T

            cross_terms = s_ijt[i, t, :] * s_ijt[i, t, :]'
            
            # for the cross-price elasticities
            G[:, :, t] = G[:, :, t] .+ alpha .* cross_terms

            # for the own-price elasticities
            for j in 1:J
                D[j, j, t] = D[j, j, t]  .+ alpha .* s_ijt[i, t, j]
            end

        end
    end

    s_jt = dropdims(sum(s_ijt, dims=1), dims=1) ./ N_draws
    D = D ./ N_draws
    G = G ./ N_draws

    derivatives = D - G

    return s_jt, derivatives, D, G
end

### function for marginal cost
function mc(w, gamma_0, gamma_1)
    marg_cost = exp.(gamma_0 .+ gamma_1 .* w[:, 3] + w[:, 4] ./ 8) # column 3: w_jt; column 4: ω_jt 
end

### function for solving equilibrium prices
function foc(p, mg_cost, x, beta_1, beta_2, beta_3, alpha, N_draws, T)
    
    sh, deriv, D, G = derivatives_sh(p, x, beta_1, beta_2, beta_3, alpha, N_draws, T)

    # get only own-price derivatives
    own_deriv = zeros(T, J)
    for t in 1:T
        for j in 1:J
            own_deriv[t, j] = deriv[j, j, t]
        end
    end

    # FOC
    FOC = (p .- mg_cost) .* own_deriv' .+ sh'
    return FOC
end

# 1. Draw the exogenous product characteristics... -----------------------

# distributions
d = MvNormal([0,0], [1 0.25; 0.25 1]);
beta_23_dist = Normal(4, 1) 

# non-random parameters
beta_1 = 1;
alpha = -2;
gamma_0 = 1/2;
gamma_1 = 1/4;

# draw betas
beta_2 = rand(beta_23_dist, N_draws)
beta_3 = rand(beta_23_dist, N_draws)

# generate ξ_jt and ω_jt
x_gen = rand(d, T*J);
xi = x_gen[1, :];
om = x_gen[2, :];

# generate matrix of characteristics X_jt and ξ_jt -- columns(1: Market ID; 2: Product ID; 3: x_jt; 4: satellite; 5: wired; 6: ξ_jt)
X_1t = hcat(1:T, ["1" for i in 1:T], abs.(randn((T, 1))), ones(T), zeros(T), xi[1:T])       # satellite  
X_2t = hcat(1:T, ["2" for i in 1:T], abs.(randn((T, 1))), ones(T), zeros(T), xi[T+1:2*T])   # satellite
X_3t = hcat(1:T, ["3" for i in 1:T], abs.(randn((T, 1))), zeros(T), ones(T), xi[2*T+1:3*T]) # wired
X_4t = hcat(1:T, ["4" for i in 1:T], abs.(randn((T, 1))), zeros(T), ones(T), xi[3*T+1:4*T]) # wired

# generate cost observables w_jt and unobservables ω_jt -- columns(1: Market ID; 2: Product ID; 3: w_jt; 4: ω_jt)
W_1t = hcat(1:T, ["1" for i in 1:T], abs.(randn((T, 1))), om[1:T])
W_2t = hcat(1:T, ["2" for i in 1:T], abs.(randn((T, 1))), om[T+1:2*T])
W_3t = hcat(1:T, ["3" for i in 1:T], abs.(randn((T, 1))), om[2*T+1:3*T])
W_4t = hcat(1:T, ["4" for i in 1:T], abs.(randn((T, 1))), om[3*T+1:4*T])

# get marginal cost
mg_cost = reshape(mc([W_1t; W_2t; W_3t; W_4t], gamma_0, gamma_1), (T, J))


# 2. Solve for the equilibrium prices for each good in each market -------
### (i) Using "fsolve":
conv = 0
prices_solve = zeros(T, J)
for t in 1:T 
    f!(p) = foc(p, mg_cost[t, :], [X_1t; X_2t; X_3t; X_4t][[t, t+T, t+2*T, t+3*T], :], beta_1, beta_2, beta_3, alpha, N_draws, 1)
    res = nlsolve(f!, 1.3 .* mg_cost[t, :], iterations = 1000)
    
    # check convergence
    conv = conv + res.f_converged

    # get prices (zero)
    prices_solve[t, :] = res.zero
end

conv == 600 # all converged

### (ii) Using the algorithm of Morrow and Skerlos (2011):
H = I(4)
p_init = reshape(mg_cost, T*J)
diff = 1
iter = 0

p_n = p_init
while diff > 1e-10
    sh_n, deriv_n, D_n, G_n = derivatives_sh(p_n, [X_1t; X_2t; X_3t; X_4t], beta_1, beta_2, beta_3, alpha, N_draws, T)
    
    # reshape prices (each line is a market, each column is a product)
    p_n = reshape(p_n, (T, J))

    # loop over markets
    Zt_n = zeros((T, J))
    for t in 1:T
        Zt_n[t, :] = inv(D_n[:, :, t]) * (H .* G_n[:,:,t]) * (p_n[t, :] - mg_cost[t, :]) - inv(D_n[:, :, t]) * sh_n[t, :]
    end
    
    # new guess
    p_n1 = mg_cost + Zt_n

    # update guess
    diff = maximum(abs.(p_n1 - p_n))
    p_n = reshape(p_n1, T*J)
    iter = iter + 1

end

prices_ms_2011 = reshape(p_n, (T, J))

# checking difference between both methods
maximum(abs.(prices_solve - prices_ms_2011)) # large maximum difference
mean(abs.(prices_solve - prices_ms_2011)) # not that high average difference

# choose Morrow and Skerlos (2011) as observed prices - Conlon and Gortmaker (2020)
p_obs = prices_ms_2011 

# 3. Calculate "observed" market shares ----------------------------------
sh_obs, deriv_obs = derivatives_sh(p_n, [X_1t; X_2t; X_3t; X_4t], beta_1, beta_2, beta_3, alpha, N_draws, T)

# build fake dataset -- columns(1: Market ID; 2: Product ID; 3: prices; 4: shares; 5: x_jt; 6: satellite; 7: wired; 8: w_jt)
G1 = hcat(X_1t[:, 1:2], p_obs[:, 1], sh_obs[:, 1], X_1t[:, 3:5], W_1t[:, 3])
G2 = hcat(X_2t[:, 1:2], p_obs[:, 2], sh_obs[:, 2], X_2t[:, 3:5], W_2t[:, 3])
G3 = hcat(X_3t[:, 1:2], p_obs[:, 3], sh_obs[:, 3], X_3t[:, 3:5], W_3t[:, 3])
G4 = hcat(X_4t[:, 1:2], p_obs[:, 4], sh_obs[:, 4], X_4t[:, 3:5], W_4t[:, 3])
data = [G1; G2; G3; G4]

# as DataFrame
df = DataFrame(data, :auto)
df = select(df, :x1 => :market_ID, :x2 => :product_ID, :x3 => :prices, :x4 => :shares, :x5 => :x, :x6 => :satellite, :x7 => :wired, :x8 => :w)

# export csv
CSV.write(joinpath(HOME_DIR, "data.csv"), df)


# 4. Check instruments in your fake data ---------------------------------
# read table
data, header = readdlm(joinpath(HOME_DIR, "data.csv"), ',', header=true)
df = DataFrame(data, vec(header))

# create instruments (using BLP approximation)
df_inst = transform(groupby(df, :market_ID), :x => ((v) -> sum(v) .- v) => :x_instrument, :w => ((v) -> sum(v) .- v) => :w_instrument)

# regressions of prices and market shares on the exogenous variables 
reg_p = lm(@formula(prices ~ x + satellite + w + x_instrument + w_instrument), df_inst)
reg_s = lm(@formula(shares ~ x + satellite + w + x_instrument + w_instrument), df_inst)
regtable(reg_p, reg_s; renderSettings = latexOutput(joinpath(HOME_DIR, "tables/reg.tex")))


# 5. Estimate the plain multinomial logit model of demand by OLS ---------
df_logit = transform(groupby(df_inst, :market_ID), :shares => ((v) -> 1 - sum(v)) => :s0)
df_logit = transform(df_logit, [:shares, :s0] => ((sj, s0) -> log.(sj) - log.(s0)) => :delta)

logit = lm(@formula(delta ~ prices + x + satellite), df_logit)
regtable(logit; renderSettings = latexOutput(joinpath(HOME_DIR, "tables/logit.tex")))


# 6. Re-estimate the multinomial logit model of demand by 2SLS -----------
iv_logit = fit(EconometricModel,
               @formula(delta ~ x + satellite + (prices ~ x + satellite + w + x_instrument + w_instrument)),
               df_logit)

regtable(iv_logit; renderSettings = latexOutput(joinpath(HOME_DIR, "tables/iv_logit.tex")))

first_stage = lm(@formula(prices ~ x + satellite + w + x_instrument + w_instrument), df_logit)
r2(first_stage)


# 7. Now estimate a nested logit model by linear IV ----------------------
### adding group share
df_nestedl = transform(groupby(df_logit, [:market_ID, :satellite]), :shares => ((v) -> v/sum(v)) => :s_jg)
df_nestedl = transform(groupby(df_nestedl, [:market_ID, :satellite]), :x => ((v) -> sum(v) .- v) => :x_g_instrument, :w => ((v) -> sum(v) .- v) => :w_g_instrument)

nested_logit = fit(EconometricModel,
                   @formula(delta ~ x + satellite + (prices + log(s_jg) + satellite&log(s_jg) ~ x + w + x_instrument + w_instrument + x_g_instrument + w_g_instrument)),
                   df_nestedl)

regtable(nested_logit; renderSettings = latexOutput(joinpath(HOME_DIR, "tables/nested_logit.tex")))

# 8. Compare the estimated own-price elasticities to the true ones -------
sigma_wired = coef(nested_logit)[5]
sigma_satellite = coef(nested_logit)[5] + coef(nested_logit)[6]

df_nestedl.sigma = @. ifelse(df_nestedl.satellite == 1, sigma_satellite, sigma_wired)
df_elast = transform(df_nestedl, [:shares, :s_jg, :sigma] => ((sj, s_jg, sigma) -> alpha ./ (1 .- sigma) .* sj .* (1 .- sigma .* s_jg .- (1 .- sigma) .* sj)) => :own_elast)

# derivatives matrix
price_deriv = zeros((J, J, T))
for t in 1:T
    for j in 1:J
        for k in 1:J
            sj = df_elast[(df_elast.market_ID .== t), :shares][j]
            sj_g = df_elast[(df_elast.market_ID .== t), :s_jg][j]
            sk = df_elast[(df_elast.market_ID .== t), :shares][k]
            sigma = df_elast[(df_elast.market_ID .== t), :sigma][j]

            if j == k # own-price
                price_deriv[j, j, t] = df_elast[(df_elast.market_ID .== t), :own_elast][j]

            elseif df_elast[(df_elast.market_ID .== t), :wired][j] == df_elast[(df_elast.market_ID .== t), :wired][k] # same group
                price_deriv[j, k, t] = -alpha .* sk .* (sj .+ sj_g .* sigma ./ (1 .- sigma))

            else # different groups
                price_deriv[j, k, t] = -alpha .* sj .* sk
            end

        end
    end
end

# derivatives to elasticities
true_elasts = zeros((J, J, T))
price_elast = zeros((J, J, T))
for t in 1:T
    for j in 1:J
        for k in 1:J
            sj = df_elast[(df_elast.market_ID .== t), :shares][j]
            pk = df_elast[(df_elast.market_ID .== t), :prices][k]
            
            price_elast[j, k, t] = pk/sj * price_deriv[j, k, t]
            true_elasts[j, k, t] = pk/sj * deriv_obs[j, k, t]            
        end
    end
end

est_own_price_elast = [price_elast[j, j, t] for j in 1:J for t in 1:T]
true_own_price_elast = [true_elasts[j, j, t] for j in 1:J for t in 1:T]

# compare
describe(est_own_price_elast)
describe(true_own_price_elast)
describe(abs.(est_own_price_elast .- true_own_price_elast))

# plot
p = scatter(est_own_price_elast, true_own_price_elast,
    xlabel="Estimated Own-Price Elasticity",
    ylabel="True Own-Price Elasticity", 
    legend=false,
    marker=:circle,
    ms=2,           # marker size
    grid=true,      # add grid
    title="Comparison of True and Estimated Own-Price Elasticities",
    aspect_ratio=1,
    alpha=0.5
)

plot!(-11:-2, -11:-2, 
    label="y = x", 
    lw=2,         # line width
    color=:red
)

savefig(p, joinpath(HOME_DIR, "tables/plot_elasticities.png"))

# diversion ratios
true_diversion = zeros((J, J, T))
estd_diversion = zeros((J, J, T))
for t in 1:T
    for j in 1:J
        for k in 1:J
            estd_diversion[j, k, t] = -price_deriv[j, k, t] / price_deriv[j, j, t]
            true_diversion[j, k, t] = -deriv_obs[j, k, t] / deriv_obs[j, j, t]            
        end
    end
end

# compare
describe(estd_diversion[estd_diversion .!= 1])
describe(true_diversion[true_diversion .!= 1])
describe(abs.(estd_diversion[estd_diversion .!= 1] .- true_diversion[true_diversion .!= 1]))
