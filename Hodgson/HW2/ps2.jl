using LinearAlgebra
using Statistics
using Distributions
using Random
using FreqTables
using Optim

Random.seed!(1)
# parameters-----------------------------------------------
P_x = [
    0.6 0.2 0.2;  # Transitions from x_t = -5
    0.2 0.6 0.2;  # Transitions from x_t = 0
    0.2 0.2 0.6   # Transitions from x_t = 5
]

x_t = [-5, 0, 5]
std_normal = Normal(0, 1)

# compute pi
N_t_pi = 1:5
pi_t = (1 ./ (N_t_pi' .+ 1)).^2 .* (10 .+ x_t).^2 .- 5

# functions -----------------------------------------------
function prob_normal_geq(cutoff, mean_n = 5, var_n = 5)
    # Create a normal distribution with given parameters
    dist = Normal(mean_n, sqrt(var_n))
    
    # Calculate P(x > cutoff)
    # This is equivalent to 1 - CDF(gamma_bar)
    prob = 1 - cdf(dist, cutoff)
    
    return prob
end

function prob_normal_leq(cutoff, mean_n = 5, var_n = 5)
    # Create a normal distribution with given parameters
    dist = Normal(mean_n, sqrt(var_n))
    
    # Calculate P(x > cutoff)
    # This is equivalent to 1 - CDF(gamma_bar)
    prob = cdf(dist, cutoff)
    
    return prob
end

function P_N(mu_bar, gamma_bar)
    # Pr(Nt+1|Nt, xt, dit = 1), N_t, N_{t+1} = 1, 2, 3, 4, 5
    P_N_d1_x1 = zeros((5, 5))
    P_N_d1_x2 = zeros((5, 5))
    P_N_d1_x3 = zeros((5, 5))

    for n_t in 1:5  # Current number of firms (1 to 5)
        for n_tp1 in 1:5  # Next period number of firms (1 to 5)
            n_other_incumbents = n_t - 1
            
            # For each possible number of other staying incumbents
            for k in 0:n_other_incumbents
                # Probability k other firms stay out of n_other_incumbents
                p_others_stay_x1 = binomial(n_other_incumbents, k) * 
                            prob_normal_leq(mu_bar[1, n_t])^k * 
                            prob_normal_geq(mu_bar[1, n_t])^(n_other_incumbents - k)
                
                p_others_stay_x2 = binomial(n_other_incumbents, k) * 
                            prob_normal_leq(mu_bar[2, n_t])^k * 
                            prob_normal_geq(mu_bar[2, n_t])^(n_other_incumbents - k)
                
                p_others_stay_x3 = binomial(n_other_incumbents, k) * 
                            prob_normal_leq(mu_bar[3, n_t])^k * 
                            prob_normal_geq(mu_bar[3, n_t])^(n_other_incumbents - k)
                
                # Total stayers = k other firms + firm i
                total_stayers = k + 1
                
                # Case: No entry needed
                if n_tp1 == total_stayers
                    if n_t < 5
                        P_N_d1_x1[n_t, n_tp1] += p_others_stay_x1 * prob_normal_geq(gamma_bar[1, n_t+1])
                        P_N_d1_x2[n_t, n_tp1] += p_others_stay_x2 * prob_normal_geq(gamma_bar[2, n_t+1])
                        P_N_d1_x3[n_t, n_tp1] += p_others_stay_x3 * prob_normal_geq(gamma_bar[3, n_t+1])
                    else
                        p_no_entry = 1
                        P_N_d1_x1[n_t, n_tp1] += p_others_stay_x1 * p_no_entry
                        P_N_d1_x2[n_t, n_tp1] += p_others_stay_x2 * p_no_entry
                        P_N_d1_x3[n_t, n_tp1] += p_others_stay_x3 * p_no_entry
                    end
                
                # Case: One firm enters
                elseif n_tp1 == total_stayers + 1
                    if n_t < 5
                        P_N_d1_x1[n_t, n_tp1] += p_others_stay_x1 * prob_normal_leq(gamma_bar[1, n_t+1])
                        P_N_d1_x2[n_t, n_tp1] += p_others_stay_x2 * prob_normal_leq(gamma_bar[2, n_t+1])
                        P_N_d1_x3[n_t, n_tp1] += p_others_stay_x3 * prob_normal_leq(gamma_bar[3, n_t+1])
                    else
                        p_entry = 0
                        P_N_d1_x1[n_t, n_tp1] += p_others_stay_x1 * p_entry
                        P_N_d1_x2[n_t, n_tp1] += p_others_stay_x2 * p_entry
                        P_N_d1_x3[n_t, n_tp1] += p_others_stay_x3 * p_entry
                    end
                end
            end
        end
    end

    # Pr(Nt+1|Nt, xt, eit = 1), N_t = 0, 1, 2, 3, 4; N_{t+1} = 1, 2, 3, 4, 5
    P_N_e1_x1 = zeros((5, 5))
    P_N_e1_x2 = zeros((5, 5))
    P_N_e1_x3 = zeros((5, 5))

    for n_t in 0:4  # Current number of firms (0 to 4)
        # Individual stay probability for incumbent firms
        p_stay_x1 = n_t > 0 ? prob_normal_leq(mu_bar[1, n_t]) : 0.0
        p_stay_x2 = n_t > 0 ? prob_normal_leq(mu_bar[2, n_t]) : 0.0
        p_stay_x3 = n_t > 0 ? prob_normal_leq(mu_bar[3, n_t]) : 0.0
        
        for n_tp1 in 1:5  # Next period number of firms (1 to 5, must include f)
            n_stay_needed = n_tp1 - 1
            
            if n_stay_needed <= n_t
                # Valid transition: n_stay_needed incumbents stay, others exit
                P_N_e1_x1[n_t+1, n_tp1] = binomial(n_t, n_stay_needed) * 
                                        p_stay_x1^n_stay_needed * 
                                        (1 - p_stay_x1)^(n_t - n_stay_needed)
                
                P_N_e1_x2[n_t+1, n_tp1] = binomial(n_t, n_stay_needed) * 
                                        p_stay_x2^n_stay_needed * 
                                        (1 - p_stay_x2)^(n_t - n_stay_needed)
                
                P_N_e1_x3[n_t+1, n_tp1] = binomial(n_t, n_stay_needed) * 
                                        p_stay_x3^n_stay_needed * 
                                        (1 - p_stay_x3)^(n_t - n_stay_needed)
            end
        end
    end

    return P_N_d1_x1, P_N_d1_x2, P_N_d1_x3, P_N_e1_x1, P_N_e1_x2, P_N_e1_x3
end

# 4 -------------------------------------------------------
gamma = 5
sigma2_gamma = 5
mu = 5
sigma2_mu = 5

sigma_mu = sqrt(sigma2_mu)
sigma_gamma = sqrt(sigma2_gamma)

# cutoffs (each row is a xt)
mu_bar0 = zeros((3, 5)) # columns: Nt = 1, 2, 3, 4, 5
gamma_bar0 = zeros((3, 5)) # columns: Nt = 0, 1, 2, 3, 4

# V_bar (rows: xt = -5, 0, 5; columns: Nt = 1, 2, 3, 4, 5)
V_bar0 = ones((3, 5))

function vfi(mu_bar, gamma_bar, V_bar, tax = 0)
    # initiate objects
    Phi_1 = zeros((3, 5))
    Phi_2 = zeros((3, 5))
    diff = 1
    iter = 0

    mu_bar_p = zeros((3, 5))
    gamma_bar_p = zeros((3, 5))
    V_bar_p = zeros((3, 5))

    P_N_d1 = Any
    P_N_e1 = Any

    while diff > 1e-10
        # get transition matrices
        P_N_d1_x1, P_N_d1_x2, P_N_d1_x3, P_N_e1_x1, P_N_e1_x2, P_N_e1_x3 = P_N(mu_bar, gamma_bar)
        P_N_d1 = [P_N_d1_x1, P_N_d1_x2, P_N_d1_x3]
        P_N_e1 = [P_N_e1_x1, P_N_e1_x2, P_N_e1_x3]

        # compute Phi matrices
        for i in 1:3
            for j in 1:5
                Phi_1[i, j] = 0.9 * (V_bar' * P_x[i, :])' * P_N_d1[i][j, :] # N_t = 1, ..., 5
                Phi_2[i, j] = 0.9 * (V_bar' * P_x[i, :])' * P_N_e1[i][j, :] # N_t = 0, ..., 4
            end
        end

        # new cutoffs
        mu_bar_p = pi_t .+ Phi_1
        gamma_bar_p = Phi_2 .- tax

        # define relevant parameters
        argument = (pi_t .+ Phi_1 .- mu) ./ sigma_mu
        Psi_values = cdf.(std_normal, argument)
        psi_values = pdf.(std_normal, argument)

        # new V_bar
        V_bar_p = (1 .- Psi_values) .* mu .+ sigma_mu .* psi_values .+ Psi_values .* (pi_t .+ Phi_1)
        
        # update guesses
        diff = max(maximum(abs.(V_bar_p .- V_bar)), maximum(abs.(mu_bar_p .- mu_bar)), maximum(abs.(gamma_bar_p .- gamma_bar)))
        V_bar = 0.1 .* V_bar_p + 0.9 .* V_bar
        mu_bar =  0.1 .* mu_bar_p + 0.9 .* mu_bar
        gamma_bar = 0.1 .* gamma_bar_p + 0.9 .* gamma_bar

        iter += 1
    end

    return mu_bar_p, gamma_bar_p, V_bar_p, P_N_d1, P_N_e1
end

mu_bar, gamma_bar, V_bar, P_N_d1, P_N_e1 = vfi(mu_bar0, gamma_bar0, V_bar0)


# 5 -------------------------------------------------------
mu_bar1, gamma_bar1, V_bar1 = vfi(ones((3, 5)), ones((3, 5)), ones((3, 5)))
mu_bar2, gamma_bar2, V_bar2 = vfi(3 .* ones((3, 5)), 3 .* ones((3, 5)), 3 .* ones((3, 5)))
mu_bar3, gamma_bar3, V_bar3 = vfi(4 .* ones((3, 5)), 5 .* ones((3, 5)), 6 .* ones((3, 5)))
mu_bar4, gamma_bar4, V_bar4 = vfi(10 .* ones((3, 5)), 10 .* ones((3, 5)), 10 .* ones((3, 5)))
mu_bar5, gamma_bar5, V_bar5 = vfi(rand(3, 5), rand(3, 5), rand(3, 5))

tolerance = 1e-8
all(isapprox(mu_bar1, mu_bar2, atol=tolerance)) &&
    all(isapprox(mu_bar1, mu_bar3, atol=tolerance)) &&
    all(isapprox(mu_bar1, mu_bar4, atol=tolerance)) &&
    all(isapprox(mu_bar1, mu_bar5, atol=tolerance))

all(isapprox(gamma_bar1, gamma_bar2, atol=tolerance)) &&
    all(isapprox(gamma_bar1, gamma_bar3, atol=tolerance)) &&
    all(isapprox(gamma_bar1, gamma_bar4, atol=tolerance)) &&
    all(isapprox(gamma_bar1, gamma_bar5, atol=tolerance))

all(isapprox(V_bar1, V_bar2, atol=tolerance)) &&
    all(isapprox(V_bar1, V_bar3, atol=tolerance)) &&
    all(isapprox(V_bar1, V_bar4, atol=tolerance)) &&
    all(isapprox(V_bar1, V_bar5, atol=tolerance))

# 6 -------------------------------------------------------
mu_3_0 = mu_bar[2, 3]
gamma_3_0 = gamma_bar[2, 4]
V_bar_3_0 = V_bar[2, 3]
V_3_0_2 = max(-2, pi_t[2, 3] + 0.9 * (V_bar' * P_x[2, :])' * P_N_d1[2][3, :])

# 7 -------------------------------------------------------
# define parameters
T = 10000
N = zeros(Int8, T+1)
entry = zeros(Int8, T)
exit = zeros(Int8, T)
stay = zeros(Int8, T)
dist_mu_gamma = Normal(5, sqrt(5))

x = zeros(Int8, T+1)
x[1] = 2

# simulate
for t in 1:T
    
    # entry
    if N[t] < 5
        gamma_t = rand(dist_mu_gamma)
        entry[t] = sum(gamma_t <= gamma_bar[x[t], N[t]+1])
    end

    # exit
    if N[t] > 0
        mu_t = rand(dist_mu_gamma, N[t])
        exit[t] = sum(mu_t .> mu_bar[x[t], N[t]])
    end

    # define stayers
    stay[t] = N[t] - exit[t]

    # update state variables
    N[t+1] = N[t] + entry[t] - exit[t]
    x[t+1] = rand(Categorical(P_x[x[t], :]))

end

mean(N)
mean(entry)
mean(exit .> 0)

# 8 -------------------------------------------------------
# re-solve equilibrium
mu_bar_tax, gamma_bar_tax, V_bar_tax, P_N_d1_tax, P_N_e1_tax = vfi(mu_bar0, gamma_bar0, V_bar0, 5)

# define parameters
N_tax = zeros(Int8, T+1)

x_tax = zeros(Int8, T+1)
x_tax[1] = 2

# simulate
for t in 1:T
    
    entry_tax = 0
    exit_tax = 0

    # entry
    if N_tax[t] < 5
        gamma_t = rand(dist_mu_gamma)
        entry_tax = sum(gamma_t <= gamma_bar_tax[x_tax[t], N_tax[t]+1])
    end

    # exit
    if N_tax[t] > 0
        mu_t = rand(dist_mu_gamma, N_tax[t])
        exit_tax = sum(mu_t .> mu_bar_tax[x_tax[t], N_tax[t]])
    end

    # update state variables
    N_tax[t+1] = N_tax[t] + entry_tax - exit_tax
    x_tax[t+1] = rand(Categorical(P_x[x_tax[t], :]))

end

mean(N_tax)

# 9 -------------------------------------------------------
freqs = freqtable(x, N) 

# entry
freqs_entry = freqtable(x[1:T], N[1:T], entry) 
e_hat = zeros((3, 5))
for i in 1:3
    for j in 1:5
        e_hat[i, j] = freqs_entry[i, j, 2] / (freqs_entry[i, j, 1] + freqs_entry[i, j, 2])
    end
end

cdf.(Normal(5, sqrt(5)), gamma_bar) # comparing to "true" one

# stay
freqs_stay = freqtable(x[1:T], N[1:T], stay) 
d_hat = zeros((3, 5))
for i in 1:3
    for j in 1:5
        for n_stay in 1:5
            d_hat[i, j] += (n_stay * freqs_stay[i, j+1, n_stay + 1]) / (j * freqs[i, j+1])
        end
    end
end

cdf.(Normal(5, sqrt(5)), mu_bar) # comparing to "true" one

# B) ------------------------------------------------------
T_sim = 100 # recommended: 100
n_sim = 1000 # recommended: 50

# forward simulate 1 incumbent
function forward_simulate_inc(gamma, sigma2_gamma, mu, sigma2_mu, und_mu_it, und_gamma_it, T_sim, x)
    
    # create shocks
    mu_it = sqrt(sigma2_mu) .* und_mu_it .+ mu
    gamma_it = sqrt(sigma2_gamma) .* und_gamma_it .+ gamma

    # value function
    V_sim = zeros((3, 5))

    for i in 1:3
        for j in 1:5

            N = zeros(Int8, T_sim+1)
            x_i = x[:, i]

            N[1] = j
            T_last = 0

            for t in 1:T_sim
            
                entry = 0
                exit = 0

                # exit (Firm 1)
                if t > 1
                    if cdf(std_normal, (mu_it[i, j, 1, t] - mu)/sqrt(sigma2_mu)) <= d_hat[x_i[t], N[t]]
                        V_sim[i, j] += 0.9^(t - 1) * pi_t[x_i[t], N[t]]
                    else
                        # value function
                        V_sim[i, j] += 0.9^(t - 1) * mu_it[i, j, 1, t]
                        break
                    end
                else
                    V_sim[i, j] += pi_t[x_i[t], N[t]]
                end

                # exit (other firms)
                if N[t] > 1
                    exit = sum(cdf.(std_normal, (mu_it[i, j, 2:N[t], t] .- mu)./sqrt(sigma2_mu)) .> d_hat[x_i[t], N[t]])
                end

                # entry
                if N[t] < 5
                    entry = sum(cdf(std_normal, (gamma_it[i, j, t] - gamma)/sqrt(sigma2_gamma)) <= e_hat[x_i[t], N[t]+1])
                end                
                
                # update state variables
                N[t+1] = N[t] + entry - exit
                T_last += 1

            end
        end
    end

    return V_sim
end

# run for n_sim incumbents
function simulate_inc(gamma, sigma2_gamma, mu, sigma2_mu, und_mu_it, und_gamma_it, T_sim, sim_inc, x)
    V_sim = zeros((3, 5, sim_inc))

    # run simulations
    for sim in 1:sim_inc
        V_sim[:, :, sim] = forward_simulate_inc(gamma, sigma2_gamma, mu, sigma2_mu, und_mu_it[:, :, :, :, sim], und_gamma_it[:, :, :, sim], T_sim, x[:, :, sim])
    end

    return mean(V_sim, dims=3)[:, :, 1]
end

# E) ------------------------------------------------------
function forward_simulate_ent(gamma, sigma2_gamma, mu, sigma2_mu, und_mu_it, und_gamma_it, T_sim, x)
    
    # create shocks
    mu_it = sqrt(sigma2_mu) .* und_mu_it .+ mu
    gamma_it = sqrt(sigma2_gamma) .* und_gamma_it .+ gamma

    # value function (now N_t = 0, ..., 4)
    V_sim = zeros((3, 5))

    for i in 1:3
        for j in 0:4

            N = zeros(Int8, T_sim+1)
            x_i = x[:, i]

            N[1] = j
            T_last = 0

            for t in 1:T_sim
            
                entry = 0
                exit = 0

                # stay (Firm 1)
                if t > 1
                    if cdf(std_normal, (mu_it[i, j+1, 1, t] - mu)/sqrt(sigma2_mu)) <= d_hat[x_i[t], N[t]]
                        V_sim[i, j+1] += 0.9^(t - 1) * pi_t[x_i[t], N[t]]
                    else
                        # value function
                        V_sim[i, j+1] += 0.9^(t - 1) * mu_it[i, j+1, 1, t]
                        break
                    end
                else
                    V_sim[i, j+1] += 0
                    entry = 1
                end

                # exit
                if t == 1
                    if N[t] > 0
                        exit = sum(cdf.(std_normal, (mu_it[i, j+1, 1:N[t], t] .- mu)./sqrt(sigma2_mu)) .> d_hat[x_i[t], N[t]])
                    end
                else
                    if N[t] > 1
                        exit = sum(cdf.(std_normal, (mu_it[i, j+1, 2:N[t], t] .- mu)./sqrt(sigma2_mu)) .> d_hat[x_i[t], N[t]])
                    end
                end

                # entry
                if t > 1 && N[t] < 5
                    entry = sum(cdf(std_normal, (gamma_it[i, j+1, t] - gamma)/sqrt(sigma2_gamma)) <= e_hat[x_i[t], N[t]+1])
                end                
                
                # update state variables
                N[t+1] = N[t] + entry - exit
                T_last += 1

            end
        end
    end

    return V_sim
end

# run for n_sim incumbents
function simulate_ent(gamma, sigma2_gamma, mu, sigma2_mu, und_mu_it, und_gamma_it, T_sim, sim_ent, x)
    V_sim = zeros((3, 5, sim_ent))

    # run simulations
    for sim in 1:sim_ent
        V_sim[:, :, sim] = forward_simulate_ent(gamma, sigma2_gamma, mu, sigma2_mu, und_mu_it[:, :, :, :, sim], und_gamma_it[:, :, :, sim], T_sim, x[:, :, sim])
    end

    return mean(V_sim, dims=3)[:, :, 1]
end

# F) ------------------------------------------------------
function objective_bbl(d_hat, e_hat, gamma, sigma2_gamma, mu, sigma2_mu, und_mu_it_inc, und_gamma_it_inc, und_mu_it_ent, und_gamma_it_ent, T_sim, n_sim, x)
    
    # incumbent
    Lambda = simulate_inc(gamma, sigma2_gamma, mu, sigma2_mu, und_mu_it_inc, und_gamma_it_inc, T_sim, n_sim, x)

    # entrant
    Lambda_E = simulate_ent(gamma, sigma2_gamma, mu, sigma2_mu, und_mu_it_ent, und_gamma_it_ent, T_sim, n_sim, x)

    # objective function
    obj_d = (cdf.(std_normal, (Lambda .- mu) ./ sqrt(sigma2_mu)) .- d_hat).^2
    obj_e = (cdf.(std_normal, (Lambda_E .- gamma) ./ sqrt(sigma2_gamma)) .- e_hat).^2

    return sum(obj_d) + sum(obj_e)
end

# define x's
x = zeros(Int8, (T_sim+1, 3, n_sim))
x[1, 1, :] .= 1
x[1, 2, :] .= 2
x[1, 3, :] .= 3

for i in 1:n_sim
    for t in 1:T_sim
        x[t+1, 1, i] = rand(Categorical(P_x[x[t, 1, i], :]))
        x[t+1, 2, i] = rand(Categorical(P_x[x[t, 2, i], :]))
        x[t+1, 3, i] = rand(Categorical(P_x[x[t, 3, i], :]))
    end
end

# initial guess
params_0 = [1.0, 10.0, -1.0, 2.5]

# Define lower bounds: set a small positive number (e.g., 1e-8) for parameters 2 and 4 to ensure positivity
lower_bounds = [-Inf, 1e-8, -Inf, 1e-8]
upper_bounds = [Inf, Inf, Inf, Inf]

# draw underlying shocks
und_mu_it = rand(std_normal, (3, 5, 5, T_sim, n_sim)) # (3x5 states, up to 5 incumbents, T_sim periods, sim_inc runs)
und_gamma_it = rand(std_normal, (3, 5, T_sim, n_sim)) # (3x5 states, 1 entrant, T_sim periods, sim_inc runs)

# define relevant parameters
obj_min(params) = objective_bbl(d_hat, e_hat, params[1], params[2], params[3], params[4], und_mu_it, und_gamma_it, und_mu_it, und_gamma_it, T_sim, n_sim, x)

# Minimize with bounds
opt = optimize(obj_min, lower_bounds, upper_bounds, params_0, Fminbox(LBFGS())) # Optim.Options(show_trace = true)
parameters = Optim.minimizer(opt)

