using Plots
using LaTeXStrings
using CSV
using DataFrames
using Optim
using LinearAlgebra
using Statistics
gr()

# parameters ----------------------------------------------
beta = 0.9
euler = 0.5775
a_t = Vector(1:5)
    
# set home directory
HOME_DIR = "C:/Users/guilh/Dropbox/ECON 600 - Industrial Organization I/Second Part/Problem Sets/HW1"

# define functions ----------------------------------------
function a_t_next(a_t, i)
    if i == 1
        a = 1
    end
    if i == 0
        a = min(5, a_t + 1)
    end

    return a
end

function vfi_i(theta_1, R, a_t = a_t, P_0 = P_0, P_1 = P_1, tol = 1e-10)
    v0 = zeros(5)
    v0_new = zeros(5)
    v1 = zeros(5)
    v1_new = zeros(5)

    # iterate
    diff = 1
    while diff > tol
        
        for a in a_t
            # i=0
            v0_new[a] = theta_1 * a + beta * (euler .+ log.(exp.(v0) + exp.(v1)))' * P_0[a, :]
            
            # i=1
            v1_new[a] = R + beta * (euler .+ log.(exp.(v0) + exp.(v1)))' * P_1[a, :]
        end

        # next iteration
        diff = max(maximum(abs.(v0 .- v0_new)), maximum(abs.(v1 .- v1_new)))
        v0 = copy(v0_new)
        v1 = copy(v1_new)
    end

    return v0, v1
end

function prob_i1(v_0, v_1, a)
    prb = exp(v_1[a]) / (exp(v_0[a]) + exp(v_1[a])) 
end

function likelihood(params, data)
    theta_1, R = params

    # compute v0, v1 for given parameters
    v0, v1 = vfi_i(theta_1, R)

    # probability of replacement for each a
    lik = zeros(nrow(data))
    for n in 1:nrow(data)
        prb = prob_i1(v0, v1, data[n, :a])
        lik[n] = data[n, :i] * prb + (1 - data[n, :i]) * (1 - prb)
    end
    
    return -sum(log.(lik))
end

h = 1e-5 
function numerical_hessian(f, x)
    n = length(x)
    hessian = zeros(n, n)
    fx = f(x)
    for i in 1:n
        for j in 1:n
            x_ij = copy(x)
            x_ij[i] += h
            x_ij[j] += h
            fxij = f(x_ij)
            
            x_i = copy(x)
            x_i[i] += h
            fxi = f(x_i)
            
            x_j = copy(x)
            x_j[j] += h
            fxj = f(x_j)
            
            hessian[i,j] = (fxij - fxi - fxj + fx) / (h^2)
        end
    end
    return hessian
end

# state transitions ---------------------------------------
# a_{t+1}
a_t0_next = a_t_next.(a_t, 0)
a_t1_next = a_t_next.(a_t, 1)

# probability matrix
P_0 = zeros((5, 5))
P_1 = zeros((5, 5))
for i in 1:5
    P_0[a_t[i], a_t0_next[i]] = 1
    P_1[a_t[i], a_t1_next[i]] = 1
end


# 3. ------------------------------------------------------
v0, v1 = vfi_i(-1, -3)

# plot
plot(1:5, v0, label=L"\bar{V}_0\left(a_t\right)", marker=:o)
plot!(1:5, v1, label=L"\bar{V}_1\left(a_t\right)", marker=:x)

xlabel!(L"a_t")
title!("Alternative-specific value functions")
savefig(joinpath(HOME_DIR, "figs/vfi_plot.png"))

# probability of replacing
prob_i1(v0, v1, 2)

# PDV for {a_t = 4, eps_0t = 1, eps_1t = -1.5}
max(v0[4] + 1, v1[4] - 1.5)

# 4. ------------------------------------------------------
# read data
data = CSV.File(joinpath(HOME_DIR, "HW1_Rust_data.asc"), header=false, delim="        ") |> DataFrame
column_names = [:a, :i]
rename!(data, column_names)

# convert to numeric
data = transform(data, :a => ByRow(x -> round(Int, parse(Float64, strip(x)))) => :a, :i => ByRow(x -> round(Int, parse(Float64, strip(x)))) => :i)

# define relevant parameters
x0 = [-1.0, -3.0]
ll(x) = likelihood(x, data)

# minimize
opt = optimize(ll, x0, BFGS())
parameters = Optim.minimizer(opt)

# compute hessian of ll at result
hessian_matrix = numerical_hessian(ll, parameters)

# std. errors
se = sqrt.(diag(inv(hessian_matrix)))

# 7. ------------------------------------------------------
ccp = combine(groupby(data, :a), :i => mean => :p_0) # i=1 is renewal action
ccp.p_1 = 1 .- ccp.p_0
ccp.va_v0 = log.(ccp.p_1) .- log.(ccp.p_0)

# sort
ccp = sort(ccp, :a)

# objective function to minimize
function hotz_miller(params, va_v0 = ccp.va_v0, A = I(5))
    
    # parameters
    theta_1, R = params
    
    # normalize u
    u = theta_1 .* a_t .- R

    # loop over different states
    rhs = zeros(5)
    for x in a_t 
        rhs[x] = u[x] + beta * P_1[x, :] ⋅ log.(ccp.p_0) - beta * P_0[x, :] ⋅ log.(ccp.p_0) # had to change P_0 and P_1 since i=1 is a=0 (Lecture 3 notation)
    end
    
    g = va_v0 .- rhs
    return g' * A * g
end

# minimize
opt = optimize(hotz_miller, x0)
parameters = Optim.minimizer(opt)
