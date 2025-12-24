#=
using Pkg
Pkg.add("DelimitedFiles")
Pkg.add("CSV")
Pkg.add("DataFrames")
=#

# INSTRUCTIONS: This script generates Figure 13c showing the evolution of order parameter S(t) for ε_max = 10%
# Run the entire script to solve the ODE system and produce evolution_S_our_model01.png
# The plot compares theoretical predictions S̃(t) with experimental data points from Mao et al.
# Adjust parameters (eta, k_2, tau) in the Parameters section to modify the model behavior.

begin
using DifferentialEquations
using SpecialFunctions
using Plots
using LaTeXStrings
using CSV
using DataFrames
using Images


kappa_lower = 0
kappa_upper = 1e5

mu0 = 0.5 * π
kappa0 = 0.01
y0 = [mu0, kappa0]

eps_max = 0.02
omega = 0
K = 1.0
tau = 0.03
tau = 0.04



# ===========================
# Parameters
# ===========================
T    = 0.5e4       # Timespan
ε    = 1.0         # Strain amplitude
K    = 1.0         # Effective stiffness
eta  = 30.0               # Viscosity constant
k1   = 1.0         # Energy parameter (tunable)
k2   = -1.0        # Energy parameter (tunable)
ω    = 0           # Frequency of the cyclic stretch (we use zero frequency because we average)


K_s = 0.7
alpha_L = 0.8
K_perp = (K_s - 1 + alpha_L)/(1 + alpha_L)
r = (1 - alpha_L)/(1 + alpha_L)

k_1 = (1 + r)^2 * (1 + K_perp / K - K_s / K)
k_2 = (r - 1) / (r + 1) * (1 - K_perp / K) / (1 + K_perp / K - K_s / K)

k_1 = 1
k_2 = -1.5
k_2 = -2.0

function eps(t, eps_max, omega)
    eps_max * (1 + cos(omega * t)) / 2
end

function Ubar(mu, k_1, k_2)
    0.5 * k_1 * (cos(2 * mu) - k_2)^2
end

function f(mu, kappa, k_1, k_2)
    if kappa < kappa_lower
        return -4 * k_1 * k_2 * sin(2 * mu) / kappa
    elseif kappa > kappa_upper
        return 2 * k_1 * (cos(2 * mu) - k_2) * sin(2 * mu)
    else
        term1 = 1 - 2 * besseli(2, kappa / 2) / ((kappa / 2) * besseli(1, kappa / 2))
        term2 = 2 / kappa + besseli(2, kappa / 2) / besseli(1, kappa / 2)
        return 2 * k_1 * (term1 * cos(2 * mu) - k_2 * term2) * sin(2 * mu)
    end
end

function g(mu, kappa, k_1, k_2)
    if kappa < kappa_lower
        return 8 * k_1 * k_2 * cos(2 * mu)
    elseif kappa > kappa_upper
        return 64 * k_1 * (k_2 * cos(2 * mu) - cos(4 * mu)) * kappa
    else
        term1 = besseli(0, kappa / 2) / besseli(1, kappa / 2)
        term2 = besseli(1, kappa / 2) / besseli(0, kappa / 2)
        num = 8 * k_1 * (r + 1) * (k_2 * cos(2 * mu) - besseli(2, kappa / 2) / besseli(1, kappa / 2) * cos(4 * mu))
        denom = (kappa / 2) * (term1 - term2) - 1
        return num / denom
    end
end

function h(kappa)
    if kappa < kappa_lower
        return 2 * kappa
    elseif kappa > kappa_upper
        return 8 * kappa^2
    else
        term1 = besseli(0, kappa / 2) / besseli(1, kappa / 2)
        term2 = besseli(1, kappa / 2) / besseli(0, kappa / 2)
        return 8 / (term1 - term2 - 2 / kappa)
    end
end

function evolution!(du, u, p, t)
    mu, kappa = u
    du[1] = (eps_max^2 * K / eta) * (1 + cos(omega * t))/2 * f(mu, kappa, k_1, k_2)
    du[2] = (eps_max^2 / eta) * K * (1 + cos(omega * t))/2 * g(mu, kappa, k_1, k_2) - K / eta  * tau^2 * h(kappa)
end

tspan = (0.0, T)
results = Dict{Float64, Tuple{Vector{Float64}, Vector{Float64}}}()

function mao_order_parameter(μ, κ)
     I0 = besseli(0, κ / 2)
     I1 = besseli(1, κ / 2)
     return (I1 * cos(2 * μ)) / I0
end


for (i, eps_max) in enumerate([0.1] * sqrt(3/2))

    function evolution!(du, u, p, t)
          mu, kappa = u
          du[1] = (eps_max^2 * K / eta) * (1 + cos(omega * t))/2 * f(mu, kappa, k_1, k_2)
          du[2] = (eps_max^2 / eta) * K * (1 + cos(omega * t))/2 * g(mu, kappa, k_1, k_2) - K / eta  * tau^2 * h(kappa)
    end
    
    prob = ODEProblem(evolution!, y0, tspan)

    sol = solve(prob, Tsit5(), saveat=0.1)
    
    μs = sol[1, :]
    κs = sol[2, :]
    ts = sol.t
        
    # Compute the order parameter S(t)
    S_MAO = [mao_order_parameter(μs[i], κs[i]) for i in 1:length(μs)]
    
    # Save in the dictionary
    results[i] = (ts, S_MAO)
    
    # Define dataframes
    df_S = DataFrame(time=ts, S=S_MAO)
    df_mu = DataFrame(time=ts, mu=μs ./ π)
    df_kappa = DataFrame(time=ts, kappa=κs)
    
     # Save into CSV files
    CSV.write("S_MAO_$(eps_max).csv", df_S)   
    CSV.write("mu_ODE_MAO_$(eps_max).csv", df_mu)
    CSV.write("kappa_ODE_MAO_$(eps_max).csv", df_kappa)
    
    # Plot μ(t)/π
    #plot(ts, μs ./ π,
    #     label=L"\mu(t) / \pi",
    #     xlabel="Time",
    #     title="μ(t)/π for ε_max = $eps_max")
    #savefig("mu_ODE_MAO_$(eps_max).png")
    
    # Plot κ(t)
    #plot(ts, κs,
    #     label=L"\kappa(t)",
    #     xlabel="Time",
    #     title="κ(t) for ε_max = $eps_max")
    #savefig("kappa_ODE_MAO_$(eps_max).png")
    
    S_MAO = [mao_order_parameter(μs[i], κs[i]) for i in 1:length(μs)]

    results[eps_max] = (ts, S_MAO)
    
end



# Combined plot of all S(t)
p = plot(xlabel=L"t/t_{\rm final}", ylabel=L"S(t)", legend=:topright,
     guidefontsize=16,    # labels on the axes
     tickfontsize=12,     # numbers on the axes
     legendfontsize=12,   # legend
     titlefontsize=20     # title
)

for (eps_max, (ts, S_MAO)) in sort(collect(results))
    ts_norm = ts ./ maximum(ts) 
    plot!(p, ts_norm, S_MAO, label="ε_max = $eps_max")
end


begin
    # Combined plot of all S(t)
    p = plot(xlabel=L"t/t_{\rm final}", legend=:topright,
        guidefontsize=16,    # labels on the axes
        tickfontsize=12,     # numbers on the axes
        legendfontsize=12,   # legend
        titlefontsize=20,     # title
        xtickfontsize=14, ytickfontsize=14)


    (ts, S_MAO) = results[1]

    ts_norm = ts ./ maximum(ts) 
    plot!(p, ts_norm, S_MAO, label=L"\tilde S(t)")

    # Experimental data set epsilon = 10%
    t_exp = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    S_exp = [-0.02117647058823513, -0.5152941176470589, -0.7929411764705883, -0.9223529411764706, -0.9247058823529412, -0.9411764705882353, -0.9647058823529411, -0.9741176470588236, -0.9482352941176468, -0.9576470588235294]

    # Superpose the experimental data
    scatter!(p, t_exp, S_exp, color=:blue, label = L"S(t)" * " (dataset)", marker=:circle, markersize=3)
    display(p)
    savefig("evolution_S_our_model01.png")
end


    # Experimental data (from Mao et al.) — time and S(t) for epsilon = 2%
    t_exp = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    S_exp = [-0.08168427263830536, -0.26481276265039433, -0.3702579010995336, -0.4605592654424039, -0.5077571815094122, -0.5441756375568474, -0.5482737032986011, -0.530812849001209, -0.5433193253123019, -0.5299493408554483]


# Experimental data set epsilon = 5%
    t_exp = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    S_exp = [-0.06937799043062198, -0.7607655502392344, -0.8564593301435408, -0.832535885167464, -0.8229665071770336, -0.8397129186602872, -0.84688995215311, -0.8564593301435409, -0.8301435406698565]


# Add to the existing plot
#scatter!(p, t_exp, S_exp, color=:green,  label = "strain 5%", marker=:diamond, markersize=3)



# Add to the existing plot
#scatter!(p, t_exp, S_exp, color=:black, marker=:square, markersize=3)


savefig("evolution_S_our_model10.png")

display(p)
end

