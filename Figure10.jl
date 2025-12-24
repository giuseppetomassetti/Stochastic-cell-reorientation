# ============================================================================
# Code to generate Figure 10: mu_comparison_second.png and kappa_comparison_second.png
# ============================================================================
"""
This code generates Figure 10, which consists of two comparison plots:
  - Plot (a): mu_comparison_second.png — mean orientation angle μ/π vs t/t_max
  - Plot (b): kappa_comparison_second.png — concentration parameter κ vs t/t_max

Both plots compare:
  - Solid lines: ODE solutions (μ(t), κ(t)) from the reduced 2D system
  - Semi-transparent/thinner lines: Fokker-Planck extracted values (μ̄(t), κ̄(t))

INSTRUCTIONS TO GENERATE THE FIGURES:
=====================================

1. Prerequisites:
   - Install Julia (https://julialang.org/)
   - Install required packages:
     ```julia
     using Pkg
     Pkg.add(["DifferentialEquations", "Plots", "SpecialFunctions", "Roots", 
              "LaTeXStrings", "DataFrames", "CSV", "LinearAlgebra"])
     ```

2. Parameters (can be modified):
   - Line 79: T = 10.0 — Time span for the simulation
   - Line 80: ε = 1.0 — Strain amplitude
   - Line 84: τ = 0.1 — Pseudo-temperature
   - Line 107: mu0 = π/4 — Initial mean orientation angle
   - Line 108: κ_values = [0.5, 2.0, 5.0] — Initial concentration parameter values to compare

3. Run the code:
   ```julia
   include("Figure10.jl")
   ```
   Or in Julia REPL:
   ```julia
   julia> include("Figure10.jl")
   ```

4. Output files:
   - mu_comparison_second.png (saved at line 339)
   - kappa_comparison_second.png (saved at line 362)

WHAT THE CODE DOES:
==================
- Uses von Mises initial distribution with μ₀ = π/4 and different κ₀ values
- Solves the Fokker-Planck equation numerically for each κ₀ value
- Extracts μ̄(t) and κ̄(t) from Fokker-Planck solutions using circular moments
- Solves the reduced 2D ODE system for each κ₀ value
- Compares ODE solutions with Fokker-Planck extracted values
- Shows how different initial concentrations affect the evolution

KEY FEATURES:
============
- Initial condition: μ₀ = π/4 (boundary of spinodal region)
- Multiple κ₀ values: [0.5, 2.0, 5.0]
- The concentration parameter κ shows non-monotonic behavior (peaks before settling)
- Good agreement between ODE and Fokker-Planck solutions
"""

using DifferentialEquations
using Plots  
using SpecialFunctions   # for besseli
using Roots
using LaTeXStrings
using DataFrames
using CSV
using LinearAlgebra


begin
# ===========================
# Parameters
# ===========================

prefix = "numerical_test_"
suffix = "_uniform"  # Suffix for output files
T    = 10.0         # Timespan
ε    = 1.0         # Strain amplitude
eps_max = ε        


τ    = 0.1       # Square root of the pseudo temperature modified, gives strange result
K    = 1.0         # Effective stiffness
η    = 1.0         # Viscosity constant
E    = K * τ^2     # Pseudo thermal energy

α    = K * ε^2 / η # Coefficient in Omega
d    = E / (η)       # Diffusion constant
r    = 0.0         # Strain ratio

K_s = 0.7
alpha_L = 0.8

K_perp = (K_s - 1 + alpha_L)/(1 + alpha_L)
r = (1 - alpha_L)/(1 + alpha_L)

k1 = (1 + r)^2 * (1 + K_perp / K - K_s / K)
k2 = (1 - r) / (r + 1) * (1 - K_perp / K) / (1 + K_perp / K - K_s / K)


kappa_lower = 0
kappa_upper = 1e10


mu0 = π / 4     # Initial mean angle at 
κ_values = [0.5 2.0 5.0] # 2.0 5.0 20.0]
κ_guess = 40 #  40Initial guess for κ

# ===========================
# Domain Discretization
# ===========================
N = 100
θ_min = 0.0
θ_max = π
dθ = (θ_max - θ_min) / N
θ_grid = [θ_min + i * dθ for i in 0:(N - 1)]  # periodic grid on [0, π)


# ===========================
# Reorientation Rate (Drift)
# ===========================
function Omega(θ, t)
    return 2 * α * k1 * (cos(2θ) + k2) * sin(2θ) 
end

# ===========================
# Fokker-Planck RHS (Method of Lines)
# ===========================
function fp_rhs!(du, u, p, t)
    for i in 1:N
        im = i == 1 ? N : i - 1
        ip = i == N ? 1 : i + 1

        adv = (Omega(θ_grid[ip], t) * u[ip] - Omega(θ_grid[im], t) * u[im]) / (2 * dθ)
        diff = K * τ^2 / η * (u[ip] - 2 * u[i] + u[im]) / dθ^2

        du[i] = -adv + diff
    end
end

# Define von Mises initial condition
function von_mises_pi(θ, μ, κ)
    return (1 / (π * besseli(0, κ/2))) * exp((κ/2) * cos(2*(θ - μ)))
end


# ===========================
# Solve the ODE Problem
# ===========================
FP_solutions = []
for kappa0 in κ_values
    u0 = von_mises_pi.(θ_grid, mu0, kappa0)
    tspan = (0.0, T)
    prob = ODEProblem(fp_rhs!, u0, tspan)
    sol = solve(prob, Tsit5(), saveat=0.1)
    push!(FP_solutions, sol)
end 

# ===========================
# Von Mises Fitting Functions
# ===========================
function compute_mu_R(p::Vector{Float64}, θ::Vector{Float64}, dθ::Float64)
    # Compute the normalization constant (integral approximation)
    Z = sum(p) * dθ
    p_norm = p ./ Z  # Normalize the density

    # Compute the second circular moment: ⟨cos(2θ)⟩ and ⟨sin(2θ)⟩
    c2 = sum(cos.(2 .* θ) .* p_norm) * dθ
    s2 = sum(sin.(2 .* θ) .* p_norm) * dθ

    # Compute the "double-angle" mean
    # Modulo operation ensures that the angle is within the range [0, 2π)
    # atan returns an angle in (-π, π], but we need it in [0, 2π) for consistency.
    μtimes2 = mod(atan(s2, c2),2*pi)

    # Now μ = 0.5 * μtimes2 maps the result to [0, π)
    μ = 0.5 * μtimes2

    # Compute the magnitude of the second moment vector
    R = sqrt(c2^2 + s2^2)

    return μ, R
end

# ===========================
# Compute κ from R
function estimate_kappa(R; κ_guess=1.0, max_κ=1e6)
    if R < 1e-6
        return 0.0
    end
    # Define the function f(k) whose root we want:
    
    return κ
end

# ===========================
# Estimate μ and κ from p(θ,t)
# ===========================
# This function estimates the mean angle μ and concentration parameter κ from the probability density p(θ,t).
# It uses the second circular moment to compute μ and then finds κ by solving the equation
# involving the modified Bessel functions of the first kind.
# The function returns the estimated values of μ and κ.
# The function uses a global variable κ_guess to store the previous estimate of κ, which is
# used as an initial guess for the next estimate. This helps in converging to the correct
# value of κ more efficiently.
# The function also updates the global lists mu_tilda_vals and kappa_tilda_vals with
# the estimated values of μ and κ for each time step.
function estimate_mu_kappa(u, θ_grid, dθ)
    μ, R = compute_mu_R(u, θ_grid, dθ)
    #f(k) = besseli(1, k/2) - R * besseli(0, k/2)
    #
    f(k) = besseli(1, k/2) - R * besseli(0, k/2)
    κ = find_zero(f, κ_guess)
    return μ, κ
end

# ===========================
# Compute μ_tilde(t), κ_tilde(t)
# ===========================

mu_FP_list = []
kappa_FP_list = []
for sol in FP_solutions
    mu_tilda_vals = Float64[]
    kappa_tilda_vals = Float64[]
    for u in sol.u
        μ, κ = estimate_mu_kappa(u, θ_grid, dθ)
        push!(mu_tilda_vals, μ)
        push!(kappa_tilda_vals, κ)
    end
    push!(mu_FP_list,mu_tilda_vals)
    push!(kappa_FP_list,kappa_tilda_vals)
end



# ===========================
# Define U(θ)
# ===========================
function U(θ)
    0.5 * k1 * (cos(2θ) + k2)^2
end

# Compute U(θ)
U_vals = [U(θ) for θ in θ_grid]

# ===========================
# Compute the solution of the ODE system
# ===========================

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
        num = 8 * k_1 * (k_2 * cos(2 * mu) - besseli(2, kappa / 2) / besseli(1, kappa / 2) * cos(4 * mu))
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

#global τ

function evolution!(du, u, p, t)
    #global τ
    mu, kappa = u
    du[1] = α * f(mu, kappa, k1, - k2)
    du[2] = α * g(mu, kappa, k1, - k2) - K * τ^2 * h(kappa)
end


mu_ODE_list = []
kappa_ODE_list = []
for (i, kappa0) in enumerate(κ_values)
    tspan = (0.0, T)
    #mu0 = π/4       # Initial mean angle
    y0 = [mu0, kappa0]
    prob = ODEProblem(evolution!, y0, tspan)
    sol_ODE = solve(prob, Tsit5(), saveat=0.1)
    mu_ODE_vals = sol_ODE[1,:]
    kappa_ODE_vals = sol_ODE[2,:]
    t_ODE_vals = sol_ODE.t
    push!(mu_ODE_list,mu_ODE_vals)
    push!(kappa_ODE_list,kappa_ODE_vals)
end


# ===========================
# Plot μ_tilda(t) and κ_tilda(t) 
# ===========================
begin
    linestyles = [:solid :dash :dot :dashdot :dashdotdot :solid]
    colors = [:red :blue :black :red :blue :black]
    p = plot(
            xlabel = L"t/t_{\rm max}",
            ylabel = L"\overline{\mu}/\pi\,, \mu/\pi",
            ls = [:solid :dash],
            color = [:blue :red],     
            legend = :topright,  
            legendfontsize = 14,
            guidefontsize = 20,       # Axis label font size  
            tickfontsize = 13,        # Tick label font size
            titlefontsize = 16, 
            lw = 2)
    for (i , kappa0) in enumerate(κ_values)
        plot!([0:0.1:T] / T,mu_ODE_list[i]/pi, ls=linestyles[i], color = colors[i], label = L"\kappa_0 = " * "$(kappa0)", lw = 2)
        plot!([0:0.1:T] / T,mu_FP_list[i]/pi, ls=linestyles[i], color = colors[i],
        label = "", lw=4, alpha = 0.5)
    end
    savefig("mu_comparison_second.png")
    display(p)

    linestyles = [:solid :dash :dot :dashdot :dashdotdot :solid]
    colors = [:red :blue :black :blue :red :black]
    p = plot(
            xlabel = L"t/t_{\rm max}",
            ylabel = L"\overline{\kappa}\,, \kappa",
            ls = [:solid :dash],
            color = [:blue :red],     
            legend = :topright,  
            legendfontsize = 14,
            guidefontsize = 20,       # Axis label font size  
            tickfontsize = 13,        # Tick label font size
            titlefontsize = 16, 
            lw = 2)
    for (i , kappa0) in enumerate(κ_values)
        plot!([0:0.1:T] / T,kappa_ODE_list[i], ls=linestyles[i], color = colors[i], label = L"\kappa_0 = " * "$(kappa0)", lw = 2)
        plot!([0:0.1:T] / T,kappa_FP_list[i], ls=linestyles[i], color = colors[i],
        label = "", lw=4, alpha = 0.5)
    end
    display(p)

    savefig("kappa_comparison_second.png")
end

   
end

