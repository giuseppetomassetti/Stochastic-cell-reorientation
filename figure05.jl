# Code for plotting Figure 5 (stationary_state_kappa.png)

# FokkerPlanck.jl
using DifferentialEquations
using Plots  
using SpecialFunctions   # for besseli
using Roots
using LaTeXStrings
using DataFrames
using CSV
using LinearAlgebra
using Printf
using Debugger



# ===========================
# Parameters
# ===========================

prefix = ""
suffix = "_stationary"  # Suffix for output files
T    = 20.0         # Timespan
ε    = 1.0         # Strain amplitude
eps_max = ε        
τ    = sqrt(0.1)    # Square root of the pseudo temperature  (works)
τ   = 0.1         # Square root of the pseudo temperature modified, gives strange result
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

N = 1000
θ_min = 0.0
θ_max = π
dθ = (θ_max - θ_min) / N
θ_grid = [θ_min + i * dθ for i in 0:(N - 1)] 


function besseli_ratio_asympt(n::Integer, x::Float64)
    X_SMALL = 1e-3   # switch to small-x asymptotic when x < X_SMALL
    X_LARGE = 1e2   # switch to large-x asymptotic when x > X_LARGE
    if x < X_SMALL
        # small-x: Iₙ(x) ~ (x/2)ⁿ / n!  =>  Iₙ/Iₙ₊₁ ~ (n+1)/(x/2) = 2(n+1)/x
        return 2*(n+1)/x
    elseif x > X_LARGE
        # large-x Debye:  Iₙ/Iₙ₊₁ ~ 1 + (2n+1)/(2x) + (4n^2−1)/(8x^2)
        return 1 + (2n+1)/(2x) + (4n^2 - 1)/(8x^2)
    else
        # “exact” fallback
        return besseli(n, x) / besseli(n+1, x)
    end
end





function estimate_mu_kappa(u, θ_grid, dθ)
    function besseli_ratio(n, x)
        # Compute the ratio of modified Bessel functions of the first kind
        return besseli(n, x) / besseli(n + 1, x)
    end
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
    function estimate_kappa(R; κ_guess=1.0, max_κ=1e6)
        if R < 1e-6
            return 0.0
        end
        # Define the function f(k) whose root we want:
        
        return κ
    end
    μ, R = compute_mu_R(u, θ_grid, dθ)
    f(k) = besseli(1, k/2) - R * besseli(0, k/2)
    f(k) = 1 / besseli_ratio_asympt(0, k/2) - R
    κ_guess = 1.0  # Initial guess for κ
    κ = find_zero(f, κ_guess)
    return μ, κ
end

# ===========================
# Define b(κ), c(κ) as in the paper
b(κ)   = besseli_ratio_asympt(1,κ/2)
h(κ)   = begin
    denom = (besseli_ratio_asympt(0,κ/2) - 1 / besseli_ratio_asympt(0,κ/2) - 2/κ)
    8 / denom
end
G(κ)   = begin
    k2_ = κ/2
    denom = k2_ * (besseli_ratio_asympt(0,k2_) - 1 / besseli_ratio_asympt(0,k2_)) - 1
    8 / denom
end
c(κ)   = b(κ) * h(κ) / G(κ)


# ===========================
# Plots the function whose zero gives kappa_*
# ===========================   

function plotf(min=0.01, max=10.0, τ=0.1, ε=1.0, k1=1.0, k2=1.0)
    tau_bar = sqrt(τ^2 / (ε^2 * k1))
    f(κ) = k2 * b(κ) - tau_bar^2 * c(κ) - 1
    kappas = range(min, stop=max, length=100)
    plot(kappas,f.(kappas), label=L"f(\kappa)", xlabel=L"\kappa", ylabel=L"f(\kappa)", legend=:topright, xscale=:log10)
end
tau_bar = sqrt(τ^2 / (ε^2 * k1))
f(κ) = k2 * b(κ) - tau_bar^2 * c(κ) - 1
plotf(0.01, 100, 10, 1, k1, k2)

k2
k1
b(0.001)
c(0.001)
k2

f(0.001)

# ===========================
# Computes the fixed point κ_*
# ===========================
# This is the value of κ that satisfies the fixed point equation in the paper.
# Note that f is a decreasing function of κ, so it has a unique zero.
function κ_star(τ, ε, k1, k2)
    κ_guess = 1e-3  # Initial guess for κ
    tau_bar = sqrt(τ^2 / (ε^2 * k1))
    f(κ) = k2 * b(κ) - tau_bar^2 * c(κ) - 1
    try
        return find_zero(f, κ_guess)
    catch
        @warn "find_zero failed, returning NaN"
        return NaN
    end
end

κ_star(10, 1, k1, k2)

 # periodic grid on [0, π)
U(θ) = 0.5 * k1 * (cos(2θ) + k2)^2

function f_eq(τ, ε, k1, k2)
    U(θ) = 0.5 * k1 * (cos(2θ) + k2)^2
    f_eq_not_renormalized = exp.(- ε^2/τ^2  * U.(θ_grid))
    partition_function = sum(f_eq_not_renormalized) * dθ
    f_eq = f_eq_not_renormalized./ partition_function
end

plot(θ_grid, f_eq(τ, ε, k1, 5))

function κ_star_tilde(τ, ε, k1, k2)
    f_equil = f_eq(τ, ε, k1, k2)
    ~ , κ_tilde = estimate_mu_kappa(f_equil, θ_grid, dθ)
    return κ_tilde
end



function ρ(κ_star, k)
    return (k * b(κ_star) - 1 ) / c(κ_star)
end

# Define von Mises initial condition
function von_mises_pi(θ, μ, κ)
    try 
        return (1 / (π * besseli(0, κ/2))) * exp((κ/2) * cos(2*(θ - μ)))
    catch
        return 1 / π
    end
end

# ===========================
# Plot of f_eq(θ) and von Mises approximation
# ===========================
begin   
    ρ_values = [1e-1^2, 5^2 * 1e-1^2 , 4 * 1.0^2]
    linestyles = [:solid, :dash, :dot, :dashdot]
    p = plot(grid = true, 
            xlabel=L"\tau", 
            ylabel=L"p_\infty(\theta), \quad \tilde p\,\left(\frac{\pi}{2}, \kappa_*\right)", 
            legend=:topright) 
    for (i, ρ_value) in enumerate(ρ_values)
        f_equil = f_eq(sqrt(ρ_value), 1, 1, k2)
        κ_star_ =  κ_star(sqrt(ρ_value), 1, 1, k2)
        if κ_star_ < 0
            @warn "κ_star is negative, skipping this ρ value"
            @warn "$(κ_star_) is too large for numerical stability"
            @info "ρ = $(ρ_value), κ_star = $(κ_star_)"
            continue
        end
        
        f_equil_approx = von_mises_pi.(θ_grid, π/2, κ_star_)
        plot!(θ_grid, f_equil, label = L"{\tau}/{\varepsilon} = " * "$(sqrt(ρ_value))", ls = linestyles[i], lw = 2, color = :blue)
        plot!(θ_grid, f_equil_approx, ls = linestyles[i], label = :none, lw = 4, alpha = 0.5, color = :blue) 
    end
    display(p)
end
savefig("comparison_equilibria_FP_ODE.png")


