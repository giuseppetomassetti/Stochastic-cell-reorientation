# ============================================================================
# Figure 12: Plot of S_tilde(μ, κ) as a function of κ
# ============================================================================
# 
# DESCRIPTION:
#   This script generates a plot of the function S_tilde(μ, κ) = 
#   (I₁(κ/2) / I₀(κ/2)) * cos(2μ) as a function of κ (kappa) for a fixed 
#   value of μ (mu). The function uses modified Bessel functions of the 
#   first kind.
#
# REQUIREMENTS:
#   - Julia packages: Plots, SpecialFunctions, LaTeXStrings
#   - Install missing packages with: using Pkg; Pkg.add(["Plots", "SpecialFunctions", "LaTeXStrings"])
#
# USAGE:
#   - Run this script in Julia: include("Figure12.jl")
#   - Or execute it directly in a Julia REPL or IDE
#
# OUTPUT:
#   - Displays the plot in a window
#   - Saves the plot as "plotStilda.png" in the current working directory
#
# PARAMETERS:
#   - κ (kappa) range: 0.1 to 20.0 (100 points)
#   - μ (mu) value: π/2 (90 degrees)
#
# ============================================================================

begin 

using Plots
using SpecialFunctions
using LaTeXStrings

# Define the function S_tilde
function S_tilde(μ, κ)
    return (besselix(1, κ / 2) / besselix(0, κ / 2)) * cos(2μ)
end

# Generate data for plotting
κ_values = range(0.1, stop=20.0, length=100)  # Range for κ
μ_values = [π/2]  # Specific values of μ to plot

# Plot the function for different values of μ
p = plot(xlabel=L"\kappa", ylabel=L"", titlefontsize=16, xlabelfontsize=18, ylabelfontsize=18, 
        legendfontsize=12,xtickfontsize=14, ytickfontsize=14, label = L"\tilde S(t)")

for μ in μ_values
    S_values = [S_tilde(μ, κ) for κ in κ_values]
    plot!(κ_values, S_values, label="")
end

# Display the plot
display(p)

savefig("plotStilda")

end