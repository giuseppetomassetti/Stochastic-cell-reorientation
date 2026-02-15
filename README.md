# Code repository for the paper "Using stochastic thermodynamics with internal variables to capture orientational spreading in cell populations undergoing cyclic stretch"

**Authors:** Rohan Abeyaratne, Sanjay Dharmaravam, Giuseppe Saccomandi, Giuseppe Tomassetti

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite it using the metadata in the [CITATION.cff](CITATION.cff) file. GitHub users can also use the "Cite this repository" button in the sidebar.

## Scripts Overview

This repository contains Julia scripts and Jupyter notebooks that generate all figures for the paper. Each script is self-contained and can be run independently.

### Figure Scripts

#### **figure04a.ipynb** (Jupyter Notebook)
- Interactive visualization of the phase portrait for the reduced dynamical system
- Allows exploration of the two-stage reorientation phenomenon
- Uses Python with NumPy, SciPy, and Matplotlib

#### **figure04b.ipynb** (Jupyter Notebook)
- Additional phase portrait analysis and visualization
- Complements Figure 4a with different parameter settings

#### **figure05.jl**
- Generates plots of the stationary state distribution for κ (concentration parameter)
- Compares theoretical predictions with numerical solutions
- Output: `stationary_state_kappa.png`

#### **figure06.ipynb** (Jupyter Notebook)
- Comprehensive analysis of the stationary state
- Comparison between Fokker-Planck solutions and ODE approximations

#### **figure07.jl**
- Plots relative error between theoretical and numerical stationary states
- Validates the accuracy of the reduced model
- Output: `stationary_state_relative_error.png`

#### **figure08.jl**
- Analysis of the evolution dynamics
- Time-dependent behavior of the orientation distribution

#### **figure09.jl**
- Comparison of different solution methods
- Validation of numerical schemes

#### **Figure10.jl**
- Detailed phase portrait analysis
- Visualization of trajectories in the (μ, κ) phase space

#### **Figure11.jl**
- Comprehensive comparison with experimental data
- Analysis of cell reorientation dynamics under different strain conditions

#### **Figure12.jl**
- Plot of the order parameter function S̃(μ, κ) as a function of κ
- Visualizes the relationship between concentration parameter and order parameter
- Output: `plotStilda.png`

#### **Figure13a.jl**
- Evolution of order parameter S(t) for ε_max = 5% strain
- Compares theoretical predictions S̃(t) with experimental data from Mao et al.
- Output: `evolution_S_our_model005.png`

#### **Figure13b.jl**
- Evolution of order parameter S(t) for ε_max = 2% strain
- Compares theoretical predictions S̃(t) with experimental data from Mao et al.
- Output: `evolution_S_our_model002.png`

#### **Figure13c.jl**
- Evolution of order parameter S(t) for ε_max = 10% strain
- Compares theoretical predictions S̃(t) with experimental data from Mao et al.
- Output: `evolution_S_our_model01.png`

#### **FigureE-9.py**
- Generates Figure E-9 for the Electronic Supplementary Material (thermodynamic quantities in the transient regime).
- Numerical simulation of the Fokker-Planck equation and thermodynamics analysis.
- Compares exact thermodynamic quantities (entropy, free energy, entropy production) with those derived from the reduced-order (von Mises) model.
- Output: `thermodynamics_comparison_full.jpg`
- Accompanied by **Thermodynamics_Explanation.md** which provides the mathematical background.

## Requirements

### Julia Scripts
- Julia 1.8 or later
- Required packages: `DifferentialEquations`, `Plots`, `SpecialFunctions`, `LaTeXStrings`, `CSV`, `DataFrames`, `Roots`, `LinearAlgebra`

Install packages with:
```julia
using Pkg
Pkg.add(["DifferentialEquations", "Plots", "SpecialFunctions", "LaTeXStrings", "CSV", "DataFrames", "Roots", "LinearAlgebra"])
```

### Jupyter Notebooks
- Python 3.7 or later
- Required packages: `numpy`, `matplotlib`, `scipy`, `ipywidgets`

Install packages with:
```bash
pip install numpy matplotlib scipy ipywidgets
```

## Usage

1. **For Julia scripts:** Run directly in Julia REPL or IDE:
   ```julia
   include("Figure13a.jl")
   ```

2. **For Jupyter notebooks:** Open in Jupyter Notebook or JupyterLab and run all cells:
   ```bash
   jupyter notebook figure04a.ipynb
   ```

Each script will generate the corresponding figure and save it as a PNG file in the current working directory. Some scripts also generate CSV files with numerical data.

## Notes

- All scripts are self-contained and can be run independently
- Parameter values can be adjusted in the "Parameters" section of each script
- Output filenames are specified within each script
- The experimental data used for comparison is embedded in the scripts (Figures 13a-c)

