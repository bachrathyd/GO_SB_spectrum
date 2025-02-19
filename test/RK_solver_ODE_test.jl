# Testgin the RK solver with Butcher Tables - compared to the Diff.Eq. ODE solver
using Plots
theme(:dark)#:vibrant:dracula:rose_pine
plotly()
#const PLOTS_DEFAULTS = Dict(:theme => :wong, :fontfamily => "Computer Modern", :label => nothing, :dpi => 600 )const PLOTS_DEFAULTS = Dict(:theme => :wong, :fontfamily => "Computer Modern", :label => nothing, :dpi => 600 )
default(size=(500, 500), titlefont=(15, "times"), legendfontsize=13, guidefont=(12, :white), tickfont=(12, :orange), guide="x", framestyle=:zerolines, yminorgrid=true, fontfamily="Computer Modern", label=nothing, dpi=600)

#using Profile
using StaticArrays
using DifferentialEquations

using LinearAlgebra


include("./src/DelaySolver.jl")
using .DelaySolver


# --- Example usage ---
Nsol = 10000       # Total number of time points.
h = 0.01              # Fixed timestep.
tv = (0:Nsol-1) .* h
const Adyn = @SMatrix [0 1;
    -10.0 -0.1]
#f(t, x) = SA[x[2], -x[1]*10.0-x[2]*0.1]
f(t, x) = Adyn * x



foo_diff(u, p, t) = f(t, u)
x = [SA[1.0, 0.5] for _ in 1:Nsol]
# # tested pairs:
# Buchertable(1) - Euler()# tökéletes
# Buchertable(2) - Midpoint()# tökéletes
# Buchertable(3) - BS3() # tökéletes
# Buchertable(4) - RK4() # tökéletes
# Buchertable(5) - DP5()  # csak kbra stimmel

# Now complete the simulation using RK4.
j = 1
@time runge_kutta_solve!(f, h, x, j, Nsol; BT=Buchertable(5), t0=0.0);
plot(tv, getindex.(x, 1))
plot!(tv, getindex.(x, 2))

probMathieu = ODEProblem(foo_diff, [1.0, 0.5], (0.0, Nsol * h))
@time sol = solve(probMathieu, Tsit5(), dt=h * 1.0000000, adaptive=false);
plot!(sol, denseplot=false,linestyle=:dash,lw=3)

@show norm(sol.u[1:end-1]-x)

@show @benchmark runge_kutta_solve!(f, h, x, j, Nsol; BT=Buchertable(4), t0=0.0)

@show @benchmark solve(probMathieu, RK4(), dt=h * 1.0, adaptive=false) # Tökéletesen ugyan azt adja egyébként

#BT = Buchertable(3)
#
#using RungeKutta
#BT = TableauGauss(Float64, 10)
#BT = TableauRK5()

