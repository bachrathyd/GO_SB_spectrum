5 + 5

using BenchmarkTools
using Plots
theme(:dark)#:vibrant:dracula:rose_pine
plotly()
#const PLOTS_DEFAULTS = Dict(:theme => :wong, :fontfamily => "Computer Modern", :label => nothing, :dpi => 600 )const PLOTS_DEFAULTS = Dict(:theme => :wong, :fontfamily => "Computer Modern", :label => nothing, :dpi => 600 )
default(size=(500, 500), titlefont=(15, "times"), legendfontsize=13, guidefont=(12, :white), tickfont=(12, :orange), guide="x", framestyle=:zerolines, yminorgrid=true, fontfamily="Computer Modern", label=nothing, dpi=600)

#using Profile
using StaticArrays
using DifferentialEquations

using LinearAlgebra

# using MDBM

using FileIO
using Suppressor

# using MDBM
#----------------------------


# Governing equation
# Governing equation

function DelayMathieu(u, h, p, t)
    # Parameters
    ζ, δ, ϵ, b, τ, T = p
    #External forcing
    F = 0.1 * (cos(2pi * t / T) .^ 10)
    # Components of the delayed differential equation
    dx = u[2]
    ddx = -(δ + ϵ * cos(2pi * t / T)) * u[1] - 2 * ζ * u[2] + b * h(p, t - τ)[1] + F
    # Update the derivative vector
    SA[dx, ddx]
end
Base.:+(a::SVector, b::Bool) = a .+ b
Base.:+(a::SVector, b::Float64) = a .+ b #TODO: where to put this?


## parameters 
ζ = 0.02          # damping coefficient
δ = 1.5#0.2          # nat. freq
ϵ = 0.15#4#5#8;#5         # cut.coeff
τ = 2pi          # Time delay
b = 0.5
T = 2pi
p = ζ, δ, ϵ, b, τ, T
#p = (ζ, ωn, k, τ,10.0)

# test simulation ---------------
#initial condition
u0 = SA[1.0, 0.0]
#history function
h(p, t) = SA[0.0; 0.0]
Tsim=100
probMathieu = DDEProblem(DelayMathieu, u0, h, (0.0, Tsim), p; constant_lags=[τ])
 
Δt=0.01
#Parameters for the solver as a Dict (it is necessary to collect it for later use)
Solver_args_BS3 = Dict(:alg => MethodOfSteps(BS3()), :adaptive => false, :verbose => false, :dt => Δt, :saveat => 0:Δt:Tsim)#
Solver_args_RK4 = Dict(:alg => MethodOfSteps(RK4()), :adaptive => false, :verbose => false, :dt => Δt, :saveat => 0:Δt:Tsim)#
Solver_args_V9 = Dict(:alg => MethodOfSteps(Vern9()), :adaptive => false, :verbose => false, :dt => Δt, :saveat => 0:Δt:Tsim)#
#Solver_args = Dict()#
@benchmark sol_BS3 = solve(probMathieu; Solver_args_BS3...)#abstol,reltol
@benchmark sol_RK4 = solve(probMathieu; Solver_args_RK4...)#abstol,reltol
@benchmark sol_V9 = solve(probMathieu; Solver_args_V9...)#abstol,reltol
scatter(sol.t[1:end-1],diff(sol.t))
plot(sol)

#last period of the long simulation:
t_select_period=0.0:0.01:T
t_select_delay=eriod=0.0:0.01:τ
sol_period=sol(sol.t[end] .- t_select_period)
sol_delay=sol(sol.t[end] .- t_select_delay)

#plot the state phase of the last segment
plot(sol_period.t,getindex.(sol_period.u,1))
plot!(sol_period.t,getindex.(sol_period.u,2))
plot!(sol_delay.t,getindex.(sol_delay.u,1))
plot!(sol_delay.t,getindex.(sol_delay.u,2))
#plot the phase space (u - du)
plot(getindex.(sol_delay.u,1),getindex.(sol_delay.u,2))
plot!(getindex.(sol_period.u,1),getindex.(sol_period.u,2))







