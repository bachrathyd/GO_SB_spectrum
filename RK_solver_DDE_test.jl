# Testgin the RK solver with Butcher Tables - compared to the Diff.Eq. ODE solver
using Plots
theme(:dark)#:vibrant:dracula:rose_pine
plotly()
#const PLOTS_DEFAULTS = Dict(:theme => :wong, :fontfamily => "Computer Modern", :label => nothing, :dpi => 600 )const PLOTS_DEFAULTS = Dict(:theme => :wong, :fontfamily => "Computer Modern", :label => nothing, :dpi => 600 )
default(size=(1500, 800), titlefont=(15, "times"), legendfontsize=13, guidefont=(12, :white), tickfont=(12, :orange), guide="x", framestyle=:zerolines, yminorgrid=true, fontfamily="Computer Modern", label=nothing, dpi=600)

#using Profile
using StaticArrays
using DifferentialEquations


using LinearAlgebra

using KrylovKit

#using Revise
include("./src/DelaySolver.jl")
#using .DelaySolver


#---------------------------- Solution with precomputed A-s -------------------------
using SemiDiscretizationMethod

function createMathieuProblem(δ, ε, b0, a1; T=2π)
    AMx = ProportionalMX(t -> @SMatrix [0.0 1.0; -δ-ε*cos(2π / T * t) -a1])
    τ1 = t -> 2π # if function is needed, the use τ1 = t->foo(t)
    BMx1 = DelayMX(τ1, t -> @SMatrix [0.0 0.0; b0 0.0])
    cVec = Additive(t -> @SVector [0.0, 0.0 * sin(4π / T * t)])
    LDDEProblem(AMx, [BMx1], cVec)
end;

τmax = 2π # the largest τ of the system

## parameters 
ζ = 0.02          # damping coefficient
δ = 1.5#0.2          # nat. freq
ϵ = 0.15#4#5#8;#5         # cut.coeff
τ = 5.0#2pi          # Time delay
b = 0.5
T = 6#2pi
mathieu_lddep = createMathieuProblem(δ, ϵ, b, ζ; T=T); # LDDE problem for Mathieu equation

## ------------------- RK ----------------------

p = 5000
Tsim=T * 10
ti = LinRange(0.0, Tsim, p + 1)
h = ti[2] - ti[1]
plot()
#for n_polinom in 1:10#It should be compatible with the RK the order of the RK method
n_polinom = 5
r = Int64((τmax + 100eps(τmax)) ÷ h + n_polinom ÷ 2 + 1)
xhist = [SA[1.0, 0.5+0.0*i/r] for i in -r:0]
xhist = [SA[1.0, 0.0] for i in -r:0]

# # tested pairs:
# Butchertable(1) + n_poli - 2 - Euler()# csak kbra stimmel (n_p 2 nél optimális)
# Butchertable(2) + n_poli - 3 - Midpoint()# csak kbra stimmel (n_p 3 nél optimális)
# Butchertable(3) - BS3()# csak kbra stimmel (n_p 4 (5) nél optimális)
# Butchertable(4) - RK4() # csak kbra stimmel (n_p 5 nél optimális)
# Butchertable(5) - DP5()  csak kbra stimmel (n_p (5)6 nál optimális)- nem sokkal kisebb a hiba mint az RK4-nél

BT = Butchertable(n_polinom)
(A_t, Bs_t, τs_t, c_t, t_all) = precomputed_coefficients(mathieu_lddep, BT[1], h, ti)
x_all = runge_kutta_solve!(A_t, Bs_t, τs_t, c_t, t_all, h, xhist, p; BT=BT, n_points=n_polinom)

t_all_plot = collect(-r:p) .* h #.- r*h
plot!(t_all_plot, getindex.(x_all, 1),lw=6,label="RK")
plot!(t_all_plot, getindex.(x_all, 2),lw=6,label="RK")

x_all = runge_kutta_solve!(A_t, Bs_t, τs_t, c_t, t_all, h, x_all[end-r:end], p; BT=BT, n_points=n_polinom+1)

t_all_plot = collect(-r:p) .* h #.- r*h
plot!(t_all_plot.+Tsim, getindex.(x_all, 1),lw=3,label="RK")
plot!(t_all_plot.+Tsim, getindex.(x_all, 2),lw=3,label="RK")

#t_fine=LinRange(t_all_plot[1],t_all_plot[end],5000)
#x_f=[interpolate_F(x_all, (t) / h + r + 1, 10) for t in t_fine]
#plot!(t_fine, getindex.(x_f, 1))
#plot!(t_fine, getindex.(x_f, 2))


## ------------------- DDE solver ----------------------
par = mathieu_lddep
u0 = SA[1.0, 0.0]
h_DDE(p, t) = SA[1.0; 0.0]
#DelayMathieu(u0,h_DDE,p,0.1)

#TODO: if constant_lags=[2π] is provided, then the discontinuites are tracked leading to non-fixed timesteps
probMathieu = DDEProblem(f_for_lddep, u0, h_DDE, (0.0, Tsim * 2.0), par)#; constant_lags=[2π]
Solver_args = Dict(:alg => MethodOfSteps(RK4()), :verbose => false, :dt => h * 1.0, :adaptive => false)
@time sol = solve(probMathieu; Solver_args...);
#@benchmark solve(probMathieu; Solver_args...)

plot!(sol, denseplot=false, linestyle=:dash, lw=3)

#scatter!(sol.t[2:end],diff(sol.t))
#plot(sol.u[end-r:end])
#plot!(x_all[end-r:end],linestyle=:dash,lw=3)
@show norm(sol.u[end-r:end] - x_all[end-r:end])
#end

plot!()


## ------------------- LMS ----------------------
for N_LMS in 3:1:10#[4]#3:8
n_p=n_polinom+1
Typ=Float64
pLMS=p*N_LMS

ti = LinRange(0.0, T*10 , pLMS + 1)
h = ti[2] - ti[1]
ti=cat(h .* (-(N_LMS-1):-1),ti,dims=1)

β=LinearMultiStepCoeff(N_LMS)

r = Int64((τmax + 100eps(τmax)) ÷ h + n_p ÷ 2 + 1+(N_LMS-1))
xhist = [SA[Typ.([1, 0])...] for i in -r:0]

(A_t, Bs_t, τs_t, c_t, t_all) = precomputed_coefficients(mathieu_lddep, [1], h, ti)
x_all_LMS = LinMultiStep_solve!(A_t, Bs_t, τs_t, c_t, t_all, h, xhist, pLMS; β=β, n_points=n_p)

t_all_plot_LMS = collect(-r:pLMS) .* h #.- r*h
plot!(t_all_plot_LMS, getindex.(x_all_LMS, 1),lw=2, linestyle=:dash,label="LMS")
#scatter!(t_all_plot_LMS, getindex.(x_all_LMS, 1))
plot!(t_all_plot_LMS, getindex.(x_all_LMS, 2),lw=2, linestyle=:dash,label="LMS")


x_all_LMS = LinMultiStep_solve!(A_t, Bs_t, τs_t, c_t, t_all, h, x_all_LMS[end-r:end], pLMS; β=β, n_points=n_p)

plot!(t_all_plot_LMS .+ Tsim, getindex.(x_all_LMS, 1),lw=2, linestyle=:dash,label="LMS")
#scatter!(t_all_plot_LMS .+ Tsim, getindex.(x_all_LMS, 1))
plot!(t_all_plot_LMS .+ Tsim, getindex.(x_all_LMS, 2),lw=2, linestyle=:dash,label="LMS")
end
plot!()
#plot(t_all_plot,getindex.(x_all_LMS[(N_LMS):end] .- x_all ,1))
