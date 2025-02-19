# Testgin the Lagrange interpolations
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
## Example usage:
cucc = []
n = 7
αv = -0.0:0.01:(n-1.0)


#-------- testing the shape functions ----------
for α in αv
    coeffs = equidistant_lagrange_coeffs(α, n)
    push!(cucc, coeffs)
end

plot(αv, getindex.(cucc, 1))
for k in 2:n
    plot!(αv, getindex.(cucc, k))
end
plot!()



#println("Interpolation coefficients at α = $α: ", coeffs)


# --- Example usage ---
using LinearAlgebra
# Create a vector of 2x2 matrices (for instance, random ones)
xv = 1:1:100
pfine = 0.00:0.01:100
N = length(xv)
foo(x) = sin(x / 4.7)
foo(x) = sin(x / 0.7)
F = [[x / 50, sin(x / 10)] for x in xv]
F = [foo(x) for x in xv]

scatter(xv, getindex.(F, 1))
#scatter!(xv,getindex.(F,1))

#plot()
for n in 1:20      # fifth-order (using 5 points)

    #scatter(xv,getindex.(F,1))
    #scatter!(xv,getindex.(F,1))


    #p = 35.2   # interpolation location (between 1 and 100)

    boo(x) = interpolate_F(F, x, n)
    plot!(pfine,foo.(pfine))
    #scatter!(pfine,boo.(pfine))
    plot!(pfine, log.(abs.(foo.(pfine) - boo.(pfine))), label=n)#yaxis=:log
end
plot!()

