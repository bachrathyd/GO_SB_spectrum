using Test
using StaticArrays
using DifferentialEquations

#include("./DelaySolver.jl")

@testset "Lagrange Coefficients" begin
    coeffs = equidistant_lagrange_coeffs(2.2, 5)
    @test length(coeffs) == 5
end

@testset "Interpolate_F" begin
    # Test with a vector of scalars
    F = sin.(collect(1.0:1:20.0))
    p = 15.2
    n = 5
    val = interpolate_F(F, p, n)
    @test isapprox(val, sin(15.2); atol=1e-10)
end

@testset "Buchertable RK4" begin
    c, A, b = Buchertable(4)
    @test length(c) == 4
    @test size(A, 1) == 4
    @test length(b) == 4
end

@testset "Runge-Kutta Solver" begin
    # Solve the ODE: dx/dt = -x, whose exact solution is exp(-t)
    f(t,x) = -x
    Nsol = 100
    h = 0.1
    x = ones(Nsol)
    runge_kutta_solve!(f, h, x, 1, Nsol; BT=Buchertable(4))
    t_final = (Nsol-1)*h
    @test isapprox(x[end], exp(-t_final); atol=1e-2)
end
