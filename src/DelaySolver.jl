### module DelaySolver
### 
using LinearAlgebra
using StaticArrays
using SemiDiscretizationMethod
using LinearSolve


#add Plots,BenchmarkTools,Statistics,LaTeXStrings,FileIO, JLD2,KrylovKit,LinearAlgebra,SemiDiscretizationMethod,DifferentialEquations,StaticArrays,Profile,Plots,GenericSchur


using Printf, Dates

using GenericSchur
#using Strided
### 
### export equidistant_lagrange_coeffs,
###     interpolate_F,
###     Butchertable,
###     runge_kutta_solve!,
###     runge_kutta_solve_delay!,
###     precomputed_coefficients,
###     f_for_lddep,
###     LinearMultiStepCoeff


Base.:+(a::SVector, b::Bool) = a .+ b
Base.:+(a::SVector, b::Float64) = a .+ b #TODO: where to put this?


################################################################################
# Interpolation Functions
################################################################################

# Global cache dictionary.
const _lagrange_cache = Dict{Tuple{Any,Int,Any},Any}()
"""
    cached_equidistant_lagrange_coeffs(α::T, n::Int) where T

Wrapper for `equidistant_lagrange_coeffs` that caches results in a dictionary.
It returns the cached value if already computed for the same (α, n) pair,
otherwise it computes, stores, and returns the new result.
"""
function cached_equidistant_lagrange_coeffs(α::T, n::Int) where {T}
    key = (α, n, T)
    if haskey(_lagrange_cache, key)
        # println("precomputed")
        return _lagrange_cache[key]
    else
        # println("new comp.")
        coeffs = equidistant_lagrange_coeffs(α, n)
        _lagrange_cache[key] = coeffs
        return coeffs
    end
end



"""
    equidistant_lagrange_coeffs(alpha::Float64, n::Int)

Compute Lagrange interpolation coefficients for an equidistant grid.
- `alpha`: evaluation point in grid units (e.g. α=2.5 means between nodes 2 and 3).
- `n`: number of grid points (polynomial of degree n-1).

Returns a vector of coefficients such that
    f(α) ≈ Σ_{j=0}^{n-1} coeffs[j+1] * f(j)
"""
# Compute Lagrange coefficients for nodes 0,1,...,n-1 at evaluation point α.
function equidistant_lagrange_coeffs(α::T, n::Int) where {T}
    coeffs = ones(T, n)
    for j in 0:(n-1)
        for k in 0:(n-1)
            if k != j
                coeffs[j+1] *= (α - k) / (j - k)
            end
        end
    end
    return coeffs
end

"""
    interpolate_F(F::AbstractArray, p::Float64, n::Int)

Interpolate the vector `F` (e.g. of matrices) at the (real) index `p`
using `n` consecutive points chosen centrally.
"""
# Interpolate F (a vector of matrices) at location p.
# Here, p is a real index (1 ≤ p ≤ length(F)).
# For central interpolation, we select n points such that the center is near p.
function interpolate_F(F::AbstractArray, p::T, n::Int) where {T}
    N = length(F)
    # Choose left index for n points, trying to center the window around p.
    #left = round(Int, p) - div(n, 2)
    left = ceil(Int, p) - div(n, 2)
    left = clamp(left, 1, N - n + 1)  # ensure indices are within bounds
    α = p - left                     # local evaluation point in [0, n-1]
    coeffs = equidistant_lagrange_coeffs(α, n)
    #coeffs = cached_equidistant_lagrange_coeffs(α, n)

    # Compute the weighted sum (each coefficient scales the corresponding matrix).
    result = zero(F[1])
    @inbounds for i in 1:n
        @inbounds result += coeffs[i] * F[left+i-1]
    end
    return result
end

################################################################################
# Butcher Table and Runge-Kutta Solvers
################################################################################
"""
LinearMultiStepCoeff(Norder::Int=4, Typ::DataType=Float64)

Compute the Adams–Bashforth coefficients for a linear multistep integrator
of the given order (number of steps).

Returns a vector `β` such that the integrator is written as

yₙ₊₁ = yₙ + h * ∑_{j=0}^{order-1} β[j+1] fₙ₋ⱼ

Coefficients are determined by solving:
∑_{j=0}^{order-1} (-j)^m * β[j+1] = 1/(m+1),  for m = 0,...,order-1.
"""
function LinearMultiStepCoeff(Norder::Int=4, Typ::DataType=Float64)
    if Norder == 1#EULER
        β = SA[Typ.([1 // 1])...]
        return β
    elseif Norder == 2
        # Adams–Bashforth 
        β = SA[Typ.([3 // 2, -1 // 2])...]
        return β
    elseif Norder == 3
        # Adams–Bashforth 
        β = SA[Typ.([23 // 12, -16 // 12, 5 // 12])...]
        return β
    elseif Norder == 4
        # Adams–Bashforth 
        β = SA[Typ.([55 // 24, -59 // 24, 37 // 24, -9 // 24])...]
        return β
    elseif Norder == 5
        # Adams–Bashforth 
        β = SA[Typ.([1901 // 720, -2774 // 720, 2616 // 720, -1274 // 720, 251 // 720])...]
        return β
    else

        k = Norder
        V = zeros(Typ, k, k)
        for m in 0:k-1
            for j in 0:k-1
                V[m+1, j+1] = Typ(-j)^Typ(m)
            end
        end
        rhs = [1 / Typ.(m + 1) for m in 0:k-1]

        prob = LinearProblem(V, rhs)#,abstol=1e-300)
        sol = solve(prob)
        β = sol.u
        #β2 = inv(V) * rhs
        #@show norm(β-β2)
        return β
    end
end

"""
LinMultiStep_solve!(A_t, Bs_t, τs_t, c_t, t_all, h, xhist::AbstractArray, p::Int; β=LinearMultiStepCoeff(), n_points=length(β) + 1)
Advance a solution for a delay problem using precomputed coefficients.
Here, `xhist` is the history vector and `p` is the number of steps to advance.
"""
#yₙ₊₁ = yₙ + h * ∑_{j=0}^{order-1} β[j+1] fₙ₋ⱼ
function LinMultiStep_solve!(A_t, Bs_t, τs_t, c_t, t_all, h, xhist::AbstractArray, p::Int; β=LinearMultiStepCoeff(), n_points=length(β) + 1)
    r = length(xhist) - 1
    xfuture = [zeros(typeof(xhist[1])) for i in 1:p]
    x = cat(xhist, xfuture; dims=1)#TODO: this should be a preserved place - to save memory and GC
    s = length(β)
    k = Vector{typeof(x[1])}(undef, s - 1 + p)# in the last point, we don't need the derivative

    for it in -(s - 1):-1
        @inbounds k[it+s] = A_t[it+s, 1] * x[it+1+r] + c_t[it+s, 1]
        for (B_loc, τ_loc) in zip(Bs_t, τs_t)
            @inbounds xidelay = interpolate_F(x, (t_all[it+s, 1] - τ_loc[it+s, 1]) / h + r, n_points)
            #xidelay = xi #TODO: ide jön a delay rész még egy forciklusban %a legrade interpoláció alapján
            @inbounds k[it+s] += B_loc[it+s, 1] * xidelay
        end
    end

    for it in 0:(p-1)
        @inbounds k[it+s] = A_t[it+s, 1] * x[it+1+r] + c_t[it+s, 1]
        for (B_loc, τ_loc) in zip(Bs_t, τs_t)
            @inbounds xidelay = interpolate_F(x, (t_all[it+s, 1] - τ_loc[it+s, 1]) / h + r, n_points)
            #xidelay = xi #TODO: ide jön a delay rész még egy forciklusban %a legrade interpoláció alapján
            @inbounds k[it+s] += B_loc[it+s, 1] * xidelay
        end

        #---------------------A.B. Linear multistep----------------------
        @inbounds x[it+1+r+1] = x[it+1+r]
        for i in 1:s
            @inbounds x[it+1+r+1] += h * β[i] * k[it+s-(i-1)]
        end
        #---------------------A.B. Linear multistep----------------------

        #x[i+1] = runge_kutta_step(f, t, x[i], h, A, b, c)
    end
    return x
end



"""
    Butchertable(Norder::Int=4)

Return a Butcher tableau (c, A, b) for an explicit RK method of order 1–7.
Uses StaticArrays.
"""
function Butchertable(Norder::Int=4, Typ::DataType=Float64)
    if Norder == 1
        # Euler (RK1)
        c = SA[Typ.([0 // 1])...]
        A = SA[
            SA[Typ.([0 // 1])...]]
        b = SA[Typ.([1 // 1])...]
        return c, A, b
    elseif Norder == 2
        # Explicit Midpoint method (RK2)
        c = SA[Typ.([0 // 1, 1 // 2])...]
        A = SA[
            SA[Typ.([0 // 1, 0 // 1])...],
            SA[Typ.([1 // 2, 0 // 1])...]]
        b = SA[Typ.([0 // 1, 1 // 1])...]
        return c, A, b
    elseif Norder == 3
        # Kutta’s 3rd-order method (RK3)
        c = SA[Typ.([0 // 1, 1 // 2, 1 // 1])...]
        A = SA[
            SA[Typ.([0 // 1, 0 // 1, 0 // 1])...],
            SA[Typ.([1 // 2, 0 // 1, 0 // 1])...],
            SA[Typ.([-1 // 1, 2 // 1, 0 // 1])...]]
        b = SA[Typ.([1 / 6, 2 / 3, 1 / 6])...]
        return c, A, b
    elseif Norder == 4
        # Classical RK4
        c = SA[Typ.([0 // 1, 1 // 2, 1 // 2, 1 // 1])...]
        A = SA[
            SA[Typ.([0 // 1, 0 // 1, 0 // 1, 0 // 1])...],
            SA[Typ.([1 // 2, 0 // 1, 0 // 1, 0 // 1])...],
            SA[Typ.([0 // 1, 1 // 2, 0 // 1, 0 // 1])...],
            SA[Typ.([0 // 1, 0 // 1, 1 // 1, 0 // 1])...]]
        b = SA[Typ.([1 // 6, 1 // 3, 1 // 3, 1 // 6])...]
        return c, A, b
    elseif Norder == 5
        # A 6-stage 5th-order method (Butcher RK5)
        c = SA[Typ.([0 // 1, 1 // 4, 1 // 4, 1 // 2, 3 // 4, 1 // 1])...]
        A = SA[
            SA[Typ.([0 // 1, 0 // 1, 0 // 1, 0 // 1, 0 // 1, 0 // 1])...],
            SA[Typ.([1 // 4, 0 // 1, 0 // 1, 0 // 1, 0 // 1, 0 // 1])...],
            SA[Typ.([1 // 8, 1 // 8, 0 // 1, 0 // 1, 0 // 1, 0 // 1])...],
            SA[Typ.([0 // 1, 0 // 1, 1 // 2, 0 // 1, 0 // 1, 0 // 1])...],
            SA[Typ.([3 // 16, -3 // 8, 3 // 8, 9 // 16, 0 // 1, 0 // 1])...],
            SA[Typ.([-3 // 7, 8 // 7, 6 // 7, -12 // 7, 8 // 7, 0 // 1])...]
        ]
        b = SA[Typ.([7 // 90, 0 // 1, 16 // 45, 2 // 15, 16 // 45, 7 // 90])...]
        return c, A, b
    elseif Norder == 6
        # 7-stage 6th‑order method (example; see note above)
        c = SA[Typ.([0 // 1, 1 // 3, 2 // 3, 1 // 3, 1 // 2, 1 // 2, 1 // 1])...]
        A = SA[
            SA[Typ.([0 // 1, 0 // 1, 0 // 1, 0 // 1, 0 // 1, 0 // 1, 0 // 1])...],
            SA[Typ.([1 // 3, 0 // 1, 0 // 1, 0 // 1, 0 // 1, 0 // 1, 0 // 1])...],
            SA[Typ.([0 // 1, 2 // 3, 0 // 1, 0 // 1, 0 // 1, 0 // 1, 0 // 1])...],
            SA[Typ.([1 // 12, 1 // 3, -1 // 12, 0 // 1, 0 // 1, 0 // 1, 0 // 1])...],
            SA[Typ.([-1 // 16, 9 // 8, -3 // 16, -3 // 8, 0 // 1, 0 // 1, 0 // 1])...],
            SA[Typ.([0 // 1, 9 // 8, -3 // 8, -3 // 4, 1 // 2, 0 // 1, 0 // 1])...],
            SA[Typ.([9 // 44, -9 // 11, 63 // 44, 18 // 11, 0 // 1, -16 // 11, 0 // 1])...]
        ]
        b = SA[Typ.([11 // 120, 0 // 1, 27 // 40, 27 // 40, -4 // 15, -4 // 15, 11 // 120])...]
        return c, A, b
        #elseif Norder == 7
        #    # 8-stage 7th‑order method (example; see note above)
        #    c = SA[0//1, 1/4, 1/4, 1/2, 1/2, 3/4, 3/4, 1//1]
        #    A = SA[
        #        SA[0//1, 0//1, 0//1, 0//1, 0//1, 0//1, 0//1, 0//1],
        #        SA[1/4, 0//1, 0//1, 0//1, 0//1, 0//1, 0//1, 0//1],
        #        SA[1/8, 1/8, 0//1, 0//1, 0//1, 0//1, 0//1, 0//1],
        #        SA[0//1, 0//1, 1/2, 0//1, 0//1, 0//1, 0//1, 0//1],
        #        SA[1/16, 0//1, 1/16, 1/4, 0//1, 0//1, 0//1, 0//1],
        #        SA[0//1, 0//1, 0//1, 1/2, 0//1, 0//1, 0//1, 0//1],
        #        SA[0//1, 0//1, 0//1, 0//1, 1/2, 0//1, 0//1, 0//1],
        #        SA[1/7, 0//1, 0//1, 3/7, 0//1, 2/7, 1/7, 0//1]
        #    ]
        #    b = SA[1/7, 0//1, 0//1, 3/7, 0//1, 2/7, 1/7, 0//1]
        #    return c, A, b
    else
        error("No default Butcher tableau for Norder = $Norder provided.")
    end
end


"""
    runge_kutta_solve!(f, h, x::AbstractArray, j::Int, Nsol::Int; BT=Butchertable(), t0=0//1)

Advance the solution of an ODE (or system) from index `j` to `Nsol`
using a Runge–Kutta method defined by the Butcher tableau `BT`.
"""

# RK solver that continues from already computed index j up to Nsol.
function runge_kutta_solve!(f, h, x::AbstractArray, j::Int, Nsol::Int; BT=Butchertable(), t0=0 // 1)
    c, A, b = BT
    s = length(b)
    k = Vector{typeof(x[1])}(undef, s)

    @inbounds for it in j:(Nsol-1)
        t = t0 + (it - 1) * h

        #---------------------RK step----------------------
        for i in 1:s
            @inbounds xi = copy(x[it])#TODO: Ez felesleges inicializálás
            for j in 1:(i-1)
                @inbounds xi += h * A[i][j] * k[j]
            end
            @inbounds k[i] = f(t + c[i] * h, xi)
        end
        @inbounds x[it+1] = x[it]
        for i in 1:s
            @inbounds x[it+1] += h * b[i] * k[i]
        end
        #---------------------RK step----------------------

        #x[i+1] = runge_kutta_step(f, t, x[i], h, A, b, c)
    end
    return x
end

"""
    runge_kutta_solve_delay!(A_t, Bs_t, τs_t, c_t, t_all, h, xhist::AbstractArray, p::Int; BT=Butchertable(), n_points)

Advance a solution for a delay problem using precomputed coefficients.
Here, `xhist` is the history vector and `p` is the number of steps to advance.
"""
function runge_kutta_solve!(A_t, Bs_t, τs_t, c_t, t_all, h, xhist::AbstractArray, p::Int; BT=Butchertable(), n_points=length(BT[1]) + 1)
    r = length(xhist) - 1
    xfuture = [ones(typeof(xhist[1])) for i in 1:p]
    x = cat(xhist, xfuture; dims=1)#TODO: this should be a preserved place - to save memory and GC
    c, A, b = BT
    s = length(c)
    k = Vector{typeof(x[1])}(undef, s)

    for it in 0:(p-1)
        #---------------------RK step----------------------
        for i in 1:s
            @inbounds xi = copy(x[it+1+r])#TODO: Ez felesleges inicializálás
            for j in 1:(i-1)
                @inbounds xi += h * A[i][j] * k[j]
            end
            #k[i] = f(t + c[i] * h, xi)

            @inbounds k[i] = A_t[it+1, i] * xi + c_t[it+1, i]
            for (B_loc, τ_loc) in zip(Bs_t, τs_t)
                @inbounds xidelay = interpolate_F(x, (t_all[it+1, i] - τ_loc[it+1, i]) / h + r + 1, n_points)
                #xidelay = xi #TODO: ide jön a delay rész még egy forciklusban %a legrade interpoláció alapján
                @inbounds k[i] += B_loc[it+1, i] * xidelay
            end

        end
        @inbounds x[it+1+r+1] = x[it+1+r]
        for i in 1:s
            @inbounds x[it+1+r+1] += h * b[i] * k[i]
        end
        #---------------------RK step----------------------

        #x[i+1] = runge_kutta_step(f, t, x[i], h, A, b, c)
    end
    return x
end

################################################################################
# Delay Problem Specific Functions
################################################################################

"""
    precomputed_coefficients(lddep, c_RK::AbstractVector, h, ti::AbstractVector)

Precompute time-shifted coefficients for a delay problem.
"""

function precomputed_coefficients(lddep::LDDEProblem, c_RK::AbstractVector, h, ti::AbstractVector)# where {T}
    #c_RK_u=unique(c_RK)#TODO: use unique points only!!! (Would be nice to notice, that "t1+c(1)" = "t2+c(0)")
    c_RK_u = c_RK
    t_all = ti .+ (c_RK_u' .* h)

    A_t = lddep.A.(t_all)
    Bs_t = [B.(t_all) for B in lddep.Bs]
    τs_t = [B.τ.(t_all) for B in lddep.Bs]
    # t_min_τ_s_t = [(t_all .- B.τ.(t_all)) ./ h+r+1 for B in lddep.Bs]
    c_t = lddep.c.(t_all)
    return (A_t, Bs_t, τs_t, c_t, t_all)
end

function f_for_lddep(u, h, p, t)
    # Parameters
    lddep = p
    du = lddep.A(t) * u + lddep.c(t)
    for B in lddep.Bs
        du += B(t) * h(p, t - B.τ(t))
    end
    return du
end

function issi_eigen(foo, u0, eigN, Niter; verbosity=0, mu_abs_error=sqrt(eps(typeof(u0[1][1]))), Niterminimum=3, dofilesave::Bool=false)
    #u0 = xhist
    #eigN = 6
    #Niter = 10
    #mu_abs_error = sqrt(eps(typeof(u0[1][1])))
    #Niterminimum = 3
    #dofilesave::Bool = false

    t = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    filename = "results.csv"
    if dofilesave
        open(filename, "a") do f
            println(f, "")
            println(f, "--------------------------------------")
            println(f, "Time,Value: $t")
        end
    end


    Niterminimum = maximum([Niterminimum, 2])
    if verbosity > 0
        println("-----------Eigen val. calc: ISSI--------------")
    end
    Nstep = length(u0)

    ## end
    S = [rand(typeof(u0[1]), Nstep) for _ in 1:eigN]
    V = copy(S)
    #S=[u0,foo(u0)]
    #for _ in 1:(eigN-2)
    #    S=[S...,foo(S[end]) ]
    #end

    H = zeros(typeof(u0[1][1]), eigN, eigN)#For Schur based calculation onlyS
    #----------------------the iteration - Start ---------------
    Threads.@threads for kS in 1:length(S)
        @inbounds V[kS] = foo(S[kS]) #TODO: ez nem jó, mert
    end

    StS = zeros(typeof(u0[1][1]), eigN, eigN)#For Schur based calculation onlyS
    StV = zeros(typeof(u0[1][1]), eigN, eigN)#For Schur based calculation onlyS
    pairwise_dot!(StS, S)
    pairwise_dot!(StV, S, V)
    H = StS \ StV
    #H .= (S' .* S) \ (V' .* S)#@strided 

    FShurr = GenericSchur.schur(H)#@strided 
    S .= FShurr.vectors' * V#@strided 
    S .= S ./ norm.(S)
    Eigvals = FShurr.values
    pshort = sortperm(Eigvals, by=abs, rev=true)
    mus_local = Eigvals[pshort]
    #----------------------the iteration - End ---------------

    mus = Vector{Any}(undef, Niter)
    musRichardson = Vector{Any}(undef, Niter)
    muswynn = Vector{Any}(undef, Niter)
    kiteration = 1
    mus[kiteration] = mus_local
    for _ in 1:Niter-1
        kiteration += 1


        #----------------------the iteration - Start ---------------
        Threads.@threads for kS in 1:length(S)
            @inbounds V[kS] = foo(S[kS]) #TODO: ez nem jó, mert
        end


        StS = zeros(typeof(u0[1][1]), eigN, eigN)#For Schur based calculation onlyS
        StV = zeros(typeof(u0[1][1]), eigN, eigN)#For Schur based calculation onlyS
        pairwise_dot!(StS, S)
        pairwise_dot!(StV, S, V)
        H = StS \ StV


        #H .= (S' .* S) \ (V' .* S)#@strided 
        #@show norm(H - H2)


        FShurr = GenericSchur.schur(H)#@strided 
        S .= FShurr.vectors' * V#@strided 
        S .= S ./ norm.(S)
        #@show norm(S[1])
        #@show norm(S[end])

        Eigvals = FShurr.values
        pshort = sortperm(Eigvals, by=abs, rev=true)
        mus_local = Eigvals[pshort]
        #----------------------the iteration - End ---------------
        mus[kiteration] = mus_local

       # if kiteration > 2
       #     musRichardson[kiteration] =aitken_extrapolation_vec(mus[kiteration-2:kiteration])
       # end
       # if kiteration > 3
       #     muswynn[kiteration] =wynn_epsilon_vec(mus[3:kiteration])
       # end


        if kiteration >= 2
        #if kiteration > 5 #now I can use musRichardson
            if dofilesave
                formatted_value = @sprintf("%.200f", abs(mus[kiteration][1]))
                open(filename, "a") do f
                    println(f, "Iter: $kiteration : mu_max_abs: $formatted_value")
                end
            end

            mu1errorlist = abs.(diff(getindex.(mus[1:kiteration], 1)))
            #mu1errorlistRichardson = abs.(diff(getindex.(musRichardson[3:kiteration], 1)))
            #mu1errorlistwynn = abs.(diff(getindex.(muswynn[4:kiteration], 1)))
            if verbosity > 1
                print("Iter: $kiteration : mu_max_abs: ", abs(mus[kiteration][1]))
                println("   difference:", mu1errorlist[end])
               # if kiteration > 2
               #     print("   Richardson (Aitken):", abs(musRichardson[kiteration][1]))
               #     println("   difference:", mu1errorlistRichardson[end])
               #     print("   wynn", abs(muswynn[kiteration][1]))
               #     println("   difference:", mu1errorlistwynn[end])
               # end
                #println("")

            end

            if kiteration >= Niterminimum && mu1errorlist[end] > mu1errorlist[end-1]

                if verbosity >= 1
                    println("Backstepping")
                    println("Iter: ", kiteration - 1, " : mu_max_abs:", abs(mus[kiteration-1][1]))
                    println(" difference:", mu1errorlist[end-1])
                end
                return mus[kiteration-1] 
                #return aitken_extrapolation_vec(mus[kiteration-3:kiteration-1])
            end
            if mu1errorlist[end] < mu_abs_error
                return mus[kiteration]
                #return aitken_extrapolation_vec(mus[kiteration-2:kiteration])
            end
        end

    end


    if verbosity == 1
        mu1errorlist = abs.(diff(getindex.(mus[1:kiteration], 1)))
        print("Iter: $kiteration : mu_max_abs: ", abs(mus[kiteration][1]))
        println("    difference:", mu1errorlist[end])
    end
    return mus[kiteration]
    #return aitken_extrapolation_vec(mus[kiteration-3:kiteration-1])
end

"""
    aitken_extrapolation(seq::Vector{Complex{T}}) where {T}

Applies Aitken’s Δ² method to the last three iterates in `seq`
to accelerate a linearly convergent sequence of complex numbers.
If fewer than three iterates are available, returns the last iterate.
"""
function aitken_extrapolation_vec(seq::AbstractArray{T}) where {T}
    return [aitken_extrapolation(getindex.(seq,n)) for n in 1:size(seq[1],1) ]
end

function aitken_extrapolation(seq::AbstractArray{Complex{T}}) where {T}
    n = length(seq)
    if n < 3
        return seq[end]
    else
        # Let x₀, x₁, x₂ be the last three iterates
        x₀ = seq[end-2]
        x₁ = seq[end-1]
        x₂ = seq[end]
        denom = x₂ - 2x₁ + x₀
        # Avoid division by (almost) zero:
        if abs(denom) < eps(T)
            return x₂
        end
        # Aitken extrapolation formula:
        return x₀ - ((x₁ - x₀)^2) / denom
    end
end



"""
    wynn_epsilon(seq::Vector{Complex{T}}) where {T}

Uses Wynn’s ε‐algorithm to accelerate convergence of an exponentially converging sequence of complex numbers.
The input `seq` is assumed to be the sequence of iterates.
Returns an extrapolated estimate of the limit.
"""

function wynn_epsilon_vec(seq::AbstractArray{T}) where {T}
    return [wynn_epsilon(getindex.(seq,n)) for n in 1:size(seq[1],1) ]
end
function wynn_epsilon(seq::Vector{Complex{T}}) where {T}
    N = length(seq)
    # Allocate a table E with dimensions (N+1) x N.
    # We set E[1, :] to correspond to ε₋₁^(n) = 0 and E[2, :] to ε₀^(n) = seq[n].
    E = Array{Complex{T}}(undef, N+1, N)
    for n in 1:N
        E[1, n] = zero(T)  # ε₋₁^(n) = 0
        E[2, n] = seq[n]   # ε₀^(n) = s_n
    end

    # Fill the table according to:
    #   ε_{k+1}^{(n)} = ε_{k-1}^{(n+1)} + 1/(ε_k^(n+1) - ε_k^(n))
    # where our table row i corresponds to k = i - 2.
    for i in 3:(N+1)
        for n in 1:(N - i + 2)
            diff = E[i-1, n+1] - E[i-1, n]
            if abs(diff) < eps(T)
                # Avoid division by zero: just propagate the value.
                E[i, n] = E[i-1, n+1]
            else
                E[i, n] = E[i-2, n+1] + 1 / diff
            end
        end
    end

    # A common choice for the extrapolated limit is ε₂m^(1) for the largest m available.
    m = fld(N+1, 2)  # integer division
    extrapolated = E[2*m, 1]
    return extrapolated
end














"""
    pairwise_dot!(D, S,V)

Computes the pairwise dot products between vectors in S and V and stores the result in
the preallocated matrix D. S and V is a vector of vectors (all of the same element type
and length) and D is an N×N matrix, where N = length(S) = length(S).

This function uses @threads and @inbounds for speed and does not allocate extra memory.
"""
function pairwise_dot!(D::AbstractMatrix{T1}, S::Vector{T}, V::Vector{T}) where {T1,T}
    N = length(S)
    Threads.@threads for i in 1:N
        @inbounds for j in 1:N
            d = dot(S[i], V[j])
            #d = S[i]'* V[j]
            D[j, i] = d
        end
    end
    return D
end
### end  # module DelaySolver


"""
    pairwise_dot!(D, S)

Computes the pairwise dot products between vectors in S and stores the result in
the preallocated matrix D. S is a vector of vectors (all of the same element type
and length) and D is an N×N matrix, where N = length(S).

This function uses @threads and @inbounds for speed and does not allocate extra memory.
"""
function pairwise_dot!(D::AbstractMatrix{T1}, S::Vector{T}) where {T1,T}
    N = length(S)
    Threads.@threads for i in 1:N
        @inbounds for j in i:N
            d = dot(S[i], S[j])
            #d = S[i]'* S[j]
            D[i, j] = d
            D[j, i] = d
        end
    end
    return D
end