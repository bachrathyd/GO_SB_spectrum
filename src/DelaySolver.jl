### module DelaySolver
### 
using LinearAlgebra
using StaticArrays
using SemiDiscretizationMethod
using LinearSolve


#add Plots,BenchmarkTools,Statistics,LaTeXStrings,FileIO, JLD2,KrylovKit,LinearAlgebra,SemiDiscretizationMethod,DifferentialEquations,StaticArrays,Profile,Plots,GenericSchur



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
    coeffs =
        equidistant_lagrange_coeffs(α, n)

    # Compute the weighted sum (each coefficient scales the corresponding matrix).
    result = zero(F[1])
    for i in 1:n
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
    #x = SizedArray{Tuple{r+1+p}}(xhist..., xfuture...)#TODO: this should be a preserved place - to save memory and GC

    #x = MVector{r+1+p,typeof(xhist[1])}
    #x[1:(r+1)] = xhist[1:(r+1)]

    c, A, b = BT
    s = length(c)
    k = Vector{typeof(x[1])}(undef, s)
    #k = SizedArray{Tuple{s}}(Vector{typeof(x[1])}(undef, s))

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



function issi_eigen(foo, u0, eigN, Niter; verbosity=0, mu_abs_error=sqrt(eps(typeof(u0[1][1]))), Niterminimum=3)
    
   # u0=xhist
   # eigN=6
   # Niterminimum=3
    Niterminimum = maximum([Niterminimum, 2])
    if verbosity > 0
        println("-----------Eigen val. calc: ISSI--------------")
    end
    Nstep = length(u0)

    ## end
    S = [rand(typeof(u0[1]), Nstep) for _ in 1:eigN]
    S = SizedArray{Tuple{eigN}}([SizedArray{Tuple{Nstep}}(rand(typeof(u0[1]), Nstep)) for _ in 1:eigN])

    V = copy(S)
    #S=[u0,foo(u0)]
    #for _ in 1:(eigN-2)
    #    S=[S...,foo(S[end]) ]
    #end

    H = zeros(typeof(u0[1][1]), eigN, eigN)#For Schur based calculation onlyS

    SS = deepcopy(H)
    VS = deepcopy(H)
    #----------------------the iteration - Start ---------------
    Threads.@threads for kS in 1:length(S)
        @inbounds V[kS] = foo(S[kS]) #TODO: ez nem jó, mert
    end


    #@btime begin
        Threads.@threads for iS in 1:length(S)
            Threads.@threads for jS in 1:length(S)
                SS[iS, jS] = S[iS]' * S[jS]
            end
        end
        Threads.@threads for iS in 1:length(S)
            Threads.@threads  for jS in 1:length(S)
                VS[iS, jS] = V[iS]' * S[jS]
            end
        end

        H = SS \ VS
    #end


   # @btime H .= (S' .* S) \ (V' .* S)#@strided

    FShurr = GenericSchur.schur(H)#@strided 
    S .= FShurr.vectors' * V#@strided 
    S .= S ./ norm.(S)
    Eigvals = FShurr.values
    pshort = sortperm(Eigvals, by=abs, rev=true)
    mus_local = Eigvals[pshort]
    #----------------------the iteration - End ---------------

    mus = Vector{Any}(undef, Niter)
    kiteration = 1
    mus[kiteration] = mus_local
    for _ in 1:Niter-1
        kiteration += 1


        #----------------------the iteration - Start ---------------
        Threads.@threads for kS in 1:length(S)
            @inbounds V[kS] = foo(S[kS]) #TODO: ez nem jó, mert
        end
        Threads.@threads for iS in 1:length(S)
            Threads.@threads for jS in 1:length(S)
                SS[iS, jS] = S[iS]' * S[jS]
            end
        end
        Threads.@threads for iS in 1:length(S)
            Threads.@threads  for jS in 1:length(S)
                VS[iS, jS] = V[iS]' * S[jS]
            end
        end

        H = SS \ VS


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

        # V = [foo(Si) for Si in S] #TODO: ez nem jó, mert
        # StS = [S[i]' * S[j] for i in 1:size(S, 1), j in 1:size(S, 1)]
        # StV = [S[i]' * V[j] for i in 1:size(S, 1), j in 1:size(S, 1)]
        # H = StS \ StV
        # FShurr = schur(H)
        # S  = [sum(FShurr.vectors[:, i] .* V) for i in 1:eigN];
        # S .= S ./ norm.(S);

        if kiteration >= 2

            mu1errorlist = abs.(diff(getindex.(mus[1:kiteration], 1)))
            if verbosity > 1
                print("Iter: $kiteration : mu_max_abs: ", abs(mus[kiteration][1]))
                println("   difference:", mu1errorlist[end])
            end

            if kiteration >= 2Niterminimum && mu1errorlist[end] > mu1errorlist[end-1]

                if verbosity >= 1
                    println("Backstepping")
                    print("Iter: ", kiteration - 1, " : mu_max_abs:", abs(mus[kiteration-1][1]))
                    println("    difference:", mu1errorlist[end-1])
                end
                return mus[kiteration-1]
            end
            if mu1errorlist[end] < mu_abs_error
                return mus[kiteration]
            end
        end

    end


    if verbosity == 1
        mu1errorlist = abs.(diff(getindex.(mus[1:kiteration], 1)))
        print("Iter: $kiteration : mu_max_abs: ", abs(mus[kiteration][1]))
        println("    difference:", mu1errorlist[end])
    end
    return mus[kiteration]
end



### end  # module DelaySolver