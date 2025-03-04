5 + 5

#]activate test_env_1
# add Plots,  LinearAlgebra,  KrylovKit,  StaticArrays,  SemiDiscretizationMethod,  LinearSolve 


# Testgin the RK solver with Butcher Tables - compared to the Diff.Eq. DDE solver
using Plots
theme(:dark)#:vibrant:dracula:rose_pine
plotly()
#const PLOTS_DEFAULTS = Dict(:theme => :wong, :fontfamily => "Computer Modern", :label => nothing, :dpi => 600 )const PLOTS_DEFAULTS = Dict(:theme => :wong, :fontfamily => "Computer Modern", :label => nothing, :dpi => 600 )
default(size=(1500, 900), titlefont=(15, "times"), legendfontsize=13, guidefont=(12, :white), tickfont=(12, :orange), guide="x", framestyle=:zerolines, yminorgrid=true, fontfamily="Computer Modern", label=nothing, dpi=600)

#using Profile
using StaticArrays
#using DifferentialEquations
using SemiDiscretizationMethod

using LinearAlgebra


using KrylovKit  #KrylovKit v0.6.1
#using Revise
include("./src/DelaySolver.jl")
#using .DelaySolver

#using Statistics
using BenchmarkTools
#using LaTeXStrings
#using FileIO, JLD2

#add Plots,BenchmarkTools,Statistics,LaTeXStrings,FileIO, JLD2,KrylovKit,LinearAlgebra,SemiDiscretizationMethod,DifferentialEquations,StaticArrays,Profile,Plots



using GenericSchur
using Quadmath

using Printf, Dates
#FileIO.save("myfile.jld2","a",a)
#b = FileIO.load("myfile.jld2","a")
#---------------------------- Solution with precomputed A-s -------------------------
function createMathieuProblem(ploc, Typ=Float64)
    τmax, ζ, δ, ϵ, τ, b, T = ploc
    AMx = ProportionalMX(t -> @SMatrix [0 1; -δ-ϵ*cos(2 * Typ(pi) / T * t) -ζ])# ::SMatrix{2,2,Typ}
    #AMx = ProportionalMX(t -> @SMatrix [0 1; -δ-ϵ*abs(cos(2 * Typ(pi) / T * t + Typ(1))) -ζ])#Convergence is saturated at order 2 (if there is no perfect hit)
    τ1 = t -> τ # if function is needed, the use τ1 = t->foo(t)
    BMx1 = DelayMX(τ1, t -> @SMatrix [0 0; b 0])
    cVec = Additive(t -> @SVector [0, 0 * sin(Typ(pi) / T * t)])
    LDDEProblem(AMx, [BMx1], cVec)
end

#@code_warntype createMathieuProblem(δ, ϵ, b, ζ; T=T, Typ=ProbType)
# -------------------- spectral method ----------------

using SemiDiscretizationMethod
using SCOL
nodes = 24;
divs = 1; # keep it as 1
method = "Chebyshev";

coll = SCOL.defineCollMethod(nodes, divs, method);

function MathieuSystem(ploc, Typ=Float64)
    τmax, ζ, δ, ϵ, τ, b, T = ploc
    α = @SMatrix [0.0 0.0; 0.0 0.0]
    β = @SMatrix [0.0 0.0; 0.0 0.0]
    γ = @SVector [0.0; 0.0]
    #return SCOL.SCOLProblem(t -> [0 1; -δ-epsilon*abs(cos(2 * Typ(pi) / T * t + Typ(1))) (-ζ)],
    #    t -> [0 0; b 0],
    #    t -> [0, 0 * sin(Typ(pi) / T * t)],
    #    t -> [0.0 0.0; 0.0 0.0], t -> [0.0 0.0; 0.0 0.0], t -> [0.0; 0.0],
    #     τ, T)
    return SCOL.SCOLProblem(t -> [0 1; -δ-ϵ*cos(2 * Typ(pi) / T * t) (-ζ)],
        t -> [0 0; b 0],
        t -> [0, 0 * sin(Typ(pi) / T * t)],
        t -> [0.0 0.0; 0.0 0.0], t -> [0.0 0.0; 0.0 0.0], t -> [0.0; 0.0],
        τ, T)
    # # #return SCOL.SCOLProblem(t -> @SMatrix [0 1; -δ-epsilon*cos(2 * Typ(pi) / T * t) (-ζ)],
    # # #t -> @SMatrix [0 0; b 0],
    # # #t -> @SVector [0, 0 * sin(Typ(pi) / T * t)],
    # # #t -> @SMatrix [0.0 0.0; 0.0 0.0], t -> @SMatrix [0.0 0.0; 0.0 0.0], t -> @SVector [0.0; 0.0],
    # # # τ, T)
end


# -------------------- spectral method ----------------

## ---------------------------------------------- Testing the spectrum of a mapping --------------------------------
#function mumax_IntmappingRK(p::Int, N_bt::Int, n_p::Int, Krylov_arg, lddep; Typ::DataType=typeof(lddep.A(0.0)[1]), verbosity::Int=0, mu_abs_error=eps(ProbType)^0.4, Niter::Int=25, eigN::Int=10,T=2 * Typ(pi),τmax=2 * Typ(pi))::Typ
function mumax_IntmappingRK(p::Int, N_bt::Int, n_p::Int, Krylov_arg, lddep; verbosity::Int=0, mu_abs_error=eps(Typ)^0.4,
    Niter::Int=25, eigN::Int=10, T::Typ=2pi, τmax::Typ=2pi, dofilesave::Bool=false)::Typ where {Typ}
    #p=100
    #N_bt=4
    #n_p=3
    #Typ=typeof(lddep.A(0.0)[1])
    #verbosity=0
    #mu_abs_error=1e-10

    ti = LinRange(0, T, p + 1)
    h = ti[2] - ti[1]

    BT_loc = Butchertable(N_bt, Typ)

    r = Int64((τmax + 100eps(τmax)) ÷ h + n_p ÷ 2 + 1)
    #xhist = [SA[Typ.([1, 0])...] for i in -r:0]
    xhist = [MVector(Typ.([1, 0])...) for i in -r:0]

    (A_t, Bs_t, τs_t, c_t, t_all) = precomputed_coefficients(lddep, BT_loc[1], h, ti)
    #@inbounds 
    foo(xh::T) where {T} = deepcopy(runge_kutta_solve!(A_t, Bs_t, τs_t, c_t, t_all, h, xh, p; BT=BT_loc, n_points=n_p)[end-r:end])::T
    #foo(xh)  = deepcopy(runge_kutta_solve!(A_t, Bs_t, τs_t, c_t, t_all, h, xh, p; BT=BT_loc, n_points=n_p)[end-r:end])
    #@code_warntype foo(xhist) 
    if !(Typ <: Union{Float16,Float32,Float64})#Typ == BigFloat#false#true#true#
        println("ISSI solve.......................................")
        mus = issi_eigen(foo, xhist, eigN, Niter, verbosity=verbosity, mu_abs_error=mu_abs_error, dofilesave=dofilesave)
        mumax1 = abs.(mus[1])[1]
        return mumax1
    else
        println("Krylov_schursolve.......................................")
        mus_sch = getindex(KrylovKit.schursolve(foo, xhist, Krylov_arg...), [3, 2, 1])
        #mus_eig = eigsolve(foo, xhist, Krylov_arg...)
        #@show abs.(mus_sch[1])
        #@show abs.(mus_eig[1])
        @show mumax1 = abs.(mus_sch[1])[1]
        return mumax1
    end
end

function mumax_IntmappingLMS(p, N_LMN, n_p, Krylov_arg, lddep; verbosity=0, mu_abs_error=eps(Typ)^0.4, Niter=25, eigN=10, T::Typ=2pi, τmax::Typ=2pi, dofilesave::Bool=false)::Typ where {Typ}
    # p=100
    # N_LMN=4
    # n_p=3
    # Typ=typeof(lddep.A(0.0)[1])
    # verbosity=0
    # mu_abs_error=1e-10
    ti = LinRange(0.0, T, p + 1)
    h = ti[2] - ti[1]
    ti = cat(h .* (-(N_LMN - 1):-1), ti, dims=1)

    β = LinearMultiStepCoeff(N_LMN)

    r = Int64((τmax + 100eps(τmax)) ÷ h + n_p ÷ 2 + 1 + (N_LMN - 1))
    #xhist = [SA[Typ.([1, 0])...] for i in -r:0]
    xhist = [MVector(Typ.([1, 0])...) for i in -r:0]

    (A_t, Bs_t, τs_t, c_t, t_all) = precomputed_coefficients(lddep, [1], h, ti)

    foo(xh) = deepcopy(LinMultiStep_solve!(A_t, Bs_t, τs_t, c_t, t_all, h, xh, p; β=β, n_points=n_p)[end-r:end])
    #foo(xhist)
    if !(Typ <: Union{Float16,Float32,Float64})#Typ == BigFloat#false#true#
        println("ISSI solve.......................................")
        mus = issi_eigen(foo, xhist, eigN, Niter, verbosity=verbosity, mu_abs_error=mu_abs_error, dofilesave=dofilesave)
        mumax1 = abs.(mus[1])[1]
        return mumax1
    else
        println("Krylov_schursolve.......................................")
        #       mus = getindex(schursolve(foo, xhist, Krylov_arg...), [3, 2, 1])
        mus = eigsolve(foo, xhist, Krylov_arg...)
        @show mumax1 = abs.(mus[1])[1]
        return mumax1
    end
end

#piq = Float128(pi)
#A = rand(BigFloat,5,5)
##x=rand(BigFloat,5)
##foo(xx)=A*xx
##foo(x)
##eigsolve(foo, x)
#A = [5. 7.; -2. -4.]
#cc=GenericSchur.schur(A)

function mumax_SD(p, N_bt, lddep)
    method = SemiDiscretization(N_bt, T / p)
    τmax = τ # the largest τ of the system

    n_Steps = Int((T + 100 * eps(T)) ÷ method.Δt)
    mapping = DiscreteMapping_LR(lddep, method, τmax,
        n_steps=n_Steps, calculate_additive=false) #The
    mumaxSD = spectralRadiusOfMapping(mapping)
end

Base.:+(a::SVector, b::Bool) = a .+ b
Base.:+(a::SVector, b::Float64) = a .+ b #TODO: where to put this?
Base.:+(a::SVector, b::T) where {T<:Real} = a .+ b #TODO: where to put this?
#Base.:+(a::SVector, b) = a .+ b #TODO: where to put this?
using KrylovKit

Neig = 10#number of required eigen values
Krylov_arg = (Neig, :LM, KrylovKit.Arnoldi(tol=1e-32, krylovdim=8 + 5, verbosity=0));
Krylov_arg = (Neig, :LM, KrylovKit.Arnoldi(tol=1e-52, krylovdim=20, verbosity=0));
Krylov_arg = (Neig, :LM, KrylovKit.Arnoldi());


fig_p_mu = scatter(xlim=(1, 1e7), ylim=(1e-30, 100), yaxis=:log10, xaxis=:log10, yticks=(10 .^ (-30.0:2.0)), xlabel="p", ylabel="μ_{max}_{error}")
fig_p_t = scatter(xlim=(1, 1e7), ylim=(1e-4, 10000), yaxis=:log10, xaxis=:log10, yticks=(10 .^ (-20.0:10.0)), xlabel="p", ylabel="t")
fig_t_mu = scatter(xlim=(1e-4, 10000), ylim=(1e-30, 100), yaxis=:log10, xaxis=:log10, yticks=(10 .^ (-30.0:2.0)), xlabel="t", ylabel="μ_{max}_{error}")



setprecision(BigFloat, 1000)
ProbType = BigFloat
τmax = 2 * ProbType(pi) # the largest τ of the system
ζ = ProbType(2 // 100)          # damping coefficient
δ = ProbType(15 // 10)#0.2          # nat. freq
#δ = ProbType(15)#0.2          # nat. freq
δ = ProbType(150)#0.2          # nat. freq
ϵ = ProbType(0.15)#4#5#8;#5         # cut.coeff
τ = 2 * ProbType(pi)          # Time delay
b = ProbType(1 // 2)
T = 2 * ProbType(pi)#2pi#ProbType(6)#
mumax_Final = big"0.02"
for δ = [1.5, 15, 150]
    p_precise = (τmax, ζ, δ, ϵ, τ, b, T)



    ProbType = Float64
    (τmax, ζ, δ, ϵ, τ, b, T) = ProbType.(p_precise)
    mathieu_lddep = createMathieuProblem(ProbType.(p_precise), ProbType) # LDDE problem for Mathieu equation



    kprec = 250

    #δ = ProbType(15 // 10)mumax_Final3 = big"0.9874287882236009555254425711554590565826255214173403120863714592465644791880350340582265172860172473126945708892340805357629378386716428389571575300722"
    #mumax_Final2 = big"0.9874287882236009555254425989605153278756154426131879"
    #mumax_Final1 = big"0.9874287882236009555254425989605347897863728041017713730870144535666577474152006756191348164723044071"#RK, 10e5 6,6
    #mumax_Final = big"0.98742878822360095552544259940189685848942960229"#RK, 10e6 6,6

    #mumax_Final = big"1.2359004793519057530277"# δ = ProbType(15)#0.2          # nat. freq
    mumax_Final = big"0.80001489200832341222"# δ = ProbType(150)#0.2          # nat. freq
    #mumax_Final = big"0.80005343460495210088373593770102"# --------------ABS--------------------δ = ProbType(150)#0.2          # nat. freq
    for kprec in [200] #[250]# [20, 50, 100, 200, 500, 1000]#[400]#
        setprecision(BigFloat, kprec)
        ProbType = BigFloat
        @show eps(ProbType)^0.4#100*sqrt(eps(ProbType)
        #ProbType = Float16
        #ProbType = Float32
        #ProbType = Float64
        #ProbType = Float128


        (τmax, ζ, δ, ϵ, τ, b, T) = ProbType.(p_precise)
        mathieu_lddep = createMathieuProblem(ProbType.(p_precise), ProbType) # LDDE problem for Mathieu equation



        #-------------------------------------------
        for p in 10 .^ collect(4:4) #5)#5:10
            @show [p, kprec]
            @time mumax_Final = mumax_IntmappingRK(p, 6, 6, Krylov_arg, mathieu_lddep, verbosity=2, eigN=10, Niter=50, mu_abs_error=1e-20, T=T, τmax=τmax, dofilesave=true)# ,mu_abs_error=0.0)
        end
    end


    function spectral_k(ploc, k_nodes::Int)

        divs = 1 # keep it as 1
        method = "Chebyshev"
        coll = SCOL.defineCollMethod(k_nodes, divs, method)
        sys = MathieuSystem(ploc, ProbType)
        DetMX = SCOL.buildDetMXsWholeState(
            sys.A,
            sys.B,
            sys.c,
            1, sys.τ, coll
        )
        StochMX = SCOL.buildStochMXsWholeState(
            sys.α,
            sys.β,
            sys.γ,
            1, sys.τ, coll
        )
        CoeffM = SCOL.CoeffMatrices(DetMX.F0, DetMX.F1, DetMX.ct, StochMX.αt, StochMX.βt, StochMX.σt)
        fm_mapping = SCOL.firstMomentMapping(CoeffM)
        #SCOL.getSpectralRadiusOfMapping(fm_mapping)
        H1 = fm_mapping.F0 \ fm_mapping.F1
        mus, vecs, info = eigsolve(H1, 6, :LM)
        eigsolve(H1, 6)
        #mus, vecs, info  = geneigsolve( ( fm_mapping.F0,fm_mapping.F1))
        return abs.(mus)[1]::Float64
    end

    #using GenericSchur
    kv = 5:1:200#200
    muv = zeros(Float64, size(kv))
    tv = zeros(Float64, size(kv))
    spectral_k(Float64.(p_precise), 10)
    @code_warntype spectral_k(10)
    for (kk, nodes) in enumerate(kv)
        @show [kk, nodes]
        #@time spectral_k(nodes)
        #@time spectral_k(nodes)
        #@show spectral_k(nodes)
        muv[kk], tv[kk] = @timed spectral_k(Float64.(p_precise), nodes)
        @show (muv[kk], tv[kk], Float64(muv[kk] - mumax_Final))
    end



    #fig_p_mu = scatter(xlim=(1, 1e7), ylim=(1e-30, 100), yaxis=:log10, xaxis=:log10, yticks=(10 .^ (-30.0:2.0)), xlabel="p", ylabel="μ_{max}_{error}")
    #fig_p_t = scatter(xlim=(1, 1e7), ylim=(1e-4, 10000), yaxis=:log10, xaxis=:log10, yticks=(10 .^ (-20.0:10.0)), xlabel="p", ylabel="t")
    #fig_t_mu = scatter(xlim=(1e-4, 10000), ylim=(1e-30, 100), yaxis=:log10, xaxis=:log10, yticks=(10 .^ (-30.0:2.0)), xlabel="t", ylabel="μ_{max}_{error}")

    plot!(fig_p_mu, kv, abs.(muv .- mumax_Final))
    plot!(fig_p_t, kv, tv)
    plot!(fig_t_mu, tv, abs.(muv .- mumax_Final))


    aaa = plot(fig_p_mu, fig_p_t, fig_t_mu, legend=:bottomright)
    display(aaa)
    #@code_warntype spectral_k(5)





    (τmax, ζ, δ, ϵ, τ, b, T) = Float64.(p_precise)
    mathieu_lddep = createMathieuProblem(Float64.(p_precise), Float64)
    kk = 10
    @time mumax_IntmappingRK(100, 6, 6, Krylov_arg, mathieu_lddep, verbosity=2, eigN=12, Niter=200, mu_abs_error=1e-60, T=T, τmax=τmax, dofilesave=true)# ,mu_abs_error=0.0)

    #
    @profview mumax_IntmappingRK(100, 6, 6, Krylov_arg, mathieu_lddep, verbosity=2, eigN=kk, Niter=10, mu_abs_error=1e-150, T=T, τmax=τmax, dofilesave=true)
    ##@profview_allocs mumax_IntmappingRK(100, 6, 6, Krylov_arg, mathieu_lddep, verbosity=2, eigN=kk, Niter=200, mu_abs_error=1e-150, T=T, τmax=τmax,dofilesave=true) sample_rate =0.1
    @code_warntype cached_equidistant_lagrange_coeffs(1.3, 5)
    #@code_warntype mumax_IntmappingRK(50, 3, 3, Krylov_arg, mathieu_lddep, verbosity=2, eigN=kk, Niter=200, mu_abs_error=1e-150, T=T, τmax=τmax)

    ## parameters
    #ProbType = BigFloat
    #TCPUlimit = 1500.0
    ##TCPUlimit = 15

    #ProbType = Float64
    #TCPUlimit = 10.0

    NmultiLMS = 1
    linW = 2
    setprecision(BigFloat, 300)
    Types2test = [Float32, Float64, Float128, BigFloat]#Float16
    TCPUlimitS = [25.0, 30.0, 180.0, 200.0] / 1
    lws = [3, 4, 5, 6]
    MarkerTypes = [:cross, :diamond, :xcross, :utriangle]#:circle



    #setprecision(BigFloat, 400)
    #Types2test = [BigFloat]
    #TCPUlimitS = [500.0]
    #lws = [6]
    #MarkerTypes = [:utriangle]


    ColorValues = [RGBA(c, 0, 1 - c, 1) for c in LinRange(0.0, 1.0, 6)]
    for (ktype, (ProbType, TCPUlimit, linW, MarkerT)) in enumerate(zip(Types2test, TCPUlimitS, lws, MarkerTypes))#[Float64]#,BigFloat]
        for thefunction2test in [mumax_IntmappingRK]#, mumax_IntmappingLMS]#[mumax_IntmappingLMS]#
            if thefunction2test == mumax_IntmappingLMS
                fplot = plot
                fplot! = plot!
            else
                fplot! = scatter!
            end
            ##thefunction2test = mumax_IntmappingLMS
            #thefunction2test = mumax_IntmappingRK


            (τmax, ζ, δ, ϵ, τ, b, T) = ProbType.(p_precise)
            mathieu_lddep = createMathieuProblem(ProbType.(p_precise), ProbType)

            #-------------------------------------------
            if false
                @time mumax_Final = mumax_IntmappingLMS(10 * 5, 5, 5, Krylov_arg, mathieu_lddep, verbosity=2, mu_abs_error=1e-25, T=T, τmax=τmax)
                @time mumax_Final = mumax_IntmappingRK(100, 5, 5, Krylov_arg, mathieu_lddep, verbosity=2, mu_abs_error=1e-25, T=T, τmax=τmax)
                if ProbType == BigFloat
                    println("----------------BigFloat---------------")
                    ##  @time mumax_Final = mumax_IntmappingLMS(50_000, 6, 6, Krylov_arg, mathieu_lddep, Typ=BigFloat,verbosity=2)#~5961 sec
                    ##  FileIO.save("precompute_mumax_Final_LMS_BigFloat400_50_000_6_6__20_20.jld2","mumax_Final",mumax_Final)
                    #@show mumax_Final = FileIO.load("precompute_mumax_Final_LMS_BigFloat400_50_000_6_6__20_20.jld2", "mumax_Final")

                    @time mumax_Final = mumax_IntmappingRK(50_000, 6, 6, Krylov_arg, mathieu_lddep, Typ=BigFloat, verbosity=2)#~5961 sec
                    FileIO.save("precompute_mumax_Final_RK_BigFloat500_25_000_6_6__20_20.jld2", "mumax_Final", mumax_Final)
                    #@show mumax_Final = FileIO.load("precompute_mumax_Final_RK_BigFloat500_25_000_6_6__20_20.jld2", "mumax_Final")
                else
                    @time mumax_Final = mumax_IntmappingLMS(30_000, 6, 6, Krylov_arg, mathieu_lddep)#~2400 sec
                end

                @show mumax_Final = FileIO.load("precompute_mumax_Final_LMS_BigFloat400_50_000_6_6__20_20.jld2", "mumax_Final")

            end
            #precompute_mumax_Final_RK_BigFloat400_50_000_6_6__20_20.jld2

            #@time mumax_Final = mumax_IntmappingRK(50_000, 6, 6, Krylov_arg, mathieu_lddep, Typ=BigFloat, verbosity=2)#~5961 sec
            #FileIO.save("precompute_mumax_Final_RK_BigFloat500_25_000_6_6__20_20.jld2","mumax_Final",mumax_Final)
            #Iter: 24 : mu_max_abs: 0.9874287882236009555254425711554590559112463494673938986021350984454678689642146229546706803433633805347709909829560156719744090691124357348225808376778
            #difference: 2.3773595406710566952941081010001925996946578656928550082414661005393979753418329082293750288521699471054777390525026457529830166930128631833893323345154e-35
            #Iter: 25 : mu_max_abs: 0.9874287882236009555254425711554590565826255214173403120863714592465644791880350340582265172860172473126945708892340805357629378386716428389571575300722
            #difference: 1.0920468038751204474030101389888813218805687832736544729549898985321567972724849675216825724793400840773412284790149616930467623445194587442081194925935e-36
            #Iter: 0 : mu_max_abs: 0.9874287882236009555254425711554590565826255214173403120863714592465644791880350340582265172860172473126945708892340805357629378386716428389571575300722
            #difference: 1.0920468038751204474030101389888813218805687832736544729549898985321567972724849675216825724793400840773412284790149616930467623445194587442081194925935e-36


            #pv = floor.(Int,2 .^(3:(0.5/Npoli):(25)))
            pv = floor.(Int, 2 .^ (2:(0.2):(25)))
            pv = floor.(Int, 2 .^ (2:(0.01):(25)))
            pv = floor.(Int, 2 .^ (5:(0.5):(22)))
            pv = floor.(Int, 2 .^ (3:(0.5):(22)))
            #pv = floor.(Int, 2 .^ (7:(1.5):(22)))
            pv = unique(pv)

            Npoliv = 12:-1:1#:5#:6#6#8#
            #Npoliv = 1:6#:5#:6#6#8#
            if thefunction2test == mumax_IntmappingLMS
                Npoliv = 6:-1:1#:5#:6#6#8#
            else
                Npoliv = 6:-1:1#:5#:6#6#8#
            end
            Npoliv = 6:-2:1
            ColorValues = [RGBA(c, ktype / length(Types2test), 1 - c, 1) for c in LinRange(0.0, 1.0, length(Npoliv))]

            BenchmarkTools.DEFAULT_PARAMETERS.samples = 300#10.0#50
            BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1

            t_mu_I = zeros(Float64, length(Npoliv), length(pv))
            tstd_mu_I = zeros(Float64, length(Npoliv), length(pv))
            #μ_I = zeros(Float64, length(Npoliv), length(pv));
            μ_I = zeros(ProbType, length(Npoliv), length(pv))


            dopreciseBenchmark = false#false;#true

            ScaleSTD = 1.0
            for (kNpoli, (Npoli, ColorVal)) in enumerate(zip(Npoliv, ColorValues))
                empty!(_lagrange_cache)
                #println("------------------------------ outter ----------------------")
                #@code_warntype thefunction2test(500, Npoli, Npoli, Krylov_arg, mathieu_lddep, verbosity=2, eigN=kk, Niter=20, mu_abs_error=1e-50, T=T, τmax=τmax)# ,mu_abs_error=0.0)
                @time thefunction2test(50, Npoli, Npoli, Krylov_arg, mathieu_lddep, verbosity=0, eigN=kk, Niter=20, mu_abs_error=1e-10, T=T, τmax=τmax)# ,mu_abs_error=0.0)
                # @code_warntype thefunction2test(500, Npoli, Npoli, Krylov_arg, mathieu_lddep, verbosity=2, eigN=kk, Niter=20, mu_abs_error=1e-50, T=T, τmax=τmax)# ,mu_abs_error=0.0)
                #println("------------------------------ outter end ----------------------")

                for (kp, p) in enumerate(pv)
                    @show (ProbType, TCPUlimit, linW)
                    @show (kp, p, kNpoli, Npoli, ColorVal)
                    if kp > 5 && t_mu_I[kNpoli, kp-1] > TCPUlimit
                        t_mu_I[kNpoli, kp:end] .= Inf
                        tstd_mu_I[kNpoli, kp:end] .= NaN
                        μ_I[kNpoli, kp:end] .= NaN
                    else

                        try

                            n_p = Npoli + 0
                            1#TODO: -1 is definitly limiting, 0 seems to be enough, 1 is definitely engough
                            if thefunction2test == mumax_IntmappingLMS
                                NmultiLMS = Npoli
                            else
                                NmultiLMS = 1
                            end
                            #println("------------------------------ Inner ----------------------")
                            #@code_warntype thefunction2test(p * NmultiLMS, Npoli, n_p, Krylov_arg, mathieu_lddep, verbosity=2, mu_abs_error=1e-70, T=T, τmax=τmax)
                            μ_I[kNpoli, kp], t_mu_I[kNpoli, kp] = @timed thefunction2test(p * NmultiLMS, Npoli, n_p, Krylov_arg, mathieu_lddep,
                                verbosity=2, eigN=12, Niter=200, mu_abs_error=1e-40, T=T, τmax=τmax, dofilesave=false)

                            #@code_warntype thefunction2test(p * NmultiLMS, Npoli, n_p, Krylov_arg, mathieu_lddep, verbosity=2, mu_abs_error=1e-70, T=T, τmax=τmax)
                            #println("------------------------------ end ----------------------")
                            @show abs(μ_I[kNpoli, kp] - mumax_Final)
                            @show t_mu_I[kNpoli, kp]
                            tstd_mu_I[kNpoli, kp] = 0.0
                            if dopreciseBenchmark
                                #t = @benchmark mumax_IntmappingRK($p, $Npoli, $n_p, $Krylov_arg, $mathieu_lddep)
                                t = @benchmark thefunction2test($p * Npoli, $Npoli, $n_p, $Krylov_arg, $mathieu_lddep, verbosity=0, mu_abs_error=1e-30)
                                @show t
                                t_mu_I[kNpoli, kp] = BenchmarkTools.median(t).time / 1e9
                                tstd_mu_I[kNpoli, kp] = BenchmarkTools.std(t).time / 1e9
                            end


                        catch err
                            println("====================================TRY/CATCH============================================")
                            println(err)
                            println("an error")
                            t_mu_I[kNpoli, kp:end] .= Inf
                            tstd_mu_I[kNpoli, kp:end] .= NaN
                            μ_I[kNpoli, kp:end] .= NaN
                            println("====================================TRY/CATCH============================================")
                        end
                    end
                    #mus_SD=[mumax_SD(p, Npoli-1, mathieu_lddep) for p in pv]
                    #plot!(pv,abs.(mus_SD .- mumax_Final),markershape=:xcross,lw=4)


                end
                muerror = abs.(μ_I .- mumax_Final)

                #muerror[muerror .< 1e-13] .= NaN
                localorder = diff(log.(abs.(μ_I[kNpoli, :] .- mumax_Final))) ./ log.(pv[2:end] ./ pv[1:end-1])
                localorder = localorder[.!isnan.(localorder)]
                # localorder = localorder[muerror[kNpoli, 2:end].>1e-12]
                localorder = median(localorder)

                println("averge order of convergenve: $localorder")
                @show ColorVal
                fig_p_mu = fplot!(fig_p_mu, pv, muerror[kNpoli, :], label=Float16(localorder), legend=:bottomright, lw=linW, linecolor=kNpoli, color=ColorVal, msc=:black, m=MarkerT)

                tt = t_mu_I[kNpoli, :]
                tterror = tstd_mu_I[kNpoli, :] .* ScaleSTD
                tts = [reverse(tt - tterror); tt + tterror]
                pvs = [reverse(pv); pv]
                muss = [reverse(muerror[kNpoli, :]); muerror[kNpoli, :]]
                #plot!(fig_p_t, pvs, tts, st=:shape, lw=0, label=false, fillalpha=0.2)#, fc=:blues
                fig_p_t = fplot!(fig_p_t, pv, t_mu_I[kNpoli, :], lw=linW, linecolor=kNpoli, color=ColorVal, msc=:black, m=MarkerT)

                #plot!(fig_t_mu, tts, muss, st=:shape, lw=0, label=false, fillalpha=0.2)#, fc=:blues

                fig_t_mu = fplot!(fig_t_mu, t_mu_I[kNpoli, :], muerror[kNpoli, :], lw=linW, linecolor=ColorVal, color=ColorVal, msc=:black, m=MarkerT)


                aaa = plot(fig_p_mu, fig_p_t, fig_t_mu, legend=:bottomright)
                display(aaa)
            end

        end
    end





end #δ
# for (kNpoli, Npoli) in enumerate(Npoliv)#5#TODO:  7 nem működik valamiért
#     #fig_p_mn(median(diff(log.(abs.(mus_Int .- mumax_Final)) / log(pv[2] / pv[1]))))
#     muerror = abs.(μ_I .- mumax_Final)
#     fig_p_mu = fplot!(fig_p_mu, pv, muerror[kNpoli, :])
# 
#     tt = t_mu_I[kNpoli, :]
#     tterror = tstd_mu_I[kNpoli, :] .* ScaleSTD
#     tts = [reverse(tt - tterror); tt + tterror]
#     pvs = [reverse(pv); pv]
#     muss = [reverse(muerror[kNpoli, :]); muerror[kNpoli, :]]
# 
#     plot!(fig_p_t, pvs, tts, st=:shape, lw=0, label=false, fillalpha=0.2)#, fc=:blues
#     fig_p_t = fplot!(fig_p_t, pv, t_mu_I[kNpoli, :])
# 
#     plot!(fig_t_mu, tts, muss, st=:shape, lw=0, label=false, fillalpha=0.2)#, fc=:blues
#     fig_t_mu = fplot!(fig_t_mu, t_mu_I[kNpoli, :], muerror[kNpoli, :])
# 
#     aaa = plot(fig_p_mu, fig_p_t, fig_t_mu, legend=:bottomright)
#     display(aaa)
# end


5 + 5











## # ------------------ Stab chart --------------
## ζ = 0.1;
## ε = 1;
## τmax = 2π;
## T = 1π;
## method = SemiDiscretization(2, T / 40);
## 
## foo(δ, b) = log(spectralRadiusOfMapping(DiscreteMapping_LR(createMathieuProblem(δ, ε, b, ζ, T=T), method, τmax,
##     n_steps=Int((T + 100eps(T)) ÷ method.Δt)))); # No additive term calculated
## using MDBM
## axis = [Axis(-1:0.2:5.0, :δ),
##     Axis(-2:0.2:1.5, :b)]
## 
## iteration = 3;
## stab_border_points = getinterpolatedsolution(MDBM.solve!(MDBM_Problem(foo, axis), iteration));
## 
## scatter(stab_border_points...)
## #,xlim=(-1.,5),ylim=(-2.,1.5),
## #    label="",title="Stability border of the delay Mathieu equation",xlabel=L"\delta",ylabel=L"b",
## #    guidefontsize=14,tickfont = font(10),markersize=2,markerstrokewidth=0)

