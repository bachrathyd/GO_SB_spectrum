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

#FileIO.save("myfile.jld2","a",a)
#b = FileIO.load("myfile.jld2","a")
#---------------------------- Solution with precomputed A-s -------------------------

function createMathieuProblem(δ, ε, b0, a1; T=2π, Typ=Float64)
    AMx = ProportionalMX(t -> @SMatrix [0 1; -δ-ε*cos(2 * Typ(pi) / T * t) -a1])
    # AMx = ProportionalMX(t -> @SMatrix [0 1; -δ-ε*abs(cos(2 * Typ(pi) / T * t)) -a1])#Convergence is saturated at order 2 (if there is no perfect hit)
    τ1 = t -> 2 * Typ(pi) # if function is needed, the use τ1 = t->foo(t)
    BMx1 = DelayMX(τ1, t -> @SMatrix [0 0; b0 0])
    cVec = Additive(t -> @SVector [0, 0 * sin(Typ(pi) / T * t)])
    LDDEProblem(AMx, [BMx1], cVec)
end

# ---------------------------------------------- Testing the spectrum of a mapping --------------------------------
function mumax_IntmappingRK(p, N_bt, n_p, Krylov_arg, mathieu_lddep; Typ=typeof(mathieu_lddep.A(0.0)[1]), verbosity=0, mu_abs_error=eps(ProbType)^0.4, Niter=25, eigN=10)

    ti = LinRange(0, T, p + 1)
    h = ti[2] - ti[1]

    BT_loc = Butchertable(N_bt, Typ)

    r = Int64((τmax + 100eps(τmax)) ÷ h + n_p ÷ 2 + 1)
    #xhist = [SA[Typ.([1, 0])...] for i in -r:0]
    xhist = [MVector(Typ.([1, 0])...) for i in -r:0]

    (A_t, Bs_t, τs_t, c_t, t_all) = precomputed_coefficients(mathieu_lddep, BT_loc[1], h, ti)
    @inbounds foo(xh) = runge_kutta_solve!(A_t, Bs_t, τs_t, c_t, t_all, h, xh, p; BT=BT_loc, n_points=n_p)[end-r:end]

    if true#Typ == BigFloat#false#
        mus = issi_eigen(foo, xhist, eigN, Niter, verbosity=verbosity, mu_abs_error=mu_abs_error)
        mumax1 = abs.(mus[1])[1]
        return mumax1
    else
        mus = getindex(schursolve(foo, xhist, Krylov_arg...), [3, 2, 1])
        @show mumax1 = abs.(mus[1])[1]
        return mumax1
    end
end

function mumax_IntmappingLMS(p, N_LMN, n_p, Krylov_arg, mathieu_lddep; Typ=typeof(mathieu_lddep.A(0.0)[1]), verbosity=0, mu_abs_error=eps(ProbType)^0.4, Niter=25, eigN=10)
    # p=100
    # N_LMN=4
    # n_p=3
    # Typ=typeof(mathieu_lddep.A(0.0)[1])
    # verbosity=0
    # mu_abs_error=1e-10
    ti = LinRange(0.0, T, p + 1)
    h = ti[2] - ti[1]
    ti = cat(h .* (-(N_LMN - 1):-1), ti, dims=1)

    β = LinearMultiStepCoeff(N_LMN)

    r = Int64((τmax + 100eps(τmax)) ÷ h + n_p ÷ 2 + 1 + (N_LMN - 1))
    #xhist = [SA[Typ.([1, 0])...] for i in -r:0]
    xhist = [MVector(Typ.([1, 0])...) for i in -r:0]

    (A_t, Bs_t, τs_t, c_t, t_all) = precomputed_coefficients(mathieu_lddep, [1], h, ti)

    foo(xh) = LinMultiStep_solve!(A_t, Bs_t, τs_t, c_t, t_all, h, xh, p; β=β, n_points=n_p)[end-r:end]
    #foo(xhist)
    if true#Typ == BigFloat#false#
        mus = issi_eigen(foo, xhist, eigN, Niter, verbosity=verbosity, mu_abs_error=mu_abs_error)
        mumax1 = abs.(mus[1])[1]
        return mumax1
    else
        mus = getindex(schursolve(foo, xhist, Krylov_arg...), [3, 2, 1])
        @show mumax1 = abs.(mus[1])[1]
        return mumax1
    end
end

using GenericSchur
using Quadmath
#piq = Float128(pi)
#A = rand(BigFloat,5,5)
##x=rand(BigFloat,5)
##foo(xx)=A*xx
##foo(x)
##eigsolve(foo, x)
#A = [5. 7.; -2. -4.]
#cc=GenericSchur.schur(A)

using SemiDiscretizationMethod
function mumax_SD(p, N_bt, mathieu_lddep)
    method = SemiDiscretization(N_bt, T / p)
    τmax = τ # the largest τ of the system

    n_Steps = Int((T + 100 * eps(T)) ÷ method.Δt)
    mapping = DiscreteMapping_LR(mathieu_lddep, method, τmax,
        n_steps=n_Steps, calculate_additive=false) #The
    mumaxSD = spectralRadiusOfMapping(mapping)
end

Base.:+(a::SVector, b::Bool) = a .+ b
Base.:+(a::SVector, b::Float64) = a .+ b #TODO: where to put this?
Base.:+(a::SVector, b::T) where {T<:Real} = a .+ b #TODO: where to put this?
#Base.:+(a::SVector, b) = a .+ b #TODO: where to put this?
using KrylovKit

Neig = 1#number of required eigen values
Krylov_arg = (Neig, :LM, KrylovKit.Arnoldi(tol=1e-32, krylovdim=8 + 5, verbosity=0));
Krylov_arg = (Neig, :LM, KrylovKit.Arnoldi(tol=1e-52, krylovdim=40, verbosity=0));
Krylov_arg = (Neig, :LM, KrylovKit.Arnoldi());


fig_p_mu = scatter(xlim=(1, 1e7), ylim=(1e-30, 100), yaxis=:log10, xaxis=:log10, yticks=(10 .^ (-30.0:2.0)), xlabel="p", ylabel="μ_{max}_{error}")
fig_p_t = scatter(xlim=(1, 1e7), ylim=(1e-4, 10000), yaxis=:log10, xaxis=:log10, yticks=(10 .^ (-20.0:10.0)), xlabel="p", ylabel="t")
fig_t_mu = scatter(xlim=(1e-4, 10000), ylim=(1e-30, 100), yaxis=:log10, xaxis=:log10, yticks=(10 .^ (-30.0:2.0)), xlabel="t", ylabel="μ_{max}_{error}")
setprecision(BigFloat, 10000)

for kprec in[500]# [20, 50, 100, 200, 500, 1000]#[400]#
    setprecision(BigFloat, kprec)
    ProbType = BigFloat
    @show eps(ProbType)^0.4#100*sqrt(eps(ProbType)
    #ProbType = Float16
    #ProbType = Float32
    #ProbType = Float64
    #ProbType = Float128

    τmax = 2 * ProbType(pi) # the largest τ of the system
    ζ = ProbType(2 // 100)          # damping coefficient
    δ = ProbType(15 // 10)#0.2          # nat. freq
    ϵ = ProbType(0.15)#4#5#8;#5         # cut.coeff
    τ = 2 * ProbType(pi)          # Time delay
    b = ProbType(1 // 2)
    T = 2 * ProbType(pi)#2pi#ProbType(6)#


    mathieu_lddep = createMathieuProblem(δ, ϵ, b, ζ; T=T, Typ=ProbType) # LDDE problem for Mathieu equation

    #-------------------------------------------
    for kk in [6]#5:10
        @show [kk, kprec]
#        @time mumax_Final = mumax_IntmappingLMS(1000*5, 5, 5, Krylov_arg, mathieu_lddep, verbosity=2, eigN=kk, Niter=200, mu_abs_error=0.0)
        @time mumax_Final = mumax_IntmappingRK(1000, 6, 6, Krylov_arg, mathieu_lddep, verbosity=2, eigN=kk, Niter=200,mu_abs_error=0.0)# ,mu_abs_error=0.0)
    end
end
## parameters
#ProbType = BigFloat
#TCPUlimit = 1500.0
##TCPUlimit = 15

#ProbType = Float64
#TCPUlimit = 10.0

NmultiLMS = 1
linW = 2

Types2test = [Float16, Float32, Float64, Float128, BigFloat]
TCPUlimitS = [4.0, 10.0, 20.0, 30.0, 100.0]
lws = [2, 3, 4, 5]
MarkerTypes = [:circle, :cross, :diamond, :xcross]

ColorValues = [RGBA(c, 0, 1 - c, 1) for c in LinRange(0.0, 1.0, 6)]
for (ktype, (ProbType, TCPUlimit, linW, MarkerT)) in enumerate(zip(Types2test, TCPUlimitS, lws, MarkerTypes))#[Float64]#,BigFloat]
    for thefunction2test in [mumax_IntmappingRK, mumax_IntmappingLMS]#[mumax_IntmappingLMS]#
        if thefunction2test == mumax_IntmappingLMS
            fplot = plot
            fplot! = plot!
        else
            fplot! = scatter!
        end
        ##thefunction2test = mumax_IntmappingLMS
        #thefunction2test = mumax_IntmappingRK

        τmax = 2 * ProbType(pi) # the largest τ of the system
        ζ = ProbType(2 // 100)          # damping coefficient
        δ = ProbType(15 // 10)#0.2          # nat. freq
        ϵ = ProbType(15 // 100)#4#5#8;#5         # cut.coeff
        τ = 2 * ProbType(pi)          # Time delay
        b = ProbType(1 // 2)
        T = 2 * ProbType(pi)#2pi#ProbType(6)#
        mathieu_lddep = createMathieuProblem(δ, ϵ, b, ζ; T=T, Typ=ProbType) # LDDE problem for Mathieu equation

        #-------------------------------------------
        @time mumax_Final = mumax_IntmappingLMS(100 * 5, 5, 5, Krylov_arg, mathieu_lddep, verbosity=2, mu_abs_error=1e-25)
        @time mumax_Final = mumax_IntmappingRK(1000, 5, 5, Krylov_arg, mathieu_lddep, verbosity=2, mu_abs_error=1e-25)

        if false
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

        mumax_Final = big"0.9874287882236009555254425711554590565826255214173403120863714592465644791880350340582265172860172473126945708892340805357629378386716428389571575300722"
        mumax_Final = big"0.9874287882236009555281246653069565850472840603735245524859608723524041405173460592718812237414602982538386862768442005172631528307123397904995178872517"
          #pv = floor.(Int,2 .^(3:(0.5/Npoli):(25)))
        pv = floor.(Int, 2 .^ (2:(0.2):(25)))
        pv = floor.(Int, 2 .^ (2:(0.01):(25)))
        pv = floor.(Int, 2 .^ (5:(0.5):(22)))
        pv = floor.(Int, 2 .^ (4:(0.5):(22)))
        pv = floor.(Int, 2 .^ (2:(0.5):(22)))
        pv = unique(pv)

        Npoliv = 12:-1:1#:5#:6#6#8#
        #Npoliv = 1:6#:5#:6#6#8#
        if thefunction2test == mumax_IntmappingLMS
            Npoliv = 6:-1:1#:5#:6#6#8#
        else
            Npoliv = 6:-1:1#:5#:6#6#8#
        end

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

            @timed thefunction2test(500, Npoli, Npoli, Krylov_arg, mathieu_lddep, verbosity=0)#, mu_abs_error=1e-30)

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
                        μ_I[kNpoli, kp], t_mu_I[kNpoli, kp] = @timed thefunction2test(p * NmultiLMS, Npoli, n_p, Krylov_arg, mathieu_lddep, verbosity=2, mu_abs_error=1e-50)
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
                    # mumax_Final= μ_I[kNpoli, kp]#
                end
                #mus_SD=[mumax_SD(p, Npoli-1, mathieu_lddep) for p in pv]
                #plot!(pv,abs.(mus_SD .- mumax_Final),markershape=:xcross,lw=4)


            end
            #mumax_Final=μ_I[kNpoli, .!isnan.(μ_I[kNpoli, :])][end]
            muerror = abs.(μ_I .- mumax_Final)

            #muerror[muerror .< 1e-13] .= NaN
            localorder = diff(log.(abs.(μ_I[kNpoli, :] .- mumax_Final))) ./ log.(pv[2:end] ./ pv[1:end-1])
            localorder = localorder[.!isnan.(localorder)]
            # localorder = localorder[muerror[kNpoli, 2:end].>1e-12]
            localorder = median(localorder)

            println("averge order of convergenve: $localorder")
            @show ColorVal
            fig_p_mu = fplot!(fig_p_mu, pv, muerror[kNpoli, :], label=Float16(localorder), legend=:bottomright, lw=linW,
                linecolor=kNpoli, color=ColorVal, msc=:black, m=MarkerT)

            fig_p_mu = fplot!(fig_p_mu, pv, muerror[kNpoli, :], label=Float16(localorder), legend=:bottomright, lw=linW,
                linecolor=kNpoli, color=ColorVal, msc=:black, m=MarkerT)

            tt = t_mu_I[kNpoli, :]
            tterror = tstd_mu_I[kNpoli, :] .* ScaleSTD
            tts = [reverse(tt - tterror); tt + tterror]
            pvs = [reverse(pv); pv]
            muss = [reverse(muerror[kNpoli, :]); muerror[kNpoli, :]]
            #plot!(fig_p_t, pvs, tts, st=:shape, lw=0, label=false, fillalpha=0.2)#, fc=:blues
            fig_p_t = fplot!(fig_p_t, pv, t_mu_I[kNpoli, :], lw=linW,
                linecolor=kNpoli, color=ColorVal, msc=:black, m=MarkerT)

            #plot!(fig_t_mu, tts, muss, st=:shape, lw=0, label=false, fillalpha=0.2)#, fc=:blues

            fig_t_mu = fplot!(fig_t_mu, t_mu_I[kNpoli, :], muerror[kNpoli, :], lw=linW,
                linecolor=ColorVal, color=ColorVal, msc=:black, m=MarkerT)

            aaa = plot(fig_p_mu, fig_p_t, fig_t_mu, legend=:bottomright)
            display(aaa)
        end

    end
end
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
## a1 = 0.1;
## ε = 1;
## τmax = 2π;
## T = 1π;
## method = SemiDiscretization(2, T / 40);
## 
## foo(δ, b0) = log(spectralRadiusOfMapping(DiscreteMapping_LR(createMathieuProblem(δ, ε, b0, a1, T=T), method, τmax,
##     n_steps=Int((T + 100eps(T)) ÷ method.Δt)))); # No additive term calculated
## using MDBM
## axis = [Axis(-1:0.2:5.0, :δ),
##     Axis(-2:0.2:1.5, :b0)]
## 
## iteration = 3;
## stab_border_points = getinterpolatedsolution(MDBM.solve!(MDBM_Problem(foo, axis), iteration));
## 
## scatter(stab_border_points...)
## #,xlim=(-1.,5),ylim=(-2.,1.5),
## #    label="",title="Stability border of the delay Mathieu equation",xlabel=L"\delta",ylabel=L"b_0",
## #    guidefontsize=14,tickfont = font(10),markersize=2,markerstrokewidth=0)

