using LinearAlgebra

using SemiDiscretizationMethod
# === Step 1: Chebyshev Nodes and Differentiation Matrix ===

"Compute Chebyshev nodes on [-1,1] and the differentiation matrix D_cheb."
function cheb_diff_matrix(M::Int)
    N = M + 1
    k = 0:M
    x = cos.(pi * k ./ M)              # Chebyshev nodes on [-1,1]
    c = [2; ones(N-2); 2] .* (-1).^(0:M) # scaling factors
    D = zeros(Float64, N, N)
    for j in 1:N
        for k in 1:N
            if j != k
                D[j,k] = (c[j]/c[k])/(x[j]-x[k])
            end
        end
        D[j,j] = -sum(D[j, setdiff(1:N,j)])
    end
    return D, x
end

"Map nodes from [-1,1] to [0,T]."
map_nodes(x, T) = (x .+ 1) .* (T/2)

# === Step 2: Barycentric Interpolation for Delayed Values ===

"Compute barycentric weights for Chebyshev nodes on [-1,1]."
function barycentric_weights_cheb(M::Int)
    N = M + 1
    w = zeros(Float64, N)
    for k in 1:N
        if k == 1 || k == N
            w[k] = 0.5 * (-1)^(k-1)
        else
            w[k] = (-1)^(k-1)
        end
    end
    return w
end

"Compute the barycentric basis function value for node j at evaluation point t."
function barycentric_basis(t, nodes, w, j)
    if isapprox(t, nodes[j], atol=1e-12)
        return 1.0
    end
    num = w[j] / (t - nodes[j])
    den = sum(w[k] / (t - nodes[k]) for k in 1:length(nodes))
    return num/den
end

"Build an interpolation matrix P such that (P*X)[i] ≈ X(t_i−τ),
 using periodic extension (if t_i−τ<0, add T)."
function interpolation_matrix(t_nodes, τ, T)
    N = length(t_nodes)
    w = barycentric_weights_cheb(N-1)  # nodes were originally from cheb_diff_matrix(M)
    P = zeros(Float64, N, N)
    for i in 1:N
        t_eval = t_nodes[i] - τ
        if t_eval < 0
            t_eval += T  # periodic extension
        end
        for j in 1:N
            P[i,j] = barycentric_basis(t_eval, t_nodes, w, j)
        end
    end
    return P
end

# === Step 3: Assemble the Discretized Operator ===

"""
compute_floquet approximates the Floquet multipliers of the linear DDE

    x′(t) = A(t)*x(t) + B(t)*x(t-τ),

using a Chebyshev collocation with M+1 nodes on [0,T].  
p contains the parameters for the problem.
f_sys_lin(x,t,p,lddep) should return a tuple (A(t), B(t)).
"""
function compute_floquet(f_sys_lin,lddep, T, τ, p, M)
    # 1. Get collocation nodes and differentiation matrix
    D_cheb, x_cheb = cheb_diff_matrix(M)
    t_nodes = map_nodes(x_cheb, T)
    N = M + 1
    D = (2/T) * D_cheb  # scale for [0,T]
    
    # 2. Build interpolation matrix for delay
    P = interpolation_matrix(t_nodes, τ, T)
    
    # 3. Assemble block operators.
    # Here we assume the state dimension is n (e.g., n = 2 for the Mathieu eq.)
    n = 2
    A_big = zeros(Float64, n*N, n*N)
    B_big = zeros(Float64, n*N, n*N)
    for i in 1:N
        t = t_nodes[i]
        A_i, B_i = f_sys_lin(zeros(n), t, p,lddep)  # linearization at 0
        A_big[((i-1)*n+1):(i*n), ((i-1)*n+1):(i*n)] .= A_i
        B_big[((i-1)*n+1):(i*n), ((i-1)*n+1):(i*n)] .= B_i
    end
    I_n = Matrix{Float64}(I, n, n)
    P_big = kron(P, I_n)
    D_big = kron(D, I_n)
    
    # The collocation system: D_big * X = (A_big + B_big*P_big) * X.
    # Rearranged: (D_big - A_big - B_big*P_big) * X = 0.
    L = D_big - A_big - B_big * P_big
    
    # 4. The eigenvalues λ of L approximate the Floquet exponents.
    # Convert to multipliers: μ = exp(λ*T)
    λ = eigvals(L)
   
   @show  pshort = sortperm( λ, by=abs, rev=true)

    μ = exp.(λ[pshort] * T)
    return μ,λ[pshort], t_nodes, L
end



function createMathieuProblem(δ, ε, b0, a1; T=2π, Typ=Float64)
    AMx = ProportionalMX(t -> @SMatrix [0 1; -δ-ε*cos(2 * Typ(pi) / T * t) -a1])
    # AMx = ProportionalMX(t -> @SMatrix [0 1; -δ-ε*abs(cos(2 * Typ(pi) / T * t)) -a1])#Convergence is saturated at order 2 (if there is no perfect hit)
    τ1 = t -> 2 * Typ(pi) # if function is needed, the use τ1 = t->foo(t)
    BMx1 = DelayMX(τ1, t -> @SMatrix [0 0; b0 0])
    cVec = Additive(t -> @SVector [0, 0 * sin(Typ(pi) / T * t)])
    LDDEProblem(AMx, [BMx1], cVec)
end

function f_Mathieu_lin(x, t, p,lddep)
    return lddep.A(t), lddep.Bs[1](t)
end
# === Example Usage ===
ProbType = Float64
τmax = 2 * ProbType(pi) # the largest τ of the system
ζ = ProbType(2 // 100)          # damping coefficient
δ = ProbType(15 // 10)#0.2          # nat. freq
ϵ = ProbType(15 // 100)#4#5#8;#5         # cut.coeff
τ = 2 * ProbType(pi)          # Time delay
b = ProbType(1 // 2)
T = 2 * ProbType(pi)#2pi#ProbType(6)#
mathieu_lddep = createMathieuProblem(δ, ϵ, b, ζ; T=T, Typ=ProbType) # LDDE problem for Mathieu equation




## Dummy linearization for the delayed Mathieu equation:
## f_Mathieu_lin(x,t,p) returns A(t) and B(t) with parameters p = (δ, ε).
#function f_Mathieu_lin(x, t, p)
#    δ, ε = p
#    # The delayed Mathieu eq:
#    #   x₁' = x₂,
#    #   x₂' = -δ*x₁ - ε*x₁_del.
#    A = [0.0 1.0; -δ 0.0]
#    B = [0.0 0.0; -ε 0.0]
#    return A, B
#end

# Define period, delay, parameters and number of collocation points:
T = 2π
τ = T    # (for example, delay equals the period)
#p = (1.5, 0.15)
p=()
M = 20   # M+1 collocation nodes

# Compute approximate Floquet multipliers:
μ, t_nodes, L = compute_floquet(f_Mathieu_lin,mathieu_lddep, T, τ, p, M)
println("Approximate Floquet multipliers:")
println(μ)

plot(log.(abs.(μ)))
