module HiPPO

using LinearAlgebra
using Polynomials
using SpecialPolynomials
using ControlSystems

export hippo_basis, reconstruct, transition, step

#=
Reconstruction based on series of coefficients.
=#

"""
    reconstruct(method::Symbol, N=0, x, ts)

Recovers the signal for timesteps `ts` given a matrix `x` of states of size `N`
    with time in its first dimension.
"""
reconstruct(method::Symbol, x, ts) = begin
    N = size(x, 2)
    eval_matrix = hippo_basis(method, N, ts)
    rec = eval_matrix * x
    return reverse(rec[:, end])
end

step(A, B, x, u) = A * x + B * u

#=
Construction of Orthogonal Polynomial Bases the HiPPO Operators are dependend on.
=#
"""
    hippo_basis(:lagt, N, vals, c=0.0; truncate_measure=true)

Constructs the Polynomial Basis for the Translated Laguerre Operator.
"""
hippo_basis(::Val{:lagt}, N, vals, c=0.0; truncate_measure=true) = begin
    eval_mat = mapreduce(hcat, 1:N) do v
        b = zeros(v)
        b[end] = 1.0
        pol = Laguerre{0}(b)
        pol.(vals)
    end
    eval_mat = eval_mat .* exp.(-vals ./ 2)
    if truncate_measure
        eval_mat[measure(:lagt).(vals).==0.0, :] .= 0.0
    end
    eval_mat = eval_mat .* exp.(-c * vals)
    return eval_mat
end

"""
    hippo_basis(:legt, N, vals, c=0.0; truncate_measure=true)
Constructs the Polynomial Basis for the Translated Legrendre Operator.
"""
hippo_basis(::Val{:legt}, N, vals; c=0.0, truncate_measure=true) = begin
    eval_mat = mapreduce(hcat, 1:N) do v
        b = zeros(v)
        b[end] = 1.0
        Legendre(b).(2 .* vals .- 1)
    end
    eval_mat = eval_mat .* transpose(sqrt.(2 * collect(0:N-1) .+ 1) .* (-1) .^ (0:N-1))
    if truncate_measure
        eval_mat[measure(:legt).(vals).==0.0, :] .= 0.0
    end
    eval_mat = eval_mat .* exp.(-c * vals)
    return eval_mat
end

"""
    hippo_basis(:legs, N, vals, c=0.0; truncate_measure=true)
Constructs the Polynomial Basis for the Scaled Legrendre Operator.
"""
hippo_basis(::Val{:legs}, N, vals; c=0.0, truncate_measure=true) = begin
    _vals = exp.(-vals)
    eval_mat = mapreduce(hcat, 1:N) do v
        b = zeros(v)
        b[end] = 1.0
        Legendre(b).(1 .- 2 .* _vals)
    end
    eval_mat = eval_mat .* transpose(sqrt.(2 * collect(0:N-1) .+ 1) .* (-1) .^ (0:N-1))
    if truncate_measure
        eval_mat[measure(:legs).(vals).==0.0, :] .= 0.0
    end
    eval_mat = eval_mat .* exp.(-c * vals)
    return eval_mat
end

hippo_basis(method::Symbol, args...; kwargs...) = hippo_basis(Val(method), args...; kwargs...)

#=
HiPPO SSM State Matrix & Input Matrix Construction
=#

"""
    transition(:lagt, N, β=1.0)
"My more recent history is more important to me."

The state-transition and input-transition matrices for the Translated Laguerre Operator.
It is based on a **Exponential Decay** measure.
"""
transition(::Val{:lagt}, N, β=1.0) = begin
    A = I(N) / 2 - tril(ones(N, N))
    B = β * ones(N)
    return A, B
end

"""
    transition(:legt, N, θ=3)
"Only my history to a point is important, but equally."

The state-transition and input-transition matrices for th Translated Legrendre Operator.
It is based on a **Moving Window** measure.
"""
transition(::Val{:legt}, N, θ=3) = begin
    B = sqrt.(2 .* (0:N-1) .+ 1)
    A_t = ones(N, N)
    for i in axes(A_t, 1)
        for j in axes(A_t, 2)
            A_t[i, j] = i < j ? (-1)^(j - i) : 1
        end
    end
    A = B' .* A_t .* B
    # return A, B
    return -A, B # if we return the flipped-sign version we make later code easier
end

"""
    transition(:legs, N)
"All of my history is equally important."

The state-transition and input-transition matrices for th Scaled Legrendre Operator.
It is based on a **Scaled Uniform** measure.
"""
transition(::Val{:legs}, N) = begin
    A = zeros(N, N)
    for n in axes(A, 1)
        for k in axes(A, 2)
            if n > k
                A[n, k] = sqrt(2 * (n - 1) + 1) * sqrt(2 * (k - 1) + 1)
            elseif n == k
                A[n, k] = n
            end
        end
    end
    B = sqrt.(2 .* (0:N-1) .+ 1)
    # return A, B
    return -A, B
end
transition(a::Symbol, args...) = transition(Val(a), args...)

#=
Definition of the measure functions for HiPPO.
=#

measure(::Union{Val{:lagt},Val{:legs}}, c=0.0) = begin
    fn = x -> ifelse(x >= 0, 1.0, 0.0) * exp(-x)
    fn_tilted = x -> exp(c * x) * fn(x)
    return fn_tilted
end
measure(::Val{:legt}, c=0.0) = begin
    fn = x -> ifelse(x > 0, 1.0, 0.0) * ifelse((1.0 - x) > 0, 1.0, 0.0)
    fn_tilted = x -> exp(c * x) * fn(x)
    return fn_tilted
end

"""
    measure(method, c=0.0)
Returns the underlying measure based on the requested method.
Note: this is needed for truncating the measure.
"""
measure(a::Symbol, args...) = measure(Val(a), args...)

"""
    get_system(method::Symbol; N=64, args...; kwargs...)
Returns a ControlSystems.jl Continous Time State Space System representing the chosen method.
"""
get_system(method::Symbol, args...) = begin
    A, B = transition(Val(method), args...)
    ControlSystems.ss(A, B, I, 0) # C=I, D=0 assumes that our state **is** our output
end

end # module HiPPO
