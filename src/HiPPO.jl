module HiPPO

using LinearAlgebra
using Polynomials
using SpecialPolynomials

export hippo_basis, reconstruct, transition, step

#=
Reconstruction based on series of coefficients.
=#

"""
    reconstruct(method::Symbol, N=0, x, ts)

Recovers the signal for timesteps `ts` given a matrix `x` of states of size `N`
    with time in its first dimension.
"""
reconstruct(method::Symbol, x, ts; c=0.0, truncate_measure=true) = begin
    N = ndims(x) > 1 ? size(x, 2) : size(x, 1)
    eval_matrix = hippo_basis(method, N, ts; c, truncate_measure)
    rec = eval_matrix * x
    return reverse(rec[:, end])
end

# step(A, B, x, u) = A * x + B * u
step(::Val{:euler}, A, B, x, u, dt) = (I + dt * A) * x + dt * B * u
step(::Val{:backeuler}, A, B, x, u, dt) = inv(I - dt * A) * x + dt * inv(I - dt * A) * B * u
step(::Val{:tustin}, A, B, x, u, dt) = inv(I - dt * A) * x + dt * inv(I - dt * A) * B * u
# α∈[0,1], with 0 = :euler, 1/2 = :tustin, 1 = :backeuler
step(::Val{:gbt}, A, B, x, u, dt; α=0.5) = inv(I - dt * α * A) * (i + dt * (1 - α) * A) * x + dt * inv(I - dt * α * A) * B * u
step(method::Symbol, args...; kwargs...) = step(Val(method), args...; kwargs...)

#=
Construction of Orthogonal Polynomial Bases the HiPPO Operators are dependend on.
=#
"""
    hippo_basis(:lagt, N, vals; c=0.0, truncate_measure=true)

Constructs the Polynomial Basis for the Translated Laguerre Operator.
"""
hippo_basis(::Val{:lagt}, N, vals; c=0.0, truncate_measure=true) = begin
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

end # module HiPPO
