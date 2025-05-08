module HiPPO

using LinearAlgebra
using Polynomials
using SpecialPolynomials

export hippo_basis, reconstruct, transition, step, get_gamma

#=
Reconstruction based on series of coefficients.
=#

"""
    reconstruct(method::Symbol, N=0, x, ts)

Recovers the signal for timesteps `ts` given a matrix `x` of states of size `N`
    with time in its first dimension.
"""
reconstruct(method::Symbol, x, ts, args...; c=0.0, truncate_measure=true, kwargs...) = begin
    N = ndims(x) > 1 ? size(x, 2) : size(x, 1)
    eval_matrix = hippo_basis(method, N, ts, args...; c, truncate_measure)
    rec = eval_matrix * x
    return reverse(rec[:, end])
end

# step(A, B, x, u) = A * x + B * u
step(::Val{:euler}, A, B, x, u, dt) = (I + dt * A) * x + dt * B * u
step(::Val{:backeuler}, A, B, x, u, dt) = inv(I - dt * A) * x + dt * inv(I - dt * A) * B * u
step(::Val{:tustin}, A, B, x, u, dt) = inv(I - dt * A) * x + dt * inv(I - dt * A) * B * u
# α∈[0,1], with 0 = :euler, 1/2 = :tustin, 1 = :backeuler
step(::Val{:gbt}, A, B, x, u, dt; α=0.5) = inv(I - dt * α * A) * (I + dt * (1 - α) * A) * x + dt * inv(I - dt * α * A) * B * u
step(method::Symbol, args...; kwargs...) = step(Val(method), args...; kwargs...)

# helper_functions
get_gamma(ht=0.6931) = log(2) / ht

#=
Construction of Orthogonal Polynomial Bases the HiPPO Operators are dependend on.
=#
"""
    hippo_basis(:lagt, N, vals; c=0.0, truncate_measure=true)

Constructs the Polynomial Basis for the Translated Laguerre Operator.
"""
hippo_basis(::Val{:lagt}, N, vals, β=1.0; c=0.0, truncate_measure=true) = begin
    eval_mat = mapreduce(hcat, 1:N) do v
        b = zeros(v)
        b[end] = 1.0
        pol = Laguerre{β - 1}(b)
        pol.(vals)
    end
    eval_mat = eval_mat .* exp.(-vals ./ 2)
    if truncate_measure
        eval_mat[measure(:lagt, c, β).(vals).==0.0, :] .= 0.0
    end
    eval_mat = eval_mat .* exp.(-c * vals)
    return eval_mat
end

"""
    hippo_basis(:legt, N, vals; c=0.0, truncate_measure=true)
Constructs the Polynomial Basis for the Translated Legrendre Operator.
"""
hippo_basis(::Val{:legt}, N, vals, θ=1; c=0.0, truncate_measure=true) = begin
    eval_mat = mapreduce(hcat, 1:N) do v
        b = zeros(v)
        b[end] = 1.0
        Legendre(b).(2 .* vals ./ θ .- 1)
    end
    eval_mat = eval_mat .* transpose(sqrt.(2 * collect(0:N-1) .+ 1) .* (-1) .^ (0:N-1))
    if truncate_measure
        eval_mat[measure(:legt, c, θ).(vals).==0.0, :] .= 0.0
    end
    eval_mat = eval_mat .* exp.(-c * vals)
    return eval_mat
end

"""
    hippo_basis(:legs, N, vals, c=0.0; truncate_measure=true)
Constructs the Polynomial Basis for the Scaled Legrendre Operator.
"""
hippo_basis(::Val{:legs}, N, vals, γ=1.0; c=0.0, truncate_measure=true) = begin
    _vals = exp.(-γ .* vals)
    eval_mat = mapreduce(hcat, 1:N) do v
        b = zeros(v)
        b[end] = 1.0
        Legendre(b).(1 .- 2 .* _vals)
    end
    eval_mat = eval_mat .* transpose(sqrt.(2 * collect(0:N-1) .+ 1) .* (-1) .^ (0:N-1))
    if truncate_measure
        eval_mat[measure(:legs, c, γ).(vals).==0.0, :] .= 0.0
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

TODO:
- currently θ is ignored, leading to the assumption that the window of interest should be [t-1,t]
"""
transition(::Val{:legt}, N, θ=1) = begin
    B = sqrt.(2 .* (0:N-1) .+ 1)
    A_t = ones(N, N)
    for n in axes(A_t, 1)
        for k in axes(A_t, 2)
            A_t[n, k] = n < k ? (-1)^(n - k) : 1
        end
    end
    A = B' .* A_t .* B
    A *= 1 / θ
    B *= 1 / θ
    return -A, B
end

"""
    transition(:legs, N)
"All of my history is equally important."

The state-transition and input-transition matrices for the Scaled Legrendre Operator.
It is based on a **Scaled Uniform** measure.

TODO:
- currently only the LTI variant exists, whch depending on the size of the embedding space
  will become constant after a certain amount of reconstruction
- the LSI variant should fix this but is non-obvious how it would be applicable to a TS of prior unknown length 
"""
transition(::Val{:legs}, N, γ=1.0) = begin
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
    B = sqrt.(2 .* (0:N-1) .+ 1) * γ
    A *= γ
    return -A, B
end
transition(a::Symbol, args...) = transition(Val(a), args...)

#=
Definition of the measure functions for HiPPO.
=#

measure(::Val{:lagt}, c=0.0, β=1.0) = begin
    fn = x -> ifelse(x >= 0, x^(β - 1) * exp(-x), 0.0)
    fn_tilted = x -> exp(c * x) * fn(x)
    return fn_tilted
end
measure(::Val{:legs}, c=0.0, γ=1.0) = begin
    fn = x -> ifelse(x >= 0, 1.0, 0.0) * exp(-γ * x)
    fn_tilted = x -> exp(c * x) * fn(x)
    return fn_tilted
end
measure(::Val{:legt}, c=0.0, θ=1) = begin
    fn = x -> ifelse(x > 0, 1.0, 0.0) * ifelse((θ - x) > 0, 1.0, 0.0)
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
Helper function to come up with a more stable initial guess for the first state transition.
"""
initial_state_guess(method, A, B, u, dt; guesses=60) = begin
    N = size(A, 1)
    guess = mapreduce(hcat, 1:guesses) do _
        state = randn(N)
        step(method, A, B, state, u, dt)
    end
    return sum(guess, dims=2) #./ guesses
end

end # module HiPPO
