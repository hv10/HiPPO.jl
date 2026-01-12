module HiPPO

using LinearAlgebra

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
step(::Val{:tustin}, A, B, x, u, dt) = inv(I - dt / 2 * A) * (I + dt / 2 * A) * x + dt * inv(I - dt / 2 * A) * B * u
#inv(I - dt * A) * x + dt * inv(I - dt * A) * B * u
# α∈[0,1], with 0 = :euler, 1/2 = :tustin, 1 = :backeuler
step(::Val{:gbt}, A, B, x, u, dt; α=0.5) = inv(I - dt * α * A) * (I + dt * (1 - α) * A) * x + dt * inv(I - dt * α * A) * B * u
step(::Val{:dss}, Λ, Bh, x, u, dt) = begin
    # compute exp_lambda_dt
    exp_lmbda = exp.(dt .* Λ) # elementwise exponential
    Bu = Bh * u # input contribution
    gain = ifelse.(abs.(Λ) .> 1e-12, (exp_lmbda .- 1) ./ Λ, dt .* ones(length(Λ)))
    z = exp_lmbda .* x .+ gain .* Bu
    return z
end

# faster step functions with precomputed LU factorization
precompute_factorization(::Val{:backeuler}, A, dt) = lu(I - dt * A)
precompute_factorization(::Val{:tustin}, A, dt) = lu(I - dt / 2 * A)
precompute_factorization(::Val{:gbt}, A, dt; α=0.5) = lu(I - dt * α * A)
precompute_factorization(method::Symbol, A, dt; kwargs...) = precompute_factorization(Val(method), A, dt; kwargs...)
process_to_dss(A, B) = begin
    Λ = eigvals(A)
    P = eigvecs(A)
    C = I(size(A, 1)) * P
    return Λ, P, P \ B, C
end

step(::Val{:backeuler}, A, B, x, u, dt, F::LU) = begin
    rhs = x + dt * B * u
    return F \ rhs
end

step(::Val{:tustin}, A, B, x, u, dt, F::LU) = begin
    rhs = (I + dt / 2 * A) * x + dt * B * u
    F \ rhs
end

step(::Val{:gbt}, A, B, x, u, dt, F::LU; α=0.5) = begin
    rhs = (I + dt * (1 - α) * A) * x + dt * B * u
    return F \ rhs
end

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
        laguerreP.(v-1, β-1, vals)
    end
    eval_mat = eval_mat .* exp.(-vals ./ 2)
    if truncate_measure
        mask = measure(:lagt, c, β).(vals) .!= 0.0
        eval_mat = eval_mat .* reshape(mask, :, 1)
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
        legendreP.(v-1, 2 .* vals ./ θ .- 1)
    end
    eval_mat = eval_mat .* transpose(sqrt.(2 * collect(0:N-1) .+ 1) .* (-1) .^ (0:N-1))
    if truncate_measure
        mask = measure(:legt, c, θ).(vals) .!= 0.0
        eval_mat = eval_mat .* reshape(mask, :, 1)
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
        legendreP.(v-1, 1 .- 2 .* _vals)
    end
    eval_mat = eval_mat .* transpose(sqrt.(2 * collect(0:N-1) .+ 1) .* (-1) .^ (0:N-1))
    if truncate_measure
        mask = measure(:legs, c, γ).(vals) .!= 0.0
        eval_mat = eval_mat .* reshape(mask, :, 1)
    end
    eval_mat = eval_mat .* exp.(-c * vals)
    return eval_mat
end

function legendreP(n, x)
    n == 0 && return one(x)
    n == 1 && return x

    Pnm1 = one(x)   # P₀
    Pn   = x        # P₁

    for k in 1:n-1
        Pnp1 = ((2k + 1) * x * Pn - k * Pnm1) / (k + 1)
        Pnm1, Pn = Pn, Pnp1
    end

    return Pn
end

function laguerreP(n, α, x)
    n == 0 && return one(x)
    n == 1 && return one(x) + α - x

    Lnm1 = one(x)
    Ln   = one(x) + α - x

    for k in 1:n-1
        Lnp1 = ((2k + 1 + α - x) * Ln - (k + α) * Lnm1) / (k + 1)
        Lnm1, Ln = Ln, Lnp1
    end

    return Ln
end



"""
    hippo_basis(:fout, N, vals; c=0.0, truncate_measure=true)
Constructs the Polynomial Basis for the FouT Operator.
"""
hippo_basis(::Val{:fout_old}, N, vals, θ=1; c=0.0, truncate_measure=true) = begin
    @assert iseven(N) "The Fourier basis (:fout) requires an even state dimension N."
    T = length(vals)
    freqs = collect(0:(N÷2-1))

    # Calculate cosine and sine components (T x N/2)
    # Python: 2*pi * k * vals (where vals is scaled by θ implicitly or explicit here)
    args = 2π .* freqs' .* (vals ./ θ)
    cos_mat = sqrt(2) .* cos.(args) # in python the axis order is flipped
    sin_mat = sqrt(2) .* sin.(args)
    # both cos_mat and sin_mat seem to be correct (content wise)

    # Normalization for the DC component (k=0)
    # In Python: cos[0] /= sqrt(2), making the first column 1.0
    cos_mat[:, 1] ./= sqrt(2)

    # Interleave cos and sin: [cos0, sin0, cos1, sin1, ...]
    # We initialize a matrix of size (Time x N)
    eval_mat = zeros(eltype(cos_mat), T, N)
    for k in 1:(N÷2)
        eval_mat[:, 2k-1] .= cos_mat[:, k]
        eval_mat[:, 2k] .= sin_mat[:, k]
    end

    if truncate_measure
        mask = measure(:fout, c, θ).(vals).!=0.0
        eval_mat = eval_mat .* reshape(mask, :, 1)
    end
    eval_mat = eval_mat .* exp.(-c .* vals)

    return eval_mat
end

function hippo_basis(::Val{:fout}, N, vals, θ=1; c=0.0, truncate_measure=true)
    @assert iseven(N) "The Fourier basis (:fout) requires an even state dimension N."
    
    # Generate pairs of columns [cos, sin] for each frequency k from 0 to N/2 - 1
    # This avoids loops with indexing mutations.
    columns = map(0:(N÷2 - 1)) do k
        # Calculate arguments scaled by length scale θ
        args = (2π * k) .* (vals ./ θ)
        
        # Generate raw Fourier components
        # Per sources, these are √2 * cos/sin [1]
        c_col = sqrt(2) .* cos.(args)
        s_col = sqrt(2) .* sin.(args)
        
        # Handle the DC component (k=0)
        # Normalizing √2*cos(0) = √2 by √2 gives the constant basis function 1.0 [1]
        if k == 0
            return hcat(c_col ./ sqrt(2), s_col)
        else
            return hcat(c_col, s_col)
        end
    end

    # Combine the list of 2-column matrices into one T x N matrix
    # reduce(hcat, ...) is highly optimized for Zygote's adjoints
    eval_mat = reduce(hcat, columns)

    # Apply truncation and exponential decay using broadcasting
    if truncate_measure
        # Assuming 'measure' is a differentiable function returning the weight/mask
        # We reshape the mask to ensure correct broadcasting against eval_mat
        m_vals = measure(:fout, c, θ).(vals)
        mask = reshape(m_vals .!= 0.0, :, 1)
        eval_mat = eval_mat .* mask
    end
    
    # Apply the "descent" or decay γ represented by c [2]
    return eval_mat .* exp.(-c .* vals)
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
- we introduced γ to make the area of good reconstruction (the time scale of the measure) configurable
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

# Currently does not consider θ. #TODO: fix
"""
FouT
"""
transition(::Val{:fout}, N, θ=1.0) = begin
    @assert iseven(N) "The Fourier basis (:fout) requires an even state dimension N."
    A = zeros(N, N)
    A[(1:N).%2 .== 1, (1:N).%2 .== 1] .= -4 # even rows and even columns (not odd as in the paper)
    A[(1:N).%2 .== 1, 1] .= -2*sqrt(2) # first row even columns
    A[1, (1:N).%2 .== 1] .= -2*sqrt(2) # first column odd rows
    A[1, 1] = -2 # top left corner is special

    # For (n-k) == 1 and k even: This targets A[3,2], A[5,4], A[7,6]...
    idx = 3:2:N
    A[CartesianIndex.(idx .+ 1, idx)] .= 2π .* div.(idx, 2)
    # For (k-n) == 1 and n even: This targets A[2,3], A[4,5], A[6,7]...
    A[CartesianIndex.(idx, idx .+ 1)] .= -2π .* div.(idx, 2)
    B = zeros(N)
    B[1:2:end] .= 2 * sqrt(2) 
    B[1] = 2 

    A *= (1 / θ)
    B *= (1 / θ)
    return A, B
end

#=
Definition of the measure functions for HiPPO.
=#
tilt_fn(c, fn) = x -> exp(c * x) * fn(x)

measure(::Val{:lagt}, c=0.0, β=1.0) = begin
    fn = x -> ifelse(x >= 0, x^(β - 1) * exp(-x), 0.0)
    fn_tilted = tilt_fn(c, fn)
    return fn_tilted
end
measure(::Val{:legs}, c=0.0, γ=1.0) = begin
    fn = x -> ifelse(x >= 0, 1.0, 0.0) * exp(-γ * x)
    fn_tilted = tilt_fn(c, fn)
    return fn_tilted
end
measure(::Val{:legt}, c=0.0, θ=1) = begin
    fn = x -> ifelse(x > 0, 1.0, 0.0) * ifelse((θ - x) > 0, 1.0, 0.0)
    fn_tilted = tilt_fn(c, fn)
    return fn_tilted
end
measure(::Val{:fout}, c=0.0, θ=1) = begin
    fn = x -> ifelse(x >= 0, 1.0, 0.0) * ifelse((θ - x) >= 0, 1.0, 0.0)
    fn_tilted = tilt_fn(c, fn)
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
