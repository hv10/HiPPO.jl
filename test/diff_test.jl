@testsnippet Diff begin
    using Zygote
    N = 4
    ts = 1:0.025:20
    y = 0.5 * collect(ts) .+ sinpi.(0.25 * collect(ts))
end

@testitem "HiPPO: :legs reconstruction is differentiable" setup = [Diff] begin
    param = HiPPO.get_gamma(5) # param = γ
    A, B = HiPPO.transition(:legs, N, param)
    up_state = accumulate(y; init=zeros(N)) do state, v
        HiPPO.step(:tustin, A, B, state, v, Float64(ts.step))
    end
    state = up_state[end]
    x_rec = HiPPO.reconstruct(:legs, state, [1.0], param)
    gs_x = Zygote.gradient(x -> sum(abs2.(y - HiPPO.reconstruct(:legs, x, ts, param))), state)
    @info "Gradient w.r.t. x" gs_x
    gs_p = Zygote.gradient(x -> sum(abs2.(y - HiPPO.reconstruct(:legs, state, ts, x))), param)
    @info "Gradient w.r.t. γ" gs_p
end

@testitem "HiPPO: :legt reconstruction is differentiable" setup = [Diff] begin
    param = 10.0 # param = θ
    A, B = HiPPO.transition(:legt, N, param)
    up_state = accumulate(y; init=zeros(N)) do state, v
        HiPPO.step(:tustin, A, B, state, v, Float64(ts.step))
    end
    state = up_state[end]
    x_rec = HiPPO.reconstruct(:legt, state, [1.0], param)
    gs_x = Zygote.gradient(x -> sum(abs2.(y - HiPPO.reconstruct(:legt, x, ts, param))), state)
    @info "Gradient w.r.t. x" gs_x
    gs_p = Zygote.gradient(x -> sum(abs2.(y - HiPPO.reconstruct(:legt, state, ts, x))), param)
    @info "Gradient w.r.t. θ" gs_p
end

@testitem "HiPPO: :lagt reconstruction is differentiable" setup = [Diff] begin
    param = 1.1 # param = β
    A, B = HiPPO.transition(:lagt, N, param)
    up_state = accumulate(y; init=zeros(N)) do state, v
        HiPPO.step(:tustin, A, B, state, v, Float64(ts.step))
    end
    state = up_state[end]
    x_rec = HiPPO.reconstruct(:lagt, state, [1.0], param)
    gs_x = Zygote.gradient(x -> sum(abs2.(y - HiPPO.reconstruct(:lagt, x, ts, param))), state)
    @info "Gradient w.r.t. x" gs_x
    gs_p = Zygote.gradient(x -> sum(abs2.(y - HiPPO.reconstruct(:lagt, state, ts, x))), param)
    @info "Gradient w.r.t. β" gs_p
end

@testitem "HiPPO: :fout reconstruction is differentiable" setup = [Diff] begin
    param = 10.0 # param = θ
    A, B = HiPPO.transition(:fout, N, param)
    up_state = accumulate(y; init=zeros(N)) do state, v
        HiPPO.step(:tustin, A, B, state, v, Float64(ts.step))
    end
    state = up_state[end]
    x_rec = HiPPO.reconstruct(:fout, state, [1.0], param)
    gs_x = Zygote.gradient(x -> sum(abs2.(y - HiPPO.reconstruct(:fout, x, ts, param))), state)
    @info "Gradient w.r.t. x" gs_x
    gs_p = Zygote.gradient(x -> sum(abs2.(y - HiPPO.reconstruct(:fout, state, ts, x))), param)
    @info "Gradient w.r.t. θ" gs_p
end