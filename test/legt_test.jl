@testitem "HiPPO :legt" setup = [Plotting] begin
    end_ts = 20
    N = 32
    θ = 60
    ts = 0:0.025:end_ts
    A, B = HiPPO.transition(:legt, N, θ)
    x = 0.5 * collect(ts) .+ sinpi.(0.05 * collect(ts)) .+ 0.25 * randn(length(ts))
    up_state = [zeros(N)]
    for v in x
        new_state = HiPPO.step(:tustin, A, B, up_state[end], v, Float64(ts.step))
        if any(isnan.(new_state))
            @warn "New State has NaN"
        end
        push!(up_state, new_state)
    end
    x_rec = HiPPO.reconstruct(:legt, up_state[end], ts)
    err = abs.(x .- x_rec) ./ abs.(x)
    err = err[2:end]
    fig = plot(err, width=:auto, title="LegT Err")
    display(fig)
    fig = plot(err[end-80:end], width=:auto, title="LegT Err Lens", canvas=UnicodePlots.DotCanvas)
    display(fig)
    fig = plot(x[end-80:end-1], width=:auto, title="LegT Recon", label="orig")
    plot!(x_rec[end-80:end-1], width=:auto, label="recon")
    display(fig)
end