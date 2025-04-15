@testitem "HiPPO :legt" setup = [Plotting] begin
    end_ts = 200
    N = 64
    θ = 60
    ts = 1:0.5:end_ts
    A, B = HiPPO.transition(:legt, N, θ)
    x = 0.15 * collect(ts) .+ sin.(0.05 * collect(ts)) .+ randn(length(ts))
    up_state = [randn(N)]
    for v in x
        new_state = HiPPO.step(:tustin, A, B, up_state[end], v, Float64(ts.step))
        if any(isnan.(new_state))
            @warn "New State has NaN"
        end
        push!(up_state, new_state)
    end
    x_rec = HiPPO.reconstruct(:legt, reduce(hcat, up_state)', ts)
    err = abs.(x .- x_rec) ./ x
    err = err[2:end]
    fig = plot(err, width=:auto, title="LegT Err")
    display(fig)
    fig = plot(err[end-80:end], width=:auto, title="LegT Err Lens", canvas=UnicodePlots.DotCanvas)
    display(fig)
    fig = plot(x, width=:auto, title="LegT Recon", label="orig")
    plot!(x_rec, width=:auto, label="recon")
    display(fig)
end