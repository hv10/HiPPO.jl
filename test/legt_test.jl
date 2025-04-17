@testitem "HiPPO :legt" setup = [Plotting] begin
    end_ts = 20
    N = 64
    θ = 10.0
    ts = 0:0.05:end_ts
    A, B = HiPPO.transition(:legt, N, θ)
    x = 0.5 * collect(ts) .+ sinpi.(0.25 * collect(ts))
    up_state = [zeros(N)]
    for v in x
        new_state = HiPPO.step(:tustin, A, B, up_state[end], v, Float64(ts.step))
        if any(isnan.(new_state))
            @info "New State has NaN" old = up_state[end] new = new_state
        end
        push!(up_state, new_state)
    end
    x_rec = HiPPO.reconstruct(:legt, up_state[end], ts, θ)
    err = abs.(x .- x_rec) ./ abs.(x)
    err = err[2:end]
    fig = plot(err, width=:auto, title="LegT Err")
    show_plot(fig, "legt_err")
    fig = plot(err[end-80:end], width=:auto, title="LegT Err Lens", canvas=UnicodePlots.DotCanvas)
    show_plot(fig, "legt_err_lens")
    fig = plot(ts, x, width=:auto, title="LegT Recon", label="orig")
    plot!(ts, x_rec, width=:auto, label="recon")
    show_plot(fig, "legt_recon")
    fig = plot(ts[end-80:end], x[end-80:end], width=:auto, title="LegT Recon Lens", label="orig")
    plot!(ts[end-80:end], x_rec[end-80:end], width=:auto, label="recon")
    show_plot(fig, "legt_recon_lens")
end