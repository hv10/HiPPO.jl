@testitem "HiPPO :legs" setup = [Plotting] begin
    end_ts = 400
    N = 64
    γ = get_gamma(5)
    ts = 0.05:0.25:end_ts
    A, B = HiPPO.transition(:legs, N, γ)
    x = 0.5 * collect(ts) .+ sinpi.(0.25 * collect(ts))
    up_state = [zeros(N)]
    for (i, v) in zip(ts, x)
        F = HiPPO.precompute_factorization(:tustin, A, Float64(ts.step))
        new_state = HiPPO.step(:tustin, A, B, up_state[end], v, Float64(ts.step), F)
        if any(isnan.(new_state))
            @warn "New State has NaN"
        end
        push!(up_state, new_state)
    end
    x_rec = HiPPO.reconstruct(:legs, up_state[end], ts, γ)
    err = abs.(x .- x_rec) ./ abs.(x)
    err = err[2:end]
    fig = plot(err, width=:auto, title="LegS Err")
    show_plot(fig, "legs_err")
    fig = plot(err[end-80:end], width=:auto, title="LegS Err Lens", canvas=UnicodePlots.DotCanvas)
    show_plot(fig, "legs_err_lens")
    fig = plot(ts, x, width=:auto, title="LegS Recon", label="orig")
    plot!(ts, x_rec, width=:auto, label="recon")
    show_plot(fig, "legs_recon")
    fig = plot(ts[end-80:end], x[end-80:end], width=:auto, title="LegS Recon Lens", label="orig")
    plot!(ts[end-80:end], x_rec[end-80:end], width=:auto, label="recon")
    show_plot(fig, "legs_recon_lens")
end