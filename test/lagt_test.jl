@testitem "HiPPO :lagt" setup = [Plotting] begin
    end_ts = 20
    N = 64
    β = 1.0
    ts = 1:0.025:end_ts
    A, B = HiPPO.transition(:lagt, N, β)
    x = 0.5 * collect(ts) .+ sinpi.(0.25 * collect(ts))
    up_state = [zeros(N)]
    for v in x
        new_state = HiPPO.step(:tustin, A, B, up_state[end], v, Float64(ts.step))
        if any(isnan.(new_state))
            @warn "New State has NaN"
        end
        push!(up_state, new_state)
    end
    x_rec = HiPPO.reconstruct(:lagt, up_state[end], ts, β)
    @show size(x_rec)
    err = abs.(x .- x_rec) ./ abs.(x)
    err = err[2:end]
    fig = plot(err, width=:auto, title="LagT Err")
    show_plot(fig, "lagt_err")
    fig = plot(err[end-80:end], width=:auto, title="LagT Err Lens", canvas=UnicodePlots.DotCanvas)
    show_plot(fig, "lagt_err_lens")
    fig = plot(ts, x, width=:auto, title="LagT Recon", label="orig")
    plot!(ts, x_rec, width=:auto, label="recon")
    show_plot(fig, "lagt_recon")
    fig = plot(ts[end-80:end], x[end-80:end], width=:auto, title="LagT Recon Lens", label="orig")
    plot!(ts[end-80:end], x_rec[end-80:end], width=:auto, label="recon")
    show_plot(fig, "lagt_recon_lens")
end