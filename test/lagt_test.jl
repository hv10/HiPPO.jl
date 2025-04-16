@testitem "HiPPO :lagt" setup = [Plotting] begin
    end_ts = 20
    N = 64
    β = 1.0
    ts = 1:0.025:end_ts
    A, B = HiPPO.transition(:lagt, N, β)
    x = 0.5 * collect(ts) .+ sinpi.(0.05 * collect(ts)) .+ 0.25 * randn(length(ts))
    up_state = [zeros(N)]
    for v in x
        new_state = HiPPO.step(:tustin, A, B, up_state[end], v, Float64(ts.step))
        if any(isnan.(new_state))
            @warn "New State has NaN"
        end
        push!(up_state, new_state)
    end
    x_rec = HiPPO.reconstruct(:lagt, up_state[end], ts)
    @show size(x_rec)
    err = abs.(x .- x_rec) ./ abs.(x)
    err = err[2:end]
    fig = plot(err, width=:auto, title="LagT Err")
    show_plot(fig, "lagt_err.txt")
    fig = plot(err[end-80:end], width=:auto, title="LagT Err Lens", canvas=UnicodePlots.DotCanvas)
    show_plot(fig, "lagt_err_lens.txt")
    fig = plot(x, width=:auto, title="LagT Recon", label="orig")
    plot!(x_rec, width=:auto, label="recon")
    show_plot(fig, "lagt_recon.txt")
end