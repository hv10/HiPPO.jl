@testitem "HiPPO :legs" setup = [Plotting] begin
    N = 8
    ts = 1:1.0:100
    A, B = HiPPO.transition(:legs, N)
    x = collect(ts) .+ sin.(0.25 * collect(ts)) .+ randn(length(ts))
    up_state = [randn(N)]
    for v in x
        new_state = HiPPO.step(:tustin, A, B, up_state[end], v, Float64(ts.step))
        if any(isnan.(new_state))
            @warn "New State has NaN"
        end
        push!(up_state, new_state)
    end
    x_rec = HiPPO.reconstruct(:legs, reduce(hcat, up_state)', ts)
    fig = plot(abs.(x .- x_rec), width=:auto)
    display(fig)
end