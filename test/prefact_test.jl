@testitem "HiPPO :legs, timing Test N" setup = [Plotting] begin
    timings = []
    for lN in 1:9
        N = 2^lN
        end_ts = 400
        γ = get_gamma(5)
        ts = 0.05:0.25:end_ts
        A, B = HiPPO.transition(:legs, N, γ)
        x = 0.5 * collect(ts) .+ sinpi.(0.25 * collect(ts))
        up_state = [zeros(N)]
        ttime = time()
        for (i, v) in zip(ts, x)
            new_state = HiPPO.step(:tustin, A, B, up_state[end], v, Float64(ts.step))
            if any(isnan.(new_state))
                @warn "New State has NaN"
            end
            push!(up_state, new_state)
        end
        ptime = time() - ttime
        ttime = time()
        F = HiPPO.precompute_factorization(:tustin, A, Float64(ts.step))
        for (i, v) in zip(ts, x)
            new_state = HiPPO.step(:tustin, A, B, up_state[end], v, Float64(ts.step), F)
            if any(isnan.(new_state))
                @warn "New State has NaN"
            end
            push!(up_state, new_state)
        end
        fact_ptime = time() - ttime
        @show ptime, fact_ptime, length(ts)
        @test ptime > fact_ptime
        push!(timings, [N, ptime, fact_ptime])
    end
    timings = reduce(hcat, timings)'
    fig = plot(timings[:, 1], timings[:, 2:end], label=["direct" "pre.comp. factorization"])
    show_plot(fig, "prefact_N")
end

@testitem "HiPPO :legs, timing Test |ts|" setup = [Plotting] begin
    timings = []
    for end_ts in 200:200:3200
        N = 128
        γ = get_gamma(5)
        ts = 0.05:0.25:end_ts
        A, B = HiPPO.transition(:legs, N, γ)
        x = 0.5 * collect(ts) .+ sinpi.(0.25 * collect(ts))
        up_state = [zeros(N)]
        ttime = time()
        for (i, v) in zip(ts, x)
            new_state = HiPPO.step(:tustin, A, B, up_state[end], v, Float64(ts.step))
            if any(isnan.(new_state))
                @warn "New State has NaN"
            end
            push!(up_state, new_state)
        end
        ptime = time() - ttime
        ttime = time()
        F = HiPPO.precompute_factorization(:tustin, A, Float64(ts.step))
        for (i, v) in zip(ts, x)
            new_state = HiPPO.step(:tustin, A, B, up_state[end], v, Float64(ts.step), F)
            if any(isnan.(new_state))
                @warn "New State has NaN"
            end
            push!(up_state, new_state)
        end
        fact_ptime = time() - ttime
        @show ptime, fact_ptime, length(ts)
        @test ptime > fact_ptime
        push!(timings, [length(ts), ptime, fact_ptime])
    end
    timings = reduce(hcat, timings)'
    @show size(timings)
    fig = plot(timings[:, 1], timings[:, 2:end], label=["direct" "pre.comp. factorization"])
    show_plot(fig, "prefact_len_ts")
end