@testitem "HiPPO :legs diagonalization" setup = [Imports] begin
    N = 64
    γ = get_gamma(5)
    A, B = HiPPO.transition(:legs, N, γ)
    Λ, P, Bh, C = HiPPO.process_to_dss(A, B)
    @show size(Λ), size(P), size(Bh), size(C)
end

@testitem "HiPPO :legt diagonalization" setup = [Imports] begin
    N = 64
    θ = 50
    A, B = HiPPO.transition(:legt, N, θ)
    Λ, P, Bh, C = HiPPO.process_to_dss(A, B)
    @show size(Λ), size(P), size(Bh), size(C)
end

@testitem "HiPPO :lagt diagonalization" setup = [Imports] begin
    N = 64
    β = 1.0
    A, B = HiPPO.transition(:lagt, N, β)
    Λ, P, Bh, C = HiPPO.process_to_dss(A, B)
    @show size(Λ), size(P), size(Bh), size(C)
end

@testitem "HiPPO :legs, :s4_dss" setup = [Imports] begin
    N = 64
    γ = get_gamma(5)
    A, B = HiPPO.transition(:legs, N, γ)
    Λ, P, Bh, C = HiPPO.process_to_dss(A, B)
    @show size(Λ), size(P), size(Bh), size(C)
end

@testitem "HiPPO :legt, :s4_dss" setup = [Plotting, Imports] begin
    N = 64
    γ = get_gamma(5)
    A, B = HiPPO.transition(:legs, N, γ)
    Λ, P, Bh, C = HiPPO.process_to_dss(A, B)
    end_ts = 400
    ts = 0.05:0.25:end_ts
    x = 0.5 * collect(ts) .+ sinpi.(0.25 * collect(ts))
    up_state = [zeros(ComplexF64, N)]
    ttime = time()
    for (i, v) in zip(ts, x)
        new_state = HiPPO.step(:dss, Λ, Bh, up_state[end], v, Float64(ts.step))
        if any(isnan.(new_state))
            @warn "New State has NaN"
        end
        push!(up_state, new_state)
    end
    @show time() - ttime
    x_rec = real.(HiPPO.reconstruct_dss(:legs, P, up_state[end], ts, γ))
    err = abs.(x .- x_rec) ./ abs.(x)
    err = err[2:end]
    fig = plot(err, width=:auto, title="DSS LegS Err")
    show_plot(fig, "dss_legs_err")
    fig = plot(err[end-80:end], width=:auto, title="DSS LegS Err Lens", canvas=UnicodePlots.DotCanvas)
    show_plot(fig, "dss_legs_err_lens")
    fig = plot(ts, x, width=:auto, title="DSS LegS Recon", label="orig")
    plot!(ts, x_rec, width=:auto, label="recon")
    show_plot(fig, "dss_legs_recon")
    fig = plot(ts[end-80:end], x[end-80:end], width=:auto, title="DSS LegS Recon Lens", label="orig")
    plot!(ts[end-80:end], x_rec[end-80:end], width=:auto, label="recon")
    show_plot(fig, "dss_legs_recon_lens")
end

@testitem "HiPPO: dss is equal to prefact. reconstruction" setup = [Imports] begin
    N = 64
    γ = get_gamma(5)
    A, B = HiPPO.transition(:legs, N, γ)
    Λ, P, Bh, C = HiPPO.process_to_dss(A, B)
    end_ts = 400
    ts = 0.05:0.25:end_ts
    x = 0.5 * collect(ts) .+ sinpi.(0.25 * collect(ts))
    up_state = [zeros(N)]
    ttime = time()
    for (i, v) in zip(ts, x)
        new_state = HiPPO.step(:dss, Λ, Bh, up_state[end], v, Float64(ts.step))
        if any(isnan.(new_state))
            @warn "New State has NaN"
        end
        push!(up_state, new_state)
    end
    @info "DSS Time" el_time = time() - ttime
    @info "Last State" P * up_state[end]
    # I know this is currently incorrect.
    # TODO: fix by approximating closest state in original basis
    # TODO: do not multiply P here as that denies the speedup of dss for the reconstruction path
    x_rec_dss = HiPPO.reconstruct(:legs, P * up_state[end], ts, γ)


    up_state = [zeros(N)]
    ttime = time()
    for (i, v) in zip(ts, x)
        F = HiPPO.precompute_factorization(:tustin, A, Float64(ts.step))
        new_state = HiPPO.step(:tustin, A, B, up_state[end], v, Float64(ts.step), F)
        if any(isnan.(new_state))
            @warn "New State has NaN"
        end
        push!(up_state, new_state)
    end
    @info "Prefact. Time" eltime = time() - ttime
    x_rec_pf = HiPPO.reconstruct(:legs, up_state[end], ts, γ)

    @info "Equal?" all((≈).(x_rec_dss, x_rec_pf; atol=1e-2))
    @info "Diff" maximum(abs.(x_rec_dss .- x_rec_pf))
    @test all(≈(x_rec_dss, x_rec_pf; atol=1e-2))
end