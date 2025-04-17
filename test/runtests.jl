using TestItemRunner

@testsnippet Plotting begin
    using Plots
    import UnicodePlots
    # unicodeplots()
    gr()
    UnicodePlots.default_size!(width=72)
    UnicodePlots.truecolors!()
    Plots.default(minorticks=2)

    show_plot(fig, outname) = begin
        if Plots.backend() == Plots.UnicodePlotsBackend()
            outname *= ".txt"
        else
            outname *= ".png"
        end
        display(fig)
        mkpath("out")
        savefig(fig, joinpath("out", outname))
    end
end

@run_package_tests
