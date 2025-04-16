using TestItemRunner

@testsnippet Plotting begin
    using Plots
    import UnicodePlots
    unicodeplots()
    UnicodePlots.default_size!(width=64)
    UnicodePlots.truecolors!()
    show_plot(fig, outname) = begin
        display(fig)
        savefig(fig, outname)
    end
end

@run_package_tests
