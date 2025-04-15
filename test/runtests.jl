using TestItemRunner

@testsnippet Plotting begin
    using Plots
    import UnicodePlots
    unicodeplots()
    Plots.default(width=:auto)
    show_plot(fig, outname) = begin
        display(fig)
        savefig(fig, outname)
    end
end

@run_package_tests
