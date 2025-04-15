using TestItemRunner

@testsnippet Plotting begin
    using Plots
    unicodeplots()
end

@run_package_tests
