# HiPPO.jl

This package tries to provide some of the functionality described in the phenomenal HiPPO Paper by Gu et al.
This online-function approximation technique is the basis for their Mamba architecture.

I write this package in my free time and for my PhD Research - therefore the quality and frequency of updates will be sparse.
If you want to contribute feel free to make a PR.

## Current Status & Planned Features:
- [ ] implement all measures & variants of HiPPO
  - [x] LegS
    - fully implemented, including *novel* $\gamma$ (halftime) parameter
  - [x] LagT
    - fully implemented, including working $\beta$ (momentum) parameter
  - [x] LegT
    - fully implemented, including working $\theta$ (windowing) parameter 
  - [ ] Fourier
- [ ] Implement Kernelization (S4 & S4D) !help needed
- [x] Remove dependence on ControlSystems.jl for applications
  - we now support discretization via the `step` function directly
