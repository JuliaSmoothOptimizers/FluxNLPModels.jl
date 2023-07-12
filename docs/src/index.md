# FluxNLPModels.jl

## Compatibility
Julia ≥ 1.6.

## How to install

This module can be installed with the following command:
```julia
pkg> add FluxNLPModels
```

## Synopsis

FluxNLPModels exposes neural network models as optimization problems conforming to the [NLPModels API](https://github.com/JuliaSmoothOptimizers/NLPModels.jl). FluxNLPModels is an interface between [Flux.jl](https://github.com/FluxML/Flux.jl)'s classification neural networks and [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).

A `FluxNLPModel` gives the user access to:
- The values of the neural network variables/weights `w`;
- The value of the objective/loss function `L(X, Y; w)` at `w` for a given minibatch `(X,Y)`;
- The gradient `∇L(X, Y; w)` of the objective/loss function at `w` for a given minibatch `(X,Y)`.

In addition, it provides tools to:
- Switch the minibatch used to evaluate the neural network;
- Retrieve the current minibatch ;
- Measure the neural network's loss at the current `w`.

# Bug reports and discussions

If you encounter any bugs or have suggestions for improvement, please open an [issue](https://github.com/JuliaSmoothOptimizers/FluxNLPModels.jl/issues). For general questions or discussions related to this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) organization, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions).

