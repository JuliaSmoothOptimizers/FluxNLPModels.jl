#TODO redo this section 
# FluxNLPModels.jl

## Compatibility
Julia ≥ 1.6.

## How to install
TODO: this section needs work since our package is not yet register
This module can be installed with the following command:
```julia
# pkg> add FluxNLPModels
# pkg> test FluxNLPModels
```

## Synopsis
FluxNLPModels exposes neural network models as optimization problems conforming to the NLPModels.jl API. FluxNLPModels is an interface between [Flux.jl](https://github.com/FluxML/Flux.jl)'s classification neural networks and [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl.git).

A `FluxNLPModel` gives the user access to:
- The values of the neural network variables/weights `w`;
- The value of the objective/loss function `L(X, Y; w)` at `w` for a given minibatch `(X,Y)`;
- The gradient `∇L(X, Y; w)` of the objective/loss function at `w` for a given minibatch `(X,Y)`.

In addition, it provides tools to:
- Switch the minibatch used to evaluate the neural network;
- Retrieve the current minibatch ;
- Measure the neural network's loss at the current `w`.

## How to use
Check the tutorials
<!-- Check the [tutorial](https://juliasmoothoptimizers.github.io/FluxNLPModels.jl/stable/). -->

# Bug reports and discussions

If you think you found a bug, feel free to open an [issue]<!--(https://github.com/JuliaSmoothOptimizers/FluxNLPModels.jl/issues). --> TODO: add repo link
Focused suggestions and requests can also be opened as issues. Before opening a pull request, please start an issue or a discussion on the topic.

If you have a question that is not suited for a bug report, feel free to start a discussion [here](#TODO). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers). Questions about any of our packages are welcome.
