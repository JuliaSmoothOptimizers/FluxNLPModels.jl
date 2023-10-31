module FluxNLPModels

using Flux, NLPModels
using Flux: onehotbatch, onecold, @epochs
export AbstractFluxNLPModel, FluxNLPModel
export reset_minibatch_train!, reset_minibatch_test!
export minibatch_next_train!, minibatch_next_test!
export accuracy, set_vars!, local_loss, update_type!

abstract type AbstractFluxNLPModel{T, S} <: AbstractNLPModel{T, S} end

""" 
    FluxNLPModel{T, S, C <: Flux.Chain} <: AbstractNLPModel{T, S}

Data structure that makes the interfaces between neural networks defined with [Flux.jl](https://fluxml.ai/) and [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).
A FluxNLPModel has fields

# Arguments
-  `meta` and `counters` retain informations about the `FluxNLPModel`;
- `chain` is the chained structure representing the neural network;
- `data_train` is the complete data training set;
- `data_test` is the complete data test;
- `size_minibatch` parametrizes the size of an training and test minibatches
- `training_minibatch_iterator` is an iterator over an training minibatches;
- `test_minibatch_iterator` is an iterator over the test minibatches;
- `current_training_minibatch` is the training minibatch used to evaluate the neural network;
- `current_minibatch_test` is the current test minibatch, it is not used in practice;
- `w` is the vector of weights/variables;
"""
mutable struct FluxNLPModel{T, S, C <: Chain, F <: Function} <: AbstractFluxNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  chain::C
  counters::Counters
  loss_f::F
  size_minibatch::Int
  training_minibatch_iterator
  test_minibatch_iterator
  current_training_minibatch
  current_test_minibatch
  rebuild # this is used to create the rebuild of flat function 
  current_training_minibatch_status
  current_test_minibatch_status
  w::S
end

"""
    FluxNLPModel(chain_ANN data_train=MLDatasets.MNIST.traindata(Float32), data_test=MLDatasets.MNIST.testdata(Float32); size_minibatch=100)

Build a `FluxNLPModel` from the neural network represented by `chain_ANN`.
`chain_ANN` is built using [Flux.jl](https://fluxml.ai/) for more details.
The other data required are: an iterator over the training dataset `data_train`, an iterator over the test dataset `data_test` and the size of the minibatch `size_minibatch`.
Suppose `(xtrn,ytrn) = Fluxnlp.data_train`
"""
function FluxNLPModel(
  chain_ANN::T,
  data_train,
  data_test;
  current_training_minibatch = [],
  current_test_minibatch = [],
  size_minibatch::Int = 100,
  loss_f::F = Flux.crossentropy,#Flux.mse, #
) where {T <: Chain, F <: Function}
  x0, rebuild = Flux.destructure(chain_ANN)
  n = length(x0)
  meta = NLPModelMeta(n, x0 = x0)
  if (isempty(data_train) || isempty(data_test))
    error("train data or test is empty")
  end
  if (isempty(current_training_minibatch) || isempty(current_test_minibatch))
    current_training_minibatch = first(data_train)
    current_test_minibatch = first(data_test)
  end

  return FluxNLPModel(
    meta,
    chain_ANN,
    Counters(),
    loss_f,
    size_minibatch,
    data_train,
    data_test,
    current_training_minibatch,
    current_test_minibatch,
    rebuild,
    nothing,
    nothing,
    x0,
  )
end

include("utils.jl")
include("FluxNLPModels_methods.jl")
end