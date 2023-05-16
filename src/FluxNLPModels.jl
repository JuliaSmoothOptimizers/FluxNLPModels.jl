module FluxNLPModels

using Flux, NLPModels
#TODO use Flux:Data vs MLUtils
# export Chain
export AbstractFluxNLPModel, FluxNLPModel
export flat_grad!
export reset_minibatch_train!, reset_minibatch_test!
export create_minibatch, set_vars!

# abstract type Chain end

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
mutable struct FluxNLPModel{T, S, C <: Flux.Chain, V, F <: Function} <: AbstractFluxNLPModel{T, S}
  meta::NLPModelMeta{T, S}
  chain::C
  counters::Counters
  data_train
  data_test
  loss_f::F
  size_minibatch::Int
  training_minibatch_iterator
  test_minibatch_iterator
  current_training_minibatch
  current_test_minibatch
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
  size_minibatch::Int = 100,
  loss_f::F = Flux.crossentropy,
) where {T <: Flux.Chain, F <: Function}
  x0, re = Flux.destructure(chain_ANN)
  n = length(x0)
  meta = NLPModelMeta(n, x0 = x0)
  if (isempty(data_train) || isempty(data_test))
    error("train data or test is empty")
  end

  #TODO we should decide if we want the minibatch to be created here to pass down, I prefere pass down
  # xtrn = data_train[1]
  # ytrn = data_train[2]

  # xtst = data_test[1]
  # ytst = data_test[2]

  # training_minibatch_iterator = create_minibatch(xtrn, ytrn, size_minibatch)
  # test_minibatch_iterator = create_minibatch(xtst, ytst, size_minibatch)

  #pass by user 
  training_minibatch_iterator = data_train
  test_minibatch_iterator = data_test
  current_training_minibatch = first(training_minibatch_iterator) #TODO add document that user has to pass in the current minibatch
  current_test_minibatch = first(test_minibatch_iterator)

  return FluxNLPModel(
    meta,
    chain_ANN,
    Counters(),
    data_train,
    data_test,
    size_minibatch,
    loss_f,
    training_minibatch_iterator,
    test_minibatch_iterator,
    current_training_minibatch,
    current_test_minibatch,
    x0,
  )
end

include("utils.jl")
include("FluxNLPModels_methods.jl")
end
