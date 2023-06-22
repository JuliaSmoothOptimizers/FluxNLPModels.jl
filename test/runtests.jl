using Test
using FluxNLPModels
using CUDA, Flux, NLPModels
using Statistics
using Flux.Data: DataLoader #TODO update this
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Base: @kwdef
using MLDatasets
using LinearAlgebra

# Helper functions
function getdata(args)
  ENV["DATADEPS_ALWAYS_ACCEPT"] = "true" # download datasets without having to manually confirm the download

  # Loading Dataset	

  xtrain, ytrain = MLDatasets.MNIST(Tx = Float32, split = :train)[:]
  xtest, ytest = MLDatasets.MNIST(Tx = Float32, split = :test)[:]

  # Reshape Data in order to flatten each image into a linear array
  xtrain = Flux.flatten(xtrain)
  xtest = Flux.flatten(xtest)

  # One-hot-encode the labels
  ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

  # Create DataLoaders (mini-batch iterators) #TODO it is passed down
  train_loader = DataLoader((xtrain, ytrain), batchsize = args.batchsize, shuffle = true)
  test_loader = DataLoader((xtest, ytest), batchsize = args.batchsize)

  return train_loader, test_loader
end

function build_model(; imgsize = (28, 28, 1), nclasses = 10)
  return Flux.Chain(Dense(prod(imgsize), 32, relu), Dense(32, nclasses))
end

@kwdef mutable struct Args
  η::Float64 = 3e-4       # learning rate
  batchsize::Int = 2    # batch size
  epochs::Int = 10        # number of epochs
  use_cuda::Bool = true   # use gpu (if cuda available)
end

args = Args() # collect options in a struct for convenience

device = cpu #TODO should we test on GPU?

@testset "FluxNLPModels tests" begin

  # Create test and train dataloaders
  train_data, test_data = getdata(args)

  # Construct model
  DN = build_model() |> device
  DNNLPModel = FluxNLPModel(DN, train_data, test_data)

  old_w, rebuild = Flux.destructure(DN)

  x1 = copy(DNNLPModel.w)

  obj_x1 = obj(DNNLPModel, x1)
  grad_x1 = NLPModels.grad(DNNLPModel, x1)

  grad_x1_2 = similar(x1)
  obj_x1_2, grad_x1_2 = NLPModels.objgrad!(DNNLPModel, x1, grad_x1_2)

  @test DNNLPModel.w == old_w
  @test obj_x1 == obj_x1_2
  println(norm(grad_x1 - grad_x1_2))
  @test norm(grad_x1 - grad_x1_2) ≈ 0.0

  # @test grad_x1 ≈ grad_x1_2
  # @test all(grad_x1  .≈ grad_x1_2)
  @test x1 == DNNLPModel.w
  @test Flux.params(DNNLPModel.chain)[1][1] == x1[1]
  @test Flux.params(DNNLPModel.chain)[1][2] == x1[2]
end

@testset "minibatch tests" begin
  # Create test and train dataloaders
  train_data, test_data = getdata(args)

  # Construct model
  DN = build_model() |> device
  DNNLPModel = FluxNLPModel(DN, train_data, test_data)
  reset_minibatch_train!(nlp)
  @test nlp.current_training_minibatch_status === nothing
  buffer_minibatch = deepcopy(nlp.current_training_minibatch)
  @test minibatch_next_train!(nlp) # should return true 
  @test !isequal(nlp.current_training_minibatch, buffer_minibatch)
end
