using Test
using FluxNLPModels
using CUDA, Flux, NLPModels
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using Base: @kwdef
using MLDatasets
using LinearAlgebra

# Helper functions
function getdata(args; T = Float32)
  ENV["DATADEPS_ALWAYS_ACCEPT"] = "true" # download datasets without having to manually confirm the download

  # Loading Dataset	

  xtrain, ytrain = MLDatasets.MNIST(Tx = T, split = :train)[:]
  xtest, ytest = MLDatasets.MNIST(Tx = T, split = :test)[:]

  # Reshape Data in order to flatten each image into a linear array
  xtrain = Flux.flatten(xtrain)
  xtest = Flux.flatten(xtest)

  # One-hot-encode the labels
  ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

  # Create DataLoaders (mini-batch iterators) 
  train_loader = DataLoader((xtrain, ytrain), batchsize = args.batchsize, shuffle = true)
  test_loader = DataLoader((xtest, ytest), batchsize = args.batchsize)

  return train_loader, test_loader
end

function build_model(; imgsize = (28, 28, 1), nclasses = 10)
  return Flux.Chain(Dense(prod(imgsize), 32, relu), Dense(32, nclasses), softmax)
end

@kwdef mutable struct Args
  η::Float64 = 3e-4       # learning rate
  batchsize::Int = 2    # batch size
  epochs::Int = 10        # number of epochs
  use_cuda::Bool = true   # use gpu (if cuda available)
end

args = Args() # collect options in a struct for convenience

device = cpu

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
  # println(norm(grad_x1 - grad_x1_2))
  @test norm(grad_x1 - grad_x1_2) ≈ 0.0

  @test x1 == DNNLPModel.w
  @test Flux.params(DNNLPModel.chain)[1][1] == x1[1]
  @test Flux.params(DNNLPModel.chain)[1][2] == x1[2]

  @test_throws Exception FluxNLPModel(DN, [], test_data) # if the train data is empty
  @test_throws Exception FluxNLPModel(DN, train_data, []) # if the test data is empty
  @test_throws Exception FluxNLPModel(DN, [], []) # if the both data is empty

  # Testing if the value of the first batch was passed it
  DNNLPModel_2 = FluxNLPModel(
    DN,
    train_data,
    test_data,
    current_training_minibatch = first(train_data),
    current_test_minibatch = first(test_data),
  )

  #checking if we can call accuracy
  train_acc = FluxNLPModels.accuracy(DNNLPModel_2; data_loader = train_data) # accuracy on train data
  test_acc = FluxNLPModels.accuracy(DNNLPModel_2) # on the test data

  @test train_acc >= 0.0
  @test train_acc <= 1.0
end

@testset "minibatch tests" begin
  # Create test and train dataloaders
  train_data, test_data = getdata(args)

  # Construct model
  DN = build_model() |> device
  nlp = FluxNLPModel(DN, train_data, test_data)
  reset_minibatch_train!(nlp)
  @test nlp.current_training_minibatch_status === nothing
  buffer_minibatch = deepcopy(nlp.current_training_minibatch)
  @test minibatch_next_train!(nlp) # should return true 
  @test minibatch_next_train!(nlp) # should return true 
  @test !isequal(nlp.current_training_minibatch, buffer_minibatch)

  reset_minibatch_test!(nlp)
  @test minibatch_next_test!(nlp) # should return true 
  @test minibatch_next_test!(nlp) # should return true 
end

@testset "Multiple precision test" begin
  # Create test and train dataloaders
  train_data, test_data = getdata(args)

  # Construct model in Float32
  DN = build_model() |> device
  nlp = FluxNLPModel(DN, train_data, test_data)

  x1 = copy(nlp.w)
  obj_x1 = obj(nlp, x1)
  grad_x1 = NLPModels.grad(nlp, x1)
  @test typeof(obj_x1) == Float32
  @test eltype(grad_x1) == Float32

  # change to Float16 
  x2 = Float16.(x1)
  obj_x2 = obj(nlp, x2)
  grad_x2 = NLPModels.grad(nlp, x2)
  # T test grad again after changing the type, using grad! method
  NLPModels.grad!(nlp, x2, grad_x2)
  @test typeof(obj_x2) == Float16
  @test eltype(grad_x2) == Float16

  # change to Float64
  x3 = Float64.(x1)
  obj_x3 = obj(nlp, x3)
  grad_x3 = NLPModels.grad(nlp, x3)
  @test typeof(obj_x3) == Float64
  @test eltype(grad_x3) == Float64

  # Construct model in Float16
  train_data_f16, test_data_f16 = getdata(args, T = Float16)
  DN_f16 = build_model() |> f16
  nlp_f16 = FluxNLPModel(DN_f16, train_data_f16, test_data_f16)

  x4 = copy(nlp_f16.w)
  obj_x4 = obj(nlp_f16, x4)
  grad_x4 = NLPModels.grad(nlp_f16, x4)

  @test typeof(obj_x4) == Float16
  @test eltype(grad_x4) == Float16

  # change to Float32 from Float16 
  x5 = Float32.(x4)
  obj_x5 = obj(nlp_f16, x5)
  grad_x5 = NLPModels.grad(nlp_f16, x5)
  @test typeof(obj_x5) == Float32
  @test eltype(grad_x5) == Float32

  # change to Float64 from Float16 
  x6 = Float64.(x4)
  obj_x6 = obj(nlp_f16, x6)
  grad_x6 = NLPModels.grad(nlp_f16, x6)
  @test typeof(obj_x6) == Float64
  @test eltype(grad_x6) == Float64

  # change to Float32 from Float128
  # expected to throw an error
  # Note we do not support BigFloat in FluxNLPModels yet!
  x7 = BigFloat.(x5)
  @test_throws Exception obj(nlp_f16, x7)

end