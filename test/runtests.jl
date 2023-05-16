using Test
using CUDA, Flux, NLPModels
using FluxNLPModels
using Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Base: @kwdef
using MLDatasets


# Helper functions
function getdata(args, device)
  ENV["DATADEPS_ALWAYS_ACCEPT"] = "true" # download datasets without having to manually confirm the download

  # Loading Dataset	
  xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
  xtest, ytest = MLDatasets.MNIST.testdata(Float32)

  # Reshape Data in order to flatten each image into a linear array
  xtrain = Flux.flatten(xtrain)
  xtest = Flux.flatten(xtest)

  # One-hot-encode the labels
  ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

  # Create DataLoaders (mini-batch iterators)
  train_loader = DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
  test_loader = DataLoader((xtest, ytest), batchsize=args.batchsize)

  return train_loader, test_loader
end

function build_model(; imgsize=(28,28,1), nclasses=10)
  return Chain( Dense(prod(imgsize), 32, relu),
                Dense(32, nclasses))
end

@kwdef mutable struct Args
  η::Float64 = 3e-4       # learning rate
  batchsize::Int = 2    # batch size
  epochs::Int = 10        # number of epochs
  use_cuda::Bool = true   # use gpu (if cuda available)
end

args = Args() # collect options in a struct for convenience

if CUDA.functional() && args.use_cuda
    @info "Training on CUDA GPU"
    CUDA.allowscalar(false)
    device = gpu
else
    @info "Training on CPU"
    device = cpu
end


@testset "FluxNLPModels tests" begin
 
  # Create test and train dataloaders
  train_loader, test_loader = getdata(args, device)

  # Construct model
  DN = build_model() |> device
  DNNLPModel = FluxNLPModel(DN, data_train = train_loader, data_test = test_loader)
  
  old_w, rebuild = Flux.destructure(DN) 
  @test DNNLPModel.w == old_w



  x1 = copy(DNNLPModel.w)
  x2 = (x -> x + 50).(Array(DNNLPModel.w))

  obj_x1 = obj(DNNLPModel, x1)
  grad_x1 = NLPModels.grad(DNNLPModel, x1)

  grad_x1_2 = similar(grad_x1)
  obj_x1_2 ,  grad_x1_2 = NLPModels.objgrad!(DNNLPModel, x1,grad_x1_2)

  @test obj_x1 == obj_x1_2
  @test grad_x1 == obj_x1_2

  @test x1 == DNNLPModel.w
  @test params(DNNLPModel.chain)[1].value[1] == x1[1]
  @test params(DNNLPModel.chain)[1].value[2] == x1[2]

  obj_x2 = obj(DNNLPModel, x2)
  grad_x2 = NLPModels.grad(DNNLPModel, x2)
  @test x2 == DNNLPModel.w
  @test params(DNNLPModel.chain)[1].value[1] == x2[1]
  @test params(DNNLPModel.chain)[1].value[2] == x2[2]

  @test obj_x1 != obj_x2
  @test grad_x1 != grad_x2




end

