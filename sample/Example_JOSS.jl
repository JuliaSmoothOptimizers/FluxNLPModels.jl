using FluxNLPModels
using CUDA, Flux, NLPModels
using Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Base: @kwdef
using MLDatasets
using JSOSolvers

# Genral functions 

function getdata(args; T = Float32) #T for types
  ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

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
  @info "The type is " typeof(xtrain)
  return train_loader, test_loader
end

const loss = logitcrossentropy

function loss_and_accuracy(data_loader, model, device; T=Float32)
  acc = T(0)
  ls = T(0.0f0)
  num = T(0)
  for (x, y) in data_loader
    x, y = device(x), device(y)
    ŷ = model(x)
    ls += loss(ŷ, y, agg = sum)
    acc += sum(onecold(ŷ) .== onecold(y)) ## Decode the output of the model
    num += size(x)[end]
  end
  return ls / num, acc / num
end

@kwdef mutable struct Args
  η::Float32 = 3e-3       # learning rate #TODO fixe this
  batchsize::Int = 512    # batch size
  epochs::Int = 10        # number of epochs
  use_cuda::Bool = true   # use gpu (if cuda available)
  verbose_freq = 10                               # logging for every verbose_freq iterations
end


# used in the callback of R2 for training deep learning model 
mutable struct StochasticR2Data
  epoch::Int
  i::Int
  max_epoch::Int
  ϵ::Float64 #TODO Fix with type T
  state
end

function cb(nlp, stats, train_loader, device, data::StochasticR2Data;)

  # Max epoch
  if data.epoch == data.max_epoch
    stats.status = :user
    return
  end
  iter = train_loader
  if data.i == 0
    next = iterate(iter)
  else
    next = iterate(iter, data.state)
  end
  data.i += 1 #flag to see if we are at first

  if next === nothing #one epoch is finished
    @info "Epoch", data.epoch
    data.i = 0
    data.epoch += 1
    return
  end
  (item, data.state) = next
  nlp.current_training_minibatch = device(item) # move to cpu or gpu
end

args = Args(; kws...) ## Collect options in a struct for convenience
device = cpu

## Create test and train dataloaders
train_loader, test_loader = getdata(args, T = myT)

@info "Constructing model and starting training"
## Construct model
model =
  Chain(
    Conv((5, 5), 1 => 6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(256 => 120, relu),
    Dense(120 => 84, relu),
    Dense(84 => 10),
  ) |> device

# now we set the model to FluxNLPModel
nlp = FluxNLPModel(model, train_loader, test_loader; loss_f = loss)

# #set the fist data sets
item = first(train_loader)
nlp.current_training_minibatch = device(item) # move to cpu or gpu
stochastic_data = StochasticR2Data(0, 0, args.epochs, atol, nothing) # data.i =0

solver_stats = JSOSolvers.R2(
  nlp;
  callback = (nlp, solver, stats) -> cb(nlp, stats, train_loader, device, stochastic_data),
)
# return stochastic_data
  ## Report on train and test
  train_loss, train_acc = loss_and_accuracy(train_loader, nlp.chain, device)
  test_loss, test_acc = loss_and_accuracy(test_loader, nlp.chain, device)