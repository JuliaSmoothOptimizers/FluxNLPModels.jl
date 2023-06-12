using FluxNLPModels
using CUDA, Flux, NLPModels

#     Ref:
#     https://github.com/FluxML/model-zoo/blob/master/vision/mlp_mnist/mlp_mnist.jl
# For tensorboard you need to install pip3 install tensorflow (python package)
# use CMD line and go to dir, then you can use tensorBoard --logir <name of the dir>

using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Base: @kwdef
using MLDatasets
# using TensorBoardLogger #the usage of TensorBoardLogger, 
using Logging: with_logger
using TensorBoardLogger: TBLogger, tb_overwrite
using JSOSolvers
using Dates

# Genral functions 

function getdata(args)
  ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

  # Loading Dataset	
  xtrain, ytrain = MLDatasets.MNIST(Tx = Float32, split = :train)[:]
  xtest, ytest = MLDatasets.MNIST(Tx = Float32, split = :test)[:]

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
  return Chain(Dense(prod(imgsize), 32, relu), Dense(32, nclasses))
end

# Note that we use the functions [Dense](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dense) so that our model is *densely* (or fully) connected and [Chain](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Chain) to chain the computation of the three layers.

# ## Loss function
const loss = logitcrossentropy

# Now, we define the loss function `loss_and_accuracy`. It expects the following arguments:
# * ADataLoader object.
# * The `build_model` function we defined above.
# * A device object (in case we have a GPU available).

function loss_and_accuracy(data_loader, model, device)
  acc = 0
  ls = 0.0f0
  num = 0
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
  η::Float32 = 3e-3       # learning rate
  batchsize::Int = 128    # batch size
  epochs::Int = 10        # number of epochs
  use_cuda::Bool = true   # use gpu (if cuda available)
  verbose_freq = 10                               # logging for every verbose_freq iterations
  tblogger = true                                 # log training with tensorboard
  save_path = "runs/output"                            # results path
end

#Main
args = Args() # collect options in a struct for convenience

#TODO create subchannle for each project
# logging by TensorBoard.jl
# if args.tblogger
#   tblogger = TBLogger(args.save_path, tb_overwrite) #TODO changing tblogger for each project 
# end

# if CUDA.functional() && args.use_cuda
#   @info "Training on CUDA GPU"
#   CUDA.allowscalar(false)
#   device = gpu
# else
#   @info "Training on CPU"
#   device = cpu
# end

# Callback to log information after every epoch
#TO see the callback run " tensorboard --logdir output " after the code is finished running, the output is the savepath
function TBCallback(train_loader, test_loader, model, epoch, device)

  ## Report on train and test
  train_loss, train_acc = loss_and_accuracy(train_loader, model, device)
  test_loss, test_acc = loss_and_accuracy(test_loader, model, device)
  println("Epoch=$epoch")
  println("  train_loss = $train_loss, train_accuracy = $train_acc")
  println("  test_loss = $test_loss, test_accuracy = $test_acc")
  if args.tblogger #&& train_steps % args.verbose_freq == 0
    with_logger(tblogger) do
      #   @info "model" params=param_dict log_step_increment=0
      @info "epoch" epoch = epoch
      @info "train" loss = train_loss acc = train_acc
      @info "test" loss = test_loss acc = test_acc
    end
  end
end

#########################################################################
### Normal Flux

function train_flux(; kws...)
  args = Args(; kws...) ## Collect options in a struct for convenience

  if CUDA.functional() && args.use_cuda
    @info "Training on CUDA GPU"
    CUDA.allowscalar(false)
    device = gpu
  else
    @info "Training on CPU"
    device = cpu
  end

  !ispath(args.save_path) && mkpath(args.save_path)

  ## Create test and train dataloaders
  train_loader, test_loader = getdata(args)

  @info "Constructing model and starting training"
  ## Construct model
  model = build_model() |> device

  ## Optimizer
  opt = Flux.setup(Flux.Descent(args.η), model)

  ## Training
  for epoch = 1:(args.epochs)
    for (x, y) in train_loader
      x, y = device(x), device(y) ## transfer data to device
      gs = gradient(m -> loss(m(x), y), model) ## compute gradient of the loss
      Flux.Optimise.update!(opt, model, gs[1]) ## update parameters
    end
    # logging
    TBCallback(train_loader, test_loader, model, epoch, device)
  end
end

#########################################################################
### SGD FluxNLPModels

function train_FluxNLPModel_SGD(; kws...)
  args = Args(; kws...) ## Collect options in a struct for convenience

  if CUDA.functional() && args.use_cuda
    @info "Training on CUDA GPU"
    CUDA.allowscalar(false)
    device = gpu
  else
    @info "Training on CPU"
    device = cpu
  end

  !ispath(args.save_path) && mkpath(args.save_path)

  ## Create test and train dataloaders
  train_loader, test_loader = getdata(args)

  @info "Constructing model and starting training"
  ## Construct model
  model = build_model() |> device

  # now we set the model to FluxNLPModel
  nlp = FluxNLPModel(model, train_loader, test_loader; loss_f = loss)
  g = similar(nlp.w) #TODO should they be here?
  w_k = copy(nlp.w)
  for epoch = 1:(args.epochs)
    for (x, y) in train_loader
      x, y = device(x), device(y) ## transfer data to device
      nlp.current_training_minibatch = (x, y)
      #   fk, g = NLPModels.objgrad!(nlp, w_k, g)
      #  fk = NLPModels.obj(nlp,w_k)
      g = NLPModels.grad(nlp, w_k)
      w_k -= args.η .* g      #   update the parameter
      FluxNLPModels.set_vars!(nlp, w_k) #TODO Not sure about this
    end
    # logging
    TBCallback(train_loader, test_loader, nlp.chain, epoch, device) #not sure to pass nlp.chain or fx
  end
end

#########################################################################
## R2 FluxNLPModels

# used in the callback of R2 for training deep learning model 
mutable struct StochasticR2Data
  epoch::Int
  i::Int
  # other fields as needed...
  max_epoch::Int
  ϵ::Float64 #TODO Fix with type T
  state
end

function cb(
  nlp,
  solver,
  stats,
  train_loader,
  test_loader,
  device,
  #   param::AbstractParameterSet,
  data::StochasticR2Data,
)

  # logging
  TBCallback(train_loader, test_loader, nlp.chain, data.epoch, device) #not sure to pass nlp.chain or fx

  # Max epoch
  if data.epoch == data.max_epoch
    stats.status = :user
    return
  end
  iter = train_loader
  if data.i == 0
    next = first(iter)
  else
    next = iterate(iter, data.state)
  end
  data.i = 1 #flag to see if we are at first

  if next === nothing #one epoch is finished
    data.i = 0
    data.epoch += 1
    return
  end
  (item, data.state) = next
  nlp.current_training_minibatch = device(item) # move to cpu or gpu
end

function train_FluxNlPModel_R2(;
  verbose = -1,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  η1 = eps(T)^(1 / 4),
  η2 = T(0.95),
  γ1 = T(1 / 2),
  γ2 = 1 / γ1,
  σmin = zero(T),# change this
  β::T = T(0),
  max_time = Inf,
  kws...,
)
  args = Args(; kws...) ## Collect options in a struct for convenience

  if CUDA.functional() && args.use_cuda
    @info "Training on CUDA GPU"
    CUDA.allowscalar(false)
    device = gpu
  else
    @info "Training on CPU"
    device = cpu
  end

  !ispath(args.save_path) && mkpath(args.save_path)

  ## Create test and train dataloaders
  train_loader, test_loader = getdata(args)

  @info "Constructing model and starting training"
  ## Construct model
  model = build_model() |> device

  # now we set the model to FluxNLPModel
  nlp = FluxNLPModel(model, train_loader, test_loader; loss_f = loss)

  #set the fist data sets
  next = first(train_loader)
  (item, state) = next
  nlp.current_training_minibatch = device(item) # move to cpu or gpu

  stochastic_data = StochasticR2Data(0, 1, args.epochs, atol, state) # data.i =1

  solver_stats = JSOSolvers.R2(
    nlp;
    atol = atol,
    rtol = rtol,
    η1 = η1,
    η2 = η2,
    γ1 = γ1,
    γ2 = γ2,
    σmin = σmin,
    β = β,
    max_time = max_time,
    verbose = verbose,
    callback = (nlp, solver, stats, nlp_param) ->
      cb(nlp, solver, stats, train_loader, test_loader, device, stochastic_data),
  )
  return stochastic_data
end

########################################################################
### Main File to Callback
#if you want the train_flux
# if args.tblogger
#   tblogger =
#     TBLogger(args.save_path * "train_flux-" * Dates.format(now(), "yyyy-mm-dd-H-M-S"), tb_overwrite) #TODO changing tblogger for each project 
# end

# train_flux()
# if args.tblogger
#   close(tblogger)
# end

# if args.tblogger #TODO add timer to this 
#   tblogger = TBLogger(
#     args.save_path * "train_FluxNLPModel_SGD-" * Dates.format(now(), "yyyy-mm-dd-H-M-S"),
#     tb_overwrite,
#   ) #TODO changing tblogger for each project 
# end

# train_FluxNLPModel_SGD() #TODO this is slow
# if args.tblogger
#   close(tblogger)
# end

if args.tblogger
  tblogger = TBLogger(
    args.save_path * "train_FluxNlPModel_R2-" * Dates.format(now(), "yyyy-mm-dd-H-M-S"),
    tb_overwrite,
  ) #TODO changing tblogger for each project 
end

train_FluxNlPModel_R2()

# closing the logger otherwise it will error out
if args.tblogger
  close(tblogger)
end
