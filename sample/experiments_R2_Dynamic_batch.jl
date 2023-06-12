using FluxNLPModels
using CUDA, NLPModels
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
using StochasticRounding



# Genral functions 

function getdata(args;T=Float32) #T for types
  ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

  # Loading Dataset	
  xtrain, ytrain = MLDatasets.MNIST(Tx = T, split = :train)[:]
  xtest, ytest = MLDatasets.MNIST(Tx = T, split = :test)[:]

  # Reshape Data in order to flatten each image into a linear array
  xtrain = Flux.flatten(xtrain)
  xtest = Flux.flatten(xtest)

  # One-hot-encode the labels
  ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)
#   return xtrain, ytrain , xtest, ytest
# end 
# function creatbatch(xtrain, ytrain , xtest, ytest,args)
  # Create DataLoaders (mini-batch iterators)
  train_loader = DataLoader((xtrain, ytrain), batchsize = args.batchsize, shuffle = true)
  test_loader = DataLoader((xtest, ytest), batchsize = args.batchsize)
  @info "The type is " typeof(xtrain)
  return train_loader, test_loader
end

# function build_model(; imgsize = (28, 28, 1), nclasses = 10,T=Float32) #TODO the rand fails for Float32sr make a new matrix then replace it 
#     return Chain( #TODO important: when using Dense(matrix of random, dimention is transposed)
#         Dense(T.(rand(Float32,32,prod(imgsize))),true, relu), # I use this way to avoid rand error
#         Dense(T.(rand(Float32, nclasses,32)),true) # The following is not correct : Dense(rand(T,32, nclasses),true) 
#         )
# end

f64(m) = Flux.paramtype(Float64, m) # similar to https://github.com/FluxML/Flux.jl/blob/d21460060e055dca1837c488005f6b1a8e87fa1b/src/functor.jl#L217

function build_model(; imgsize = (28, 28, 1), nclasses = 10,T=Float32)
  m =  Chain(Dense(prod(imgsize), 32, relu), Dense(32, nclasses))
  if T == Float64
    m = f64(m)
  end
  return m
end

# m16 = f16(m64)


# Note that we use the functions [Dense](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dense) so that our model is *densely* (or fully) connected and [Chain](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Chain) to chain the computation of the three layers.

# ## Loss function
const loss = logitcrossentropy
const loss2 = Flux.mse
# Now, we define the loss function `loss_and_accuracy`. It expects the following arguments:
# * ADataLoader object.
# * The `build_model` function we defined above.
# * A device object (in case we have a GPU available).

function loss_and_accuracy(data_loader, model, device,T)
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
  batchsize::Int = 1000    # batch size
  epochs::Int = 10        # number of epochs
  use_cuda::Bool = true   # use gpu (if cuda available)
  verbose_freq = 10                               # logging for every verbose_freq iterations
  tblogger = true                                 # log training with tensorboard
  save_path = "runs/output"                            # results path
end


#function to get dictionary of model parameters
function fill_param_dict!(dict, m, prefix)
    if m isa Chain
        for (i, layer) in enumerate(m.layers)
            fill_param_dict!(dict, layer, prefix*"layer_"*string(i)*"/"*string(layer)*"/")
        end
    else
        for fieldname in fieldnames(typeof(m))
            val = getfield(m, fieldname)
            if val isa AbstractArray
                val = vec(val)
            end
            dict[prefix*string(fieldname)] = val
        end
    end
end

function TBCallback(train_loader, test_loader, model, epoch, device;T=Float32)

  ## Report on train and test
  train_loss, train_acc = loss_and_accuracy(train_loader, model, device,T)
  test_loss, test_acc = loss_and_accuracy(test_loader, model, device,T)
  # println("Epoch=$epoch")
  # println("  train_loss = $train_loss, train_accuracy = $train_acc")
  # println("  test_loss = $test_loss, test_accuracy = $test_acc")

  param_dict = Dict{String, Any}()
  fill_param_dict!(param_dict, model, "")

  if args.tblogger #&& train_steps % args.verbose_freq == 0
    with_logger(tblogger) do
      @info "model" params=param_dict log_step_increment=0
      @info "epoch" epoch = epoch
      @info "train" loss = train_loss acc = train_acc
      @info "test" loss = test_loss acc = test_acc
    end
  end
end
args = Args() # collect options in a struct for convenience


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
  data::StochasticR2Data;
  myT=Float32
)

  # logging
  TBCallback(train_loader, test_loader, nlp.chain, data.epoch, device;T=myT) #not sure to pass nlp.chain or fx
  # Max epoch
  if data.epoch == data.max_epoch
    stats.status = :user
    return
  end

  b=2
  j=0
  item_minibatch = []
  iter = train_loader

  while j <= b 
    j=j+1
    if data.i == 0
      next = iterate(iter)
    # elseif data.i %  3 !=0
    #   return 
    else
      next = iterate(iter, data.state)
    end
    data.i += 1 #flag to see if we are at first

    if next === nothing #one epoch is finished
      @info "Epoch", data.epoch
      data.i = 0
      data.epoch += 1
      break
    end

    (item, data.state) = next
    item_minibatch.append(item)
  # @info "The data "
  end
  if(item_minibatch !==nothing)
    nlp.current_training_minibatch = device(item_minibatch) # move to cpu or gpu
  end
end


#TODO  IDEA for batch size updating

# Pseudo code:
# data = load_full_data()
# batch_size = 10
# while true
#     minibatch = data[rand(1:end, batch_size)]
#     loss, grad = withgradient(evaluate_model, minibatch)
#     if should_stop(loss) 
#         break
#     end
#     optimize_step!(model, gradient, learning_rate)
#     batch_size= min(1000, batchsize +2)

# end





# Frames White
# :dancer::skin-tone-2:  3 days ago
#  how would make sure don't revisit same data points?)
# You don't really have to. Normally ML training trans in each point dozens to thousands of times. So just randomly sampling each training step by the law if large numbers says it's going to average out the same as shuffling every time plus doing every item
# .but if you want you can always just track the last index of the item you took.
# Pseudo code:
# data = load_full_data()
# batch_size = 10
# batch_start = 1
# while true
#     batch_end = batch_start + batch_size
#     minibatch = data[batch_start:min(batch_end, end)]
    
#     loss, grad = withgradient(evaluate_model, minibatch)
#     if should_stop(loss) 
#         break
#     end
#     optimize_step!(model, gradient, learning_rate)
#     batch_size= min(1000, batchsize +2)
#     batch_start = batch_end > length(data) ? batch_end : 1
# end
# :heart:
# 1



# Lucas Pereira
#   3 days ago
# these are valid suggestions. but what I recommended would look something like dataloaders = [Dataloader(1:numberOfSamples, batchsize = bs) for bs in [64, 128, 256 ..., 1024]]. And then, in each epoch, you use the batches of integers to access the data. You guarantee to visit every sample exactly once every epoch. And, after using the same dataloader for a few epochs, you move on to the next one in the vector dataloaders. In general:
# batchSizes = [2 ^ f for f in 6:10] # 64, 128, ..., 1024
# dataLoaders = [Dataloader(1:numberOfSamples |> collect, batchsize = bs) for bs in batchSizes]
# loaderCounter = 1
# for epoch in 1:numberOfEpochs
#   # move on to next dataloader every 10 epochs
#   epoch % 10 == 0 && loaderCounter += 1
#   for intBatch in dataLoaders[loaderCounter]
#     dataBatch = data[intBatch]
#     # use batch. update model...
#   end
# end




# b=4
# # item_minibatch_x = similar(item[1])
# # item_minibatch_y = similar(item[2])
# j=0
# while j <= b 
#   j=j+1
#   next = iterate(iter, state)
#   if next === nothing #one epoch is finished
#     break
#   end
#   (item, state) = next
#   if(j ==0)
#     item_minibatch_x =item[1]
#     item_minibatch_y = item[2]

#   else
#     item_minibatch_x= hcat(item_minibatch_x,item[1])
#     item_minibatch_y= hcat(item_minibatch_y, item[2]) #TODO issue it turns onehotarray to bool

#   end
# end

# item_minibatch = (item_minibatch_x,item_minibatch_y)
# println(typeof(item_minibatch))
# println(typeof(item))



function train_FluxNlPModel_R2(;
  myT=Float32, #used to define the type
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

  #R2 parameter
  # verbose = -1,
  atol = myT(0.0001) #eps(myT)
  rtol = myT(0.001) #eps(myT)
  η1 = myT(0.01)#eps(myT)^(1 / 4)
  η2 = myT(0.99)
  γ1 = myT(1 / 2)
  γ2 = 1 / γ1
  σmin = rand(myT)#zero(myT)# change this
  # β = myT(0.9)
  β = myT(0)
  max_time = Inf

  !ispath(args.save_path) && mkpath(args.save_path)

  ## Create test and train dataloaders
  train_loader, test_loader = getdata(args,T=myT)

  @info "Constructing model and starting training"
  ## Construct model
  model = build_model(T= myT) |> device
  @info "The type of model  is " typeof(model)

  # now we set the model to FluxNLPModel
  nlp = FluxNLPModel(model, train_loader, test_loader; loss_f = loss) #TODO add the device here for the data mini-batch
  @info "The type of nlp.w  is " typeof(nlp.w)

  # #set the fist data sets
  item  = first(train_loader)
  nlp.current_training_minibatch = device(item) # move to cpu or gpu
  stochastic_data = StochasticR2Data(0, 0, args.epochs, atol, nothing) # data.i =0
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
    verbose = 1,
    max_iter= 3000,
    callback = (nlp, solver, stats) -> cb(nlp, solver, stats, train_loader, test_loader, device, stochastic_data; myT=myT),
  )
  # return stochastic_data
end





# ----------------------------------#
#       R2
# ----------------------------------#

# for myT in [Float16,Float32,Float64,Float32sr] #SR fails ERROR: ArgumentError: Sampler for this object is not defined
for myT in [Float64]
    if args.tblogger #TODO add timer to this 
      global   tblogger = TBLogger(
            args.save_path * "/FluxNLPModel_R2/_"*string(myT)*"_" * Dates.format(now(), "yyyy-mm-dd-H-M-S"),
            tb_overwrite,
        ) #TODO changing tblogger for each project 
    end
    train_FluxNlPModel_R2(;myT=myT) #TODO this is slow
    if args.tblogger
    close(tblogger)
    end
end
