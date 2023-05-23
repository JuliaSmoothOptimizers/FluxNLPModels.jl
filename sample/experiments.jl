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

  # Create DataLoaders (mini-batch iterators)
  train_loader = DataLoader((xtrain, ytrain), batchsize = args.batchsize, shuffle = true)
  test_loader = DataLoader((xtest, ytest), batchsize = args.batchsize)
  @info "The type is " typeof(xtrain)
  return train_loader, test_loader
end

function build_model(; imgsize = (28, 28, 1), nclasses = 10,T=Float32)
#   return Chain(Dense(prod(imgsize), 32, relu), Dense(32, nclasses))
    return Chain( #TODO important: when using Dense(matrix of random, dimention is transposed)
        Dense(rand(T,32,prod(imgsize)),true, relu),
        Dense(rand(T, nclasses,32),true) # The following is not correct : Dense(rand(T,32, nclasses),true) 
        )
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
  η::Float32 = 3e-3       # learning rate #TODO fixe this
  batchsize::Int = 2    # batch size
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

function TBCallback(train_loader, test_loader, model, epoch, device)

  ## Report on train and test
  train_loss, train_acc = loss_and_accuracy(train_loader, model, device)
  test_loss, test_acc = loss_and_accuracy(test_loader, model, device)
  println("Epoch=$epoch")
  println("  train_loss = $train_loss, train_accuracy = $train_acc")
  println("  test_loss = $test_loss, test_accuracy = $test_acc")

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
### SGD FluxNLPModels

function train_FluxNLPModel_SGD(; T= Float32 ,kws...)
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
  train_loader, test_loader = getdata(args,T=T) #the type is chages

  @info "Constructing model and starting training"
  ## Construct model
  model = build_model(T=T) |> device

  @info "The type of model  is " typeof(model)

  # now we set the model to FluxNLPModel
  nlp = FluxNLPModel(model, train_loader, test_loader; loss_f = loss)
  g = similar(nlp.w) #TODO should they be here?
  w_k = copy(nlp.w)
  @info "The type of nlp.w  is " typeof(nlp.w)

  for epoch = 1:(args.epochs)
    for (x, y) in train_loader
      x, y = device(x), device(y) ## transfer data to device
      nlp.current_training_minibatch = (x, y)
      g = NLPModels.grad(nlp, w_k)
      w_k -= T(args.η) .* g      #   update the parameter
      FluxNLPModels.set_vars!(nlp, w_k) #TODO Not sure about this
    end
    # logging
    TBCallback(train_loader, test_loader, nlp.chain, epoch, device) #not sure to pass nlp.chain or fx
  end
end



for myT in [Float16,Float32,Float64,Float32sr]
# for myT in [Float32]
    if args.tblogger #TODO add timer to this 
      global   tblogger = TBLogger(
            args.save_path * "/FluxNLPModel_SGD/_"*string(myT)*"_" * Dates.format(now(), "yyyy-mm-dd-H-M-S"),
            tb_overwrite,
        ) #TODO changing tblogger for each project 
    end
    train_FluxNLPModel_SGD(;T=myT) #TODO this is slow
    if args.tblogger
    close(tblogger)
    end
end


# #TODO 
# #changing the type from Float16 to Float64
# parameters = Flux.params(predict)

# println("type of the parameters", typeof.(parameters)) 

# cfun(x::AbstractArray) = Float64.(x); 
# cfun(x) = x; #Noop for stuff which is not arrays (e.g. activation functions)
# m64 = Flux.fmap(cfun, predict);

# println("type of the parameters", typeof.(Flux.params(m64))) 


# f16(m) = Flux.paramtype(Float16, m) # similar to https://github.com/FluxML/Flux.jl/blob/d21460060e055dca1837c488005f6b1a8e87fa1b/src/functor.jl#L217

# m16 = f16(m64)
# println("type of the parameters", typeof.(Flux.params(m16))) 