

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
using LinearAlgebra


myT = Float64
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

function build_model(; imgsize = (28, 28, 1), nclasses = 10,T=Float32) #TODO the rand fails for Float32sr make a new matrix then replace it 
  return Chain( #TODO important: when using Dense(matrix of random, dimention is transposed)
      Dense(T.(zeros(Float32,32,prod(imgsize))),true, relu), # I use this way to avoid rand error
      Dense(T.(zeros(Float32, nclasses,32)),true) # The following is not correct : Dense(rand(T,32, nclasses),true) 
      )
end


# Note that we use the functions [Dense](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Dense) so that our model is *densely* (or fully) connected and [Chain](https://fluxml.ai/Flux.jl/stable/models/layers/#Flux.Chain) to chain the computation of the three layers.

# ## Loss function
const loss = logitcrossentropy
const loss2 = Flux.mse

function loss_and_accuracy(data_loader, model, device,T)
  acc = T(0)
  ls = T(0.0f0)
  num = T(0)
  for (x, y) in data_loader
    x, y = device(x), device(y)
    # println(typeof(x))
    ŷ = model(x)
    # println(typeof(ŷ))

    ls += loss(ŷ, y, agg = sum)    
    # println(typeof(ls))

    acc += sum(onecold(ŷ) .== onecold(y)) ## Decode the output of the model
    # println(typeof(acc))

    num += size(x)[end]
    # println(typeof(num))

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



args = Args() ## Collect options in a struct for convenience
device = cpu




## Create test and train dataloaders
train_loader, test_loader = getdata(args,T=myT)

@info "Constructing model and starting training"
## Construct model
model = build_model(T= myT) |> device
@info "The type of model  is " typeof(model)

# now we set the model to FluxNLPModel
nlp = FluxNLPModel(model, train_loader, test_loader; loss_f = loss) #TODO add the device here for the data mini-batch
@info "The type of nlp.w  is " typeof(nlp.w)


x = copy(nlp.meta.x0)
∇fk= similar(nlp.meta.x0)
c= 0

while c<10
  # ∇fk= NLPModels.grad(nlp, x)       
    gk= NLPModels.grad(nlp, x)   
    # ∇fk= similar(nlp.meta.x0)
    
    grad!(nlp, x, ∇fk)

    
    x .= x .- (∇fk)

    norm_∇fk = norm(∇fk)
    norm_gk= norm(gk)

    c= c+1;
    println(norm_∇fk,"-------------", norm_gk)
end



# #######################################################
# w1, re = Flux.destructure(model)
# x, y = first(train_loader)

# function local_L(rebuld,x,y, w::AbstractVector{T}) where {T}
#   # increment!(nlp, :neval_obj) #TODO not sure 
#   # set_vars!(nlp, w)
#   chain = rebuld(w)
 
#   return loss(chain(x), y)
# end


# c= 0

# while c<10
#   g = gradient(w_g->local_L(re,x,y, w_g) , w1);

#   println(norm(g[1])) 
#   w1 .= w1 .- g[1];
#   c=c+1
# end






#        """
#     f, g = objgrad!(nlp, x, g)

# Evaluate ``f(x)`` and ``∇f(x)`` at `x`. `g` is overwritten with the
# value of ``∇f(x)``.
# """
# function objgrad!(nlp, x, g)
#   @lencheck nlp.meta.nvar x g
#   f = obj(nlp, x)
#   grad!(nlp, x, g)
#   return f, g
# end



# julia> function pow(x, n)
#          r = 1
#          for i = 1:n
#            r *= x
#          end
#          return r
#        end
# pow (generic function with 1 method)

# julia> gradient(x -> pow(x, 3), 5)
# (75.0,)

# julia> pow2(x, n) = n <= 0 ? 1 : x*pow2(x, n-1)
# pow2 (generic function with 1 method)

# julia> gradient(x -> pow2(x, 3), 5)




  # set_vars!(nlp, w)
  # x, y = nlp.current_training_minibatch
  # param = Flux.params(nlp.chain)
  # gs = gradient(() -> nlp.loss_f(nlp.chain(x), y), param) # compute gradient  

  # i = 1
  # j= 1
  # for p in param
  #   buff, re = Flux.destructure(gs[p])
  #   j = i + size(buff)[1]
  #   g[i:j-1 ] = buff
  #   i = j+1
    
  # end
  # return g
