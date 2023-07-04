# FluxNLPModels.jl Tutorial
## Setting up 
This step-by-step example assumes prior knowledge of [Julia](https://julialang.org/) and [Flux.jl](https://github.com/FluxML/Flux.jl).
See the [Julia tutorial](https://julialang.org/learning/) and the [Flux.jl tutorial](https://fluxml.ai/Flux.jl/stable/models/quickstart/#man-quickstart) for more details.


We have aligned this tutorial to [MLP_MNIST](https://github.com/FluxML/model-zoo/blob/master/vision/mlp_mnist/mlp_mnist.jl) example and reused some of their functions.

### What we cover in this tutorial

We will cover the following:

- Define a Neural Network (NN) Model in Flux, 
  - Fully connected model
- Define or set the loss function
- Data loading
  - MNIST 
  - Divide the data into train and test
- Define a method for calculating accuracy and loss
- Transfer the NN model to FluxNLPModel 
- Using FluxNLPModels and access 
  - Gradient of current weight
  - Objective (or loss) evaluated at current weights 


### Packages needed
```@example FluxNLPModel
using FluxNLPModels
using Flux, NLPModels
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using MLDatasets
using JSOSolvers
```

### Setting Neural Network (NN) Model

First, a NN model needs to be define in Flux.jl.
Our model is very simple: It consists of one "hidden layer" with 32 "neurons", each connected to every input pixel. Each neuron has a sigmoid nonlinearity and is connected to every "neuron" in the output layer. Finally, softmax produces probabilities, i.e., positive numbers that add up to 1.

We have two ways of defining the models:

1. **Direct Definition**: You can directly define the model in your code, specifying the layers and their connections using Flux's syntax. This approach allows for more flexibility and customization.
   ```@example FluxNLPModel
    model = Flux.Chain(Dense(28^2=> 32, relu), Dense(32=>10)) 
   ```

2. **Method-Based Definition**: Alternatively, you can create a method that returns the model. This method can encapsulate the specific architecture and parameters of the model, making it easier to reuse and manage. It provides a convenient way to define and initialize the model when needed.
   ```@example FluxNLPModel
    function build_model(; imgsize = (28, 28, 1), nclasses = 10)
      return Chain(Dense(prod(imgsize), 32, relu), Dense(32, nclasses)) 
    end
   ```



Both approaches have their advantages, and you can choose the one that suits your needs and coding style.

### Loss function

We can define any loss function that we need, here we use Flux build-in logitcrossentropy function. 
```@example FluxNLPModel
## Loss function
const loss = Flux.logitcrossentropy
```

We also define a loss function `loss_and_accuracy`. 
```@example FluxNLPModel
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
```


### Load datasets and define minibatch 
In this section, we will cover the process of loading datasets and defining minibatches for training your model using Flux. Loading and preprocessing data is an essential step in machine learning, as it allows you to train your model on real-world examples.

We will specifically focus on loading the MNIST dataset. We will divide the data into training and testing sets, ensuring that we have separate data for model training and evaluation.

Additionally, we will define minibatches, which are subsets of the dataset that are used during the training process. Minibatches enable efficient training by processing a small batch of examples at a time, instead of the entire dataset. This technique helps in managing memory resources and improving convergence speed.



```@example FluxNLPModel
function getdata(batchsize)
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
  train_loader = DataLoader((xtrain, ytrain), batchsize = batchsize, shuffle = true)
  test_loader = DataLoader((xtest, ytest), batchsize = batchsize)

  return train_loader, test_loader
end
```


### Transfering to FluxNLPModels

```@example FluxNLPModel
  device = cpu
  train_loader, test_loader = getdata(128)

  ## Construct model
  model = build_model() |> device

  # now we set the model to FluxNLPModel
  nlp = FluxNLPModel(model, train_loader, test_loader; loss_f = loss)
```



 
## Tools associated with a FluxNLPModel
The problem dimension `n`, where `w` ∈ ℝⁿ:
```@example FluxNLPModel
n = nlp.meta.nvar
```

### Get the current network weights:
```@example FluxNLPModel
w = nlp.w
```

### Evaluate the loss function (i.e. the objective function) at `w`:
```@example FluxNLPModel
using NLPModels
NLPModels.obj(nlp, w)
```
The length of `w` must be `nlp.meta.nvar`.

### Evaluate the gradient at `w`:
```@example FluxNLPModel
g = similar(w)
NLPModels.grad!(nlp, w, g)
```