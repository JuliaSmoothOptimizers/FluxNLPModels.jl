var documenterSearchIndex = {"docs":
[{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [FluxNLPModels]","category":"page"},{"location":"reference/#FluxNLPModels.FluxNLPModel","page":"Reference","title":"FluxNLPModels.FluxNLPModel","text":"FluxNLPModel{T, S, C <: Flux.Chain} <: AbstractNLPModel{T, S}\n\nData structure that makes the interfaces between neural networks defined with Flux.jl and NLPModels. A FluxNLPModel has fields\n\nArguments\n\nmeta and counters retain informations about the FluxNLPModel;\nchain is the chained structure representing the neural network;\ndata_train is the complete data training set;\ndata_test is the complete data test;\nsize_minibatch parametrizes the size of an training and test minibatches\ntraining_minibatch_iterator is an iterator over an training minibatches;\ntest_minibatch_iterator is an iterator over the test minibatches;\ncurrent_training_minibatch is the training minibatch used to evaluate the neural network;\ncurrent_minibatch_test is the current test minibatch, it is not used in practice;\nw is the vector of weights/variables;\n\n\n\n\n\n","category":"type"},{"location":"reference/#FluxNLPModels.FluxNLPModel-Union{Tuple{F}, Tuple{T}, Tuple{T, Any, Any}} where {T<:Flux.Chain, F<:Function}","page":"Reference","title":"FluxNLPModels.FluxNLPModel","text":"FluxNLPModel(chain_ANN data_train=MLDatasets.MNIST.traindata(Float32), data_test=MLDatasets.MNIST.testdata(Float32); size_minibatch=100)\n\nBuild a FluxNLPModel from the neural network represented by chain_ANN. chain_ANN is built using Flux.jl for more details. The other data required are: an iterator over the training dataset data_train, an iterator over the test dataset data_test and the size of the minibatch size_minibatch. Suppose (xtrn,ytrn) = Fluxnlp.data_train\n\n\n\n\n\n","category":"method"},{"location":"reference/#FluxNLPModels.accuracy-Union{Tuple{AbstractFluxNLPModel{T, S}}, Tuple{S}, Tuple{T}} where {T, S}","page":"Reference","title":"FluxNLPModels.accuracy","text":"accuracy(nlp::AbstractFluxNLPModel)\n\nCompute the accuracy of the network nlp.chain on the entire test dataset. data_loader can be overwritten to include other data,  device is set to cpu \n\n\n\n\n\n","category":"method"},{"location":"reference/#FluxNLPModels.minibatch_next_test!-Tuple{AbstractFluxNLPModel}","page":"Reference","title":"FluxNLPModels.minibatch_next_test!","text":"minibatch_next_test!(nlp::AbstractFluxNLPModel)\n\nSelects the next minibatch from nlp.test_minibatch_iterator.   Returns the new current status of the iterator nlp.current_test_minibatch. minibatch_next_test! aims to be used in a loop or method call. if return false, it means that it reach the end of the mini-batch\n\n\n\n\n\n","category":"method"},{"location":"reference/#FluxNLPModels.minibatch_next_train!-Tuple{AbstractFluxNLPModel}","page":"Reference","title":"FluxNLPModels.minibatch_next_train!","text":"minibatch_next_train!(nlp::AbstractFluxNLPModel)\n\nSelects the next minibatch from nlp.training_minibatch_iterator.   Returns the new current status of the iterator nlp.current_training_minibatch. minibatch_next_train! aims to be used in a loop or method call. if return false, it means that it reach the end of the mini-batch\n\n\n\n\n\n","category":"method"},{"location":"reference/#FluxNLPModels.reset_minibatch_test!-Tuple{AbstractFluxNLPModel}","page":"Reference","title":"FluxNLPModels.reset_minibatch_test!","text":"reset_minibatch_test!(nlp::AbstractFluxNLPModel)\n\nIf a data_loader (an iterator object is passed to FluxNLPModel) then  Select the first test minibatch for nlp.\n\n\n\n\n\n","category":"method"},{"location":"reference/#FluxNLPModels.reset_minibatch_train!-Tuple{AbstractFluxNLPModel}","page":"Reference","title":"FluxNLPModels.reset_minibatch_train!","text":"reset_minibatch_train!(nlp::AbstractFluxNLPModel)\n\nIf a data_loader (an iterator object is passed to FluxNLPModel) then  Select the first training minibatch for nlp.\n\n\n\n\n\n","category":"method"},{"location":"reference/#FluxNLPModels.set_vars!-Union{Tuple{S}, Tuple{T}, Tuple{AbstractFluxNLPModel{T, S}, AbstractVector{T}}} where {T<:Number, S}","page":"Reference","title":"FluxNLPModels.set_vars!","text":"set_vars!(model::AbstractFluxNLPModel{T,S}, new_w::AbstractVector{T}) where {T<:Number, S}\n\nSets the vaiables and rebuild the chain\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModels.grad!-Union{Tuple{S}, Tuple{T}, Tuple{AbstractFluxNLPModel{T, S}, AbstractVector{T}, AbstractVector{T}}} where {T, S}","page":"Reference","title":"NLPModels.grad!","text":"g = grad!(nlp, w, g)\n\nEvaluate ∇f(w), the gradient of the objective function at w in place.\n\nArguments\n\nnlp::AbstractFluxNLPModel{T, S}: the FluxNLPModel data struct;\nw::AbstractVector{T}: is the vector of weights/variables;\ng::AbstractVector{T}: the gradient vector.\n\nOutput\n\ng: the gradient at point w.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModels.obj-Union{Tuple{S}, Tuple{T}, Tuple{AbstractFluxNLPModel{T, S}, AbstractVector{T}}} where {T, S}","page":"Reference","title":"NLPModels.obj","text":"f = obj(nlp, w)\n\nEvaluate f(w), the objective function of nlp at w.\n\nArguments\n\nnlp::AbstractFluxNLPModel{T, S}: the FluxNLPModel data struct;\nw::AbstractVector{T}: is the vector of weights/variables.\n\nOutput\n\nf_w: the new objective function.\n\n\n\n\n\n","category":"method"},{"location":"reference/#NLPModels.objgrad!-Union{Tuple{S}, Tuple{T}, Tuple{AbstractFluxNLPModel{T, S}, AbstractVector{T}, AbstractVector{T}}} where {T, S}","page":"Reference","title":"NLPModels.objgrad!","text":"objgrad!(nlp, w, g)\n\nEvaluate both f(w), the objective function of nlp at w, and ∇f(w), the gradient of the objective function at w in place.\n\nArguments\n\nnlp::AbstractFluxNLPModel{T, S}: the FluxNLPModel data struct;\nw::AbstractVector{T}: is the vector of weights/variables;\ng::AbstractVector{T}: the gradient vector.\n\nOutput\n\nf_w, g: the new objective function, and the gradient at point w.\n\n\n\n\n\n","category":"method"},{"location":"#FluxNLPModels.jl","page":"Home","title":"FluxNLPModels.jl","text":"","category":"section"},{"location":"#Compatibility","page":"Home","title":"Compatibility","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Julia ≥ 1.6.","category":"page"},{"location":"#How-to-install","page":"Home","title":"How to install","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This module can be installed with the following command:","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add FluxNLPModels","category":"page"},{"location":"#Synopsis","page":"Home","title":"Synopsis","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"FluxNLPModels exposes neural network models as optimization problems conforming to the NLPModels API. FluxNLPModels is an interface between Flux.jl's classification neural networks and NLPModels.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"A FluxNLPModel gives the user access to:","category":"page"},{"location":"","page":"Home","title":"Home","text":"The values of the neural network variables/weights w;\nThe value of the objective/loss function L(X, Y; w) at w for a given minibatch (X,Y);\nThe gradient ∇L(X, Y; w) of the objective/loss function at w for a given minibatch (X,Y).","category":"page"},{"location":"","page":"Home","title":"Home","text":"In addition, it provides tools to:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Switch the minibatch used to evaluate the neural network;\nRetrieve the current minibatch ;\nMeasure the neural network's loss at the current w.","category":"page"},{"location":"#Bug-reports-and-discussions","page":"Home","title":"Bug reports and discussions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you encounter any bugs or have suggestions for improvement, please open an issue. For general questions or discussions related to this repository and the JuliaSmoothOptimizers organization, feel free to start a discussion here.","category":"page"},{"location":"tutorial/#FluxNLPModels.jl-Tutorial","page":"Tutorial","title":"FluxNLPModels.jl Tutorial","text":"","category":"section"},{"location":"tutorial/#Setting-up","page":"Tutorial","title":"Setting up","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"This step-by-step example assumes prior knowledge of Julia and Flux.jl. See the Julia tutorial and the Flux.jl tutorial for more details.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"We have aligned this tutorial to MLP_MNIST example and reused some of their functions.","category":"page"},{"location":"tutorial/#What-we-cover-in-this-tutorial","page":"Tutorial","title":"What we cover in this tutorial","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"We will cover the following:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Define a Neural Network (NN) Model in Flux, \nFully connected model\nDefine or set the loss function\nData loading\nMNIST \nDivide the data into train and test\nDefine a method for calculating accuracy and loss\nTransfer the NN model to FluxNLPModel \nUsing FluxNLPModels and access \nGradient of current weight\nObjective (or loss) evaluated at current weights ","category":"page"},{"location":"tutorial/#Packages-needed","page":"Tutorial","title":"Packages needed","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using FluxNLPModels\nusing Flux, NLPModels\nusing Flux.Data: DataLoader\nusing Flux: onehotbatch, onecold, @epochs\nusing Flux.Losses: logitcrossentropy\nusing MLDatasets\nusing JSOSolvers","category":"page"},{"location":"tutorial/#Setting-Neural-Network-(NN)-Model","page":"Tutorial","title":"Setting Neural Network (NN) Model","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"First, a NN model needs to be define in Flux.jl. Our model is very simple: It consists of one \"hidden layer\" with 32 \"neurons\", each connected to every input pixel. Each neuron has a sigmoid nonlinearity and is connected to every \"neuron\" in the output layer. Finally, softmax produces probabilities, i.e., positive numbers that add up to 1.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"One can create a method that returns the model. This method can encapsulate the specific architecture and parameters of the model, making it easier to reuse and manage. It provides a convenient way to define and initialize the model when needed.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"function build_model(; imgsize = (28, 28, 1), nclasses = 10)\n  return Chain(Dense(prod(imgsize), 32, relu), Dense(32, nclasses)) \nend","category":"page"},{"location":"tutorial/#Loss-function","page":"Tutorial","title":"Loss function","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"We can define any loss function that we need, here we use Flux build-in logitcrossentropy function. ","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"## Loss function\nconst loss = Flux.logitcrossentropy","category":"page"},{"location":"tutorial/#Load-datasets-and-define-minibatch","page":"Tutorial","title":"Load datasets and define minibatch","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"In this section, we will cover the process of loading datasets and defining minibatches for training your model using Flux. Loading and preprocessing data is an essential step in machine learning, as it allows you to train your model on real-world examples.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"We will specifically focus on loading the MNIST dataset. We will divide the data into training and testing sets, ensuring that we have separate data for model training and evaluation.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Additionally, we will define minibatches, which are subsets of the dataset that are used during the training process. Minibatches enable efficient training by processing a small batch of examples at a time, instead of the entire dataset. This technique helps in managing memory resources and improving convergence speed.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"function getdata(bs)\n  ENV[\"DATADEPS_ALWAYS_ACCEPT\"] = \"true\"\n\n  # Loading Dataset\t\n  xtrain, ytrain = MLDatasets.MNIST(Tx = Float32, split = :train)[:]\n  xtest, ytest = MLDatasets.MNIST(Tx = Float32, split = :test)[:]\n\n  # Reshape Data in order to flatten each image into a linear array\n  xtrain = Flux.flatten(xtrain)\n  xtest = Flux.flatten(xtest)\n\n  # One-hot-encode the labels\n  ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)\n\n  # Create DataLoaders (mini-batch iterators)\n  train_loader = DataLoader((xtrain, ytrain), batchsize = bs, shuffle = true)\n  test_loader = DataLoader((xtest, ytest), batchsize = bs)\n\n  return train_loader, test_loader\nend","category":"page"},{"location":"tutorial/#Transfering-to-FluxNLPModels","page":"Tutorial","title":"Transfering to FluxNLPModels","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"  device = cpu\n  train_loader, test_loader = getdata(128)\n\n  ## Construct model\n  model = build_model() |> device\n\n  # now we set the model to FluxNLPModel\n  nlp = FluxNLPModel(model, train_loader, test_loader; loss_f = loss)","category":"page"},{"location":"tutorial/#Tools-associated-with-a-FluxNLPModel","page":"Tutorial","title":"Tools associated with a FluxNLPModel","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"The problem dimension n, where w ∈ ℝⁿ:","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"n = nlp.meta.nvar","category":"page"},{"location":"tutorial/#Get-the-current-network-weights:","page":"Tutorial","title":"Get the current network weights:","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"w = nlp.w","category":"page"},{"location":"tutorial/#Evaluate-the-loss-function-(i.e.-the-objective-function)-at-w:","page":"Tutorial","title":"Evaluate the loss function (i.e. the objective function) at w:","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using NLPModels\nNLPModels.obj(nlp, w)","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"The length of w must be nlp.meta.nvar.","category":"page"},{"location":"tutorial/#Evaluate-the-gradient-at-w:","page":"Tutorial","title":"Evaluate the gradient at w:","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"g = similar(w)\nNLPModels.grad!(nlp, w, g)","category":"page"},{"location":"tutorial/#Train-a-neural-network-with-JSOSOlvers.R2","page":"Tutorial","title":"Train a neural network with JSOSOlvers.R2","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"max_time = 60. # run at most 1min\ncallback = (nlp, \n            solver, \n            stats) -> FluxNLPModels.minibatch_next_train!(nlp)\n\nsolver_stats = R2(nlp; callback, max_time)\ntest_accuracy = FluxNLPModels.accuracy(nlp) #check the accuracy","category":"page"}]
}
