---
# title: 'FluxNLPModels and KnetNLPModels.jl: Wrapper and connector for deep learning models into JSOSolvers.jl'
Title: 'FluxNLPModels and KnetNLPModels.jl: Wrappers and Connectors for Deep Learning Models in JSOSolvers.jl'

tags:
  - Julia
  - Machine learning
  - Smooth-optimization
authors:
    - name: Farhad Rahbarnia
    corresponding: true
    affiliation: 1
  - name: Paul Raynaud
    # orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  # - name: Nathan Allaire
  #   corresponding: true # (This is how to denote the corresponding author)
  #   affiliation: 1
affiliations:
 - name: GERAD, Montréal, Canada
   index: 1
 - name: GSCOP, Grenoble, France
   index: 2
date: 19 May 2023
bibliography: paper.bib
---

# Summary
<!-- Edit one 
Julia language supports many machine learning and more specifically deep learning (DL) libraries. Knet [Knet.jl](https://github.com/denizyuret/Knet.jl)  and Flux (ref) are among the top libraries that are used by practicitonrs of D and machine learning researchers. These packages generally designed to be a standalone modules which includes:
- deep neural networks modelling;
- support standard training and test datasets (from MLDatasets.jl in Julia);
- several loss-functions, which may be evaluated from a mini-batch of a dataset;
- evaluate the accuracy of a neural network from a test dataset;
- GPU support of any operation performed by a neural network;
- state-of-the-art optimizers: SGD, Nesterov, Adagrad, Adam (refs), which are sophisticate stochastic line-search around first order derivatives of the loss-function.

DL models and optimization can be seen as an unconstrained non-inear optimizations. However, usually Flux and Knet due to their standalone nature lack interfaces with pure optimization frameworks such as JSOSolver (ref). JSOSolvers expects the problem to be defined in a specific way as defined by [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl.). Turining DL models (espically the more complecated one) is a difficualt  and time consuming task for a users. 

We introdue two packages: KnetNLPModels.jl and FluxNLPModels.jl which provide a wrapper for DL models for JSOSolvers unconstrained models. The user only need to call the wrapper object with their Dl models and sample inputs and outputs. It allows user to take advantage of the Knet's and Flux's interfaces, such as:
- standard training and test datasets
- several lost functions or user-defined
- ability to divide datasets into user-defined-size minibatches
- support GPU/CPU interface
- different percisions
- initilization routins for weights
- TensorBoard.jl
- exisiting sample benchmarks 
- data preprosessing


The interfaces for KnetNLPModels and FluxNLPModels differe slightly due to different functionallities they would have.   -->


<!-- Edit 2 -->
<!-- 
The Julia language supports various machine learning libraries, including [Knet.jl](https://github.com/denizyuret/Knet.jl) and Flux (reference), which are widely used by practitioners and researchers in deep learning and machine learning. These libraries offer standalone modules that encompass several features, such as deep neural network modeling, support for standard training and test datasets (via MLDatasets.jl in Julia), various loss functions evaluatable from dataset mini-batches, accuracy evaluation on test datasets, GPU support for neural network operations, and state-of-the-art optimizers like SGD, Nesterov, Adagrad, and Adam (references) that utilize sophisticated stochastic line search around first-order derivatives of the loss function.

Deep learning models and optimization can be regarded as unconstrained nonlinear optimizations. However, Flux and Knet, given their standalone nature, lack interfaces with pure optimization frameworks like JSOSolver (reference). JSOSolvers expects problems to be defined in a specific manner as outlined by [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl). Training and using JSOSolvers for DL models, especially complex ones, can be a challenging and time-consuming task for users.

To address this issue, we present two packages: KnetNLPModels.jl and FluxNLPModels.jl. These packages provide wrappers for DL models within JSOSolvers unconstrained models. Users only need to call the wrapper object with their DL models, sample inputs, and outputs. This approach allows users to leverage the interfaces provided by Knet and Flux, including standard training and test datasets, multiple predefined or user-defined loss functions, the ability to partition datasets into user-defined minibatches, GPU/CPU support, different precisions, weight initialization routines, TensorBoard.jl integration, existing sample benchmarks, and data preprocessing capabilities.

The interfaces of KnetNLPModels and FluxNLPModels differ slightly to accommodate their respective functionalities. -->

<!-- Edit 3
 -->
Several deep learning (DL) libraries have been developed for the Julia language, including [Knet.jl](https://github.com/denizyuret/Knet.jl), and [Flux.jl](reference). Both are widely used by practitioners and researchers in deep learning and machine learning (this needs a reference or a measure to back up the statement). Those libraries offer facilities for deep neural network modeling, access to standard training and test datasets (via MLDatasets.jl, among others), various loss functions that may be evaluated on dataset mini batches, accuracy evaluation facilities on test datasets, GPU support for neural network operations, and state-of-the-art optimizers, including the stochastic gradient method, Nesterov acceleration, Adagrad, and Adam (references) that rely solely on first derivatives of the sampled loss.

Deep network training can be regarded as an unconstrained nonlinear optimization problem. However, DL libraries, including Knet.jl and Flux.jl, given their standalone nature and the traditional nature of stochastic algorithms, lack interfaces with general optimization frameworks such as [JSOSolvers.jl](reference) in which descent of a certain objective is enforced. JSOSolvers.jl expects a model to be conform to the [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) API.

*Training and using JSOSolvers.jl for DL models, especially complex ones, can be a challenging and time-consuming task for users.* (I don’t understand what this means)

We present two packages, KnetNLPModels.jl and FluxNLPModels.jl, that expose DL models as optimization problems conforming to the NLPModels.jl API. Users instantiate a model using a deep network, sample inputs, and outputs. Sampled objective value and gradient are evaluated using the facilities included in Knet.jl and Flux.jl. This approach allows users to leverage the interfaces provided by the DL libraries, including standard training and test datasets, predefined or user-defined loss functions, the ability to partition datasets into user-defined minibatches, GPU/CPU support, use of various floating-point systems, weight initialization routines, [TensorBoard.jl](link) integration, existing sample benchmarks, and data preprocessing capabilities.

We hope the decoupling of the modeling tool from the optimization solvers will allow users and researchers to employ a wide variety of optimization solvers, including a range of existing solvers not traditionally applied to deep network training.

*We still need some statements saying that we did just that and comment on the numerical results as compared to SG or another method.*



















<!-- KnetNLPModels.jl tackles this issue by implementing a KnetNLPModel, an unconstrained smooth optimization model. -->

<!-- KnetNLPModel gather a neural network modelled with Knet, a loss function, a dataset and implement interface's methods related to unconstrained models with Knet's functionnalities. -->
KnetNLPModel benefits from the JuliaSmoothOptimizers ecosystem and is not limitted to the Knet solvers
It has access to:
- [JSOSolvers.jl](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl) optimizers, which train the neural network by considering the weights as variables;
- augmented optimization models such as quasi-Newton models (LBFGS or LSR1).


# Example -- name to change 
The following example (ref) covers traing a DNN model on MNIST (ref) data sets. 
It includes loading the data, defining DNN model, here Lenet-5 (ref), setting the mini-batches, and traing using R2 solver from JSOSolvers. 

The main step is to transfer a Knet model to a KnetNLP model, this can be achived by:
```julia
# LeNet is defined model for Knet 
LeNetNLPModel = KnetNLPModel(
        LeNet;
        data_train = (xtrn, ytrn),
        data_test = (xtst, ytst),
        size_minibatch = minibatchSize,
    )
```
 ```KnetNLPModel``` takes **LeNet**, a small DNN model defined in Knet using ```Chainnnll```, training and test data, as well as user-defined batchsize.

Once the KnetNLP model is created, solvers from JSOSolver can be used. Here is an example using R2 solver 
```julia
solver_stats = R2(
        LeNetNLPModel;
        callback = (nlp, solver, stats, nlp_param) ->
            cb(nlp, solver, stats, stochastic_data),
    )
```
For more information on R2 solver, refer to (ref)

To change the mini-batch data and update the epochs, a callback method may be constructed and passed on to the R2 solver.

```julia
function cb(
    nlp,
    solver,
    stats,
    data::StochasticR2Data,
)
    # Max epoch
    if data.epoch == data.max_epoch
        stats.status = :user
        return
    end
    data.i = KnetNLPModels.minibatch_next_train!(nlp)
    if data.i == 1   # once one epoch is finished     
        # reset
        data.grads_arr = []
        data.epoch += 1
        acc = KnetNLPModels.accuracy(nlp) # accracy of the minibatch on the test Data
        train_acc = Knet.accuracy(nlp.chain; data = nlp.training_minibatch_iterator) 
    end
end
```
We used a stuct to pass on different values and keep track of the accuracy during the training.

To check the accuracy of the training or test data, use:
```julia
train_acc = Knet.accuracy(nlp.chain; data = nlp.training_minibatch_iterator) 
```


To allow use of GPU we need the ```Knet.array_type``` to be set, we can achive that using:
```julia
if CUDA.functional()
    Knet.array_type[] = CUDA.CuArray{T}
else
    Knet.array_type[] = Array{T}
end
```


# Statement of need


# Acknowledgements

This work has been supported by the NSERC Alliance grant 544900-19 in collaboration with Huawei-Canada



# References