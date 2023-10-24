using CUDA
"""
    set_vars!(model::AbstractFluxNLPModel{T,S}, new_w::AbstractVector{T}) where {T<:Number, S}

Sets the vaiables and rebuild the chain
"""
function set_vars!(nlp::AbstractFluxNLPModel{T, S}, new_w::AbstractVector) where {T <: Number, S}
  nlp.w .= new_w
  type_ind = findfirst(x->x == eltype(new_w),nlp.Types)
  nlp.chain[type_ind] = nlp.rebuild[type_ind](nlp.w)
end

function local_loss(nlp::AbstractFluxNLPModel{T, S}, x, y, w::AbstractVector{T}) where {T, S}
  # increment!(nlp, :neval_obj) #TODO not sure 
  type_ind = findfirst(x->x == eltype(w),nlp.Types)
  nlp.chain[type_ind] = nlp.rebuild[type_ind](w)
  return nlp.loss_f(nlp.chain[type_ind](x), y)
end

function local_loss(nlp::AbstractFluxNLPModel{T, S}, rebuild, x, y, w::AbstractVector) where {T, S}
  model = rebuild(w)
  return nlp.loss_f(model(x), y)
end

"""
    accuracy(nlp::AbstractFluxNLPModel)

Compute the accuracy of the network `nlp.chain` on the entire test dataset. data_loader can be overwritten to include other data, 
device is set to cpu 
"""
function accuracy(
  nlp::AbstractFluxNLPModel{T, S};
  model = nlp.chain[findfirst(x->x==nlp.Types[1],nlp.Types)],
  data_loader = nlp.test_minibatch_iterator,
  device = cpu,
  myT = nlp.Types[1],
) where {T, S}
  acc = myT(0)
  num = myT(0)
  for (x, y) in data_loader
    x, y = device(x), device(y)
    ŷ = model(x)
    acc += sum(onecold(ŷ) .== onecold(y)) ## Decode the output of the model
    num += size(x)[end]
  end
  return acc / num #TODO make sure num is not zero
end

"""
    reset_minibatch_train!(nlp::AbstractFluxNLPModel)

If a data_loader (an iterator object is passed to FluxNLPModel) then 
Select the first training minibatch for `nlp`.
"""
function reset_minibatch_train!(nlp::AbstractFluxNLPModel)
  nlp.current_training_minibatch = first(nlp.training_minibatch_iterator)
  nlp.current_training_minibatch_status = nothing
end

"""
    minibatch_next_train!(nlp::AbstractFluxNLPModel)

Selects the next minibatch from `nlp.training_minibatch_iterator`.  
Returns the new current status of the iterator `nlp.current_training_minibatch`.
`minibatch_next_train!` aims to be used in a loop or method call.
if return false, it means that it reach the end of the mini-batch
"""
function minibatch_next_train!(nlp::AbstractFluxNLPModel; device = cpu)
  iter = nlp.training_minibatch_iterator
  if nlp.current_training_minibatch_status === nothing
    next = iterate(iter)
  else
    next = iterate(iter, nlp.current_training_minibatch_status)
  end

  if next === nothing #end of the loop 
    reset_minibatch_train!(nlp)
    return false
  end
  (item, nlp.current_training_minibatch_status) = next
  nlp.current_training_minibatch = device(item)
  return true
end

"""
    reset_minibatch_test!(nlp::AbstractFluxNLPModel)

If a data_loader (an iterator object is passed to FluxNLPModel) then 
Select the first test minibatch for `nlp`.
"""
function reset_minibatch_test!(nlp::AbstractFluxNLPModel)
  nlp.current_test_minibatch = first(nlp.test_minibatch_iterator)
  nlp.current_test_minibatch_status = nothing
end

"""
    minibatch_next_test!(nlp::AbstractFluxNLPModel)

Selects the next minibatch from `nlp.test_minibatch_iterator`.  
Returns the new current status of the iterator `nlp.current_test_minibatch`.
`minibatch_next_test!` aims to be used in a loop or method call.
if return false, it means that it reach the end of the mini-batch
"""
function minibatch_next_test!(nlp::AbstractFluxNLPModel; device = cpu)
  iter = nlp.test_minibatch_iterator
  if nlp.current_test_minibatch_status === nothing
    next = iterate(iter)
  else
    next = iterate(iter, nlp.current_test_minibatch_status)
  end

  if next === nothing #end of the loop 
    reset_minibatch_test!(nlp)
    return false
  end
  (item, nlp.current_test_minibatch_status) = next
  nlp.current_test_minibatch = device(item)
  return true
end

"""
find_type_index(nlp::AbstractFluxNLPModel,w::AbstractVector)

find index of nlp.Types corresponding to element types of w. Returns error if not found.
"""
function find_type_index(nlp::AbstractFluxNLPModel,w::AbstractVector)
  type_ind = findfirst(x->x == eltype(w),nlp.Types)
  type_ind === nothing && error("$(eltype(w)) is not a format supported for weights, supported formats are $(nlp.Types)")
  return type_ind
end

"""
    test_types_consistency(Types,data_train,data_test)

  Tests FluxNLPModel input FP formats consistency.
"""
function test_types_consistency(Types::Vector{DataType},data_train,data_test)
  issorted(Types,by = x-> -eps(x)) || error("models should be provided by increasing precision FP formats")
  train_type = eltype(first(data_train)[1])
  test_type = eltype(first(data_test)[1])
  Types[1] == train_type || @warn "train data FP format ($train_type) doesn't match lowest precision FP format of model weight ($(Types[1]))"
  Types[1] == test_type || @warn "test data FP format ($test_type) doesn't match lowest precision FP format of model weight ($(Types[1]))"
end

"""
    test_types_consistency(chain_ANN,data_train,data_test)

  Tests FluxNLPModel loader and NN device consistency.
"""
function test_devices_consistency(chain_ANN::Vector{T},data_train,data_test) where {T <: Chain}
  d = Flux.destructure.(chain_ANN)
  weights = [del[1] for del in d]
  is_chain_gpu = [typeof(w) <: CuArray for w in weights]
  if !in(sum(is_chain_gpu),[0,length(chain_ANN)])
    @error "Chain models should all be on the same device."
  end
  is_all_chain_gpu = is_chain_gpu[1]
  is_train_gpu = typeof(first(data_train)[1]) <: CuArray
  is_test_gpu = typeof(first(data_test)[1]) <: CuArray
  @show is_chain_gpu is_all_chain_gpu is_train_gpu is_test_gpu
  if is_all_chain_gpu != is_train_gpu
    @error "train loader and models are not on the same device."
  end
  if is_all_chain_gpu != is_test_gpu
    @error "test loader and models are not on the same device."
  end
end
