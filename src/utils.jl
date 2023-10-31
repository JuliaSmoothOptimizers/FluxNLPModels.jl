"""
    update_type!(nlp::AbstractFluxNLPModel{T, S}, w::AbstractVector{V}) where {T,V, S}


Sets the vaiables and rebuild the chain to a sepecific type defined by weigths 
"""
function update_type!(nlp::AbstractFluxNLPModel{T, S}, w::AbstractVector{V}) where {T, V, S}
  if V == Float16
    local_chain = f16(nlp.chain)
  elseif V == Float64
    local_chain = f64(nlp.chain)
  elseif V == Float32
    local_chain = f32(nlp.chain)
  else
    error("The package only support Float16, Float32 and Float64")
  end

  nlp.chain = local_chain
  nlp.w, nlp.rebuild = Flux.destructure(nlp.chain)
end

"""
    set_vars!(model::AbstractFluxNLPModel{T,S}, new_w::AbstractVector{T}) where {T<:Number, S}

Sets the vaiables and rebuild the chain
"""
function set_vars!(
  nlp::AbstractFluxNLPModel{T, S},
  new_w::AbstractVector{V},
) where {T <: Number, S, V}
  nlp.w .= new_w
  nlp.chain = nlp.rebuild(nlp.w)
end

function local_loss(nlp::AbstractFluxNLPModel{T, S}, x, y, w::AbstractVector{V}) where {T, S, V}
  # increment!(nlp, :neval_obj) #TODO not sure 
  nlp.chain = nlp.rebuild(w)
  return nlp.loss_f(nlp.chain(x), y)
end

"""
    accuracy(nlp::AbstractFluxNLPModel)

Compute the accuracy of the network `nlp.chain` on the entire test dataset. data_loader can be overwritten to include other data, 
device is set to cpu 
"""
function accuracy(
  nlp::AbstractFluxNLPModel{T, S};
  model = nlp.chain,
  data_loader = nlp.test_minibatch_iterator,
  device = cpu,
  myT = Float32,
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
