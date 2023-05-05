"""
    flag_dim(x)
#TODO to this better  
Returns true if x has 3 dimensions.
This function is used to reshape X in `create_minibatch(X, Y, minibatch_size)` in case x has only 3 dimensions.
"""
flag_dim(x) = length(size(x)) == 3

"""
    create_minibatch(X, Y, minibatch_size)

Create a minibatch's iterator of the data `X`, `Y` of size `1/minibatch_size * length(Y)`.
"""
function create_minibatch(x_data, y_data, minibatch_size; shuffle::Bool)
  mb = minibatch(
    x_data,
    y_data,
    minibatch_size;
    xsize = (size(x_data, 1), size(x_data, 2), size(x_data, 3), :),
  )

  mb = DataLoader((x_data, y_data), batchsize=minibatch_size, shuffle=shuffle)

  return mb
end


"""
    reset_minibatch_train!(nlp::AbstractFluxNLPModel)

Select the first training minibatch for `nlp`.
"""
function reset_minibatch_train!(nlp::AbstractFluxNLPModel)
  nlp.current_training_minibatch = first(nlp.training_minibatch_iterator)
  nlp.i_train = 1
end

"""
  rand_minibatch_train!(nlp::AbstractFluxNLPModel)

Select a training minibatch for `nlp` randomly.
"""
function rand_minibatch_train!(nlp::AbstractFluxNLPModel)
  nlp.i_train = rand(1:(nlp.training_minibatch_iterator.imax))
  nlp.current_training_minibatch = iterate(nlp.training_minibatch_iterator, nlp.i_train)
end

"""
    minibatch_next_train!(nlp::AbstractFluxNLPModel)

Selects the next minibatch from `nlp.training_minibatch_iterator`.  
Returns the new current location of the iterator `nlp.i_train`.
If it returns 1, the current training minibatch is the first of `nlp.training_minibatch_iterator` and the previous minibatch was the last of `nlp.training_minibatch_iterator`.
`minibatch_next_train!` aims to be used in a loop or method call.
Refer to FluxNLPModelProblems.jl for more use cases.
"""
function minibatch_next_train!(nlp::AbstractFluxNLPModel)
  nlp.i_train += nlp.size_minibatch # update the i by mini_batch size
  result = iterate(nlp.training_minibatch_iterator, nlp.i_train)
  if result === nothing
    # reset to the begining 
    reset_minibatch_train!(nlp)
  else
    (next, indice) = result
    nlp.current_training_minibatch = next
  end

  return nlp.i_train
end

"""
    reset_minibatch_test!(nlp::AbstractFluxNLPModel)

Select a new test minibatch for `nlp` at random.
"""
function rand_minibatch_test!(nlp::AbstractFluxNLPModel)
  nlp.i_test = rand(1:(nlp.test_minibatch_iterator.imax))
  nlp.current_test_minibatch = iterate(nlp.test_minibatch_iterator, nlp.i_test)
end

"""
    reset_minibatch_test!(nlp::AbstractFluxNLPModel)

Select the first test minibatch for `nlp`.
"""
function reset_minibatch_test!(nlp::AbstractFluxNLPModel)
  nlp.current_test_minibatch = first(nlp.test_minibatch_iterator)
  nlp.i_test = 1
end

"""
    minibatch_next_test!(nlp::AbstractFluxNLPModel)

Selects the next minibatch from `test_minibatch_iterator`.
Returns the new current location of the iterator `nlp.i_test`.
If it returns 1, the current training minibatch is the first of `nlp.test_minibatch_iterator` and the previous minibatch was the last of `nlp.test_minibatch_iterator`.
`minibatch_next_test!` aims to be used in a loop or method call - refere to FluxNLPModelProblems.jl for more use cases
"""
function minibatch_next_test!(nlp::AbstractFluxNLPModel)
  nlp.i_test += nlp.size_minibatch #TODO in the futue we might want to have different size for minbatch test vs train
  result = iterate(nlp.test_minibatch_iterator, nlp.i_test)

  if result === nothing
    # reset to the begining 
    reset_minibatch_test!(nlp)
  else
    (next, indice) = result
    nlp.current_test_minibatch = next
  end

  return nlp.i_test

end

"""
    accuracy(nlp::AbstractFluxNLPModel)

Compute the accuracy of the network `nlp.chain` on the entire test dataset.
"""
#TODO add the accuracy
# accuracy(nlp::AbstractFluxNLPModel) = Flux.accuracy(nlp.chain; data = nlp.test_minibatch_iterator)

"""
    set_vars!(model::AbstractFluxNLPModel{T,S}, new_w::AbstractVector{T}) where {T<:Number, S}
"""
function set_vars!(
  model::AbstractFluxNLPModel{T, S},
  new_w::AbstractVector{T},
) where {T <: Number, S}

  #Flattening 
  old_w, rebuild = Flux.destructure(model.chain)
  # model two
  model.chain = rebuild(new_w)

end
