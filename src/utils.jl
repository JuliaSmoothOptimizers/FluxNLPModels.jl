
# """
#     create_minibatch(X, Y, minibatch_size)

# Create a minibatch's iterator of the data `X`, `Y` of size `1/minibatch_size * length(Y)`.
# """
# function create_minibatch(x_data, y_data, minibatch_size; shuffle::Bool)
#   mb = minibatch(
#     x_data,
#     y_data,
#     minibatch_size;
#     xsize = (size(x_data, 1), size(x_data, 2), size(x_data, 3), :),
#   )
#   mb = DataLoader((x_data, y_data), batchsize = minibatch_size, shuffle = shuffle)
#   return mb
# end

# """
#     reset_minibatch_train!(nlp::AbstractFluxNLPModel)

# Select the first training minibatch for `nlp`.
# """
# function reset_minibatch_train!(nlp::AbstractFluxNLPModel)
#   nlp.current_training_minibatch = first(nlp.training_minibatch_iterator)
# end

# """
#     reset_minibatch_test!(nlp::AbstractFluxNLPModel)

# Select the first test minibatch for `nlp`.
# """
# function reset_minibatch_test!(nlp::AbstractFluxNLPModel)
#   nlp.current_test_minibatch = first(nlp.test_minibatch_iterator)
# end

"""
    accuracy(nlp::AbstractFluxNLPModel)

Compute the accuracy of the network `nlp.chain` on the entire test dataset.
"""
#TODO add the accuracy

"""
    set_vars!(model::AbstractFluxNLPModel{T,S}, new_w::AbstractVector{T}) where {T<:Number, S}

Sets the vaiables and rebuild the chain
"""
function set_vars!(
  nlp::AbstractFluxNLPModel{T, S},
  new_w::AbstractVector{T},
) where {T <: Number, S} #TODO test T 

  #Flattening 
  old_w, rebuild = Flux.destructure(nlp.chain) #TODO IMPORTANT check what happens if it started with float32 and now I do float64
  nlp.w = new_w
  # model two
  nlp.chain = rebuild(new_w)
  return old_w # return the old wieghts #TODO not sure if we need this but it would be good to keep track
end

"""
    flat_grad!(nlp,g)

    calculate the gradient and return 1 dimentional vector 
# Arguments
- `nlp::AbstractFluxNLPModel{T, S}`: the FluxNLPModel data struct
- `w::AbstractVector{T}`: is the vector of weights/variables;
-`g::AbstractVector{T}`: the gradient vector

# Output
- `f_w`, `g`: the new objective function, and the grad at point x
"""

function flat_grad!(nlp::AbstractFluxNLPModel{T, S}, g::AbstractVector{T}) where {T, S}
  x, y = nlp.current_training_minibatch
  param = Flux.params(nlp.chain)
  gs = gradient(() -> nlp.loss_f(nlp.chain(x), y), param) # compute gradient  

  for p in param
    buff, re = Flux.destructure(gs[p])
    append!(g, buff)
  end
  return g
end
