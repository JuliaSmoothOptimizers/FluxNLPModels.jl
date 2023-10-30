"""
    f = obj(nlp, w)

Evaluate `f(w)`, the objective function of `nlp` at `w`. if `w` and `nlp` precision different, we advance to match the the type of `w`

# Arguments
- `nlp::AbstractFluxNLPModel{T, S}`: the FluxNLPModel data struct;
- `w::AbstractVector{V}`: is the vector of weights/variables. The reason for V here is to allow different precision type for weight and models 

# Output
- `f_w`: the new objective function.

"""
function NLPModels.obj(nlp::AbstractFluxNLPModel{T, S}, w::AbstractVector{V}) where {T, V, S}
  x, y = nlp.current_training_minibatch

  if (T != V)  # we check if the types are the same, 
    update_type!(nlp, w)
    if eltype(x) != V #TODO check if the user have changed the typed ?
      x = V.(x)
    end
  end

  set_vars!(nlp, w)
  increment!(nlp, :neval_obj)
  return nlp.loss_f(nlp.chain(x), y)
end

"""
    g = grad!(nlp, w, g)

Evaluate `∇f(w)`, the gradient of the objective function at `w` in place.

# Arguments
- `nlp::AbstractFluxNLPModel{T, S}`: the FluxNLPModel data struct;
- `w::AbstractVector{T}`: is the vector of weights/variables;
- `g::AbstractVector{T}`: the gradient vector.

# Output
- `g`: the gradient at point `w`.

"""
function NLPModels.grad!(
  nlp::AbstractFluxNLPModel{T, S},
  w::AbstractVector{V},
  g::AbstractVector{T},
) where {T, V, S}
  @lencheck nlp.meta.nvar w g
  x, y = nlp.current_training_minibatch

  if (T != V)  # we check if the types are the same, 
    update_type!(nlp, w)
    g = V.(g)
    if eltype(x) != V #TODO check if the user have changed the typed ?
      x = V.(x)
    end
  end

  increment!(nlp, :neval_grad)
  g .= gradient(w_g -> local_loss(nlp, x, y, w_g), w)[1]
  return g
end

"""
    objgrad!(nlp, w, g)

Evaluate both `f(w)`, the objective function of `nlp` at `w`, and `∇f(w)`, the gradient of the objective function at `w` in place.

# Arguments
- `nlp::AbstractFluxNLPModel{T, S}`: the FluxNLPModel data struct;
- `w::AbstractVector{T}`: is the vector of weights/variables;
- `g::AbstractVector{T}`: the gradient vector.

# Output
- `f_w`, `g`: the new objective function, and the gradient at point w.

"""
function NLPModels.objgrad!(
  nlp::AbstractFluxNLPModel{T, S},
  w::AbstractVector{V},
  g::AbstractVector{T},
) where {T,V, S}
  @lencheck nlp.meta.nvar w g

  if (T != V)  # we check if the types are the same, 
    update_type!(nlp, w)
    g = V.(g)
    if eltype(x) != V #TODO check if the user have changed the typed ?
      x = V.(x)
    end
  end

  increment!(nlp, :neval_obj)
  increment!(nlp, :neval_grad)
  set_vars!(nlp, w)

  x, y = nlp.current_training_minibatch
  f_w = nlp.loss_f(nlp.chain(x), y)
  g .= gradient(w_g -> local_loss(nlp, x, y, w_g), w)[1]

  return f_w, g
end
