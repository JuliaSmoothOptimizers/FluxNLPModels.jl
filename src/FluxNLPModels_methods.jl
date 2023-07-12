"""
    f = obj(nlp, w)

Evaluate `f(w)`, the objective function of `nlp` at `w`.

# Arguments
- `nlp::AbstractFluxNLPModel{T, S}`: the FluxNLPModel data struct;
- `w::AbstractVector{T}`: is the vector of weights/variables.

# Output
- `f_w`: the new objective function.

"""
function NLPModels.obj(nlp::AbstractFluxNLPModel{T, S}, w::AbstractVector{T}) where {T, S}
  increment!(nlp, :neval_obj)
  set_vars!(nlp, w)
  x, y = nlp.current_training_minibatch
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
  w::AbstractVector{T},
  g::AbstractVector{T},
) where {T, S}
  @lencheck nlp.meta.nvar w g
  increment!(nlp, :neval_grad)
  x, y = nlp.current_training_minibatch
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
  w::AbstractVector{T},
  g::AbstractVector{T},
) where {T, S}
  @lencheck nlp.meta.nvar w g
  increment!(nlp, :neval_obj)
  increment!(nlp, :neval_grad)
  set_vars!(nlp, w)

  x, y = nlp.current_training_minibatch
  f_w = nlp.loss_f(nlp.chain(x), y)
  g .= gradient(w_g -> local_loss(nlp, x, y, w_g), w)[1]

  return f_w, g
end
