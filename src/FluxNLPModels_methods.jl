"""
    f = obj(nlp, x)

Evaluate `f(x)`, the objective function of `nlp` at `x`.

# Arguments
- `nlp::AbstractFluxNLPModel{T, S}`: the FluxNLPModel data struct
- `w::AbstractVector{T}`: is the vector of weights/variables;

# Output
- `f_w`: the new objective function
"""
function NLPModels.obj(nlp::AbstractFluxNLPModel{T, S}, w::AbstractVector{T}) where {T, S}
  increment!(nlp, :neval_obj)
  set_vars!(nlp, w)
  x, y = nlp.current_training_minibatch
  f_w = nlp.loss_f(nlp.chain(x), y)
  return f_w
end

"""
    g = grad!(nlp, x, g)

Evaluate `∇f(x)`, the gradient of the objective function at `x` in place.
# Arguments
- `nlp::AbstractFluxNLPModel{T, S}`: the FluxNLPModel data struct
- `w::AbstractVector{T}`: is the vector of weights/variables;
-`g::AbstractVector{T}`: the gradient vector

# Output
- `g`: the gradient at point x
"""
function NLPModels.grad!(
  nlp::AbstractFluxNLPModel{T, S},
  w::AbstractVector{T},
  g::AbstractVector{T},
) where {T, S}
  @lencheck nlp.meta.nvar w g
  increment!(nlp, :neval_grad)
 
  g = gradient(w_g->local_loss(nlp, w_g) , w)
  return g[1]
end

function local_loss(nlp::AbstractFluxNLPModel{T, S}, w::AbstractVector{T}) where {T, S}
  # increment!(nlp, :neval_obj) #TODO not sure 
  set_vars!(nlp, w)
  x, y = nlp.current_training_minibatch
  f_w = nlp.loss_f(nlp.chain(x), y)
  return f_w
end



"""
    objgrad!(nlp, x, g)

    Evaluate both `f(x)`, the objective function of `nlp` at `x` and `∇f(x)`, the gradient of the objective function at `x` in place.

# Arguments
- `nlp::AbstractFluxNLPModel{T, S}`: the FluxNLPModel data struct
- `w::AbstractVector{T}`: is the vector of weights/variables;
-`g::AbstractVector{T}`: the gradient vector

# Output
- `f_w`, `g`: the new objective function, and the gradient at point x

"""
function NLPModels.objgrad!(
  nlp::AbstractFluxNLPModel{T, S},
  w::AbstractVector{T},
  g::AbstractVector{T},
) where {T, S}
  @lencheck nlp.meta.nvar w g
  #both updates
  increment!(nlp, :neval_obj)
  increment!(nlp, :neval_grad)

  set_vars!(nlp, w)

  x, y = nlp.current_training_minibatch
  f_w = nlp.loss_f(nlp.chain(x), y)

  g = gradient(w_g->local_loss(nlp, w_g) , w)

  return f_w, g[1]
end
