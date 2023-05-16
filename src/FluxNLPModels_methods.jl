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
  f_w = nlp.chain(nlp.current_training_minibatch)
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
  set_vars!(nlp, w)
  g = flat_grad!(nlp, g)
  return g
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
  f_w = nlp.chain(nlp.current_training_minibatch)
  g = flat_grad!(nlp, g)
  return f_w, g
end
