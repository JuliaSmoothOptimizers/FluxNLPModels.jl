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
  set_vars!(nlp, w)

  x, y = nlp.current_training_minibatch
  param = Flux.params(nlp.chain)
  gs = gradient(() -> nlp.loss_f(nlp.chain(x), y), param) # compute gradient  

  for p in param
    buff, re = Flux.destructure(gs[p])
    append!(g, buff)
  end
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

  x, y = nlp.current_training_minibatch
  f_w = nlp.loss_f(nlp.chain(x), y)

  param = Flux.params(nlp.chain)
  gs = gradient(() -> nlp.loss_f(nlp.chain(x), y), param) # compute gradient  #TODO maybe I use F_w

  for p in param
    buff, re = Flux.destructure(gs[p])
    append!(g, buff)
  end
  return f_w, g
end
