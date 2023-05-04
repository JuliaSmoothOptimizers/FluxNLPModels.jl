"""
    f = obj(nlp, x)

Evaluate `f(x)`, the objective function of `nlp` at `x`.
"""
function NLPModels.obj(nlp::AbstractFluxNLPModel{T, S}, w::AbstractVector{T}) where {T, S}
  increment!(nlp, :neval_obj)
  set_vars!(nlp, w)
  #TODO check If I need to reconstruct it 
  f_w = nlp.chain(nlp.current_training_minibatch)
  return f_w
end

"""
    g = grad!(nlp, x, g)

Evaluate `∇f(x)`, the gradient of the objective function at `x` in place.
"""
function NLPModels.grad!(
  nlp::AbstractFluxNLPModel{T, S}, #TODO add loss function to NLP
  w::AbstractVector{T},
  g::AbstractVector{T},
) where {T, S}
  @lencheck nlp.meta.nvar w g
  increment!(nlp, :neval_grad)
  set_vars!(nlp, w)  #TODO does it update it ? or does it do   m = rebuild(flat) # we can rebuild with a new weights here we use flat wigth
  
  x,y = nlp.current_training_minibatch #TODO check this
  param = Flux.params(nlp.chain) # model's trainable parameters, #TODO I need to improve this 
  gs = gradient(() -> nlp.loss(nlp.chain(x), y), param) # compute gradient
  
  for p in param
      buff , re  = Flux.destructure(gs[p])
      append!(g, buff);    
  end

  return g
end
