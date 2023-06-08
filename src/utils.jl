"""
    accuracy(nlp::AbstractFluxNLPModel)

Compute the accuracy of the network `nlp.chain` on the entire test dataset.
"""
#TODO add the accuracy

"""
    set_vars!(model::AbstractFluxNLPModel{T,S}, new_w::AbstractVector{T}) where {T<:Number, S}

Sets the vaiables and rebuild the chain
"""
function set_vars!(nlp::AbstractFluxNLPModel{T, S}, new_w::AbstractVector{T}) where {T <: Number, S} #TODO test T 
    nlp.w .= new_w
    nlp.chain = nlp.rebuild(nlp.w)
end


function local_loss(nlp::AbstractFluxNLPModel{T, S},x,y, w::AbstractVector{T}) where {T, S}
    # increment!(nlp, :neval_obj) #TODO not sure 
    nlp.chain = nlp.rebuild(w)
    return nlp.loss_f(nlp.chain(x), y)
  end
