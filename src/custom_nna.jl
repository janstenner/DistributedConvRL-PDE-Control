using Random
using Setfield: @set
using Zygote: ignore
import CUDA: device
using MacroTools: @forward

Base.@kwdef struct CustomNeuralNetworkApproximator{M,O} <: AbstractApproximator
    model::M
    optimizer::O = nothing
end

# some model may accept multiple inputs
(app::CustomNeuralNetworkApproximator)(args...; kwargs...) = app.model(args...; kwargs...)

@forward CustomNeuralNetworkApproximator.model Flux.testmode!,
Flux.trainmode!,
Flux.params,
device

functor(x::CustomNeuralNetworkApproximator) =
    (model=x.model,), y -> CustomNeuralNetworkApproximator(y.model, x.optimizer)

RLBase.update!(app::CustomNeuralNetworkApproximator, gs) =
    Flux.Optimise.update!(app.optimizer, Flux.params(app), gs)

Base.copyto!(dest::CustomNeuralNetworkApproximator, src::CustomNeuralNetworkApproximator) =
    Flux.loadparams!(dest.model, Flux.params(src))