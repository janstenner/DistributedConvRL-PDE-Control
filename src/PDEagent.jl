using ReinforcementLearning
using Flux
using StableRNGs
using IntervalSets
using Setfield: @set
using Zygote: ignore
using CUDA

export create_agent, CustomDDPGPolicy


function create_NNA(;na, ns, use_gpu, is_actor, init, copyfrom = nothing, nna_scale, drop_middle_layer, learning_rate = 0.001)
    nna_size_actor = Int(floor(10 * nna_scale))
    nna_size_critic = Int(floor(20 * nna_scale))

    if is_actor
        if drop_middle_layer
            n = Chain(
                Dense(ns, nna_size_actor, relu; init = init),
                Dense(nna_size_actor, na, tanh; init = init),
            )
        else
            n = Chain(
                Dense(ns, nna_size_actor, relu; init = init),
                Dense(nna_size_actor, nna_size_actor, relu; init = init),
                Dense(nna_size_actor, na, tanh; init = init),
            )
        end
    else
        if drop_middle_layer
            n = Chain(
                Dense(ns + na, nna_size_critic, relu; init = init),
                Dense(nna_size_critic, 1; init = init),
            )
        else
            n = Chain(
                Dense(ns + na, nna_size_critic, relu; init = init),
                Dense(nna_size_critic, nna_size_critic, relu; init = init),
                Dense(nna_size_critic, 1; init = init),
            )
        end
    end

    nna = NeuralNetworkApproximator(
        model = use_gpu ? n |> gpu : n,
        optimizer = Flux.ADAM(learning_rate),
    )

    if copyfrom !== nothing
        copyto!(nna, copyfrom) 
    end

    nna
end

function create_agent(;action_space, state_space, use_gpu, rng, y, p, batch_size,
                    start_steps, start_policy, update_after, update_freq, act_limit, act_noise, nna_scale = 1, drop_middle_layer = false, memory_size = 0, trajectory_length = 1000, mono = false, learning_rate = 0.001)
    
    init = Flux.glorot_uniform(rng)
    
    behavior_actor = create_NNA(na = size(action_space)[1], ns = size(state_space)[1], use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, learning_rate = learning_rate)
    behavior_critic = create_NNA(na = size(action_space)[1], ns = size(state_space)[1], use_gpu = use_gpu, is_actor = false, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, learning_rate = learning_rate)
    target_actor = create_NNA(na = size(action_space)[1], ns = size(state_space)[1], use_gpu = use_gpu, is_actor = true, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, learning_rate = learning_rate)
    target_critic = create_NNA(na = size(action_space)[1], ns = size(state_space)[1], use_gpu = use_gpu, is_actor = false, init = init, nna_scale = nna_scale, drop_middle_layer = drop_middle_layer, learning_rate = learning_rate)

    copyto!(behavior_actor, target_actor)  # force sync
    copyto!(behavior_critic, target_critic)  # force sync

    if mono
        reward_size = 1
    else
        reward_size = size(action_space)[1]
    end
    
    Agent(
        policy = CustomDDPGPolicy(
            action_space = action_space,
            state_space = state_space,

            rng = rng,

            target_actor = target_actor,
            target_critic = target_critic,
            behavior_actor = behavior_actor,
            behavior_critic = behavior_critic,

            use_gpu = use_gpu,

            y = y,
            p = p,
            batch_size = batch_size,
            start_steps = start_steps,
            start_policy = start_policy,
            update_after = update_after,
            update_freq = update_freq,
            act_limit = act_limit,
            act_noise = act_noise,
            memory_size = memory_size,
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = trajectory_length,
            state = Float32 => size(state_space)[1],
            action = Float32 => size(action_space)[1],
            reward = Float32 => reward_size,
        ),
    )
end

Base.@kwdef mutable struct CustomDDPGPolicy{
    BA<:NeuralNetworkApproximator,
    BC<:NeuralNetworkApproximator,
    TA<:NeuralNetworkApproximator,
    TC<:NeuralNetworkApproximator,
    P,
    R,
} <: AbstractPolicy

    action_space::Space
    state_space::Space

    rng::R

    behavior_actor::BA
    behavior_critic::BC
    target_actor::TA
    target_critic::TC

    use_gpu::Bool

    y::Float32
    p::Float32
    batch_size::Int
    start_steps::Int
    start_policy::P
    update_after::Int
    update_freq::Int
    act_limit::Float64
    act_noise::Float64
    memory_size::Int

    update_step::Int = 0
    actor_loss::Float32 = 0.0f0
    critic_loss::Float32 = 0.0f0
end

Flux.functor(x::CustomDDPGPolicy) = (
    ba = x.behavior_actor,
    bc = x.behavior_critic,
    ta = x.target_actor,
    tc = x.target_critic,
),
y -> begin
    x = @set x.behavior_actor = y.ba
    x = @set x.behavior_critic = y.bc
    x = @set x.target_actor = y.ta
    x = @set x.target_critic = y.tc
    x
end


function (policy::CustomDDPGPolicy)(env; learning = true, test = false)
    if learning
        policy.update_step += 1
    end

    if policy.update_step <= policy.start_steps
        policy.start_policy(env)
    else
        D = device(policy.behavior_actor)
        s = DynamicStyle(env) == SEQUENTIAL ? state(env) : state(env, player)

        s = send_to_device(D, s)
        
        if test == false
            actions = policy.behavior_actor(s)
        else
            actions = zeros(size(policy.action_space))

            for i in 1:length(actions)
                actions[i] = policy.behavior_actor(s[:,i])[1]
            end
        end

        actions = actions |> send_to_host

        if learning
            actions[1:end-policy.memory_size,:] += randn(policy.rng, size(actions[1:end-policy.memory_size,:])) .* policy.act_noise
            actions = clamp.(actions, -policy.act_limit, policy.act_limit)
        else
            actions = clamp.(actions, -policy.act_limit, policy.act_limit)
        end

        actions
    end
end

function (policy::CustomDDPGPolicy)(stage::AbstractStage, env::AbstractEnv)
    nothing
end

function RLBase.update!(
    policy::CustomDDPGPolicy,
    traj::CircularArraySARTTrajectory,
    ::AbstractEnv,
    ::PostEpisodeStage,
)
    policy.update_step = 0
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    p::CustomDDPGPolicy,
    ::AbstractEnv,
    ::PreEpisodeStage,
)
    if length(trajectory) > 0
        for i in 1:size(p.state_space)[2]
            pop!(trajectory[:state])
            pop!(trajectory[:action])
            if haskey(trajectory, :legal_actions_mask)
                pop!(trajectory[:legal_actions_mask])
            end
        end
    end
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    policy::CustomDDPGPolicy,
    env::AbstractEnv,
    ::PreActStage,
    action,
)
    s = policy isa NamedPolicy ? state(env, nameof(policy)) : state(env)

    #We assume that action_space[2] == state_space[2]
    for i in 1:size(s)[2]
        push!(trajectory[:state], s[:,i])
        push!(trajectory[:action], action[:,i])
        if haskey(trajectory, :legal_actions_mask)
            lasm =
                policy isa NamedPolicy ? legal_action_space_mask(env, nameof(policy)) :
                legal_action_space_mask(env)
            push!(trajectory[:legal_actions_mask], lasm)
        end
    end
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    policy::CustomDDPGPolicy,
    env::AbstractEnv,
    ::PostActStage,
)
    r = policy isa NamedPolicy ? reward(env, nameof(policy)) : reward(env)

    #We assume that state_space[2] == size(r)[1]
    for i in 1:size(r)[1]
        push!(trajectory[:reward], r[i])
        push!(trajectory[:terminal], is_terminated(env))
    end
end

function RLBase.update!(
    trajectory::AbstractTrajectory,
    policy::CustomDDPGPolicy,
    env::AbstractEnv,
    ::PostEpisodeStage,
)
    # Note that for trajectories like `CircularArraySARTTrajectory`, data are
    # stored in a SARSA format, which means we still need to generate a dummy
    # action at the end of an episode.

    s = policy isa NamedPolicy ? state(env, nameof(policy)) : state(env)
    a = zeros(size(policy.action_space))

    for i in 1:size(s)[2]
        push!(trajectory[:state], s[:,i])
        push!(trajectory[:action], a[:,i])
        if haskey(trajectory, :legal_actions_mask)
            lasm =
                policy isa NamedPolicy ? legal_action_space_mask(env, nameof(policy)) :
                legal_action_space_mask(env)
            push!(trajectory[:legal_actions_mask], lasm)
        end
    end
end

function RLBase.update!(
    policy::CustomDDPGPolicy,
    traj::CircularArraySARTTrajectory,
    ::AbstractEnv,
    ::PreActStage,
)
    length(traj) > policy.update_after || return
    policy.update_step % policy.update_freq == 0 || return
    inds, batch = sample(policy.rng, traj, BatchSampler{SARTS}(policy.batch_size))
    update!(policy, batch)
end

function RLBase.update!(policy::CustomDDPGPolicy, batch::NamedTuple{SARTS})
    s, a, r, t, snext = batch

    A = policy.behavior_actor
    C = policy.behavior_critic
    Aₜ = policy.target_actor
    Cₜ = policy.target_critic

    y = policy.y
    p = policy.p

    # r = vec(r)'
    # new_t = Array{Float32}(undef, 1, 0)
    # for i in 1:length(t)
    #     new_t = hcat(new_t, t[i] * ones(size(policy.action_space)[2])')
    # end
    
    # s, a, r, t, snext = send_to_device(device(policy), (s, a, r, new_t, snext))

    s, a, r, t, snext = send_to_device(device(policy), (s, a, r, t, snext))

    anext = Aₜ(snext)
    qₜ = Cₜ(vcat(snext, anext)) |> vec
    #qₜ = - abs.( (s .* 50 - 30 .* Aₜ(snext)) / 50 ) |> vec
    qnext = r .+ y .* (1 .- t) .* qₜ
    a = Flux.unsqueeze(a, ndims(a)+1)

    gs1 = gradient(Flux.params(C)) do
        q = C(vcat(s, a)) |> vec
        loss = mean((qnext .- q) .^ 2)
        ignore() do
            policy.critic_loss = loss
        end
        loss
    end

    update!(C, gs1)

    gs2 = gradient(Flux.params(A)) do
        loss = -mean(C(vcat(s, A(s))))
        #loss = abs.( (s .* 500 - 30 .* A(s)) / 500 )[1]
        ignore() do
            policy.actor_loss = loss
        end
        loss
    end

    #Flux.Optimise.update!(Descent(), Flux.params(A), gs2)
    update!(A, gs2)

    # polyak averaging
    for (dest, src) in zip(Flux.params([Aₜ, Cₜ]), Flux.params([A, C]))
        dest .= p .* dest .+ (1 - p) .* src
    end
end

struct ZeroPolicy <: AbstractPolicy
    action_space
end

(p::ZeroPolicy)(env) = zeros(size(p.action_space))