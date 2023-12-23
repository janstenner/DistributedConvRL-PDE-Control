using ReinforcementLearning
using DataFrames
using Statistics
using UnicodePlots:lineplot

export PDEhook

Base.@kwdef mutable struct PDEhook <: AbstractHook

    rewards::Vector{Float64} = Float64[]
    rewards_compare::Vector{Float64} = Float64[]
    reward::Float64 = 0.0
    ep = 1

    is_display_on_exit::Bool = true
    use_random_init::Bool = false
    collect_history::Bool = false
    collect_NNA::Bool = true
    collect_bestDF::Bool = true

    min_best_episode = 0
    bestNNA = nothing
    bestDF = DataFrame()
    bestreward = -1000000.0
    bestepisode = 0
    currentNNA = nothing
    currentDF = DataFrame()
    history = []
    errored_episodes = []
    error_detection = function(y) return false end
end

Base.getindex(h::PDEhook) = h.rewards

function (hook::PDEhook)(::PreExperimentStage, agent, env)
    if hook.collect_NNA && hook.currentNNA === nothing
        hook.currentNNA = deepcopy(agent.policy.behavior_actor)
        hook.bestNNA = deepcopy(agent.policy.behavior_actor)
    end
end

function (hook::PDEhook)(::PreEpisodeStage, agent, env)
    if hook.use_random_init
        env.y0 = generate_random_init()
        env.y = env.y0

        env.state = env.featurize(; env = env)
    end
end

function (hook::PDEhook)(::PostActStage, agent, env)
    hook.reward += mean(reward(env))

    if hook.collect_bestDF
        tmp = DataFrame()
        insertcols!(tmp, :timestep => env.steps)
        insertcols!(tmp, :action => [vec(env.action)])
        insertcols!(tmp, :p => [send_to_host(env.p)])
        insertcols!(tmp, :y => [send_to_host(env.y)])
        insertcols!(tmp, :reward => [reward(env)])
        append!(hook.currentDF, tmp)
    end
end

function (hook::PDEhook)(::PostEpisodeStage, agent, env)
    if env.time >= env.te && hook.ep >= hook.min_best_episode
        push!(hook.rewards_compare, hook.reward)
        if hook.collect_NNA && length(hook.rewards_compare) >= 1 && hook.reward >= maximum(hook.rewards_compare)
            copyto!(hook.bestNNA, agent.policy.behavior_actor)
            hook.bestreward = hook.reward
            hook.bestepisode = hook.ep
            if hook.collect_bestDF
                hook.bestDF = copy(hook.currentDF)
            end
        end
    end

    if env.time < env.te
        if hook.error_detection(env.y)
            push!(hook.errored_episodes, hook.ep)
        end
    end
    
    if hook.collect_history
        push!(hook.history, hook.currentDF)
    end
    hook.currentDF = DataFrame()

    hook.ep += 1

    push!(hook.rewards, hook.reward)
    hook.reward = 0
    
    if hook.collect_NNA
        copyto!(hook.currentNNA, agent.policy.behavior_actor)
    end
end

function (hook::PDEhook)(::PostExperimentStage, agent, env)
    if hook.is_display_on_exit && !isempty(hook.rewards)
        println(lineplot(hook.rewards, title="Total reward per episode", xlabel="Episode", ylabel="Score"))
    end
end