using LinearAlgebra
using ReinforcementLearning
using IntervalSets
using StableRNGs
using PlotlyJS
using Blink
using ReinforcementLearning:update!
import DataStructures as DS
using FileIO, JLD2
using FFTW
using Random, Distributions

include(pwd() * "/src/PDEagent.jl")
include(pwd() * "/src/PDEenv.jl")
include(pwd() * "/src/PDEhook.jl")
include(pwd() * "/src/plotting.jl")
include(pwd() * "/src/StopCondition.jl")


te = 5.0
t0 = 0.0
dt = 0.1
oversampling = 30
min_best_episode = 1

te_plot = 50.0
dt_plot = dt
t_action = 20.0
dt_slowmo = dt

check_max_value = "y"
max_value = 30.0

dx = Lx / nx
sim_space = Space(fill(-1..1, (nx)))
xx = collect(dx:dx:Lx)

# agent tuning parameters
memory_size = 0
nna_scale = 0.6
nna_scale_critic = 7.0
drop_middle_layer = true
temporal_steps = 1
action_punish = 0.002#0.2
delta_action_punish = 0.002#0.5
window_size = 1
use_gpu = false
action_space = Space(fill(-1..1, (1 + memory_size, length(actuator_positions))))
nu = 0.2
use_radau = true
agent_power = 7.5

y0_1D_standard = [4 <= i <= 44 ? 0.5 : 0.0 for i = 1:nx ]

coeff_matrix = [0.0 -0.5 0.0 0.5 0.0;
                0.0 1.0 -2.0 1.0 0.0;
                1.0 -4.0 6.0 -4.0 1.0]
        
δX = [1/dx 1/dx^2 1/dx^4]

# additional agent parameters
rng = StableRNG(seed)
Random.seed!(seed)
y = 0.99f0
p = 0.995f0
batch_size = 3
start_steps = 6
start_policy = ZeroPolicy(action_space)
update_after = 10
update_freq = 1
update_loops = 20
reset_stage = POST_EPISODE_STAGE
learning_rate = 0.0005
learning_rate_critic = 0.001
act_limit = 1.0
act_noise = 1.2
trajectory_length = 150_000


boundary_condition = "periodic"

function prepare_gaussians(sigma = 0.8; norm_mode = 1)
    temp = []

    extra_length = 50

    t = collect(dx - extra_length*dx : dx : Lx + extra_length*dx)

    for (i, position) in enumerate(sensor_positions)
        p = (1 / (sqrt(2 * pi * sigma))) * exp.(-(((t .- position * dx) * 1 ).^2 / 2 * sigma^2 ))

        if norm_mode == 1
            p ./= sum(p)
        else
            p ./= maximum(p)
        end

        pleft = p[1 : extra_length]
        pright = p[extra_length + nx + 1 : end]
        p = p[extra_length + 1 : extra_length + nx]
        p[end+1-length(pleft) : end] += pleft
        p[1:length(pright)] += pright


        push!(temp, p)
    end

    return temp    
end

gaussians = prepare_gaussians(sigma_sensors)
gaussians_actuators = prepare_gaussians(sigma_actuators; norm_mode = 2)
gaussians_actuators = gaussians_actuators[actuators_to_sensors]

kx = vcat(0:nx/2-1, 0, -nx/2+1:-1)  # integer wavenumbers: exp(2*pi*kx*x/L)
alpha = 2*pi*kx/Lx             # real wavenumbers:    exp(alpha*x)
D = 1im*alpha                  # D = d/dx operator in Fourier space
L = alpha.^2 - alpha.^4         # linear operator -D^2 - D^4 in Fourier space
G = -0.5*D
dt2  = dt/2
dt32 = 3*dt/2
A_inv = (ones(nx) - dt2*L).^(-1)
B =  ones(nx) + dt2*L
FFT! = plan_fft!((1+0im)*ones(nx), flags=FFTW.ESTIMATE)
IFFT! = plan_ifft!((1+0im)*ones(nx), flags=FFTW.ESTIMATE)

d1 = 1im*alpha
d2 = - alpha.^2

function do_step(env)
    dt_oversample = dt / oversampling
    dt2  = dt_oversample/2
    dt32 = 3*dt_oversample/2
    A_inv = (ones(nx) - dt2*L).^(-1)
    B =  ones(nx) + dt2*L

    u = env.y
    u = (1+0im)*u       # force u to be complex

    Nn = G.*fft(u.^2)
    Nn1  = copy(Nn)  # -u u_x (spectral), notation Nn = N^n     = N(u(n dt))
    FFT!*u        # transform u to spectral

    for n = 1:oversampling
        Nn1 .= Nn   # shift nonlinear term in time
        Nn .= u         # put u into N in prep for comp of nonlineat
        
        IFFT!*Nn       # transform Nu to gridpt values, in place
        Nn .= Nn.*Nn   # collocation calculation of u^2
        FFT!*Nn        # transform Nu back to spectral coeffs, in place

        Nn .= G.*Nn

        # loop fusion! Julia translates the folling line of code to a single for loop. 
        u .= A_inv .*(B .* u .+ dt32.*Nn .- dt2.*Nn1 + dt_oversample * fft(env.p)) + dt_oversample * fft(μ * cos.((2+pi.+xx/(Lx/2))))
    end
    
    IFFT!*u
    real(u)
end

function reward_function(env)
    y = env.y .* 6
    
    sensors = zeros(length(actuator_positions))

    #convolution
    for i in 1:length(actuator_positions)
        sensors[i] = abs.(dot(y, gaussians[actuators_to_sensors[i]])).^1.3 / (max_value * 3)
    end
    sensor_rewards = - abs.(sensors)

    # println(sensor_rewards)
    # println(- action_punish * (env.action[1,:] .* env.action[1,:]))
    # println(- delta_action_punish * (env.delta_action[1,:] .* env.delta_action[1,:]))
    # println("")

    return sensor_rewards - action_punish * env.action[1,:].^2 - delta_action_punish * env.delta_action[1,:].^2

    #return 0.3 * sensor_rewards .- 0.008 * sum(y.*env.p) / 4
    #return 0.3 * sensor_rewards .- (D + P)

    #return [- abs(D + P) * dt / length(actuator_positions) for i in actuator_positions]
end

function reward_function_null(env)
    return env.reward
end

function featurize(y0 = nothing, t0 = nothing; env = nothing)
    if isnothing(env)
        y = y0
    else
        y = env.y
    end

    sensors = zeros(length(sensor_positions))

    #convolution
    for i in 1:length(sensor_positions)
        sensors[i] = dot(y, gaussians[i]) / (max_value)
    end

    window_half_size = Int(floor(window_size/2))
    result = vcat([circshift(sensors, i)' for i in 0-window_half_size:0+window_half_size]...)

    result = result[:,actuators_to_sensors]

    if temporal_steps > 1
        if isnothing(env)
            resulttemp = result
            for i in 1:temporal_steps-1
                result = vcat(result, resulttemp)
            end
        else
            result = vcat(result, env.state[1:end-size(result)[1]-memory_size,:])
        end
    end

    if memory_size > 0
        if isnothing(env)
            result = vcat(result, zeros(memory_size, length(actuator_positions)))
        else
            result = vcat(result, env.action[end-(memory_size-1):end,:])
        end
    end

    return result
end

function prepare_action(action0 = nothing, t0 = nothing; env = nothing) 
    if isnothing(env)
        action =  action0
    else
        action = env.action
    end

    p = zeros(nx)

    for i in 1:length(actuator_positions)
        p = p .+ agent_power * action[1,i] * gaussians_actuators[i]
    end

    return p
end


# PDEenv can also take a custom y0 as a parameter. Example: PDEenv(y0=y0_sawtooth, ...)
function initialize_setup(;use_random_init = false)

    global env = PDEenv(do_step = do_step, 
                reward_function = reward_function,
                featurize = featurize,
                prepare_action = prepare_action,
                y0 = y0_1D_standard,
                te = te, t0 = t0, dt = dt, 
                sim_space = sim_space, 
                action_space = action_space,
                oversampling = oversampling,
                use_radau = use_radau,
                max_value = max_value,
                check_max_value = check_max_value)

    global agent = create_agent(action_space = action_space,
                        state_space = env.state_space,
                        use_gpu = use_gpu, 
                        rng = rng,
                        y = y, p = p, batch_size = batch_size, 
                        start_steps = start_steps, 
                        start_policy = start_policy,
                        update_after = update_after, 
                        update_freq = update_freq,
                        update_loops = update_loops,
                        reset_stage = reset_stage,
                        act_limit = act_limit, 
                        act_noise = act_noise,
                        nna_scale = nna_scale,
                        nna_scale_critic = nna_scale_critic,
                        drop_middle_layer = drop_middle_layer,
                        memory_size = memory_size,
                        trajectory_length = trajectory_length,
                        learning_rate = learning_rate,
                        learning_rate_critic = learning_rate_critic)

    global hook = PDEhook(min_best_episode = min_best_episode, use_random_init = use_random_init)
end

function generate_random_init()
    number_sin = 8
    a_i = rand(Uniform(-1, 1), number_sin)
    a_i = a_i / norm(a_i)
    y0 = zeros(nx)
    x_axis = collect(dx:dx:Lx)
    for i in 1:number_sin
        y0 += a_i[i] * sin.(i * x_axis / (2*pi))
    end
    y0 = y0 * 30 / norm(y0)
end

initialize_setup()

# plotrun(use_best = false, plot3D = true)

function train(use_random_init = true; loops = 8)
    hook.use_random_init = use_random_init

    No_Steps = 800

    agent.policy.act_noise = act_noise
    for i = 1:loops
        println("")
        println(agent.policy.act_noise)
        run(agent, env, StopAfterEpisodeWithMinSteps(No_Steps), hook)
        println(hook.bestreward)
        agent.policy.act_noise = agent.policy.act_noise * 0.2

        hook.rewards = clamp.(hook.rewards, -3000, 0)
    end
end

function train_multi(No_Episodes = 2800)
    global best_rewards = []
    best_rewards = Vector{Float64}(best_rewards)
    n_experiment = 0

    while true
        n = 1
        n_experiment += 1
        global rng = StableRNG(abs(rand(Int)))
        initialize_setup(use_random_init = true)

        println("")
        println("--------- STARTING EXPERIMENT # $n_experiment ---------")
        println("")

        while n <= No_Episodes
            inner_No_Episodes = 50
            loops = 14
            agent.policy.act_noise = 0.15
            i = 1
            while i <= loops && n <= No_Episodes
                run(agent, env, StopAfterEpisode(inner_No_Episodes), hook)
                println(hook.bestreward)
                agent.policy.act_noise = agent.policy.act_noise * 0.9
                println(agent.policy.act_noise)

                hook.rewards = clamp.(hook.rewards, -3000, 0)
                n += inner_No_Episodes
                i += inner_No_Episodes
            end
        end

        push!(best_rewards, hook.bestreward)
        save(n_experiment)

        FileIO.save(dirpath * "/saves/best_rewards.jld2","best_rewards",best_rewards)
        # a = FileIO.load("./scripts/KS_spectral/sparse_8/saves/best_rewards.jld2","best_rewards")

        println("")
        println("--------- BEST REWARD: $(hook.bestreward) ---------")
        println("")
    end
end

#train()

#plotrun(use_best = false)
#plotrun()
#plotrun(plot_best = true)
#plotrun(plot3D = true)

#plot_heat(p_te = 200.0, p_t_action = 100.0)
#plot_heat(p_te = 200.0, p_t_action = 100.0, plot_separate = true, from = 90, to = 115)
#plot_heat(p_te = 200.0, p_t_action = 100.0, use_best = false)
#plot_heat(plot_best = true)


function load(number = nothing)
    if isnothing(number)
        global hook = FileIO.load(dirpath * "/saves/hook.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/agent.jld2","agent")
        #global env = FileIO.load(dirpath * "/saves/env.jld2","env")
    else
        global hook = FileIO.load(dirpath * "/saves/hook$number.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/agent$number.jld2","agent")
        #global env = FileIO.load(dirpath * "/saves/env$number.jld2","env")
    end
end

function save(number = nothing)
    isdir(dirpath * "/saves") || mkdir(dirpath * "/saves")

    if isnothing(number)
        FileIO.save(dirpath * "/saves/hook.jld2","hook",hook)
        FileIO.save(dirpath * "/saves/agent.jld2","agent",agent)
        #FileIO.save(dirpath * "/saves/env.jld2","env",env)
    else
        FileIO.save(dirpath * "/saves/hook$number.jld2","hook",hook)
        FileIO.save(dirpath * "/saves/agent$number.jld2","agent",agent)
        #FileIO.save(dirpath * "/saves/env$number.jld2","env",env)
    end
end

#FileIO.save("KSResults.jld2", "KSResults", KSResults)
#global KSResults = FileIO.load("KSResults.jld2", "KSResults")

