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


# env parameters

te = 80.0
t0 = 0.0
dt = 0.1
oversampling = 15
min_best_episode = 10

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
nna_scale = 3.2
drop_middle_layer = true
temporal_steps = 1
action_punish = 0.0#0.0002#0.2
delta_action_punish = 0.0#0.0002#0.5
use_gpu = false
action_space = Space(fill(-1..1, (length(actuator_positions))))
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
batch_size = 10
start_steps = -1
start_policy = ZeroPolicy(action_space)
update_after = 100
update_freq = 10
act_limit = 1.0
act_noise = 0.1
trajectory_length = 700
learning_rate = 0.0001


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
    u = env.y

    Nn1  = G.*fft(u.*u) # -u u_x (spectral), notation Nn = N^n     = N(u(n dt))
    ufft  = fft(u)        # transform u to spectral

    Nn  = G.*fft(real(u).^2) # compute Nn = -u u_x

    ufft = A_inv .* (B .* ufft + dt32*Nn - dt2*Nn1 + dt * fft(env.p))

    real(ifft(ufft))
end

function do_step2(env)
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
        u .= A_inv .*(B .* u .+ dt32.*Nn .- dt2.*Nn1 + dt_oversample * fft(env.p)) 
    end
    
    IFFT!*u
    real(u)
end

function do_step_with_reward(env)
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

    D_reward = 0
    P_reward = 0

    dy1 = d1 .* u
    dy2 = d2 .* u
    u_temp = copy(u)

    for n = 1:oversampling
        Nn1 .= Nn   # shift nonlinear term in time
        Nn .= u         # put u into N in prep for comp of nonlineat
        
        IFFT!*Nn       # transform Nu to gridpt values, in place
        Nn .= Nn.*Nn   # collocation calculation of u^2
        FFT!*Nn        # transform Nu back to spectral coeffs, in place

        Nn .= G.*Nn

        # loop fusion! Julia translates the folling line of code to a single for loop. 
        u .= A_inv .*(B .* u .+ dt32.*Nn .- dt2.*Nn1 + dt_oversample * fft(env.p))

        dy1 .= d1 .* u
        dy2 .= d2 .* u
        u_temp .= u
        IFFT!*dy1
        IFFT!*dy2
        IFFT!*u_temp
        D_reward += (1/n) * (mean(real(dy2).^2) - D_reward)
        P_reward += (1/n) * (mean(real(dy1).^2) + mean(real(u_temp).*env.p) - P_reward)
    end

    reward = - abs( D_reward + P_reward ) * dt / length(actuator_positions)

    env.reward = [reward for i in actuator_positions]
    
    IFFT!*u
    real(u)
end

function reward_function(env)
    y = abs.(env.y)
    
    sensors = zeros(length(actuator_positions))

    #convolution
    for i in 1:length(actuator_positions)
        sensors[i] = dot(y, gaussians[actuators_to_sensors[i]]) / (max_value * 1600)
    end
    sensor_rewards = - abs.(sensors)

    # println(sensor_rewards)
    # println(- action_punish * (env.action[1,:] .* env.action[1,:]))
    # println(- delta_action_punish * (env.delta_action[1,:] .* env.delta_action[1,:]))
    # println("")

    return [mean(sensor_rewards - action_punish * env.action[:].^2 - delta_action_punish * env.delta_action[:].^2)]

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
        sensors[i] = dot(y, gaussians[i]) / max_value
    end

    result = sensors'
    result = result[:,actuators_to_sensors]
    result = reshape(result[:], (length(sensor_positions),1))

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
        p = p .+ agent_power * action[i] * gaussians_actuators[i]
    end

    return p
end


# PDEenv can also take a custom y0 as a parameter. Example: PDEenv(y0=y0_sawtooth, ...)
function initialize_setup(;use_random_init = false)

    global env = PDEenv(do_step = do_step2, 
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
                        act_limit = act_limit, 
                        act_noise = act_noise,
                        nna_scale = nna_scale,
                        drop_middle_layer = drop_middle_layer,
                        memory_size = memory_size,
                        trajectory_length = trajectory_length,
                        mono = true,
                        learning_rate = learning_rate)

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


function load(number = nothing)
    if isnothing(number)
        global hook = FileIO.load(dirpath * "/saves/hook.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/agent.jld2","agent")
        #global env = FileIO.load(dirpath * "/saves/env.jld2","env")
    else
        global hook = FileIO.load(dirpath * "/saves/hook$number.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/agent$number.jld2","agent")
        global env = FileIO.load(dirpath * "/saves/env$number.jld2","env")
    end
end

function save(number = nothing)
    isdir(dirpath * "/saves") || mkdir(dirpath * "./saves")

    if isnothing(number)
        FileIO.save(dirpath * "/saves/hook.jld2","hook",hook)
        FileIO.save(dirpath * "/saves/agent.jld2","agent",agent)
        FileIO.save(dirpath * "/saves/env.jld2","env",env)
    else
        FileIO.save(dirpath * "/saves/hook$number.jld2","hook",hook)
        FileIO.save(dirpath * "/saves/agent$number.jld2","agent",agent)
        FileIO.save(dirpath * "/saves/env$number.jld2","env",env)
    end
end

function train(use_random_init = false; loops = 20)
    hook.use_random_init = use_random_init
    outer_loops = loops

    for j = 1:outer_loops
        No_Episodes = 40
        inner_loops = 80
        agent.policy.act_noise = 0.8
        for i = 1:inner_loops
            println("")
            println(agent.policy.act_noise)
            run(agent, env, StopAfterEpisode(No_Episodes), hook)
            println(hook.bestreward)
            agent.policy.act_noise = agent.policy.act_noise * 0.9

            hook.rewards = clamp.(hook.rewards, -3000, 0)
        end
    end
end