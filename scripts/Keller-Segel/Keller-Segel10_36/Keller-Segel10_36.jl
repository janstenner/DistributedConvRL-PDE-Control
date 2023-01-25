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

#dir variable
dirpath = string(@__DIR__)
open(dirpath * "/.gitignore", "w") do io
    println(io, "frames/*")
    println(io, "video_output/*")
end

# env parameters

seed = 126

te = 8.0
t0 = 0.0
dt = 0.003
oversampling = 50
min_best_episode = 20

te_plot = 5.0
dt_plot = dt
t_action = 2.0
dt_slowmo = dt

Lx = 10.0
nx = 200
dx = Lx / nx
sim_space = Space(fill(-1..1, (2, nx)))
xx = collect(dx:dx:Lx)

# sensor positions - 
sensor_positions = collect(3:5:nx)
actuator_positions = sensor_positions[3:end-2]
#actuators_to_sensors = [1,4,7,10]
actuators_to_sensors = collect(3:38)

# agent tuning parameters
memory_size = 0
nna_scale = 2.0
temporal_steps = 2
action_punish = 0.0# 0.02
delta_action_punish = 0.0# 0.05
window_size = 5
use_gpu = false
action_space = Space(fill(-1..1, (1 + memory_size, length(actuator_positions))))
nu = 0.2
use_radau = true

y0_2D_standard = [4 <= i <= 44 ? 0.5 : 0.0 for i = 1:nx ]
y0_2D_standard = ones(nx)
y0_2D_standard = vcat(y0_2D_standard', (1.01) .* y0_2D_standard')

coeff_matrix = [-0.5/dx 0.0 0.5/dx;
                1.0/dx^2 -2.0/dx^2 1.0/dx^2]
        
δX = [1/dx 1/dx^2]

# additional agent parameters
rng = StableRNG(seed)
Random.seed!(seed)
y = 0.99f0
p = 0.995f0
batch_size = 100
start_steps = -1
start_policy = RandomPolicy(action_space; rng = rng)
update_after = 1
update_freq = 10
act_limit = 1.0
act_noise = 0.1
trajectory_length = 10_000


boundary_condition = "periodic"

function prepare_gaussians(sigma = 0.8)
    temp = []

    extra_length = 50

    t = collect(dx - extra_length*dx : dx : Lx + extra_length*dx)

    for (i, position) in enumerate(sensor_positions)
        p = exp.(-(((t .- position * dx) * 1 ).^2 / 2 * sigma^2 ))

        pleft = p[1 : extra_length]
        pright = p[extra_length + nx + 1 : end]
        p = p[extra_length + 1 : extra_length + nx]
        #p[end+1-length(pleft) : end] += pleft
        #p[1:length(pright)] += pright


        push!(temp, p)
    end

    return temp    
end

function prepare_rectangles(half_window_size = 2)
    temp = []

    t = collect(dx : dx : Lx)

    for (i, position) in enumerate(sensor_positions)
        p = zeros(length(t))

        p[position - half_window_size : position + half_window_size] .= 1.0

        push!(temp, p)
    end

    return temp    
end

gaussians = prepare_rectangles(2)
gaussians_actuators = gaussians[actuators_to_sensors]

kx = vcat(0:nx/2-1, 0, -nx/2+1:-1)  # integer wavenumbers: exp(2*pi*kx*x/L)
alpha = 2*pi*kx/Lx             # real wavenumbers:    exp(alpha*x)

Lu = ones(nx) - alpha.^2
Lv = ones(nx) + alpha.^2

FFT! = plan_fft!((1+0im)*ones(nx), flags=FFTW.ESTIMATE)
IFFT! = plan_ifft!((1+0im)*ones(nx), flags=FFTW.ESTIMATE)

d1 = 1im*alpha
d2 = - alpha.^2

function do_step_wrong(env)
    dt_oversample = dt / oversampling
    dt2  = dt_oversample/2
    dt32 = 3*dt_oversample/2
    Au_inv = (ones(nx) - dt2*Lu).^(-1)
    Av_inv = (ones(nx) - dt2*Lv).^(-1)
    Bu =  ones(nx) + dt2*Lu
    Bv =  ones(nx) + dt2*Lv

    u = env.y[1,:]
    v = env.y[2,:]
    u = (1+0im)*u
    v = (1+0im)*v

    u_pwr2 = u.^2
    FFT!*u
    FFT!*v

    u1 = d1.*u
    IFFT!*u1
    v1 = d1.*v
    IFFT!*v1
    v2 = d2.*v
    IFFT!*v2

    u1[1] = 0.0
    v1[end] = 0.0
    u1[1] = 0.0
    v1[end] = 0.0

    Nn_u = 5.6 * u1 .* v1 - 5.6 * u .* v2 - u_pwr2
    FFT!*Nn_u
    Nn_v = u
    Nn1_u  = copy(Nn_u)
    Nn1_v  = copy(Nn_v)

    for n = 1:oversampling
        Nn1_u .= Nn_u   # shift nonlinear term in time
        Nn1_v .= Nn_v

        u_pwr2 .= u
        IFFT!*u_pwr2
        u_pwr2 .= u_pwr2.*u_pwr2
        u1 .= d1.*u
        IFFT!*u1
        v1 .= d1.*v
        IFFT!*v1
        v2 .= d2.*v
        IFFT!*v2

        u1[1] = 0.0
        v1[end] = 0.0
        u1[1] = 0.0
        v1[end] = 0.0

        Nn_u = 5.6 * u1 .* v1 - 5.6 * u .* v2 - u_pwr2
        FFT!*Nn_u

        Nn_v .= u

        # loop fusion! Julia translates the folling line of code to a single for loop. 
        u .= Au_inv .*(Bu .* u .+ dt32.*Nn_u .- dt2.*Nn1_u)
        v .= Av_inv .*(Bv .* v .+ dt32.*Nn_v .- dt2.*Nn1_v + dt_oversample * fft(env.p))
    end
    
    IFFT!*u
    IFFT!*v
    vcat(real(u)', real(v)')
end

function f(y, p, t)
    u = y[1,:]
    v = y[2,:]

    U = [circshift(u, 1) u circshift(u, -1)]
    V = [circshift(v, 1) v circshift(v, -1)]

    U[1,1] = U[1,2]
    U[end,3] = U[end,2]
    V[1,1] = V[1,2]
    V[end,3] = V[end,2]

    du = coeff_matrix * U'
    dv = coeff_matrix * V'

    v .= dv[2,:] - v + u + p
    u .= du[2,:] + u - 5.6 * du[1,:] .* dv[1,:] - 5.6 * u .* dv[2,:] - u.^2

    vcat(u', v')
end

function do_step(env)
    tspan = (env.time, env.time + env.dt)
    prob = ODEProblem(f, env.y, tspan, env.p)
    sol = solve(prob, RK4(), reltol=1e-8, abstol=1e-8)
    last(sol.u)
end

function reward_function(env)
    y = env.y
    
    sensors = zeros(length(actuator_positions))

    #convolution
    for i in 1:length(actuator_positions)
        sensors[i] = dot(y[1,:].^2, gaussians[actuators_to_sensors[i]]) / 4
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

function featurize(y0 = nothing, t0 = nothing; env = nothing)
    if isnothing(env)
        y = y0
    else
        y = env.y
    end

    sensors = zeros(2, length(sensor_positions))

    #convolution
    for i in 1:length(sensor_positions)
        sensors[1, i] = dot(y[1,:], gaussians[i]) / 4
        sensors[2, i] = dot(y[2,:], gaussians[i]) / 4
    end

    window_half_size = Int(floor(window_size/2))
    result1 = vcat([circshift(sensors[1,:], i)' for i in 0-window_half_size:0+window_half_size]...)
    result1 = result1[:,actuators_to_sensors]
    result2 = vcat([circshift(sensors[2,:], i)' for i in 0-window_half_size:0+window_half_size]...)
    result2 = result2[:,actuators_to_sensors]

    result = vcat(result1, result2)

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
        p = p .+ 10 * action[1,i] * gaussians_actuators[i]
    end

    return p
end


# PDEenv can also take a custom y0 as a parameter. Example: PDEenv(y0=y0_sawtooth, ...)
function initialize_setup(;use_random_init = false)

    global env = PDEenv(do_step = do_step,
                reward_function = reward_function,
                featurize = featurize,
                prepare_action = prepare_action,
                y0 = y0_2D_standard,
                te = te, t0 = t0, dt = dt, 
                sim_space = sim_space, 
                action_space = action_space,
                oversampling = oversampling,
                use_radau = use_radau)

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
                        memory_size = memory_size,
                        trajectory_length = trajectory_length)

    global hook = PDEhook(min_best_episode = min_best_episode, use_random_init = use_random_init)
end

function generate_random_init()
    number_sin = Int(ceil(Lx/3))
    a_i = rand(Uniform(-1, 1), number_sin*2)
    a_i = a_i / norm(a_i)
    y0 = ones(2, nx)
    x_axis = collect(dx:dx:Lx)
    for i in 1:number_sin
        y0[1,:] += 1 * a_i[i] * sin.(i * x_axis / (2*pi*(Lx/22)))
        y0[2,:] += 1 * a_i[i + number_sin] * sin.(i * x_axis / (2*pi*(Lx/22)))
    end
    y0
end

initialize_setup()

# plotrun(use_best = false, plot3D = true)

function train(use_random_init = false)
    hook.use_random_init = use_random_init

    while true
        No_Episodes = 50
        loops = 14
        agent.policy.act_noise = 0.15
        for i = 1:loops
            run(agent, env, StopAfterEpisode(No_Episodes), hook)
            println(hook.bestreward)
            agent.policy.act_noise = agent.policy.act_noise * 0.9
            println(agent.policy.act_noise)

            hook.rewards = clamp.(hook.rewards, -3000, 0)
        end
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
#plot_heat(p_te = 200.0, p_t_action = 100.0, use_best = false)
#plot_heat(plot_best = true)


function load(number = nothing)
    if isnothing(number)
        global hook = FileIO.load(dirpath * "/saves/hook.jld2","hook")
        global agent = FileIO.load(dirpath * "/saves/agent.jld2","agent")
        global env = FileIO.load(dirpath * "/saves/env.jld2","env")
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
