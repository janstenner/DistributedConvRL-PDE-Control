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
using SparseArrays

#--- disable printing of warnings thrown by SciMLBase about backwards compatibility of return codes that will be deprecated in Julia 1.9
import Logging
Logging.disable_logging(Logging.Warn)

include(pwd() * "/src/fluid_rk4.jl")
include(pwd() * "/src/PDEagent.jl")
include(pwd() * "/src/PDEenv.jl")
include(pwd() * "/src/PDEhook.jl")
include(pwd() * "/src/plotting.jl")
include(pwd() * "/src/StopCondition.jl")


# train till 6 is enough for 32 and 16

nu = 0.00005

Lx  = 1;            
Ly  = 1;
nx  = 128;
if evaluation
    seed = 76
    nx  = 256
end  
ny  = nx;
dx  = Lx/nx;        
dy  = Ly/ny;
sim_space = Space(fill(-1..1, (nx, ny)))



te = 6.0
t0 = 0.0
dt = 0.02
oversampling = floor(16*nx*dt)

min_best_episode = 1

te_plot = 10.0
dt_plot = dt
t_action = 2.0
dt_slowmo = dt

check_max_value = "reward"
max_value = 3.0


# sensor positions - 
sensor_positions = [[i,j] for i in 1:Int(nx/sensors_per_axis):nx for j in 1:Int(ny/sensors_per_axis):ny]
actuator_positions = [[i,j] for i in 1:Int(nx/sensors_per_axis):nx for j in 1:Int(ny/sensors_per_axis):ny]
actuators_to_sensors = collect(1:length(sensor_positions))

# agent tuning parameters
memory_size = 0
nna_scale = 1.8
nna_scale_critic = 17.0
drop_middle_layer = true
temporal_steps = 1
action_punish = 0.002#0.0005
delta_action_punish = 0.002#0.0005
window_size = 3
use_gpu = false
action_space = Space(fill(-1..1, (1 + memory_size, length(actuator_positions))))
use_radau = true
agent_power = 70.0

# additional agent parameters
rng = StableRNG(seed)
Random.seed!(seed)
y = 0.99f0
p = 0.995f0
batch_size = 3
start_steps = 10
start_policy = ZeroPolicy(action_space)
update_after = 10
update_freq = 1
update_loops = 20
reset_stage = POST_EPISODE_STAGE
learning_rate = 0.0005
learning_rate_critic = 0.001
act_limit = 1.0
act_noise = 1.2
trajectory_length = 1_800_000


## For Dealiasing [padding] in wavespace
# -- use ifpad = 0 for no de-aliasing
# -- use ifpad = 1 to invoke de-aliasing
ifpad = 1

nxp = nx*3/2; nyp = ny*3/2; # padded

## Fourier wavenumber
kx  = [0:(nx/2); (-nx/2 + 1):(-1)]/Lx*2*pi;
ky  = [0:(ny/2); (-ny/2 + 1):(-1)]/Ly*2*pi;
kxp = [0:(nxp/2); (-nxp/2 + 1):(-1)]/Lx*2*pi;
kyp = [0:(nyp/2); (-nyp/2 + 1):(-1)]/Ly*2*pi;

kx2 = kx.^2
ky2 = ky.^2
kxp2 = kxp.^2
kyp2 = kyp.^2

kx2ky2 = [ky2[i] + kx2[j] for i in 1:nx, j in 1:ny]
kx_repeat = vcat([kx' for i in 1:ny]...)
ky_repeat = hcat([ky for i in 1:nx]...)

if gpu_env
    kx2ky2 = CuArray(kx2ky2)
    kx_repeat = CuArray(kx_repeat)
    ky_repeat = CuArray(ky_repeat)
end

## Grid generation
x1 = range(0,Lx,length=nx+1); 
y1 = range(0,Ly,length=ny+1); 

x1 = x1[1:nx]
y1 = y1[1:ny]

xx,yy = meshgrid(x1,y1);

y0_2D_standard = ic(4, rng)
y0_2D_standard = gpu_env ? CuArray(y0_2D_standard) : y0_2D_standard


function prepare_gaussians(;variance = 0.04, norm_mode = 1)
    temp = []

    for (i, position) in enumerate(sensor_positions)
        
        p = real(ifft(taylorvtx(position[1]*dx - dx, position[2]*dy - dy, variance, 1.0)))
        p[p.<0.1] .= 0.0
        if norm_mode == 1
            p ./= sum(p)
        else
            p ./= maximum(p)
        end
        p = sparse(p)

        push!(temp, gpu_env ? CuArray(p) : p)
    end

    return temp    
end

gaussians = prepare_gaussians(variance = variance)
gaussians_actuators = prepare_gaussians(variance = variance, norm_mode = 2)
gaussians_actuators = gaussians_actuators[actuators_to_sensors]

function do_step(env)
    dt_oversample = dt / oversampling
    y = env.y

    for n = 1:oversampling
        y = rk4(y, env.p, dt_oversample)
    end

    env.y = y
end

function f(y, p, t)
    rhs(y, p)
end

tol = 1e-8
tol = 1e0

function do_step2(env)
    tspan = (env.time, env.time + env.dt)
    prob = ODEProblem(f, env.y, tspan, env.p)
    sol = solve(prob, RK4(), reltol=tol, abstol=tol)
    last(sol.u)
end

function reward_function(env)
    y = real(ifft(env.y))
    
    sensors = zeros(length(actuator_positions))

    #y = abs.(y)
    #factor = 10_000 * (nx/100) * (ny/100)
    #convolution
    for i in 1:length(actuator_positions)
        sensors[i] = send_to_host(abs.(dot(y, gaussians[actuators_to_sensors[i]])).^1.1 ./ (320))
    end
    sensor_rewards = - abs.(sensors)

    return sensor_rewards - action_punish * env.action[1,:].^2 - delta_action_punish * env.delta_action[1,:].^2
end

function featurize(y0 = nothing, t0 = nothing; env = nothing)
    if isnothing(env)
        y = real(ifft(y0))
    else
        y = real(ifft(env.y))
    end

    sensors = zeros(sensors_per_axis, sensors_per_axis)
    #factor = 4096 / (nx*ny)

    #convolution
    for i in 1:length(sensor_positions)
        sensors[Int(floor((i-1) / sensors_per_axis) + 1), Int((i-1) % sensors_per_axis) + 1] = send_to_host(dot(y, gaussians[i]) ./ 70)
    end

    window_half_size = Int(floor(window_size/2))
    result = vcat([reshape(circshift(sensors, [i,j])',(1,length(sensor_positions)))
                                                for i in 0-window_half_size:0+window_half_size
                                                for j in 0-window_half_size:0+window_half_size]...)
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

    p = gpu_env ? CuArray(zeros(nx,ny)) : zeros(nx,ny)

    for i in 1:length(actuator_positions)
       p = p .+ agent_power * action[1,i] * gaussians_actuators[i]
    end

    return fft(p)
end

function error_detection(y)
    y = real(ifft(y))
    y_x = abs.(circshift(y, [1,0]) - y)
    y_y = abs.(circshift(y, [0,1]) - y)

    if maximum(y_x) > 10.0 || maximum(y_y) > 10.0
        return true
    end

    return false
end



mutable struct NegatePolicy <: AbstractPolicy
    action_space
    start_steps
    start_policy
    update_step
end

function (p::NegatePolicy)(env) 
    
    p.update_step += 1

    if p.update_step <= p.start_steps
        result = p.start_policy(env)
    else
        result = zeros(size(p.action_space))
        for i in 1:length(result)
            result[i] = - env.state[i]
        end

        result = clamp.(result, -1.0, 1.0)
    end

    result
end

function RLBase.update!(
    policy::NegatePolicy,
    traj::CircularArraySARTTrajectory,
    ::AbstractEnv,
    ::PostEpisodeStage,
)
    policy.update_step = 0
end

function create_agent_negate(;action_space, state_space, start_steps = 0, start_policy, trajectory_length = 1)
    Agent(
        policy = NegatePolicy(
            action_space,
            start_steps,
            start_policy,
            0
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = trajectory_length,
            state = Float32 => size(state_space),
            action = Float32 => size(action_space),
            reward = Float32 => size(action_space)[2],
        ),
    )
end



# PDEenv can also take a custom y0 as a parameter. Example: PDEenv(y0=y0_sawtooth, ...)
function initialize_setup(;use_random_init = false)

    global env = PDEenv(do_step = do_step2,
                reward_function = reward_function,
                featurize = featurize,
                prepare_action = prepare_action,
                y0 = y0_2D_standard,
                te = te, t0 = t0, dt = dt, 
                sim_space = sim_space, 
                action_space = action_space,
                oversampling = oversampling,
                use_radau = use_radau,
                max_value = max_value,
                check_max_value = check_max_value,
                use_gpu = gpu_env)

    global agent_negate = create_agent_negate(action_space = action_space,
                                            state_space = env.state_space,
                                            start_steps = start_steps,
                                            start_policy = start_policy)

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

    global hook = PDEhook(min_best_episode = min_best_episode,
                        use_random_init = use_random_init,
                        collect_history = false,
                        collect_bestDF = false,
                        error_detection = error_detection)

    global hook2 = PDEhook(min_best_episode = 0,
                    use_random_init = false,
                    collect_history = false,
                    collect_bestDF = false,
                    collect_NNA = false)
end

function generate_random_init()
    if evaluation
        result = ic(4, rng)
    else
        result = ic(3, rng)
    end
    result = gpu_env ? CuArray(result) : result
    result
end

initialize_setup()

# plotrun(use_best = false, plot3D = true)

function testrun(;viz = true, use_best = false, negate = false, no_action = false, video = false)
    if negate
        temp_start_steps = agent_negate.policy.start_steps
        agent_negate.policy.start_steps  = 50
    end

    if no_action
        negate = true
        temp_start_steps = agent_negate.policy.start_steps
        agent_negate.policy.start_steps = 10000
    end

    hook.use_random_init = true

    if use_best
        copyto!(agent.policy.behavior_actor, hook.bestNNA)
        temp_start_steps = agent.policy.start_steps
        agent.policy.start_steps  = 50
        
        temp_update_after = agent.policy.update_after
        agent.policy.update_after = 10000

        temp_noise = agent.policy.act_noise
        agent.policy.act_noise = 0.0
    end

    if video
        rm(dirpath * "/frames/", recursive=true, force=true)
        mkdir(dirpath * "/frames")
    end

    use_best && (agent.policy.update_step = 0)
    negate && (agent_negate.policy.update_step = 0)
    (use_best || negate) && (global energy = [])
    (use_best || negate) && (energy_sum = 0.0)

    (use_best || negate) || hook(PRE_EXPERIMENT_STAGE, agent, env)
    (use_best || negate) || agent(PRE_EXPERIMENT_STAGE, env)

    if viz || video
        if viz
            w = Window()
        end
        colorscale = [[0, "blue"], [0.5, "yellow"], [1, "red"], ]
        ymax = 30
        layout = Layout(
                plot_bgcolor="#f1f3f7",
                coloraxis = attr(cmin = -ymax, cmid = 0, cmax = ymax, colorscale = colorscale),
            )
    end

    loop_end_outer = (use_best || negate) ? 1 : 5
    loop_end_inner = (use_best || negate)  ? 1 : 3
    (use_best || negate)  || (agent.policy.act_noise = act_noise)

    for j in 1:loop_end_outer
        for i in 1:loop_end_inner
            reset!(env)
            (use_best || negate) || agent(PRE_EPISODE_STAGE, env)
            (use_best || negate) || hook(PRE_EPISODE_STAGE, agent, env)

            #env.y = yyy

            if viz || video
                p = plot(heatmap(z=send_to_host(real(ifft(env.y))), coloraxis="coloraxis"), layout)
                if viz
                    body!(w,p)
                end
            end

            n = 1
            while !is_terminated(env)
                negate ? action = agent_negate(env) : action = agent(env)

                #println(action)
                (use_best || negate) || agent(PRE_ACT_STAGE, env, action)
                (use_best || negate) || hook(PRE_ACT_STAGE, agent, env, action)

                env(action)

                if viz || video
                    n == 50 && (global pic50 = send_to_host(real(ifft(env.y))))
                    n == 250 && (global pic250 = send_to_host(real(ifft(env.y))))

                    react!(p, [heatmap(z=send_to_host(real(ifft(env.y))), coloraxis="coloraxis")], layout)
                    sleep(0.05)

                    if video
                        savefig(p, dirpath * "/frames/a$(lpad(string(n), 3, '0')).png"; width=1000, height=800)
                    end
                end
                n += 1

                (use_best || negate) || agent(POST_ACT_STAGE, env)
                (use_best || negate) || hook(POST_ACT_STAGE, agent, env)

                if use_best || negate
                    temp_energy = sum(abs.(send_to_host(real(ifft(env.y))))) / (nx*ny)
                    println(temp_energy)
                    energy_sum += temp_energy
                    push!(energy, temp_energy)
                else
                    println(mean(env.reward))
                end
            end
            (use_best || negate) || agent(POST_EPISODE_STAGE, env)
            (use_best || negate) || hook(POST_EPISODE_STAGE, agent, env)
            println("")
            println("---------------------------------------------------------")
            println("")
        end

        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        (use_best || negate) ? println(energy_sum) : println(hook.bestreward)
        (use_best || negate) || (agent.policy.act_noise = agent.policy.act_noise * 0.2)
        (use_best || negate) || println(agent.policy.act_noise)
        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    end
    (use_best || negate) || hook(POST_EXPERIMENT_STAGE, agent, env)

    negate && (agent_negate.policy.start_steps = temp_start_steps)

    if use_best
        copyto!(agent.policy.behavior_actor, hook.currentNNA)
        agent.policy.start_steps = temp_start_steps

        agent.policy.act_noise = temp_noise

        agent.policy.update_after = temp_update_after
    end

    if video
        isdir(dirpath * "/video_output") || mkdir(dirpath * "/video_output")
        rm(dirpath * "/video_output/output.mp4", force=true)
        run(`ffmpeg -framerate 16 -i "$(dirpath)/frames/a%03d.png" -c:v libx264 -crf 21 -an -pix_fmt yuv420p10le "$(dirpath)/video_output/output.mp4"`)
    end
end

frame = 1

function train(use_random_init = true; loops = 6)
    hook.use_random_init = use_random_init

    No_Steps = 580

    agent.policy.act_noise = act_noise
    for i = 1:loops
        println("")
        println(agent.policy.act_noise)
        run(agent, env, StopAfterEpisodeWithMinSteps(No_Steps), hook)
        println(hook.bestreward)
        agent.policy.act_noise = agent.policy.act_noise * 0.6

        hook.rewards = clamp.(hook.rewards, -3000, 0)
    end
end
#train()

function train_multi(No_Episodes = 17)
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
            inner_No_Episodes = 1
            loops = 18
            agent.policy.act_noise = 0.17
            i = 1
            while i <= loops && n <= No_Episodes
                run(agent, env, StopAfterEpisode(inner_No_Episodes), hook)
                println(hook.bestreward)
                agent.policy.act_noise = agent.policy.act_noise * 0.7
                println(agent.policy.act_noise)

                hook.rewards = clamp.(hook.rewards, -30000, 0)
                n += inner_No_Episodes
                i += 1
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

#run(agent_negate, env, StopAfterEpisode(1), hook2);

#plotrun(use_best = false)
#plotrun()
#plotrun(plot_best = true)
#plotrun(plot3D = true)

#plot_heat(p_te = 200.0, p_t_action = 100.0)
#plot_heat(p_te = 200.0, p_t_action = 100.0, use_best = false)
#plot_heat(plot_best = true)


function load(number = nothing)
    if isnothing(number)
        global hook = FileIO.load(dirpath * "/saves/hook.jld2","hook");
        global agent = FileIO.load(dirpath * "/saves/agent.jld2","agent");
        #global env = FileIO.load(dirpath * "/saves/env.jld2","env");
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

#FileIO.save("corrupted_y.jld2","corrupted_y",corrupted_y)
#corrupted_y = FileIO.load("corrupted_y.jld2","corrupted_y")
#yyy = corrupted_y[280]

#FileIO.save("FluidResults.jld2", "FluidResults", FluidResults)
#global FluidResults = FileIO.load("FluidResults.jld2", "FluidResults")
#env.y0 = CuArray(FluidResults["y0"])
#env.y0 = CuArray(FluidResults["y0_128"])