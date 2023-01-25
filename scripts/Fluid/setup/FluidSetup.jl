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

include(pwd() * "/src/fluid_rk4.jl")
include(pwd() * "/src/PDEagent.jl")
include(pwd() * "/src/PDEenv.jl")
include(pwd() * "/src/PDEhook.jl")
include(pwd() * "/src/plotting.jl")


# env parameters

nu = 0.0005

Lx  = 1;            
Ly  = 1;
nx  = 64;    
ny  = nx;
dx  = Lx/nx;        
dy  = Ly/ny;
sim_space = Space(fill(-1..1, (nx, ny)))


# sensor positions - 
sensor_positions = [[i,j] for i in 1:Int(nx/sensors_per_axis):nx for j in 1:Int(ny/sensors_per_axis):ny]
actuator_positions = [[i,j] for i in 1:Int(nx/sensors_per_axis):nx for j in 1:Int(ny/sensors_per_axis):ny]
actuators_to_sensors = collect(1:length(sensor_positions))


te = 6.0
t0 = 0.0
dt = 0.02
oversampling = floor(16*nx*dt)

min_best_episode = 10

te_plot = 10.0
dt_plot = dt
t_action = 2.0
dt_slowmo = dt

check_max_value = "reward"
max_value = 0.025

# agent tuning parameters
memory_size = 0
nna_scale = 0.4
drop_middle_layer = true
temporal_steps = 1
action_punish = 0.0005
delta_action_punish = 0.0005
window_size = 1
use_gpu = false
action_space = Space(fill(-1..1, (1 + memory_size, length(actuator_positions))))
use_radau = true
agent_power = 70.0

# additional agent parameters
rng = StableRNG(seed)
Random.seed!(seed)
y = 0.99f0
p = 0.995f0
batch_size = 300
start_steps = 10
start_policy = ZeroPolicy(action_space)
update_after = 100
update_freq = 50
act_limit = 1.0
act_noise = 0.1
trajectory_length = 40_000


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

y0_2D_standard = ic(3, rng)
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

    y = abs.(y)
    #factor = 10_000 * (nx/100) * (ny/100)
    #convolution
    for i in 1:length(actuator_positions)
        sensors[i] = send_to_host(dot(y, gaussians[actuators_to_sensors[i]]) ./ (100 * 600))
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
                        act_limit = act_limit, 
                        act_noise = act_noise,
                        nna_scale = nna_scale,
                        drop_middle_layer = drop_middle_layer,
                        memory_size = memory_size,
                        trajectory_length = trajectory_length)

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
    result = ic(3, rng)
    result = gpu_env ? CuArray(result) : result
    result
end

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

function train(use_random_init = false; loops = 40)
    hook.use_random_init = use_random_init


    No_Episodes = 2
    loops = loops
    agent.policy.act_noise = 0.13
    for i = 1:loops
        println("")
        println(agent.policy.act_noise)
        run(agent, env, StopAfterEpisode(No_Episodes), hook)
        println(hook.bestreward)
        agent.policy.act_noise = agent.policy.act_noise * 0.7
        global last_an = agent.policy.act_noise

        hook.rewards = clamp.(hook.rewards, -30000, 0)
    end

end

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

        if video
            rm(dirpath * "/frames/", recursive=true, force=true)
            mkdir(dirpath * "/frames")
        end
    end

    use_best && (agent.policy.update_step = 0)
    negate && (agent_negate.policy.update_step = 0)
    (use_best || negate) && (global rewards = Vector{Float64}())
    (use_best || negate) && (reward_sum = 0.0)

    (use_best || negate) || hook(PRE_EXPERIMENT_STAGE, agent, env)
    (use_best || negate) || agent(PRE_EXPERIMENT_STAGE, env)

    if viz
        w = Window()
        colorscale = [[0, "blue"], [0.5, "yellow"], [1, "red"], ]
        ymax = 30
        layout = Layout(
                plot_bgcolor="#f1f3f7",
                coloraxis = attr(cmin = -ymax, cmid = 0, cmax = ymax, colorscale = colorscale),
            )
    end

    loop_end_outer = (use_best || negate) ? 1 : 5
    loop_end_inner = (use_best || negate)  ? 1 : 2
    (use_best || negate)  || (agent.policy.act_noise = 0.18)

    for j in 1:loop_end_outer
        for i in 1:loop_end_inner
            reset!(env)
            (use_best || negate) || agent(PRE_EPISODE_STAGE, env)
            (use_best || negate) || hook(PRE_EPISODE_STAGE, agent, env)

            #env.y = yyy

            if viz
                p = plot(heatmap(z=send_to_host(real(ifft(env.y))), coloraxis="coloraxis"), layout)
                body!(w,p)
            end

            n = 1
            while !is_terminated(env)
                negate ? action = agent_negate(env) : action = agent(env)

                #println(action)
                (use_best || negate) || agent(PRE_ACT_STAGE, env, action)
                (use_best || negate) || hook(PRE_ACT_STAGE, agent, env, action)

                env(action)

                if viz
                    n == 50 && (global pic50 = send_to_host(real(ifft(env.y))))
                    n == 250 && (global pic250 = send_to_host(real(ifft(env.y))))

                    react!(p, [heatmap(z=send_to_host(real(ifft(env.y))), coloraxis="coloraxis")], layout)
                    sleep(0.05)

                    if use_best && video
                        savefig(p, dirpath * "/frames/a$(lpad(string(n), 3, '0')).png"; width=1000, height=800)
                    end
                end
                n += 1

                (use_best || negate) || agent(POST_ACT_STAGE, env)
                (use_best || negate) || hook(POST_ACT_STAGE, agent, env)

                println(mean(env.reward))

                if use_best || negate
                    reward_sum += mean(env.reward)
                    push!(rewards, mean(env.reward))
                end
            end
            (use_best || negate) || agent(POST_EPISODE_STAGE, env)
            (use_best || negate) || hook(POST_EPISODE_STAGE, agent, env)
            println("")
            println("---------------------------------------------------------")
            println("")
        end

        println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        (use_best || negate) ? println(reward_sum) : println(hook.bestreward)
        (use_best || negate) || (agent.policy.act_noise = agent.policy.act_noise * 0.7)
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

        if video
            isdir(dirpath * "/video_output") || mkdir(dirpath * "/video_output")
            rm(dirpath * "/video_output/output.mp4", force=true)
            run(`ffmpeg -framerate 16 -i "$(dirpath)/frames/a%03d.png" -c:v libx264 -crf 21 -an -pix_fmt yuv420p10le "$(dirpath)/video_output/output.mp4"`)
        end
    end
end