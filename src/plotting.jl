using PlotlyJS
using Blink

function plot_heat(;use_best = true, p_dt = nothing, p_te = 30.0, p_t_action = 20.0, plot_best = false, use_random_init = true, plot_separate = false, from = nothing, to = nothing)

    if !plot_best
        reset!(env)

        if use_random_init
            env.y = generate_random_init()
        end

        global y0 = env.y

        env.te = p_te
        if !isnothing(p_dt) env.dt = p_dt end

        if isnothing(from)
            from = 0
        end
    
        if isnothing(to)
            to = p_te
        end

        temp_start_steps = agent.policy.start_steps
        agent.policy.start_steps = -1
        if use_best
            copyto!(agent.policy.behavior_actor, hook.bestNNA)
        end
    end
    
    if !plot_best
        x_axis = collect(0:env.dt:p_te+env.dt)
        y_axis = collect(dx:dx:Lx)

        if length(size(env.y)) > 1
            yresults = zeros(size(env.y)[1], length(y_axis), length(x_axis))
        else
            yresults = zeros(1, length(y_axis), length(x_axis))
        end
        presults = zeros(length(y_axis), length(x_axis))
        rresults = zeros(length(x_axis))

        if length(size(env.y)) > 1
            yresults[:,:,1] = env.y
        else
            yresults[1,:,1] = env.y
        end
        presults[:,1] = env.p

        i = 2
        done = false

        while !done
            if env.time < p_t_action
                action = zeros(size(action_space))
            else
                action = agent.policy(env, learning = false)
            end
            env(action)

            if length(size(env.y)) > 1
                yresults[:,:,i] = env.y
            else
                yresults[1,:,i] = env.y
            end
            presults[:,i] = env.p
            rresults[i] = mean(env.reward)

            i += 1
            done = is_terminated(env)
        end

        yresults = Array(yresults)
        presults = Array(presults)

        global yresults = yresults
        global presults = presults
        global rresults = rresults
        global y_axis = y_axis
        global x_axis = x_axis
    else
        global yresults2 = reduce(hcat, hook.bestDF.y)
        global yresults = zeros(1, size(yresults2)[1], size(yresults2)[2])
        yresults[1,:,:] = yresults2
        global presults = reduce(hcat, hook.bestDF.p)
        global rresults = map(mean, hook.bestDF.reward)
        global y_axis = collect(dx:dx:Lx)
        global x_axis = collect(0:env.dt:hook.bestDF.timestep[end]/env.dt)
        
        if isnothing(from)
            from = 0
        end
    
        if isnothing(to)
            to = Lx
        end
    end

    if length(size(env.y)) > 1
        yplots = size(yresults)[1]
    else
        yplots = 1
    end

    colorscale = [[0, "blue"], [0.5, "white"], [1, "red"], ]
    ymax = max(abs(minimum(yresults[:,:, Int(floor(from/env.dt))+1:Int(floor(to/env.dt))+1 ])), abs(maximum(yresults[:,:, Int(floor(from/env.dt))+1:Int(floor(to/env.dt))+1 ])))
    pmax = max(abs(minimum(presults[:, Int(floor(from/env.dt))+1:Int(floor(to/env.dt))+1 ])), abs(maximum(presults[:, Int(floor(from/env.dt))+1:Int(floor(to/env.dt))+1 ])))
    layout = Layout(
                plot_bgcolor="#f1f3f7",
                font=attr(family="Computer Modern", size=20, color="black"),
                #title = "plot_best = $plot_best,  use_best = $use_best,   best NN: $(hook.bestepisode)",
                coloraxis = attr(cmin = -ymax, cmid = 0, cmax = ymax, colorscale = colorscale),
                coloraxis2 = attr(cmin = -pmax, cmid = 0, cmax = pmax, colorscale = colorscale),
                # yaxis = attr(title="y", gridcolor = "#aaaaaa", linecolor = "#000000"),
                # xaxis = attr(title="t", gridcolor = "#aaaaaa", linecolor = "#000000", range = [from,to])
            )

    if plot_separate
        for i in 1:size(yresults)[1]
            layout = Layout(
                plot_bgcolor="#ffffff",
                font=attr(family="Computer Modern", size=26, color="black"),
                coloraxis = attr(cmin = -ymax, cmid = 0, cmax = ymax, colorscale = colorscale),#, showscale=false),
                coloraxis2 = attr(cmin = -pmax, cmid = 0, cmax = pmax, colorscale = colorscale),
                yaxis = attr(title="y", gridcolor = "#aaaaaa", linecolor = "#000000"),
                xaxis = attr(title="t", gridcolor = "#aaaaaa", linecolor = "#000000", range = [from,to])
            )
            p = plot(heatmap(x = x_axis, y = y_axis, z = yresults[i,:,:], coloraxis="coloraxis"), layout)
            display(p)
        end
        layout = Layout(
                plot_bgcolor="#ffffff",
                font=attr(family="Computer Modern", size=26, color="black"),
                coloraxis = attr(cmin = -ymax, cmid = 0, cmax = ymax, colorscale = colorscale),
                coloraxis2 = attr(cmin = -pmax, cmid = 0, cmax = pmax, colorscale = colorscale),
                yaxis = attr(title="p", gridcolor = "#aaaaaa", linecolor = "#000000"),
                xaxis = attr(title="t", gridcolor = "#aaaaaa", linecolor = "#000000", range = [from,to])
            )
        p = plot(heatmap(x = x_axis, y = y_axis, z = presults, coloraxis="coloraxis2"), layout)
        display(p)
        p = plot(scatter(x = x_axis, y = rresults .* (-1)), layout)
        display(p)
    else
        p = make_subplots(rows=2 + yplots, cols=1, shared_xaxes=true)
        
        for i in 1:size(yresults)[1]
            add_trace!(p, heatmap(x = x_axis, y = y_axis, z = yresults[i,:,:], coloraxis="coloraxis"), row = i, col = 1)
        end
        add_trace!(p, heatmap(x = x_axis, y = y_axis, z = presults, coloraxis="coloraxis2"), row = yplots + 1, col = 1)
        add_trace!(p, scatter(x = x_axis, y = rresults), row = yplots + 2, col = 1)
        
        
        #p = plot(heatmap(x = x_axis, y = y_axis, z = yresults, coloraxis="coloraxis"))
        relayout!(p, layout.fields)
        display(p)
    end

    if !plot_best
        if use_best
            copyto!(agent.policy.behavior_actor, hook.currentNNA)
        end
        #reset!(env)
        agent.policy.start_steps = temp_start_steps
        if !isnothing(p_dt) env.dt = dt end
        env.te = te
    end
end

function plot_sensors(;gaussians = gaussians, actuators_to_sensors = nothing)
    lll = []
    t = collect(1:nx) .* dx

    if isnothing(actuators_to_sensors)
        for curve in gaussians
            push!(lll, scatter(x=t, y=curve))
        end
    else
        for curve in gaussians[actuators_to_sensors]
            push!(lll, scatter(x=t, y=curve))
        end
    end

    plot(Vector{AbstractTrace}(lll))
end

function plot_sums(;use_best = true, p_dt = 0.1, p_te = 30.0, p_t_action = 20.0)
    reset!(env)
    env.te = p_te
    env.dt = p_dt
    temp_start_steps = agent.policy.start_steps
    agent.policy.start_steps = -1
    if use_best
        copyto!(agent.policy.behavior_actor, hook.bestNNA)
    end

    layout = Layout(
            plot_bgcolor="#f1f3f7",
            title = "KS Equation with " * boundary_condition * " boundary conditions",
            margin=attr(l=10, r=10, b=30, t=30, pad=0)
        )
    
    x_axis = collect(0:p_dt:p_te)

    yresults = zeros(length(x_axis))
    presults = zeros(length(x_axis))

    yresults[1] = sum(abs.(env.y))
    presults[1] = sum(abs.(env.p))

    i = 2
    done = false

    while !done
        if env.time < p_t_action
            action = zeros(size(action_space))
        else
            action = agent.policy(env, learning = false)
        end
        env(action)

        yresults[i] = sum(abs.(env.y))
        presults[i] = sum(abs.(env.p))

        i += 1
        done = is_terminated(env)
    end

    global yresults = yresults
    global presults = presults
    global y_axis = y_axis
    global x_axis = x_axis

    p = make_subplots(rows=2, cols=1, shared_xaxes=true, vertical_spacing=0.1, subplot_titles=["Sum of Y" "Sum of P"])
    add_trace!(p, scatter(x = x_axis, y = yresults, mode="lines"), row = 1, col = 1)
    add_trace!(p, scatter(x = x_axis, y = presults, mode="lines"), row = 2, col = 1)
    #p = [p1; p2]
    relayout!(p, layout.fields)
    display(p)

    if use_best
        copyto!(agent.policy.behavior_actor, hook.currentNNA)
    end
    reset!(env)
    agent.policy.start_steps = temp_start_steps
    env.dt = dt
    env.te = te
end

function plot_actions(;use_best = true, p_dt = 0.1, p_te = 30.0, p_t_action = 20.0)
    reset!(env)
    env.te = p_te
    env.dt = p_dt
    temp_start_steps = agent.policy.start_steps
    agent.policy.start_steps = -1
    if use_best
        copyto!(agent.policy.behavior_actor, hook.bestNNA)
    end

    layout = Layout(
            plot_bgcolor="#f1f3f7",
            title = "KS Equation with " * boundary_condition * " boundary conditions",
            margin=attr(l=10, r=10, b=30, t=30, pad=0)
        )
    
    x_axis = collect(0:p_dt:p_te)

    results = zeros(length(x_axis),size(action_space)[2])

    results[1,:] = vec(env.action)

    i = 2
    done = false

    while !done
        if env.time < p_t_action
            action = zeros(size(action_space))
        else
            action = agent.policy(env, learning = false)
        end
        env(action)

        results[i,:] = vec(env.action)

        i += 1
        done = is_terminated(env)
    end

    global results = results

    p = plot(results)
    #p = [p1; p2]
    relayout!(p, layout.fields)
    display(p)

    if use_best
        copyto!(agent.policy.behavior_actor, hook.currentNNA)
    end
    reset!(env)
    agent.policy.start_steps = temp_start_steps
    env.dt = dt
    env.te = te
end

function plotrun(;use_best = true, plot3D = false, plot_rewards = true, plot_best = false, use_random_init = true)
    reset!(env)

    if use_random_init
        env.y = generate_random_init()
    end

    env.te = te_plot
    env.dt = dt_plot
    temp_start_steps = agent.policy.start_steps
    agent.policy.start_steps = -1
    if use_best
        copyto!(agent.policy.behavior_actor, hook.bestNNA)
    end

    if plot3D
        layout = Layout(
                title = "KS Equation with " * boundary_condition * " boundary conditions<br><sub>NN from Episode " * string(hook.bestepisode) * "</sub>",
                font=attr(size=15),
                autosize=false,
                    legend = attr(
                    x=1,
                    y=1.02,
                    yanchor="bottom",
                    xanchor="right",
                    orientation="h"
                ),
                scene=attr(
                    xaxis=attr(backgroundcolor="#f1f3f7",gridcolor="#E2E4E6"),
                    yaxis=attr(backgroundcolor="#f1f3f7",gridcolor="#E2E4E6",range=[0,101]),
                    zaxis = attr(backgroundcolor="#f1f3f7",gridcolor="#E2E4E6",
                                range = [-2.0,6.5],
                                nticks = 5),
                    plot_bgcolor="#f1f3f7",
                    camera=attr(
                        up=attr(x=0, y=0, z=1),
                        eye=attr(x=-0.9, y=2.9, z=0.5),
                        center=attr(x=0.35, y=0, z=-0.2)
                    ),
                    aspectratio=attr(x=1.6, y=2.4, z=1),
                    aspectmode="manual",
                )
            )
        
        x_axis = collect(0:dx:Lx)

        rm(dirpath * "/frames/", recursive=true, force=true)
        mkdir(dirpath * "/frames")

        trace1 = scatter(x=x_axis, y=[100 for i in x_axis], z=env.y, mode="lines", name = "state", line=attr(color=env.y, colorscale="Bluered", width = 4), type="scatter3d")
        p = plot([trace1], layout, config=PlotConfig(staticPlot=true))
        savefig(p, dirpath * "/frames/muell.png"; width=1000, height=800)

        sleep(0.1)

        rm(dirpath * "/frames/muell.png", force=true)
        i = 1
        oldstates = DS.Queue{Vector}()

        while !is_terminated(env)
        #while i < 4
            DS.enqueue!(oldstates, env.y)
            length(oldstates) >= 100 ? DS.dequeue!(oldstates) : nothing

            if env.time < t_action
                action = zeros(size(action_space))
            else
                if env.dt == dt_plot
                    env.dt = dt_slowmo
                end
                action = agent.policy(env, learning = false)
            end
            env(action)

            layout["title"] = "KS Equation with " * boundary_condition * " boundary conditions, dt = " * string(env.dt) * ", Time: " * string(env.time) * "s<br><sub>NN from Episode " * string(hook.bestepisode) * "</sub>"
            
            trace1 = scatter(x=x_axis, y=[100 for i in x_axis], z=env.y, mode="lines", name = "state", line=attr(color=env.y, colorscale="Bluered", width = 4), type="scatter3d")
            trace2 = scatter(x=x_axis, y=[100 for i in x_axis], z=vec(env.p), mode="lines", name = "action", line=attr(color="green", width = 4), type="scatter3d")


            traces = [trace2,trace1]
            j = 1
            for oldstate in Iterators.reverse(oldstates)
                push!(traces, scatter(x=x_axis, y=[(100 - j) for i in x_axis], z=oldstate, line=attr(color=oldstate, colorscale="Bluered"), type="scatter3d", mode="lines", opacity=0.06 + 0.6 * j^(-1), showlegend=false))
                j += 1
            end

            p = plot(traces, layout, config=PlotConfig(staticPlot=true))
            sleep(0.01)
            savefig(p, dirpath * "/frames/a$(lpad(string(i), 3, '0')).png"; width=1200, height=800)
            i += 1
        end
    else
        layout = Layout(
                plot_bgcolor="#f1f3f7",
                #title = "KS Equation with " * boundary_condition * " boundary conditions<br><sub>NN from Episode " * string(hook.bestepisode) * "</sub>",
                yaxis_range = [-4, 4],
                legend = attr(
                    x=1,
                    y=1.02,
                    yanchor="bottom",
                    xanchor="right",
                    orientation="h"
                ),
                margin=attr(l=100, r=80, b=80, t=100, pad=10)
            )
        
        w = Window()

        x_axis = collect(0:dx:Lx)

        traces = []
        rewards = []

        if plot_best
            df = hook.bestDF[hook.bestDF.timestep .== 1, :]

            push!(traces, scatter(x = x_axis, y = df.y[1], mode="lines", name = "state"))
            push!(traces, scatter(x = x_axis, y = df.p[1], mode="lines", name = "action"))
        else
            push!(traces, scatter(x = x_axis, y = env.y, mode="lines", name = "state"))
            push!(traces, scatter(x = x_axis, y = vec(env.p), mode="lines", name = "action"))
        end

        traces = Array{GenericTrace}(traces)

        if plot_rewards
            if plot_best
                push!(rewards, mean(df.reward[1]))
            else
                push!(rewards, mean(reward(env)))
            end
            p1 = plot(traces)
            p2 = plot(scatter(y = rewards, mode="lines", name = "reward"))
            p = [p1 p2]
            relayout!(p, layout.fields)
        else
            p = plot(traces, layout, config=PlotConfig(staticPlot=true))
        end
        body!(w,p)

        rm(dirpath * "/frames/", recursive=true, force=true)
        mkdir(dirpath * "/frames")

        i = 1
        done = false
        rewards = []

        while !done
            traces = []

            if plot_best
                df = hook.bestDF[hook.bestDF.timestep .== i, :]
                y_temp = df.y[1]
                p_temp = df.p[1]
    
                push!(traces, scatter(x = x_axis, y = y_temp, mode="lines", name = "state"))
                push!(traces, scatter(x = x_axis, y = p_temp, mode="lines", name = "action"))
            else
                if env.time < t_action
                    action = zeros(size(action_space))
                else
                    if env.dt == dt_plot
                        env.dt = dt_slowmo
                    end
                    action = agent.policy(env, learning = false)
                end
                env(action)

                push!(traces, scatter(x = x_axis, y = env.y, mode="lines", name = "state"))
                push!(traces, scatter(x = x_axis, y = vec(env.p), mode="lines", name = "action"))
            end

            traces = Array{GenericTrace}(traces)
            
            if plot_rewards
                if plot_best
                    df = hook.bestDF[hook.bestDF.timestep .== i, :]
                    push!(rewards, mean(df.reward[1]))
                else
                    push!(rewards, mean(reward(env)))
                end
                p1 = plot(traces)
                p2 = plot(scatter(y = rewards, mode="lines", name = "reward"))
                p = [p1 p2]
                relayout!(p, layout.fields)
                body!(w,p)
            else
                react!(p, traces, layout)
            end

            sleep(0.02)
            savefig(p, dirpath * "/frames/a$(lpad(string(i), 3, '0')).png"; width=1300, height=800)
            i += 1
            if plot_best
                done = i == maximum(hook.bestDF.timestep)
            else
                done = is_terminated(env)
            end
        end
    end

    isdir(dirpath * "/video_output") || mkdir(dirpath * "/video_output")
    rm(dirpath * "/video_output/output.mp4", force=true)
    run(`ffmpeg -framerate 16 -i "$(dirpath)/frames/a%03d.png" -c:v libx264 -crf 21 -an -pix_fmt yuv420p10le "$(dirpath)/video_output/output.mp4"`)
    # rm(dirpath * "/video_output/output.gif", force=true)
    # run(`ffmpeg -framerate 25 -i "$(dirpath)/frames/a%03d.png" -filter_complex "[0:v] fps=20,scale=w=880:h=-1,split [a][b];[a] palettegen=stats_mode=single [p];[b][p] paletteuse=new=1" "$(dirpath)/video_output/output.gif"`)

    if use_best
        copyto!(agent.policy.behavior_actor, hook.currentNNA)
    end
    reset!(env)
    agent.policy.start_steps = temp_start_steps
    env.dt = dt
    env.te = te
end




function plot_rewards(res_y = 100, res_action = 80, max_value = 30.0)

    results = zeros(res_y, res_action)

    for i in 1:res_y
        for j in 1:res_action
            y = max_value * i/res_y .* ones(size(sim_space))
            gpu_env && (y = CuArray(y))
            action = j/res_action .* ones(length(actuator_positions))

            results[i,j] = reward_function(nothing; test = Dict("y" => y, "action" => action, "delta_action" => action))[1]
        end
    end

    plot(heatmap(z=results))
end