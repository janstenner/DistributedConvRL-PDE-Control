using ProgressMeter
#####
# A custom stop condition that first counts the steps and if they are reached, it lets the current episode play to its end
#####

mutable struct StopAfterEpisodeWithMinSteps{Tl}
    step::Int
    cur::Int
    progress::Tl
end

function StopAfterEpisodeWithMinSteps(step; cur = 1, is_show_progress = true)
    if is_show_progress
        progress = Progress(step, 1)
        ProgressMeter.update!(progress, cur)
    else
        progress = nothing
    end
    StopAfterEpisodeWithMinSteps(step, cur, progress)
end

function (s::StopAfterEpisodeWithMinSteps)(agent, env)
    if !isnothing(s.progress)
        # https://github.com/timholy/ProgressMeter.jl/pull/131
        # next!(s.progress; showvalues = [(Symbol(s.tag, "/", :STEP), s.cur)])
        next!(s.progress)
    end

    @debug s.tag STEP = s.cur

    res = false
    if (s.cur >= s.step)
        #now check if the episode is over
        if is_terminated(env)
            res = true
        end
    end
    s.cur += 1
    res
end
