dirpath = string(@__DIR__)
seed = 59


# Set this to false if you want to train an agent yourself.
# Otherwise this script will load up a pre-trained agent and do an evaluation run.
evaluation = true


Lx = 200.0
nx = 240

# sensor and actor positions and gaussian curve parameters
sensor_positions = collect(1:3:nx)
actuator_positions = collect(1:3:nx)
actuators_to_sensors = collect(1:80)
sigma_sensors = 1.0
sigma_actuators = 1.0

#amount of inhomogenous disturbance
μ = 0.0

include(pwd() * "/scripts/KS/setup/KSSetup.jl")


if evaluation
    load()
    plot_heat(p_te = 200.0, p_t_action = 100.0)
else
    train(; loops = 6)
    save()
end