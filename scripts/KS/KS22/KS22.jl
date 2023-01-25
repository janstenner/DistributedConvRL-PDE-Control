seed = 55
dirpath = string(@__DIR__)

Lx = 22.0
nx = 96

# sensor and actor positions and gaussian curve parameters
sensor_positions = collect(1:12:nx)
actuator_positions = collect(1:12:nx)
actuators_to_sensors = collect(1:8)
sigma_sensors = 0.7
sigma_actuators = 0.7

#amount of inhomogenous disturbance
μ = 0.0

include(pwd() * "/scripts/KS/setup/KSSetup.jl")

initialize_setup()


# use train(true) to train an agent yourself or load() to load up a pre-trained agent
train(true; loops = 14)
#load()

plot_heat(p_te = 200.0, p_t_action = 100.0)