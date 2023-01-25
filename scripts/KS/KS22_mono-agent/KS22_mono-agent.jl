seed = 33
dirpath = string(@__DIR__)

# here we try to train a single agent that takes all 8 sensors as input and outputs all 8 actions
# this fails to train

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

include(pwd() * "/scripts/KS/setup/KSmonoSetup.jl")

initialize_setup()

# now we try to train the agent but it does not go well
train(true; loops = 3)

plot_heat(p_te = 200.0, p_t_action = 100.0)