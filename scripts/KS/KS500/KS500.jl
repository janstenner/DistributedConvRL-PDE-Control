seed = 13
dirpath = string(@__DIR__)

Lx = 500.0
nx = 600

# sensor and actor positions and gaussian curve parameters
sensor_positions = collect(1:3:nx)
actuator_positions = collect(1:3:nx)
actuators_to_sensors = collect(1:200)
sigma_sensors = 1.0
sigma_actuators = 1.0

#amount of inhomogenous disturbance
μ = 0.0

include(pwd() * "/scripts/KS/setup/KSSetup.jl")

initialize_setup()


# here we load an agent that was previously trained on an environment with Lx = 200 (this works since the sensors and actuators share the same distance and parameters of the gaussian curves)
load()

plot_heat(p_te = 200.0, p_t_action = 100.0)