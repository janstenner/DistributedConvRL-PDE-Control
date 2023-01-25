seed = 126
dirpath = string(@__DIR__)

Lx = 10.0
nx = 100

# sensor and actuator positions
sensor_positions = collect(3:5:nx)
actuator_positions = sensor_positions[3:end-2]
actuators_to_sensors = collect(3:18)

include(pwd() * "/scripts/Keller-Segel/setup/KellerSegelSetup.jl")

initialize_setup()


# use train(true) to train an agent yourself or load() to load up a pre-trained agent
train(true; loops = 10)
#load()

plot_heat()