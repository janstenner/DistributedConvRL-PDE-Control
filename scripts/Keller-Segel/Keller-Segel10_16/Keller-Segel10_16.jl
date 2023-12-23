dirpath = string(@__DIR__)
seed = 155

# Set this to false if you want to train an agent yourself.
# Otherwise this script will load up a pre-trained agent and do an evaluation run.
evaluation = true

Lx = 10.0
nx = 100

# sensor and actuator positions
sensor_positions = collect(3:5:nx)
actuator_positions = sensor_positions[3:end-2]
actuators_to_sensors = collect(3:18)

include(pwd() * "/scripts/Keller-Segel/setup/KellerSegelSetup.jl")



if evaluation
    load()
    plot_heat(from=14)
else
    train(; loops = 13)
    save()
end