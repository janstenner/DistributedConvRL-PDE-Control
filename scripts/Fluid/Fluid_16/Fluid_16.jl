dirpath = string(@__DIR__)
seed = 661


# Set this to false if you want to train an agent yourself.
# Otherwise this script will load up a pre-trained agent and do an evaluation run.
evaluation = true


#Set this to false if you do not have a GPU that supports CUDA
gpu_env = true

#set the number of sensors per axis (this is also the amount of actors per axis)
sensors_per_axis = 16
#variance of the gaussian curve for convolution
variance = 0.04


include(pwd() * "/scripts/Fluid/setup/FluidSetup.jl")


if evaluation
    load()
    testrun(use_best = true)
    plot(abs.(energy))
else
    train(; loops = 6)
    save()
end