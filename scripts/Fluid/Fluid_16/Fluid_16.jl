seed = 76
dirpath = string(@__DIR__)


#Set this to false if you do not have a GPU that supports CUDA
gpu_env = true

#set the number of sensors per axis (this is also the amount of actors per axis)
sensors_per_axis = 16
#variance of the gaussian curve for convolution
variance = 0.04


include(pwd() * "/scripts/Fluid/setup/FluidSetup.jl")


initialize_setup()

# use train(true) to train an agent yourself or load() to load up a pre-trained agent
train(true; loops = 20)
#load()

testrun(use_best = true)

plot(abs.(rewards))