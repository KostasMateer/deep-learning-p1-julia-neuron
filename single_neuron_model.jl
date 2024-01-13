# Code reviewed by Niya Ma, everything looks good
# Code reviewed by Hannah Kanjan, everything looks good

using Distributions: Normal

include("neuron.jl") # Revise.includet doesn't play nice with structs, so Neuron is in its own file.

# y=x activation for a linear neuron
function linear_activation(input)
    return input
end

# derivative of a linear neuron's activation function
function linear_derivative(activation)
    return 1
end

# σ(x) activation for a sigmoid neuron
function sigmoid_activation(input)
    return 1 ./ (1 .+ exp.(-input))
end

# derivative of a sigmoid neuron's activation function
function sigmoid_derivative(activation)
    return activation*(1-activation)
end

# create a neuron with linear activation and random initial weights
function LinearNeuron(input_dimension::Integer)
    weights = rand(Normal(0,1), input_dimension)
    bias = rand(Normal(0,1))
    Neuron(weights, bias, linear_activation, linear_derivative)
end

# create a neuron with sigmoid activation and random initial weights
function SigmoidNeuron(input_dimension::Integer)
    weights = rand(Normal(0,1), input_dimension)
    bias = rand(Normal(0,1))
    Neuron(weights, bias, sigmoid_activation, sigmoid_derivative)
end

# finds the model's output on a signle data point
# input: neuron, point (represented as a vector)
# output: number
function predict(model::Neuron, data_point::AbstractVector{<:Real})
    weighted_sum = sum(model.weights.*data_point)+model.bias
    model.activation(weighted_sum)
end

# finds the model's output on a collection of data points
# input: neuron, data_set (array where each row represents a point)
# output: vector of predictions for each point
function predict(model::Neuron, data_set::AbstractMatrix{<:Real})
    weighted_sum = sum(data_set.*model.weights', dims=2) .+ model.bias
    model.activation(weighted_sum)
end
