# Reviewed by Niya Ma, everything looks good, in line 41 and 44 gradient function used 2 instead of -2, not sure why it still works. 
# Reviewed by Hannah Kanjan. She also said everything looks good. She also does not know why the gradient function works without the -2.

# finds the mean-squared-error loss between predictions and targets
# input: vector of predictions, vector of targets
# output: number
function MSE(predictions::AbstractVector{<:Real}, targets::AbstractVector{<:Real})
    n = length(predictions)
    (1/n) * sum((predictions.-targets).^2)
end

# finds the fraction of points classified correctly if predictions are rounded to 0/1
# only applicable for classification models
# input: vector of predictions, vector of targets
# output: number
function accuracy(predictions::AbstractVector{<:Real}, targets::AbstractVector{<:Real})
    sum(round.(predictions) .== targets) / length(targets)
end

# finds the gradient of loss w.r.t. model parameters on a data set
# input: model, input array (one data point per row), target vector (length = #points)
# output: a vector of length data_dim + 1, where entries 1:d are weight
#         partial derivatives, and entry d+1 is the bias partial derivative
function gradient(model::Neuron, inputs::AbstractMatrix{<:Real},
                  targets::AbstractVector{<:Real})
    N = size(inputs, 1)
    
    predictions = predict(model, inputs)

    derivatives = []
    for pred in predictions
        derivatives = push!(derivatives, model.derivative(pred))
    end
    
    errors = predictions .- targets
    
    gradients = []
    for i in 1:length(model.weights)
        gradient = 1/N * (2 * sum((errors .* inputs[:,i]) .* derivatives))
        gradients = push!(gradients, gradient)
    end
    gradients = push!(gradients, 1/N * (2 * sum(errors)))
    return vec(gradients)
end

# changes the model's weights and bias to take a step in the -gradient direction
# input: gradient is a vector of length #weights + 1, where entries 1:d are weight
#        partial derivatives, and entry d+1 is the bias partial derivative
function update!(model::Neuron, gradient::Vector{<:Any}, step_size::Real)
    model.weights = model.weights .- (step_size .* gradient[1:end-1])
    model.bias = model.bias - (step_size * gradient[end])
end

# gradient descent: repeatedly updates the model by small steps in the -gradient direction
# inputs: inputs/targets are a data set to train on
#         losses/gradients/parameters specify vectors for logging itermediate values
#         if these are nothing, we just update the model
#         if they are vectors, then at each iteration, we should push the current values
function train!(model::Neuron, inputs::AbstractMatrix{<:Real},
                targets::AbstractVector{<:Real}, iterations::Integer,
                step_size::Real; losses::Union{Vector,Nothing}=nothing,
                gradients::Union{Vector,Nothing}=nothing,
                parameters::Union{Vector,Nothing}=nothing)

    for _ in 1:iterations
        grads = gradient(model, inputs, targets)
        update!(model, grads, step_size)
        if gradients isa Vector
            push!(gradients, grads)
        end
        if losses isa Vector
           losses = push!(losses, MSE(vec(predict(model, inputs)), targets))
        end
        if parameters isa Vector
            push!(parameters, model.weights)
            push!(parameters, model.bias)
        end
    end
    model, losses, gradients, parameters
end
