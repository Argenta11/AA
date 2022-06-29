using DelimitedFiles
using Statistics
using Flux
using Flux.Losses


function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    unique_classes = unique(classes)

    if size(unique_classes, 1) == 2
        bool_array = Array{Bool, 2}(undef, size(feature, 1), 1)
        bool_array[:, 1] .= (feature.==classes[2])
        return bool_array
    else
        bool_array = falses(size(feature,1), size(unique_classes, 1))
        for i in 1:(size(unique_classes,1 ))
            bool_array[:,i] = convert(Array{Bool,1}, feature.== unique_classes[i])
        end
        return bool_array
    end

end

function oneHotEncoding(feature::AbstractArray{<:Any,1})
    oneHotEncoding(feature, unique(feature))
end

function oneHotEncoding(feature::AbstractArray{Bool,1})
    return reshape(feature, (:,1))
end

function calculateMinMaxNormalizationParameters(input::AbstractArray{<:Real,2})

    min = minimum(input, dims = 1)
    max = maximum(input, dims = 1)
    return(min, max)

end

function calculateZeroMeanNormalizationParameters(input::AbstractArray{<:Real,2})
    
    media = mean(input, dims = 1)
    desv = std(input, dims = 1)
    return(media, desv)
end

function normalizeMinMax!(input::AbstractArray{<:Real,2}, parameters::NTuple{2, AbstractArray{<:Real,2}})
    
    input .-= parameters[1]
    input ./= (parameters[2] - parameters[1])

end

normalizeMinMax!(input::AbstractArray{<:Real,2}) = normalizeMinMax!(input, calculateMinMaxNormalizationParameters(input))

function normalizeMinMax!(input::AbstractArray{<:Real,2})
    parameters = calculateMinMaxNormalizationParameters(input)
    normalizeMinMax!(input, parameters)
end

function normalizeMinMax(input::AbstractArray{<:Real,2}, parameters::NTuple{2, AbstractArray{<:Real,2}})
    new_matrix = copy(input)
    normalizeMinMax!(copy, parameters)
    return new_matrix
end

normalizeMinMax(input::AbstractArray{<:Real,2}) = normalizeMinMax(input, calculateMinMaxNormalizationParameters(input))

function normalizeZeroMean!(input::AbstractArray{<:Real,2}, parameters::NTuple{2, AbstractArray{<:Real,2}})
    return (input.-parameters[1])./parameters[2]
end

function normalizeZeroMean!(input::AbstractArray{<:Real,2})
    return normalizeZeroMean!(input, calculateZeroMeanNormalizationParameters(input))
end

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)

    if size(outputs,1) == 1
        
        return outputs .>= threshold
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=1)
        outputs = falses(size(outputs))
        outputs[indicesMaxEachInstance] .= true
    end

    return (outputs)

end

function accuracy(targets::AbstractArray{Bool, 1}, outputs::AbstractArray{Bool, 1})
    @assert (size(targets,1)==size(outputs,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo número de filas"
    
    return count(targets .== outputs)/size(outputs, 1)
end

function accuracy(targets::AbstractArray{Bool, 2}, outputs::AbstractArray{Bool, 2})

    if size(targets, 2) == 1
        accuracy(targets[:,1], outputs[:,1])
    elseif size(targets, 2) > 2
        count = 0
        for i in 1:(size(targets, 1))
            if targets[i, :] == outputs[i, :]
                count += 1
            end
        end
        return count/size(targets, 1)
    end
end

function accuracy(targets::AbstractArray{Bool, 1}, outputs::AbstractArray{<:Real,1}, threshold::Real=0.5)
    outputs = outputs .> 0.5 
    return accuracy(targets, outputs)
end

function accuracy(targets::AbstractArray{Bool,2}, outputs::AbstractArray{<:Real,2})
    classComparison = targets .== outputs
    correctClassifications = all(classComparison, dims=2)
    return mean(correctClassifications)
end

function rna_clasification(topology::AbstractArray{<:Int,1}, targets, outputs)

    ann = Flux.Chain()
    numInputsLayer = size(targets, 1)
    
    for numOutputsLayer = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, σ))
        numInputsLayer = numOutputsLayer
    end

    if size(outputs, 1) == 1
        ann = Flux.Chain(ann..., Dense(numInputsLayer, size(outputs,1), σ))
    else 
        ann = Chain(ann..., Dense(numInputsLayer, size(outputs,1)))
        ann = Flux.Chain(ann..., softmax)
    end

    return ann
end

function train(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01)
    inputs = transpose(dataset[1]) # Matriz transpuesta de 4 filas y 150 columnas
    targets = transpose(dataset[2]) # Matriz transpuesta de 3 filas y 150 columnas

    ann = rna_clasification(topology, inputs, targets)

    loss(inputs, targets) = (size(targets,1) == 1) ? Losses.binarycrossentropy(ann(inputs), targets) : Losses.crossentropy(ann(inputs), targets); 

    for _ in 1:maxEpochs
        Flux.train!(loss, params(ann), [(inputs, targets)], ADAM(learningRate))
    end
    return (ann, loss)
end

function train(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01)
    outputs = reshape(dataset[2], (:,1))

    train(topology, (dataset[1], outputs), maxEpochs, minLoss, learningRate)
end

topology = [4,3]
learningRate = 0.01
minLoss = 0
numMaxEpochs = 1000

dataset = readdlm("iris.data",',');

inputs = dataset[:,1:4]; # Coge todas las filas (:) y las 4 primeras columnas (en iris, las entradas)

targets = dataset[:,5]; # Matriz bidimensional de 1 columna (array)

@assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo número de filas"

inputs = convert(Array{Float32,2},inputs); # Se convierten todos los valores en Float32
targets = oneHotEncoding(targets)

inputs = normalizeMinMax!(inputs)

(ann, loss_array) = train(topology, (inputs, targets), numMaxEpochs, minLoss, learningRate)
