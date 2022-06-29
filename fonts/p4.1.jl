using DelimitedFiles
using Statistics
using Flux
using Flux.Losses
using Random
using Random:seed!
using Plots

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

function train(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}},
     maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01, maxEpochsVal::Int=20, validationSet::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=([undef], [undef]),
     testSet::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=([undef], [undef]))

    trainingInputs = transpose(dataset[1]) # Matriz transpuesta de 4 filas y 150 columnas
    trainingTargets = transpose(dataset[2]) # Matriz transpuesta de 3 filas y 150 columnas

    testInputs = transpose(testSet[1])
    testTargets = transpose(testSet[2])

    if (isassigned(validationSet[1], 2))
        validationInputs = transpose(validationSet[1])
        validationTargets = transpose(validationSet[2])
    end


    # Se comprueban que el numero de patrones (filas) coincide tanto en entrenamiento, validacion y test
    @assert (size(trainingInputs, 2) == size(trainingTargets, 2))
    @assert (size(testInputs, 2) == size(testTargets, 2))
    
    if (isassigned(validationSet[1], 2))
        @assert (size(validationInputs, 2) == size(validationTargets, 2))
    end

    # Se comprueba que haya el mismo numero de columnas 
    if (isassigned(validationSet[1], 2))
            @assert (size(trainingInputs, 1) == size(validationInputs,1) == size(testInputs, 1))
            @assert (size(trainingTargets, 1) == size(validationTargets,1) == size(testTargets, 1))
    else 
        @assert (size(trainingInputs, 1)  == size(testInputs, 1))
        @assert (size(trainingTargets, 1) == size(testTargets, 1))
    end

    ann = rna_clasification(topology, trainingInputs, trainingTargets)

    loss(x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x), y) : Losses.crossentropy(ann(x), y); 

    
    trainingLosses = Float32[]
    trainingAccuracies = Float32[]
    validationLosses = Float32[]
    validationAccuracies = Float32[]
    testLosses = Float32[]
    testAccuracies = Float32[]
    numEpochs = 0;

    function calculateMetrics()

        trainingLoss = loss(trainingInputs, trainingTargets)
        testLoss = loss(testInputs, testTargets)

        trainingOutputs = classifyOutputs(ann(trainingInputs))
        testOutputs = classifyOutputs(ann(testInputs))
        
        trainingAccuracy = accuracy(trainingOutputs', trainingTargets')
        testAccuracy = accuracy(testOutputs', testTargets')
        if (isassigned(validationSet[1], 2))

            validationLoss = loss(validationInputs, validationTargets)

            validationOutputs = classifyOutputs(ann(validationInputs))
            validationAccuracy = accuracy(validationOutputs', validationTargets')

            #println("Epoch ", numEpochs, ": Training loss: ", round(trainingLoss, digits = 4), "\t, accuracy: ", round(100*trainingAccuracy, digits=3), "\t % - Validation loss: ", round(validationLoss, digits = 4), "\t, accuracy: ", round(100*validationAccuracy, digits=3), "\t % - Test loss: ", round(testLoss, digits = 4), "\t, accuracy: ", round(100*testAccuracy, digits=3), " %");

            return (trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy)
        else
            #println("Epoch ", numEpochs, ": Training loss: ", trainingLoss, ", accuracy: ", round(100*trainingAccuracy, digits=2) , ", % - Test loss: ", testLoss, ", accuracy: ", 100*testAccuracy, " %");

            return (trainingLoss, trainingAccuracy, testLoss, testAccuracy)
        end
        
    end

    (isassigned(validationSet[1], 2)) ?
        (trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy) = calculateMetrics() :
        (trainingLoss, trainingAccuracy, testLoss, testAccuracy) = calculateMetrics()
    

    push!(trainingLosses, trainingLoss)
    push!(trainingAccuracies, trainingAccuracy)

    push!(testLosses, testLoss)
    push!(testAccuracies, testAccuracy)

    if (isassigned(validationSet[1], 2))
        push!(validationLosses, validationLoss)
        push!(validationAccuracies, validationAccuracy)
        bestValLoss = validationLoss
    end

    numEpochsVal = 0; 
    bestAnn = deepcopy(ann)

    while (numEpochs < maxEpochs) && (trainingLoss > minLoss) && (numEpochsVal < maxEpochsVal)
        Flux.train!(loss, params(ann), [(trainingInputs, trainingTargets)], ADAM(learningRate))
        
        numEpochs +=1

        (isassigned(validationSet[1], 2)) ?
        (trainingLoss, trainingAccuracy, validationLoss, validationAccuracy, testLoss, testAccuracy) = calculateMetrics() :
        (trainingLoss, trainingAccuracy, testLoss, testAccuracy) = calculateMetrics()
    
        push!(trainingLosses, trainingLoss)
        push!(trainingAccuracies, trainingAccuracy)

        push!(testLosses, testLoss)
        push!(testAccuracies, testAccuracy)

        if (isassigned(validationSet[1], 2))
            push!(validationLosses, validationLoss)
            push!(validationAccuracies, validationAccuracy)

            if (validationLoss < bestValLoss)
                bestValLoss = validationLoss
                numEpochsVal = 0
                bestAnn = deepcopy(ann)
            else 
                numEpochsVal += 1
            end
        end

    end
        if (isassigned(validationSet[1], 2))
            return (bestAnn, trainingLosses, validationLosses, testLosses, trainingAccuracies, validationAccuracies, testAccuracies)
        else
            return (ann, trainingLosses, testLosses, trainingAccuracies, testAccuracies)
        end
end

function train(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
        maxEpochs::Int=1000, minLoss::Real=0, learningRate::Real=0.01, maxEpochsVal::Int=20, validationSet::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=([undef], [undef]),
        testSet::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=([undef], [undef]))

    train(topology, (dataset[1], reshape(dataset[2], (:,1))), maxEpochs, minLoss, learningRate, maxEpochsVal, (validationSet[1], reshape(validationSet[2], (:,1))),  (testSet[1], reshape(testSet[2], (:,1))))
end


function holdOut(N::Int64, P::Float64)
    @assert ((P>=0.) & (P<=1.))
    n = Int(round(N*P, RoundUp, digits=0))

    full_array = randperm(N)

    test_index = full_array[1:n]
    trainIdex = full_array[n+1:N]

    @assert (size(test_index,1) + size(trainIdex,1) == N)

    return (trainIdex, test_index)
end

function holdOut(N::Int64, Pval::Float64, Ptest::Float64)
    @assert ((Pval>=0.) & (Pval<=1.))
    @assert ((Ptest>=0.) & (Ptest<=1.))
    @assert ((Pval+Ptest)<=1.)

    trainValidationIndex, testIndex = holdOut(N, Ptest) # Se genera el array de indices de entrenamiento y validacion y los indices destinados para test
    
    trainingIndex, validationIndex = holdOut(length(trainValidationIndex), Pval * N / length(trainValidationIndex)) # Se generan nuevos indices para el array de indices

    @assert (size(trainValidationIndex[trainingIndex], 1) + size(trainValidationIndex[validationIndex], 1) + size(testIndex,1) == N )

    return (trainValidationIndex[trainingIndex], trainValidationIndex[validationIndex], testIndex)
end

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert(length(outputs) == length(targets))

    acc = accuracy(outputs, targets) 
    errorRate = 1. - acc

    recall = mean(outputs[targets]) # Sensibilidad
    specificity = mean(.!outputs[.!targets]) # Especificidad
    precision = mean(targets[outputs]) 
    NPV = mean(.!targets[.!outputs])

    if isnan(recall) && isnan(precision)
        recall = 1.
        precision = 1.
    elseif isnan(specificity) && isnan(NPV)
        specificity = 1.
        NPV = 1.
    end

    recall = isnan(recall) ? 0. : recall
    precision = isnan(precision) ? 0. : precision
    specificity = isnan(specificity) ? 0. : specificity
    NPV = isnan(NPV) ? 0. : NPV

    F1 = (recall==precision==0.) ? 0. : 2*(recall * precision)/(recall+precision)

    confMatrix = Array{Int64,2}(undef, 2, 2)

    confMatrix[1,1] = sum(.!targets .& .!outputs) # Verdaderos negativos
    confMatrix[1,2] = sum(.!targets .& outputs) # Falsos negativos
    confMatrix[2,1] = sum(targets .& .!outputs) # Falsos positivos
    confMatrix[2,2] = sum(targets .& outputs) # Verdaderos positivos

    return (accuracy, errorRate, recall, specificity, precision, NPV,F1, confMatrix)
end

function confusionMatrix(outputs:: AbstractArray{<:Real}, targets:: AbstractArray{<:Real},umbral:: AbstractArray{<:Real}=0.5)
    salidas = zeros(size(outputs))
    for i in size(outputs)
        salidas[i] = (outputs[i]>umbral)
    end
    return confusionMatrix(salidas,targets)
end

function unocontratodos(inputs::AbstractArray{Float64, 2}, targets::AbstractArray{Bool,2})
    numClasses = size(targets,2);
    numInstances = size(inputs, 1)
    @assert(numClasses>2)
    outputs = Array{Float32,2}(undef, numInstances, numClasses)

    for numClass in 1:numClasses
        model = fit(inputs, targets[:,[numClass]])
        outputs[:,numClass] .= model(inputs)
    end

    outputs = softmax(outputs')'
    vmax = maximum(outputs, dims=2)
    outputs = (outputs .== vmax)
    return outputs
end


function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}, weighted::Bool)
    @assert (size(outputs) == size(targets))
    nClasses=size(targets, 2)
    @assert (nClasses !=2)
    if (nClasses==1)
        return confusionMatrix(outputs[:,1],targets[:,1])
    else
        @assert (all(sum(outputs, dims=2).==1))

        recall = zeros(nClasses);
        specificity = zeros(nClasses);
        precision = zeros(nClasses);
        NPV = zeros(nClasses);
        F1 = zeros(nClasses);
        confMatrix = Array{Int64,2}(undef, nClasses, nClasses)

        numInstancesFromEachClass = vec(sum(targets, dims=1))
        
        for numClass in findall(numInstancesFromEachClass.>0)
            #COMPROBAR EL ORDEN(accuracy, error_rate, recall, speciticity, precision, negative_predictive_value,F1_score, matriz_confusion)
            (_, _, recall[numClass], specificity[numClass], precision[numClass], NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass], targets[:,numClass]);
        end

        for numClassTarget in 1:numClasses
            for numClassOutput in 1:numClasses
                confMatrix[numClassTarget, numClassOutput] = sum(targets[:,numClassTarget] .& outputs[:,numClassOutput])
            end
        end

        if !weighted
            numClassesWithInstances = sum(numInstancesFromEachClass.>0)
            recall = sum(recall)/numClassesWithInstances
            specificity = sum(specificity)/numClassesWithInstances
            precision = sum(precision)/numClassesWithInstances
            NPV = sum(NPV)/numClassesWithInstances
            F1 = sum(F1)/numClassesWithInstances
        else
            weights = numInstancesFromEachClass./sum(numInstancesFromEachClass)
            recall = sum(weights.*recall)
            specificity = sum(weights.*specificity)
            precision = sum(weights.*precision)
            NPV = sum(weights.*NPV)
            F1 = sum(weights.*F1)
        end

        acc = accuracy(outputs, targets)
        errorRate = 1 - acc;
        return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
    end
end

confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}, weighted::Bool) = confusionMatrix(classifyOutputs(outputs')', targets, weighted)

function confusionMatrix(outputs::AbstractArray{Any}, targets::AbstractArray{Any}, weighted::Bool)
    @assert(all([in(output, unique(targets)) for output in outputs]))
    classes = unique(targets)
    return confusionMatrix(oneHotEncoding(outputs,classes) ,oneHotEncoding(targets,classes) , weighted);
end

printConfusionMatrix(outputs::AbstractArray{Float32,2}, targets::AbstractArray{Bool,2}, weighted::Bool) = printConfusionMatrix(convert(Array{Float64,2}, outputs),targets,weighted);

function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}, weighted::Bool)
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = 
confusionMatrix(outputs, targets, weighted);
    numClasses = size(confMatrix,1);
    writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; 
println(""); );
    writeHorizontalLine(); 
    print("\t| ");

    if (numClasses==2) 
        println(" - \t + \t|"); 
    else
        print.("Cl. ", 1:numClasses, "\t| "); 
    end;
    println("");
    writeHorizontalLine();
    for numClassTarget in 1:numClasses
        # print.(confMatrix[numClassTarget,:], "\t"); 
        if (numClasses==2)
            print(numClassTarget == 1 ? " - \t| " : " + \t| "); 
        else
            print("Cl. ", numClassTarget, "\t| "); 
        end;
        print.(confMatrix[numClassTarget,:], "\t| "); 
        println("");
        writeHorizontalLine(); 
    end;
    println("Accuracy: ", acc);
    println("Error rate: ", errorRate);
    println("Recall: ", recall);
    println("Specificity: ", specificity);
    println("Precision: ", precision);
    println("Negative predictive value: ", NPV);
    println("F1-score: ", F1);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, 
confMatrix);
end;

printConfusionMatrix(outputs::AbstractArray{Float64,2}, targets::AbstractArray{Bool,2},
weighted::Bool) =  printConfusionMatrix(classifyOutputs(outputs')', targets, 
weighted)

topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda 
learningRate = 0.01; # Tasa de aprendizaje
maxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0.2; # Porcentaje de patrones que se usaran para validacion 
testRatio = 0.2; # Porcentaje de patrones que se usaran para test 
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el
minLoss = 0.1

dataset = readdlm("iris.data",',');

inputs = convert(Array{Float32,2}, dataset[:,1:4]); # Se convierten todos los valores en Float32
targets = oneHotEncoding(dataset[:,5]) # Matriz bidimensional de 1 columna (array)

@assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo número de filas"

numClasses = size(targets, 2)

@assert(numClasses>2)

normalizeMinMax!(inputs)
(trainingIndices, validationIndices, testIndices) = holdOut(size(inputs,1), validationRatio, testRatio)

trainingInputs    = inputs[trainingIndices,:]; 
validationInputs  = inputs[validationIndices,:]; 
testInputs        = inputs[testIndices,:]; 
trainingTargets   = targets[trainingIndices,:]; 
validationTargets = targets[validationIndices,:]; 
testTargets       = targets[testIndices,:];

outputs = Array{Float64,2}(undef, size(inputs,1), numClasses)
(ann, trainingLosses, validationLosses, testLosses, trainingAccuracies, validationAccuracies, testAccuracies) = train(topology, (trainingInputs, trainingTargets), maxEpochs, minLoss, learningRate, maxEpochsVal, (validationInputs, validationTargets), (testInputs, testTargets))

trainingOutputs = collect(ann(trainingInputs')')
printConfusionMatrix(trainingOutputs, trainingTargets, true) 
println("Results in the validation set:")
validationOutputs = collect(ann(validationInputs')');
printConfusionMatrix(validationOutputs, validationTargets, true); 
println("Results in the test set:")
testOutputs = collect(ann(testInputs')');
printConfusionMatrix(testOutputs, testTargets, true);
println("Results in the whole dataset:")
outputs = collect(ann(inputs')');
printConfusionMatrix(outputs, targets, true);