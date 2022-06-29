using DelimitedFiles

dataset = readdlm("iris.data",',');

inputs = dataset[:,1:4]; # Coge todas las filas (:) y las 4 primeras columnas (en iris, las entradas)
targets = dataset[:,5]; # Matriz bidimensional de 1 columna (array)

inputs = convert(Array{Float32,2},inputs); # Se convierten todos los valores en Float32

unique_targets = unique(targets)
new_targets = zeros(size(targets,1), size(unique_targets,1))

if size(unique_targets, 1) == 2
    new_targets = Array{Bool, 2}(undef, size(inputs, 1), 1)
    new_targets[:, 1] .= (targets.==unique_targets[2])
else
    for i in 1:(size(unique_targets , 1))
        new_targets[:,i] = convert(Array{Bool,1}, targets.== unique_targets[i])
    end
end

using Statistics

min = minimum(inputs, dims = 1)
max = maximum(inputs, dims = 1)
media = mean(inputs, dims = 1)
desv = std(inputs, dims = 1)

normalizacion(v::Float32, m::Float32, d::Float32) = ((v-m)/d)

inputs = normalizacion.(inputs, media, desv)

println(inputs);
println(new_targets);

@assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo número de filas"