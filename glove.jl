"""

=========== IMPORTANT ! =========

Original implementation of Glove does not use any "<UNK>" tag during training.
It does calculate the "<UNK>" keyword symbol by taking the average of the last 
100 words - including Word Embedding and Context Embedding !  

So in order to obtain this special tag just finish and save the training.
Later just calculate the mentioned procedure.
"""

ENV["JULIA_PYTHONCALL_EXE"] = "/home/kguenel/miniconda3/envs/tokenizers/bin/python3"

cd(@__DIR__)
using Pkg
Pkg.activate("/home/kguenel/Glove")

using Dates
using Logging
using LinearAlgebra
using Statistics
using Printf
using Base.Threads
using Distances
using Random

Random.seed!(1234)

using OhMyREPL


# using XLEs

using CUDA
CUDA.device!(0)
using Flux
using Optimisers
using Zygote


using Flux: DataLoader
using Flux: @layer, create_bias
using BSON: @save, @load
using Test
using ProgressMeter


struct FileLogger <: AbstractLogger
    io::IO
end

function Logging.shouldlog(::FileLogger, level, _module, group, id)
    return level >= Logging.Info
end

function Logging.min_enabled_level(::FileLogger)
    return Logging.Info
end

function Logging.handle_message(j::FileLogger, level, message,
                                _module, group, id, file, line; kwargs...)
    println(j.io, "[$level] $message")
end


"""reading glove's cooccurrence data 
"""
function readCooccurrence(file::String)
    file = open(file, "r")
    data = read(file)
    buffer = IOBuffer(data)

    word1 = Int32[]; word2 = Int32[]; vals  = Float64[]

    while !eof(buffer)
        push!(word1, read(buffer, Int32))
        eof(buffer) ? break : nothing
        push!(word2, read(buffer, Int32))
        eof(buffer) ? break : nothing
        push!(vals, read(buffer, Float64))
    end
    return word1, word2, vals
end 

function get_weights(cooccurs::Vector{Float64}; XMAX::Int64=10, ALPHA::Real=.75)
    x = Float32(1.) ::Float32
    log_occurs = Float32.(log.(cooccurs)) ::Vector{Float32}
    weights = Array{Float32}(undef, length(log_occurs));
    @inbounds for i in eachindex(cooccurs)
        weights[i] = cooccurs[i] < XMAX ? (cooccurs[i] / XMAX) ^ ALPHA : x
    end
    return weights, log_occurs
end


struct Glove
    WE::Embedding
    CE::Embedding
    wbias::Embedding
    cbias::Embedding
end
@layer Glove

function (glove::Glove)(w::T, c::T) where {T}
    words = glove.WE(w)
    context = glove.CE(c)
    wbias = glove.wbias(w) 
    cbias = glove.cbias(c) 
    s = sum(words .* context, dims=1)
    return (s + wbias + cbias) |> vec
end

function loss(model::Glove, word::T, context::T, cooccurs::R, weight::R) where {T, R}
    # t = 2 ; half = .5
    loss = (model(word, context) .- cooccurs).^ 2
    return mean(loss .* weight) * .5
end


function createWeight(VSIZE::Integer, IN_DIM::Integer)
    W = Matrix{Float32}(undef, VSIZE, IN_DIM)
    RAND_MAX = 2147483647
    for i in axes(W, 2)
        for j in axes(W, 1)
            W[j, i] = (rand() / RAND_MAX - 0.5) / IN_DIM
        end
    end
    return W .* .02
end

createUniform(VSIZE::Int64, IN_DIM::Int64) = 2 * 0.01 * rand(VSIZE, IN_DIM) .- 0.01


function createParams(VSIZE::Int64, IN_DIM::Int64; init::Function=createUniform)
    WE = Embedding(VSIZE, IN_DIM, init=init);
    CE = Embedding(VSIZE, IN_DIM, init=init);
    wbias = Embedding(VSIZE, 1, init=init);
    cbias = Embedding(VSIZE, 1, init=init);
    return Glove(WE, CE, wbias, cbias)
end


function my_train!(glove::Glove, train_data::DataLoader, opt_state; epochs::Int=35)
    p = Progress(epochs; color=:darkblue, showspeed=true)
    generate_showvalues(epoch, loss) = () -> [(:Epoch, epoch), (:Loss, loss)]
    for epoch in 1:epochs
        trn_losses = Float32[];
        for(word, context, y, weight) in train_data
            word, context, y, weight =  (word, context, y, weight) .|> gpu
            loss_, ∇glove = Flux.withgradient(glove, word, context, y, weight) do g, wrd, ctx, y, wei
                # loss(g, wrd |> gpu, ctx |> gpu, y |> gpu, wei |> gpu)
                loss(g, wrd, ctx, y, wei)
            end
            Optimisers.update!(opt_state, glove, ∇glove[1]);
            push!(trn_losses, loss_)
        end

        # @info "Epoch: $(epoch),
        # Loss: $(mean(trn_losses)), "
        # flush(log_file)
        loss_ = mean(trn_losses)
        next!(p; showvalues = generate_showvalues(epoch, loss_))

        # if mean(trn_losses) < loss_min
            # @printf "Saving model from epoch %i" epoch
            # model_state = Flux.state(glove |> cpu)
            # @save "./models/text8/$(IN_DIM)/$(epoch)-seed-6666.bson" model_state opt_state
        #    epochs_no_improve = 0
        #    loss_min = mean(trn_losses)
        # else
        #    @info "Skipping saving"
        #    epochs_no_improve += 1
        #    if epochs_no_improve == n_epochs_stop
        #        @warn "Stopping training :  no improvement!"
        #        break
        #    end
        # end
        # @printf "\n"

    end
end


epochs = 10
freq   = 10
# λ = 1e-2
λ = 2e-3
@info "Learining Rate : $λ"
distro = :glorot_uniform
# distros = map(Symbol, [glorot_uniform, glorot_normal, kaiming_uniform,
#                        kaiming_normal, truncated_normal, orthogonal, identity_init)


# distro = :kaiming_uniform

root_file = "/mnt/depo/github/GloVe/"
vfile = root_file * "vocab.txt"
@info "Reading Vocabulary File : $(vfile)"
vocab = split.(readlines(vfile), ' ')
# vocab table
V = vocab .|> first .|> string
# frequency table
# F = vocab .|> last .|> string .|> i -> parse(Int64, i)
# special token for unknown words

global XMAX = 10
global ALPHA = .75
global VSIZE = length(V)
global IN_DIM = 64


w2i = Dict(word => idx for (idx, word) in enumerate(V))
i2w = Dict(idx => word for (idx, word) in enumerate(V))


file  = root_file * "cooccurrence.bin"
# file  = "./wmt.cooccurrence.shuf.bin"
@info "Reading Cooccurrence File :  $(file)"
left, right, cooccurs = readCooccurrence(file)
weights, log_occurs = get_weights(cooccurs);


@info "Creating Training Data"
# idx = randperm(MersenneTwister(1234), 30000000)
global BSIZE =  div(length(left), 1000) # seems to be the optimal batch
# global BSIZE =  div(length(left), 2000)
train_data = DataLoader((left, right, log_occurs, weights), batchsize=BSIZE, partial=true, shuffle=true)
# @info "Total Number of Batches: " div(length(left), BSIZE)

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
log_path = "/home/phd/Documents/Thesis/GloVe/$freq/"
logfile = "$(log_path)/$(λ)_$(timestamp).log"
log_file = open(logfile, "w+")
global_logger(FileLogger(log_file))

flush(log_file)

# for distro in distros

    @info "Creating Model"
    # or createWeight for gloVe style initialization
    glove = createParams(VSIZE, IN_DIM; init=Flux.eval(distro)) |> gpu
#     glove = Glove(WE, CE, wbias, cbias) |> gpu
    @info "Distro: $(distro)"
    @info "Setting Optimisers"


    rule = Optimisers.OptimiserChain(Optimisers.ADAM(λ))
                                     # Optimisers.WeightDecay(1f-8),
                                     # Optimisers.ClipGrad(1));
    opt_state = Optimisers.setup(rule, glove);

    @info "Started Training"
    my_train!(glove, train_data, opt_state, epochs=epochs)


    contextVectors = glove.CE.weight |> Array
    wordVectors = glove.WE.weight  |> Array

    r, c = size(contextVectors)
    G = reduce(hcat, [wordVectors, contextVectors])
    G = reshape(G, r, c, 2)
    L = mean(G, dims=3)[:, :, 1] # |> XLEs.unit


    # cos = pairwise(CosineDist(), M, L)

    save_at = "./$freq/$distro/$(λ)/ "
    !isdir(save_at) ? mkpath(save_at) : nothing
    @info "Writing to $save_at"
    writeEmbeds(save_at * "text8_cosine_loss_$(λ)", V, L, vinfo=false)
    flush(log_file)
# end

function writeEmbeds(file::String, voc::Array{String}, embed::M; vinfo::Bool=false) where {M}
    if size(embed, 1) < size(embed, 2)
        @warn "Need to convert the embedding matrix format from column to raw major"
        embed = Matrix(permutedims(embed))
    end
    @info "Writing Embedding file as .txt - will consume too much space."
    s = open(file * ".vec", "w+")
    lines = length(voc)
    if vinfo
        write(s, string(lines) * " " * string(size(embed, 2)) * "\n")
    end
    for i in 1:lines
        write(s, voc[i] * " " * join(string.(embed[i, :]), " ") * "\n")
    end
    close(s)
end


close(log_file)
