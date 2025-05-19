ENV["JULIA_PYTHONCALL_EXE"] = "/home/kguenel/miniconda3/envs/tokenizers/bin/python3"


cd(@__DIR__)
using Pkg
Pkg.activate("/home/kguenel/Glove")

using .Iterators
using Dates
using Logging
using LinearAlgebra
using Statistics
using Printf
using Base.Threads
using Distances
using Random

Random.seed!(1234)

# using OhMyREPL


# using XLEs

using CUDA
CUDA.device!(1)
using Flux
using Optimisers
using Zygote

using Flux: glorot_uniform, glorot_normal, kaiming_uniform,
    kaiming_normal, truncated_normal, orthogonal, identity_init
using Flux: DataLoader
using Flux: Embedding
using Flux: Losses


using BytePairEncoding

using BSON: @save, @load
using Test
using ProgressMeter


using TensorBoardLogger
using TensorBoardLogger: with_logger


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

mutable struct Norm

	WE::Float32
	CE::Float32
	wbias::Float32
	cbias::Float32
	counter::Int32
	Norm(WE=.0, CE=.0, wbias=.0, cbias=.0, counter=0) = new(WE, CE, wbias, cbias, counter)
end

function accumulateNorm!(model, n::Norm)
	n.WE = (model.WE.weight |> norm) + n.WE 
	n.CE = (model.CE.weight |> norm) + n.CE 
	n.wbias = (model.wbias.weight |> norm) + n.wbias
	n.cbias = (model.cbias.weight |> norm) + n.cbias
	n.counter +=1 
	return nothing 
end


function mapnorm!(f, n::Norm)
    n.WE = f(n.WE)
    n.CE = f(n.CE)
    n.wbias = f(n.wbias)
    n.cbias = f(n.cbias)
    return n
end

function calculatNorm!(n::Norm)
	counter = n.counter
	n = mapnorm!(x -> x / counter, n )
	return nothing
end


function resetNorm!(n::Norm)
	n = mapnorm!(x -> x = 0., n)
	n.counter = 0
	return nothing
end

struct GloveBag
    WE::EmbeddingBag
    CE::EmbeddingBag
    wbias::EmbeddingBag
    cbias::EmbeddingBag
end
Flux.@layer GloveBag

function lossBag(model::GloveBag, word::T, context::T, cooccurs::R, weight::R) where {T, R}
    loss = model(word, context) - cooccurs
    return mean((loss).^2 .* weight) * .5
end

function (glove::GloveBag)(w::T, c::T) where {T}
    words = glove.WE(w)
    context = glove.CE(c)
    wbias = glove.wbias(w) |> permutedims
    cbias = glove.cbias(c) |> permutedims
    s = sum(words .* context, dims=1) |> permutedims
    return s + wbias + cbias
end

function createParamsBag(VSIZE::Int64, IN_DIM::Int64; init::Function=createUniform)
    WE = EmbeddingBag(VSIZE => IN_DIM; init=Flux.identity_init(gain=5));
    CE = EmbeddingBag(VSIZE => IN_DIM;init=Flux.identity_init(gain=5));
    wbias = EmbeddingBag(VSIZE => 1; init=Flux.identity_init(gain=5));
    cbias = EmbeddingBag(VSIZE => 1; init=Flux.identity_init(gain=5));
    return WE, CE, wbias, cbias
end


struct Glove
    WE::Embedding
    CE::Embedding
    wbias::Embedding
    cbias::Embedding
end
Flux.@layer Glove

function loss(model::Glove, word::T, context::T, cooccurs::R, weight::R) where {T, R}
    # t = 2 ; half = .5
    loss = (model(word, context) - cooccurs).^ 2
    return mean(loss .* weight) * .5
end

function (glove::Glove)(w::T, c::T) where {T}
    words = glove.WE(w)
    context = glove.CE(c)
    wbias = glove.wbias(w) |> permutedims
    cbias = glove.cbias(c) |> permutedims
    s = sum(words .* context, dims=1) |> permutedims
    return s + wbias + cbias
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
    return WE, CE, wbias, cbias
end

function fill_param_dict!(dict, m, prefix)
	dict[prefix * "AE"] = m.WE.weight
	dict[prefix * "WE"] = m.CE.weight
	dict[prefix * "wbias"] = m.wbias.weight
	dict[prefix * "cbias"] = m.cbias.weight
	return nothing
end

function fill_norm_dict!(dict, normholder, prefix)
	dict[prefix * "AE"] = normholder.WE
	dict[prefix * "WE"] = normholder.CE
	dict[prefix * "wbias"] = normholder.wbias
	dict[prefix * "cbias"] = normholder.cbias
	return nothing
end

function TBCallBack(model, loss_::Float32, normholder::Norm)

	param_dict = Dict{String, Any}()
	norm_dict = Dict{String, Any}()

	fill_param_dict!(param_dict, model, "")
	fill_norm_dict!(norm_dict, normholder, "")
	with_logger(logger) do 
		@info "Model" params=param_dict log_step_increment=1
		@info "Train" params=loss_ log_step_increment=1
		@info "Norm" params=norm_dict log_step_increment=1
	end

end

function TBCallBackGrads(∇model, ∇normholder::Norm)

	param_dict = Dict{String, Any}()
	norm_dict = Dict{String, Any}()

	fill_param_dict!(param_dict, ∇model, "∇")
	fill_norm_dict!(norm_dict, ∇normholder, "∇")
	with_logger(logger) do 
		@info "∇Model" params=param_dict log_step_increment=1
		@info "∇Norm" params=norm_dict log_step_increment=1
	end
end

function my_train!(glove::Glove, train_data::DataLoader, opt_state; epochs::Int=35)
    p = Progress(epochs; color=:darkblue, showspeed=true)
    generate_showvalues(epoch, loss) = () -> [(:Epoch, epoch), (:Loss,  loss)]
    normholder = Norm()
    ∇normholder = Norm()
    for epoch in 1:epochs
        trn_losses = Float32[];
        grads = []; ∇glove = nothing;
        for(word, context, y, weight) in train_data
            loss_, ∇glove = Flux.withgradient(glove, word, context, y, weight) do g, wrd, ctx, y, wei
                loss(g, wrd |> gpu, ctx |> gpu, y |> gpu, wei |> gpu)          
            end
 			Optimisers.update!(opt_state, glove, ∇glove[1]);
            push!(trn_losses, loss_)
            accumulateNorm!(glove, normholder)
            accumulateNorm!(∇glove[1], ∇normholder)
        end

        push!(grads, ∇glove[1])
        loss_ = mean(trn_losses)
        next!(p; showvalues = generate_showvalues(epoch, loss_))

        calculatNorm!(normholder)
        calculatNorm!(∇normholder)

        TBCallBackGrads(∇glove[1], ∇normholder)
        TBCallBack(glove, Float32(loss_), normholder)

        map(resetNorm!, [normholder, ∇normholder])
    end
end

function trainBag!(glove::GloveBag, train_data::DataLoader, opt_state; epochs::Int=35)
    p = Progress(epochs; color=:darkblue, showspeed=true)
    generate_showvalues(epoch, loss) = () -> [(:Epoch, epoch), (:Loss,  loss)]
    normholder = Norm()
    ∇normholder = Norm()
    batch = 1
    for epoch in 1:epochs
        trn_losses = Float32[];
        grads = []; ∇glove = nothing;
        for(word, context, y, weight) in train_data
			word = collect(left_w2i[idx] for idx in  word)
			context = collect(right_w2i[idx] for idx in  context)
            loss_, ∇glove = Flux.withgradient(glove, word, context, y, weight) do g, wrd, ctx, y, wei           
                lossBag(g, wrd |> gpu, ctx |> gpu, y |> gpu, wei |> gpu)          
            end
 			Optimisers.update!(opt_state, glove, ∇glove[1]);
            push!(trn_losses, loss_)
            accumulateNorm!(glove, normholder)
            accumulateNorm!(∇glove[1], ∇normholder)
            println(batch, "/ $(length(train_data))")
            batch += 1
        end

        push!(grads, ∇glove[1])
        loss_ = mean(trn_losses)
        next!(p; showvalues = generate_showvalues(epoch, loss_))

        calculatNorm!(normholder)
        calculatNorm!(∇normholder)

        TBCallBackGrads(∇glove[1], ∇normholder)
        TBCallBack(glove, Float32(loss_), normholder)

        map(resetNorm!, [normholder, ∇normholder])
        batch = 1
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

root_file = "/home/kguenel/Documents/github/GloVe/"
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
global VSIZE = length(V) + 6
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
global BSIZE =  div(length(left), 10000)
# global BSIZE =  div(length(left), 2000)
train_data = DataLoader((left, right, log_occurs, weights), batchsize=BSIZE, partial=true, shuffle=true)
# @info "Total Number of Batches: " div(length(left), BSIZE)


	
@info "Creating Model"
    # or createWeight for gloVe style initialization
WE, CE, wbias, cbias = createParams(VSIZE, IN_DIM; init=eval(distro))
glove = Glove(WE, CE, wbias, cbias) |> gpu;

rule = Optimisers.OptimiserChain(Optimisers.ADAM(λ))
                                 # Optimisers.WeightDecay(1f-8),
                                 # Optimisers.ClipGrad(1));
opt_state = Optimisers.setup(rule, glove);

logger = TBLogger(root_file * "content/log_$(λ)")

@info "Started Training"
my_train!(glove, train_data, opt_state, epochs=epochs)


### the world of tokenizers 


left_word = collect(i2w[idx] for idx in left) 
right_word = collect(i2w[idx] for idx in right) 


left_words, right_words = map(unique, [left_word, right_word])



enc = BytePairEncoding.load_tiktoken_encoder("cl100k_base")


left_tiktokens = Vector{Vector{Int64}}(undef, length(left_words))
right_tiktokens = Vector{Vector{Int64}}(undef, length(right_words))

@showprogress [left_tiktokens[idx] = enc.encode(word) for (idx, word) in enumerate(left_words)]
@showprogress [right_tiktokens[idx] = enc.encode(word) for (idx, word) in enumerate(right_words)]


tvocab = enc.vocab.list.vector
tiktoken_w2i = Dict(word => idx for (idx, word) in enumerate(tvocab))
tiktoken_i2w = Dict(idx => word for (idx, word) in enumerate(tvocab))


# prototype 


global left_w2i = Dict(word => idx for (word, idx) in zip(left_words, left_tiktokens))
left_i2w = Dict(idx => word for (word, idx) in zip(left_words, left_tiktokens))

global right_w2i = Dict(word => idx for (word, idx) in zip(left_words, right_tiktokens))
right_i2w = Dict(idx => word for (word, idx) in zip(left_words, right_tiktokens))

F = union(left_tiktokens, right_tiktokens) |> flatten |> unique

# global VSIZE = length(F)
global VSIZE = length(tvocab)

tok2i = Dict(tok_id => id for (id, tok_id) in enumerate(F))

dataloader = DataLoader((left_word, right_word, log_occurs, weights), batchsize=1024, partial=true, shuffle=true)
# dataloader = DataLoader((left_tiktokens, right_tiktokens, log_occurs, weights), batchsize=BSIZE, partial=true, shuffle=true)

@info "Creating Bagging Model"
    # or createWeight for gloVe style initialization

WE, CE, wbias, cbias = createParamsBags(VSIZE, IN_DIM; init=Flux.eval(distro))
gloveBag = GloveBag(WE, CE, wbias, cbias) |> gpu;

rule = Optimisers.OptimiserChain(Optimisers.ADAM(λ))
                                 # Optimisers.WeightDecay(1f-8),
                                 # Optimisers.ClipGrad(1));
opt_state = Optimisers.setup(rule, gloveBag);

logger = TBLogger(root_file * "content/log_all_BagModel_$(λ)")

@info "Started Training"
trainBag!(gloveBag, dataloader, opt_state, epochs=epochs)
	






λ = 1e-2

WE, CE, wbias, cbias = createParamsBag(VSIZE, IN_DIM; init=eval(distro))
gloveBag = GloveBag(WE, CE, wbias, cbias) |> gpu;

rule = Optimisers.OptimiserChain(Optimisers.ADAM(λ),
                                 # Optimisers.WeightDecay(1f-8),
                                 Optimisers.ClipGrad(1));
opt_state = Optimisers.setup(rule, gloveBag);


logger = TBLogger(root_file * "content/log_BagModel_$(λ)")

word, context, logs, weight = first(dataloader)
word = collect(left_w2i[idx] for idx in  word)
context = collect(right_w2i[idx] for idx in  context)

# convert word lists into smaller lists
small_word_list = Vector{Vector{Int64}}(undef, length(word))
small_context_list = Vector{Vector{Int64}}(undef, length(word))

@showprogress for (id, (wlist, clist)) in enumerate(zip(word, context))
	w_aux = Vector{Int64}(undef, length(wlist))
	c_aux = Vector{Int64}(undef, length(clist))
	for (i, w) in enumerate(wlist)
		w_aux[i] = tok2i[w]
	end

	for (i, c) in enumerate(clist)
		c_aux[i] = tok2i[c]
	end

	small_word_list[id] = w_aux
	small_context_list[id] = c_aux
end



word = small_word_list
context = small_context_list

normholder = Norm()
∇normholder = Norm()
trn_losses = Float32[];

@showprogress for epoch in 1:10
    grads = []; ∇glove = nothing;
	loss_, ∇glove = Flux.withgradient(gloveBag, word, context, logs, weight) do g, wrd, ctx, y, wei           
            lossBag(g, wrd |> gpu, ctx |> gpu, y |> gpu, wei |> gpu)          
        end
	Optimisers.update!(opt_state, gloveBag, ∇glove[1]);
	push!(trn_losses, loss_)
	accumulateNorm!(gloveBag, normholder)
    accumulateNorm!(∇glove[1], ∇normholder)

	push!(grads, ∇glove[1])
    # loss_ = mean(trn_losses)


    calculatNorm!(normholder)
    calculatNorm!(∇normholder)

    TBCallBackGrads(∇glove[1], ∇normholder)
    TBCallBack(gloveBag, Float32(loss_), normholder)
    map(resetNorm!, [normholder, ∇normholder])
end


































