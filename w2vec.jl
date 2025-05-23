cd(@__DIR__)

using Pkg
Pkg.activate("/home/kguenel/Glove")

using Random
using .Iterators
using .Threads
using Statistics
using LinearAlgebra

using NNlib
using CUDA
using Flux
using Zygote
using Optimisers
using StatsBase


using Flux: @layer
using Flux: frequencies, DataLoader
using BSON: @load, @save

using ProgressMeter

using ExplainableAI
using TensorBoardLogger
using TensorBoardLogger: with_logger

CUDA.device!(1)

rng = Random.default_rng()
Random.seed!(rng, 0)


### multi-threaded negative sampler 


xpower(x) = x^.75

struct NegativeSampler
    vocab::Vector{Int}
    probs::Vector{Float64}
    w2i::Dict{Int,Int}
    rngs::Vector{MersenneTwister}  # One RNG per thread
    sample_size::Int
    power::Float64
end 


function NegativeSampler(corpus::Vector{Int}; sample_size=5, power=0.75)    
    word_counts = frequencies(corpus)
    vocab = collect(keys(word_counts))
    counts = values(word_counts)
    pow = xpower.(counts)
    normalizer = reduce(+, pow) # normalizer for probability distro.
    probs = pow ./ normalizer
    w2i = Dict(word => i for (i, word) in enumerate(vocab))
    # Initialize one RNG per thread
    rngs = [MersenneTwister(rand(UInt)) for _ in 1:nthreads()]
    @info "Negative Sampler Object is being generated..."
    return NegativeSampler(vocab, probs, w2i, rngs, sample_size, power)
end



function get_negative_samples(sampler::NegativeSampler, target_word::Int; num_samples=nothing)
    num_samples = isnothing(num_samples) ? sampler.sample_size : num_samples
    target_idx = get(sampler.w2i, target_word, -1)
    
    # Thread-safe sampling using threadid() to get the correct RNG
    tid = threadid() 
    local_rng = sampler.rngs[tid]
    
    # Create adjusted probabilities
    if target_idx > 0
        adjusted_probs = copy(sampler.probs)
        adjusted_probs[target_idx] = 0.0
        adjusted_probs ./= sum(adjusted_probs)
    else
        adjusted_probs = sampler.probs
    end
    
    # Sample using thread-local RNG
    sample_indices = sample(local_rng, 1:length(sampler.vocab), Weights(adjusted_probs), num_samples; replace=false)
    return sampler.vocab[sample_indices]
end


function get_negative_samples_batch(sampler::NegativeSampler, target_words::Vector{Int}; num_samples=nothing)
    num_samples = isnothing(num_samples) ? sampler.sample_size : num_samples
    results = Vector{Vector{Int}}(undef, length(target_words))
    
    @threads for i in eachindex(target_words)
        results[i] = get_negative_samples(sampler, target_words[i]; num_samples)
    end
    
    return reduce(hcat, results)
end



get_frequencies(corpus::String)::Dict = frequencies(split(corpus))


function subsample_frequent_words(corpus::String)
    filtered_corpus = String[]
    wordCounts = get_frequencies(corpus)
    tot_wordCoutns = sum(values(wordCounts))
    wordCounts = Dict(word => wordCounts[word] / tot_wordCoutns for (word, _) in wordCounts)
    @showprogress for word in split(corpus)
        rand() < (1 + sqrt(wordCounts[word] * 1e3)) * 1e-3 / wordCounts[word] ? push!(filtered_corpus, word) : nothing
    end
    return join(filtered_corpus, " ")
end



function positiveSampler(corpus::Vector{Int}; window_size::Int=4)
    tot_words = length(corpus)
    center = Int[]; context = Int[]; 
    @showprogress for i in 1:length(corpus)
        fcontextWord = max(1, i - window_size)
        lcontextWord = min(i + window_size, tot_words)
        for j in fcontextWord:lcontextWord
            if i != j
                push!(center, corpus[i])
                push!(context, corpus[j])
            end
        end
    end
    return center, context
end


################# MODEL ###################

struct Word2Vec
    V::Embedding
    U::Embedding    
end
@layer Word2Vec

function Word2Vec(VSIZE::Int, IN_DIM::Int)
    V = Embedding(VSIZE, IN_DIM)
    U = Embedding(VSIZE, IN_DIM)
    return Word2Vec(V, U)
end

(m::Word2Vec)(center, context, neg_samples) = m.V(center), m.U(context), m.U(neg_samples)

function loss(model::Word2Vec, center::V, context::V, neg_samples::M) where {V, M}

    # DxB,   DxB,     DxKxB  
    inputs, posouts, negouts = model(center, context, neg_samples)

    # inputs = reshape(inputs, 1, dsize, bsize) # (D, 1, B)
    # logsigmoid(dot(inputs, posouts))
    # pos_scores = sum(log.(sigmoid.(sum(dot(inputs, posouts))))) # scalar
    pos_scores = -logsigmoid(sum(inputs .* posouts, dims=1))
    # need to reshape the inputs
    dsize, bsize = size(inputs)
    inputs = reshape(inputs, dsize, 1, bsize) # (D, 1, B)
    negouts= permutedims(negouts, (1, 3, 2))
    neg_scores = sum(logsigmoid(-batched_mul(inputs, negouts)), dims=3)

    return -(pos_scores + neg_scores) / bsize
end


# Training loop
function train!(model::Word2Vec, dataloader::DataLoader, neg_samples, opt_state; epochs=10)
    p = Progress(epochs; color=:darkblue, showspeed=true)
    generate_showvalues(epoch, loss) = () -> [(:Epoch, epoch), (:Loss, loss)]
    bsize = dataloader.batchsize
    for epoch in 1:epochs
        trn_losses = Float32[];
        for(words, contexts) in dataloader
            negs = get_negative_samples_batch(neg_samples, contexts; num_samples=10);
            words, contexts, negs =  (words, contexts, negs) .|> gpu
            loss_, ∇model = Flux.withgradient(model, words, contexts, negs) do m, wrd, ctx, negs
                loss(m, wrd, ctx, negs)
            end
            Optimisers.update!(opt_state, model, ∇model[1]);
            push!(trn_losses, loss_)
            println(epoch, ':', loss_)
        end
        loss_ = mean(trn_losses)
        next!(p; showvalues = generate_showvalues(epoch, loss_))
    end
end


################### MAIN ########################

root_file = "/mnt/depo/github/GloVe/"
text = root_file * "text8"

corpus = read(text, String)

filteredCorpus = subsample_frequent_words(corpus)


vocab = filteredCorpus |> split |> unique .|> string

w2i = Dict(word => idx for (idx, word) in enumerate(vocab))
i2w = Dict(idx => word for (idx, word) in enumerate(vocab))

# transfer the filtered corpus to intCorpus 
intCorpus = collect(w2i[word] for word in split(filteredCorpus));


@info "Generating Positive Samples:"
centers, contexts = positiveSampler(intCorpus, window_size=8)
@info "Generator for Negative Samples is being generated"
neg_samples = NegativeSampler(intCorpus, sample_size=10);



VSIZE = length(vocab)

model = Word2Vec(VSIZE, 300) |> gpu

rule = Optimisers.OptimiserChain(Optimisers.ADAM(3e-4))
                                     # Optimisers.WeightDecay(1f-8),
                                     # Optimisers.ClipGrad(1));
opt_state = Optimisers.setup(rule, model);

dataloader = DataLoader((centers, contexts), batchsize=16, shuffle=true, partial=false)

ctr, ctx = first(dataloader)

vals_loss = []
@showprogress for (ctr, ctx) in dataloader
    negs = get_negative_samples_batch(neg_samples, ctx; num_samples=10);
    ctr, ctx , negs = (ctr, ctx , negs) .|> gpu
    push!(vals_loss, loss(model, ctr, ctx, negs))
end


@time for _ in 1:1000
    negs = get_negative_samples_batch(neg_samples, ctx |> cpu ; num_samples=10);
    ctr, ctx , negs = (ctr, ctx , negs) .|> gpu
    loss_, ∇model = Flux.withgradient(model, ctr, ctx, negs) do m, wrd, ctx, negs
                loss(m, wrd, ctx, negs)
        end
    Optimisers.update!(opt_state, model, ∇model[1]);
    println(loss_)
end


train!(model, dataloader, neg_samples, opt_state; epochs=10)


