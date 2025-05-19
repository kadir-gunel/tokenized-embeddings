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
using Distributions


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



# Tokenize and prepare data
function preprocess(corpus)
    words = split(lowercase(corpus))
    vocab = Dict(word => i for (i, word) in enumerate(unique(words)))
    idx_to_word = [word for (word, _) in sort(collect(vocab), by = x -> x[2])]
    word_indices = [vocab[word] for word in words]
    return word_indices, vocab, idx_to_word
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


"""
# Generate skip-gram pairs
function generate_pairs(corpus::String, w2i::Dict, word_indices; window_size=4, sample_size=8)
    center = []; context = []; negs = []
    neg_sampler = sample_negatives(corpus, w2i; sample_size=sample_size)
    @showprogress for i in 1:length(word_indices)
        fcontextWord = max(1, i - window_size)
        lcontextWord = min(i + window_size, length(word_indices))
        for j in fcontextWord:lcontextWord
            if i != j
                push!(center, word_indices[i])
                push!(context, word_indices[j])
                push!(negs, collect(first(neg_sampler)))
            end
        end
    end
    return center, context
end
"""

function get_distribution(corpus::Vector{Int})
    sample_probs = Dict{Int, Float32}()
    wordCounts = frequencies(corpus)
    normFactor = sum(v^.75 for v in values(wordCounts))
    sample_probs = Dict(word => Float32(count^.75 / normFactor) for (word, count) in wordCounts)
    vocab = wordCounts |> keys |> collect
    probs = collect(sample_probs[w] for w in vocab)
    dist = Categorical(probs)
    return vocab, dist
end

"""
function sample_negatives(corpus::Vector{T}, 
                          center::Vector{R},
                          context::Vector{R};
                          sample_size::Int=25) where {T, R}

    neg_samples = Vector{Vector{Int64}}(undef, length(center))
    ucenter = center |> unique
    vocab, distribution = get_distribution(corpus)

    @showprogress for (id, cidx) in enumerate(ucenter)
        forbiddens = unique(context[center .== cidx])
        negs = Int64[]
        while length(negs) < sample_size
            w = vocab[rand(distribution)]
            if !(w in forbiddens)
                push!(negs, w)
            end
        end
        neg_samples[id] = negs
    end
    return negs
    # return repeated(vocab[rand(dist)] for _ in 1:sample_size)
end
"""

function sample_negatives(corpus::Vector{Int}; sample_size::Int=25)
    sample_probs = Dict{Int, Float32}()
    wordCounts = frequencies(corpus)
    normFactor = sum(v^.75 for v in values(wordCounts))
    sample_probs = Dict(word => Float32(count^.75 / normFactor) for (word, count) in wordCounts)
    vocab = wordCounts |> keys |> collect
    probs = collect(sample_probs[w] for w in vocab)
    dist = Categorical(probs)
    return repeated(vocab[rand(dist)] for _ in 1:sample_size)
end

function sample_negatives(target, vocab::Vector{Int64}, distribution; sample_size::Int=25)
    neg_samples = Int[]
    while length(neg_samples) < sample_size
        w = vocab[rand(dist)]
        w != target ? push!(neg_samples, w) : nothing
    end
    return neg_samples
end


function generate_neg_samples(neg_samples; bsize::Int=64)
    x = collect(first.(neg_samples.x) for _ in 1:bsize)
    reduce(hcat, x)
end

function generate_neg_samplesMT(neg_samples; bsize::Int64)

    x = Vector{Vector{eltype(first(neg_samples.x))}}(undef, bsize)
    @threads for i in 1:bsize
        x[i] = collect(first.(neg_samples.x))
    end
    reduce(hcat, x)

end 

function generate_pairs(corpus::Vector{Int}; window_size::Int=4) # , sample_size::Int=25)
    tot_words = length(corpus)
    center = Int32[]; context = Int32[]; # neg_samples = []
    # vocab, distribution = get_distribution(corpus)
    @showprogress for i in 1:length(corpus)
        fcontextWord = max(1, i - window_size)
        lcontextWord = min(i + window_size, tot_words)
        for j in fcontextWord:lcontextWord
            if i != j
                push!(center, corpus[i])
                push!(context, corpus[j])
                # push!(neg_samples, sample_negatives(corpus[j], vocab, distribution; sample_size=sample_size))
            end
        end
    end
    return center, context # , neg_samples
end

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
    pos_scores = sum(log.(sigmoid.(sum(dot(inputs, posouts))))) # scalar

    # need to reshape the inputs
    dsize, bsize = size(inputs)
    inputs = reshape(inputs, 1, dsize, bsize) # (D, 1, B)

    neg_scores = sum(logsigmoid(-batched_mul(inputs, negouts)))

    return -(pos_scores + neg_scores) / bsize
end


# Forward + loss (negative sampling)
function loss_fn(model::Word2Vec, center, context, negative_samples)
    v_c = model.input_embeddings[:, center]
    v_o = model.output_embeddings[:, context]

    # Positive score
    pos_score = log(sigmoid(dot(v_c, v_o)))

    # Negative score
    neg_score = 0.0
    for neg in negative_samples
        v_n = model.output_embeddings[:, neg]
        neg_score += log(sigmoid(-dot(v_c, v_n)))
    end

    return - (pos_score + neg_score)
end

# Training loop
function train!(model::Word2Vec, dataloader::DataLoader, neg_samples, opt_state; epochs=10)
    p = Progress(epochs; color=:darkblue, showspeed=true)
    generate_showvalues(epoch, loss) = () -> [(:Epoch, epoch), (:Loss, loss)]
    bsize = dataloader.batchsize
    for epoch in 1:epochs
        trn_losses = Float32[];
        negs = generate_neg_samplesMT(neg_samples; bsize=bsize)
        for(words, contexts) in dataloader
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
centers, contexts = generate_pairs(intCorpus, window_size=8)
@info "Generator for Negative Samples is being generated"
neg_samples = sample_negatives(intCorpus, sample_size=5);


VSIZE = length(vocab)

model = Word2Vec(VSIZE, 128) |> gpu

rule = Optimisers.OptimiserChain(Optimisers.ADAM(1e-2))
                                     # Optimisers.WeightDecay(1f-8),
                                     # Optimisers.ClipGrad(1));
opt_state = Optimisers.setup(rule, model);

dataloader = DataLoader((centers, contexts), batchsize=1024*32, shuffle=true, partial=false)

ctr, ctx = first(dataloader)
@time negs = generate_neg_samplesMT(neg_samples; bsize=1024 * 32);

ctr, ctx , negs = (ctr, ctx , negs) .|> gpu

loss(model, ctr, ctx, negs)

@time for _ in 1:1000
    negs = generate_neg_samplesMT(neg_samples; bsize=1024 * 32);
    ctr, ctx , negs = (ctr, ctx , negs) .|> gpu
    loss_, ∇model = Flux.withgradient(model, ctr, ctx, negs) do m, wrd, ctx, negs
                loss(m, wrd, ctx, negs)
        end
    Optimisers.update!(opt_state, model, ∇model[1]);
    println(loss_)
end



# Example usage
corpus = "the quick brown fox jumps over the lazy dog"
word_indices, vocab, idx_to_word = preprocess(corpus)
pairs = generate_pairs(word_indices, 2)
vocab_size = length(vocab)
embed_dim = 10
model = Word2Vec(vocab_size, embed_dim)
train!(model, pairs, vocab_size, epochs=100, lr=0.05)

# Getting embeddings
embedding_matrix = model.input_embeddings






"""

struct Embedding{T}
    W::T
end

Embedding(vocab_size::Integer, ex3mbedding_size::Integer) = Embedding(randn(Float32, embedding_size, vocab_size))
@layer Embedding

(m::Embedding)(x) = m.W[:, x]

struct DotProduct{T}
    fᵤ::T
    fᵥ::T
end

@layer DotProduct

(m::DotProduct)(x::Tuple{Integer,Integer}) = m.fᵤ(x[1]) ⋅ m.fᵥ(x[2])

(m::DotProduct)(x,y) = sum(m.fᵤ(x) .* m.fᵥ(y))

vocab_length = 10_000
embedding_size = 16
# function main(vocab_length = 10_000, embedding_size = 300)
  encodder = Embedding(vocab_length, embedding_size)
  decodder = Embedding(vocab_length, embedding_size)
  model = DotProduct(encodder, decodder)
  model_zip(x::Integer, y::Integer) = model((x, y))

  opt = Flux.Optimise.Descent()

  function loss(model,
                target_word_index,
                context_word_index,
                negative_sample_word_indices)

    l1 = - sum(log.(sigmoid.(model(target_word_index,
                                   context_word_index))))

    l2 = - sum(log.(sigmoid.(-model(target_word_index,
                                    negative_sample_word_indices))))
    l1 + l2
  end


#  @time begin
#     for idx in 1:50
        target_idx = rand(1:vocab_length)
        context_idx = rand(1:vocab_length, 16)
        neg_idx = rand(1:vocab_length, 16, 15)

        ps = Flux.params(model)

        gs = Flux.gradient(ps) do
          l = loss(model, target_idx, context_idx, neg_idx)
        end
        Flux.Optimise.update!(opt, ps, gs)
#    end
#  end

# end
"""

