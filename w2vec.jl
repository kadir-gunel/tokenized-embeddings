cd(@__DIR__)

using Pkg
Pkg.activate("/home/kguenel/Glove")


using .Libc
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
using ParameterSchedulers
using StatsBase

using Flux: @layer
using Flux: flatten, frequencies, DataLoader
using ParameterSchedulers: Scheduler
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


function subsample_frequent_words(corpus::String; minfreq::Int=10)
    filtered_corpus = String[]
    wordCounts = get_frequencies(corpus)
    filter!(w -> w.second > minfreq, wordCounts)
    tot_wordCounts = sum(values(wordCounts))
    # wordCounts = Dict(word => wordCounts[word] / tot_wordCounts for (word, _) in wordCounts)
    @showprogress for word in split(corpus)
        if haskey(wordCounts, word)
            # rand() < (1 + sqrt(wordCounts[word] * 1e3)) * 1e-3 / wordCounts[word] ? push!(filtered_corpus, word) : nothing
            rand() < (1 + sqrt(wordCounts[word] * 1e3 / tot_wordCounts)) * (1e-3 * tot_wordCounts) / wordCounts[word] ? push!(filtered_corpus, word) : nothing
        end
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
    pos_scores = logsigmoid(sum(inputs .* posouts, dims=1))
    # need to reshape the inputs
    dsize, bsize = size(inputs)
    inputs = reshape(inputs, 1, dsize, bsize) # (D, 1, B)
    # negouts= permutedims(negouts, (1, 3, 2))
    neg_scores = sum(logsigmoid(flatten(-batched_mul(inputs, negouts))), dims=1)

    return -mean(pos_scores + neg_scores)
end


# Training loop
function train!(model::Word2Vec, dataloader::DataLoader, neg_samples, opt_state; epochs=10)
    p = Progress(epochs; color=:white, showspeed=true)
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

filteredCorpus = subsample_frequent_words(corpus; minfreq=4)

vocab = filteredCorpus |> split |> unique .|> string

w2i = Dict(word => idx for (idx, word) in enumerate(vocab))
i2w = Dict(idx => word for (idx, word) in enumerate(vocab))

# transfer the filtered corpus to intCorpus 
intCorpus = collect(w2i[word] for word in split(filteredCorpus));


@info "Generating Positive Samples:"
centers, contexts = positiveSampler(intCorpus, window_size=5)
@info "Generator for Negative Samples is being generated"
neg_samples = NegativeSampler(intCorpus, sample_size=5);


VSIZE = length(vocab)

model = Word2Vec(VSIZE, 100) |> gpu

# n = 8 # AccumGrad(n),
# const lr = 1e-2
rule = Optimisers.OptimiserChain(# Optimisers.AccumGrad(16),
                                 Optimisers.ADAM(.025) # (7e-2),
                                 Optimisers.ClipGrad(1),
                                 Optimisers.WeightDecay(1e-3))

# rule = Optimisers.OptimiserChain(Optimisers.ADAM(2e-3))
                                     # Optimisers.WeightDecay(1f-8),
                                     # Optimisers.ClipGrad(1));
opt_state = Optimisers.setup(rule, model);
# sched = ParameterSchedulers.Stateful(Step(lr, 9e-1, 128))

dataloader = DataLoader((centers, contexts), batchsize=4096, shuffle=true, partial=false)

avgloss = Float32[]
duration = Float32[]
nextlr = lr
@time for (iter, (ctr, ctx)) in enumerate(dataloader)

    negs = get_negative_samples_batch(neg_samples, ctx; num_samples=5);
    ctr, ctx , negs = (ctr, ctx , negs) .|> gpu
    start_time = time()
    loss_, ∇model = Flux.withgradient(model, ctr, ctx, negs) do m, wrd, ctx, negs
                loss(m, wrd, ctx, negs)
        end
    end_time = time()
    push!(duration, round(length(ctx) * (start_time/end_time), digits=3))
    push!(avgloss, loss_)
    if iter % 16 == 0
        @info "Loss: $(round(mean(avgloss), digits=3)), \t Tokens/sec : $(mean(duration)) \t  LR: $(nextlr) \n"
        empty!(duration)
        empty!(avgloss)
    end

    Optimisers.update!(opt_state, model, ∇model[1]);
    nextlr = ParameterSchedulers.next!(sched)
    Optimisers.adjust!(opt_state, nextlr)
    


end


# train!(model, dataloader, neg_samples, opt_state; epochs=5)
bsize = 4
report_every = 100
initial_alpha = 0.025
min_alpha = 0.0001
iterations = collect(1:5)
processed_words = 0
start_time = time()
avgloss = Float32[]
data_idx = collect(1:length(centers))
for iter in iterations
    #shuffle at each iteration
    idx = shuffle!(data_idx)
    data = collect(zip(centers[idx], contexts[idx]))

    for i in collect(1:bsize:length(data))
        batch = data[i:i+bsize-1]
        ctr, ctx = first.(batch), last.(batch)
        negs = get_negative_samples_batch(neg_samples, ctx; num_samples=5);
        ctr, ctx , negs = (ctr, ctx , negs) .|> gpu
        loss_, ∇model = Flux.withgradient(model, ctr, ctx, negs) do m, wrd, ctx, negs
                loss(m, wrd, ctx, negs)
        end
        processed_words += length(batch)
        # linear learning rate decay
        prog = (iter * length(data) + i) / (length(iterations) * length(data))
        alpha = maximum([min_alpha, initial_alpha * (1 - prog)])
        
        
        push!(avgloss, loss_)
        Optimisers.update!(opt_state, model, ∇model[1]);
        Optimisers.adjust!(opt_state, alpha)

        if processed_words % report_every == 0
            elapsed = time() - start_time
            words_per_sec = processed_words / elapsed
            remaining = (length(data) * length(iterations) - processed_words) / words_per_sec

            @info " 
            Iter : $(iter / length(iterations))\t
            Alpha: $(alpha)\t
            Loss: $(round(mean(avgloss), digits=3))\t
            Progress: $(100 * prog)\t
            words/sec: $(words_per_sec)\t
            ETA: $(remaining)\n
            "
            empty!(avgloss)

        end

    end
end



















