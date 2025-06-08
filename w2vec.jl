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
using SafeTensors

using ProgressMeter

using ExplainableAI
using TensorBoardLogger
using TensorBoardLogger: with_logger

CUDA.device!(1)

rng = Random.default_rng()
Random.seed!(rng, 0)



### faster negative sampler using Vose's algorithm
struct CPUNegativeSampler
    vocab::Vector{Int32}
    probs::Vector{Float32}
    alias_table::Tuple{Vector{Int32}, Vector{Float32}}
    sample_size::Int32
    power::Float32
    rngs::Vector{MersenneTwister}  # One RNG per thread
    w2i::Dict{Int32,Int32}
end

function CPUNegativeSampler(corpus::Vector{Int}; sample_size=5, power=0.75f0)
    word_counts = frequencies(corpus)
    vocab = Int32.(collect(keys(word_counts)))
    counts = Float32.(collect(values(word_counts)))
    
    probs = counts .^ power
    probs ./= sum(probs)
    
    alias_table = alias_setup(probs)
    w2i = Dict(word => i for (i, word) in enumerate(vocab))
    
    # Initialize one RNG per thread
    rngs = [MersenneTwister(rand(UInt)) for _ in 1:Threads.nthreads()]
    
    CPUNegativeSampler(
        vocab,
        probs,
        alias_table,
        Int32(sample_size),
        Float32(power),
        rngs,
        w2i
    )
end

function get_negative_samples_cpu(                                                                                                                                               
           sampler::CPUNegativeSampler,                                                                                                                                                 
           target_words::Vector{Int32};                                                                                                                                                 
           num_samples=nothing                                                                                                                                                          
       )                                                                                                                                                                                
           num_samples = isnothing(num_samples) ? sampler.sample_size : Int32(num_samples)                                                                                              
           batch_size = length(target_words)                                                                                                                                            
           results = Matrix{Int32}(undef, num_samples, batch_size)                                                                                                                      
                                                                                                                                                                                        
           J, q = sampler.alias_table                                                                                                                                                   
                                                                                                                                                                                        
           Threads.@threads for i in 1:batch_size                                                                                                                                       
               tid = Threads.threadid()                                                                                                                                                 
               local_rng = sampler.rngs[tid]  # Thread-local RNG                                                                                                                        
               target_idx = get(sampler.w2i, target_words[i], -1)                                                                                                                       
                                                                                                                                                                                        
               for j in 1:num_samples                                                                                                                                                   
                   # Alias method sampling                                                                                                                                              
                   u = rand(local_rng, Float32)                                                                                                                                         
                   k = rand(local_rng, 1:length(sampler.vocab))                                                                                                                         
                   idx = u < q[k] ? k : J[k]                                                                                                                                            
                                                                                                                                                                                        
                   # Rejection sampling for target word                                                                                                                                 
                   while idx == target_idx                                                                                                                                              
                       u = rand(local_rng, Float32)                                                                                                                                     
                       k = rand(local_rng, 1:length(sampler.vocab))                                                                                                                     
                       idx = u < q[k] ? k : J[k]                                                                                                                                        
                   end                                                                                                                                                                  
                                                                                                                                                                                        
                   results[j, i] = sampler.vocab[idx]                                                                                                                                   
               end                                                                                                                                                                      
           end                                                                                                                                                                          
                                                                                                                                                                                        
           return results                                                                                                                                                               
end                                     

function alias_setup(probs::Vector{Float32})
    n = length(probs)
    J = Vector{Int32}(undef, n)  # Alias table
    q = Vector{Float32}(undef, n) # Prob table
    
    # Use Vectors as stacks (with push!/pop!)
    smaller = Int[]
    larger = Int[]
    
    # Initialize
    for i in 1:n
        q[i] = probs[i] * n
        if q[i] < 1.0f0
            push!(smaller, i)
        else
            push!(larger, i)
        end
    end
    
    # Build alias table
    while !isempty(smaller) && !isempty(larger)
        small = pop!(smaller)
        large = pop!(larger)
        
        J[small] = large
        q[large] = q[large] - (1.0f0 - q[small])
        
        if q[large] < 1.0f0
            push!(smaller, large)
        else
            push!(larger, large)
        end
    end
    
    # Handle remaining probabilities
    for i in smaller
        q[i] = 1.0f0
    end
    for i in larger
        q[i] = 1.0f0
    end
    
    return (J, q)
end


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

# (m::Word2Vec)(center, context, neg_samples) = m.V(center), m.U(context), m.U(neg_samples)

(m::Word2Vec)(center, context) = m.V(center), m.U(context)

function newloss(model::Word2Vec, inputs::V, contexts::M, y::M) where {V, M}
    inputs, contexts = model(inputs, contexts)
    dim, bsize = inputs |> size
    inputs = reshape(inputs, 1, dim, bsize)
    ŷ = flatten(batched_mul(inputs, contexts))
    return Flux.logitbinarycrossentropy(ŷ, y)
end

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


S_SIZE = 25
BSIZE =  4096 * 16 # 10_000 #4096 * 8
DIMS = 200

@info "Generating Positive Samples:"
centers, contexts = positiveSampler(intCorpus, window_size=8)
@info "Generator for Negative Samples is being generated"
sampler = CPUNegativeSampler(intCorpus, sample_size=S_SIZE);
# neg_samples = NegativeSampler(intCorpus, sample_size=S_SIZE);

VSIZE = length(vocab)

model = Word2Vec(VSIZE, DIMS) |> gpu

# n = 8 # AccumGrad(n),
# const lr = 1e-2
rule = Optimisers.OptimiserChain(# Optimisers.AccumGrad(16),
                                 Optimisers.ADAM(25e-3), # (5e-2),
                                 Optimisers.ClipGrad(1))
                                 # Optimisers.WeightDecay(1e-3))

# rule = Optimisers.OptimiserChain(Optimisers.ADAM(2e-3))
                                     # Optimisers.WeightDecay(1f-8),
                                     # Optimisers.ClipGrad(1));
opt_state = Optimisers.setup(rule, model);
# sched = ParameterSchedulers.Stateful(Step(25e-3, 9e-1, 4))

dataloader = DataLoader((centers, contexts), batchsize=BSIZE, shuffle=true, partial=false)


avgloss = Float32[]
duration = Float32[]
nextlr = 25e-3
runnning_loss = 0.

y = vcat(ones(Int64, 1, BSIZE), zeros(Int64, S_SIZE, BSIZE))
# y = reshape(y, S_SIZE + 1, 1, BSIZE)

@time for (iter, (ctr, ctx)) in enumerate(dataloader)
    # println("Iteration:", iter)
    # negs = get_negative_samples_batch(neg_samples, ctx; num_samples=S_SIZE);
    neg_ctx = get_negative_samples_cpu(sampler, Int32.(ctx));
    ctx = vcat(permutedims(ctx), neg_ctx)
    ctr, ctx, y = (ctr, ctx, y) |> gpu
    start_time = time()
    loss_, ∇model = Flux.withgradient(model, ctr, ctx, y) do m, ctr, ctx, y 
                newloss(m, ctr, ctx, y)
    end
    Optimisers.update!(opt_state, model, ∇model[1]);
    end_time = time()
    push!(duration, round(length(ctx) * (start_time/end_time), digits=3))
    push!(avgloss, loss_)
    if iter % 1_000 == 0
        @info "Loss: $(round(mean(avgloss), digits=3)), \t Tokens/sec : $(mean(duration))\n"
        @info "$(iter / length(dataloader)) percent completed \n"
        empty!(duration)
        empty!(avgloss)
    end
    # nextlr = ParameterSchedulers.next!(sched)
    # Optimisers.adjust!(opt_state, nextlr)
end


# train!(model, dataloader, neg_samples, opt_state; epochs=5)
bsize = 4096 * 64 # 10_000 # 10_000_000
report_every = bsize
initial_alpha = 25e-3
min_alpha = 1e-4
iterations = collect(0:4)
processed_words = 0
start_time = time()
avgloss = Float32[]
data_idx = collect(1:length(centers))
for iter in iterations
    #shuffle at each iteration
    @info "Iteration: $(iter)\n"
    idx = shuffle!(data_idx)
    data = collect(zip(centers[idx], contexts[idx]))
    outputs = vcat(ones(Int64, 1, bsize), zeros(Int64, S_SIZE, bsize))
    for i in collect(1:bsize:div(length(data), bsize) * bsize)
        batch = data[i:i+bsize-1]
        ctr, ctx = first.(batch), last.(batch)
        # negs = get_negative_samples_batch(neg_samples, ctx; num_samples=S_SIZE);
        neg_ctx = get_negative_samples_cpu(sampler, Int32.(ctx));
        ctx = vcat(permutedims(ctx), neg_ctx) # first row positive, rest rows negatives
        ctr, ctx, outputs = (ctr, ctx, outputs) |> gpu
        loss_, ∇model = Flux.withgradient(model, ctr, ctx, outputs) do m, ctr, ctx, outputs
                newloss(m, ctr, ctx, outputs)
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
            Iteration : $(iter) / $(length(iterations))\t
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


U = model.U.weight
V = model.V.weight

params = Dict("U" => U, "V" => V)
f = "path2word2vec.safetensors"
SafeTensors.serialize(f, params)


# for loading 

loaded = SafeTensors.deserialize(f)

















