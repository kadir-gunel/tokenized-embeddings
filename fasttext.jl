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
using StatsBase
using MurmurHash3

using Flux: @layer
using Flux: flatten, frequencies
# using BSON: @load, @save
using SafeTensors


using Wandb, Logging
using ProgressMeter

CUDA.device!(1)

rng = Random.default_rng()
Random.seed!(rng, 0)


xpower(x) = x^.75
get_frequencies(corpus::String)::Dict = frequencies(split(corpus))


### faster negative sampler using Vose's algorithm
struct NegativeSampler
    vocab::Vector{Int32}
    probs::Vector{Float32}
    alias_table::Tuple{Vector{Int32}, Vector{Float32}}
    sample_size::Int32
    power::Float32
    rngs::Vector{MersenneTwister}  # One RNG per thread
    w2i::Dict{Int32,Int32}
end

function NegativeSampler(corpus::Vector{Int}; sample_size=5, power=0.75f0)
    word_counts = frequencies(corpus)
    vocab = Int32.(collect(keys(word_counts)))
    counts = Float32.(collect(values(word_counts)))
    
    probs = counts .^ power
    probs ./= sum(probs)
    
    alias_table = alias_setup(probs)
    w2i = Dict(word => i for (i, word) in enumerate(vocab))
    
    # Initialize one RNG per thread
    rngs = [MersenneTwister(rand(UInt)) for _ in 1:Threads.nthreads()]
    
    NegativeSampler(
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
           sampler::NegativeSampler,                                                                                                                                                 
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

function subsample_frequent_words(corpus::String; minfreq::Int=10)
    filtered_corpus = String[]
    wordCounts = get_frequencies(corpus)
    filter!(w -> w.second > minfreq, wordCounts)
    tot_wordCounts = sum(values(wordCounts))
    @showprogress for word in split(corpus)
        if haskey(wordCounts, word)
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

function generate_subwords(word::String; min_n::Int=3, max_n::Int=6)
    # Add word boundaries
    word = lowercase(word)
    subwords = Set{String}()
    n_chars = length(word)
    if length(word) <= min_n
    	return ["<EMPTY>"] # special tag
	end
    # Input validation
    min_n = max(1, min_n)  # Ensure min_n is at least 1
    max_n = min(max_n, n_chars)  # Don't exceed word length
    
    # Extract n-grams of lengths min_n to max_n
    for n in min_n:max_n
        for i in 1:(n_chars - n + 1)
            subword = word[i:i+n-1]
            push!(subwords, subword)
        end
    end
    
    return collect(subwords)
end

function generate_subwords_parallel(words::Vector{String}; min_n::Int=3, max_n::Int=6)
    # Preallocate the output vector
    subwords_batch = Vector{Vector{String}}(undef, length(words))
    
    # Use @threads with a static schedule for better performance with uneven workloads
    Threads.@threads :static for i in eachindex(words)
        subwords_batch[i] = generate_subwords(words[i]; min_n=min_n, max_n=max_n)
    end
    
    return subwords_batch
end




function generate_subword_indices(words::Vector{String}, B::Int; ngram_range=(3, 6))
    batch_size = length(words)
    max_subwords_per_word = 20  # Adjust based on max word length
    indices = Matrix{Int}(undef, max_subwords_per_word, batch_size)
    counts = zeros(Int, batch_size)  # Track subword counts per word

    @threads :static for i in 1:batch_size
        word = words[i]
        subword_count = 0

        # Generate n-grams for the current word
        for n in ngram_range[1]:ngram_range[2]
            for j in 1:(length(word) - n + 1)
                if subword_count >= max_subwords_per_word
                    @warn "Word $(word) exceeds max_subwords_per_word ($max_subwords_per_word)"
                    break
                end
                
                subword = word[j:j+n-1]
                subword_count += 1

                # Hash subword to fixed bucket size (using ncodeunits for Unicode safety)
                hash_val = MurmurHash3.mmhash32(ncodeunits(subword), pointer(subword), UInt32(0))
                indices[subword_count, i] = abs(Int(hash_val)) % B + 1
            end
        end

        counts[i] = subword_count
        # Pad remaining slots with zeros (optional)
        for k in (subword_count+1):max_subwords_per_word
            indices[k, i] = 0
        end
    end

    return indices, counts
end


################# MODEL ###################

struct FastText
    V::Embedding
    S::EmbeddingBag   # S for Subwords
end
@layer FastText

function FastText(VSIZE::Int, Bucket::Int, IN_DIM::Int)
    V = Embedding(VSIZE, IN_DIM)
    S = EmbeddingBag(Bucket => IN_DIM)
    return FastText(V, S)
end

(m::FastText)(center, context, subwords) = m.V(center), m.V(context) , m.S(subwords)

function loss(model::FastText, inputs::V, contexts::M, subwords::M, y::M) where {V, M}
    inputs, contexts, subwords = model(inputs, contexts, subwords)
    inputs = (inputs + subwords) ./ 2
    dim, bsize = inputs |> size
    inputs = reshape(inputs, 1, dim, bsize)
    ŷ = flatten(batched_mul(inputs, contexts))
    return Flux.logitbinarycrossentropy(ŷ, y)
end

function train!(model::FastText, centers, contexts, sampler::NegativeSampler; initial_alpha=25e-3, min_alpha=1e-4, iterations=collect(0:4))
    report_every = BSIZE
    processed_words = 0
    start_time = time()
    avgloss = Float32[]

    data_idx = collect(1:length(centers))
    for iter in iterations
        #shuffle at each iteration
        @info "Iteration: $(iter)\n"
        idx = shuffle!(data_idx)
        data = collect(zip(centers[idx], contexts[idx]))
        outputs = vcat(ones(Int64, 1, BSIZE), zeros(Int64, S_SIZE, BSIZE))
        for i in collect(1:BSIZE:div(length(data), BSIZE) * BSIZE)
            batch = data[i:i+BSIZE-1]
            ctr, ctx = first.(batch), last.(batch)
            neg_ctx = get_negative_samples_cpu(sampler, Int32.(ctx));
            ctx = vcat(permutedims(ctx), neg_ctx) # first row positive, rest rows negatives
            ctr, ctx, outputs = (ctr, ctx, outputs) |> gpu
            loss_, ∇model = Flux.withgradient(model, ctr, ctx, outputs) do m, ctr, ctx, outputs
                    loss(m, ctr, ctx, outputs)
            end
            processed_words += length(batch)
            # linear learning rate decay
            prog = (iter * length(data) + i) / (length(iterations) * length(data))
            alpha = maximum([min_alpha, initial_alpha * (1 - prog)])
            
            Optimisers.update!(opt_state, model, ∇model[1]);
            Optimisers.adjust!(opt_state, alpha)

            push!(avgloss, loss_)

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
end 

################### MAIN ########################

root_file = "/mnt/depo/github/GloVe/"
text = root_file * "text8"

corpus = read(text, String)
filteredCorpus = subsample_frequent_words(corpus; minfreq=3)
vocab = filteredCorpus |> split |> unique .|> string

# Example Usage
word = "apple"
subwords = generate_subwords(word) 
	

function tokenize(sentence, min_n=3, max_n=6)
    words = split(lowercase(sentence), r"\s+") .|> string
    return [generate_subwords(word, min_n=min_n, max_n=max_n) for word in words]
end


function hash_subword(subword::String)::Int
    pnt = pointer(subword)
    len = sizeof(subword)
    return (abs(Int(MurmurHash3.mmhash32(len, pnt, UInt32(0))))) % B + 1
end

r = Int[]
for sw in subwords
	for word in sw
		push!(r, hash_subword(word) % B + 1)
	end 
end


w2i = Dict(word => idx for (idx, word) in enumerate(vocab))
i2w = Dict(idx => word for (idx, word) in enumerate(vocab))

# transfer the filtered corpus to intCorpus 
intCorpus = collect(w2i[word] for word in split(filteredCorpus));


global S_SIZE = 25
global BSIZE =  4096 * 8 # 10_000 #4096 * 8
global DIMS = 384
global VSIZE = length(vocab)
initial_alpha = 25e-3
# min_alpha = 1e-4
# iterations = collect(0:4)

@info "Generating Positive Samples:"
centers, contexts = positiveSampler(intCorpus, window_size=8)
@info "Generator for Negative Samples is being generated"
sampler = NegativeSampler(intCorpus, sample_size=S_SIZE);



model = Word2Vec(VSIZE, DIMS) |> gpu
rule = Optimisers.OptimiserChain(Optimisers.RADAM(initial_alpha), # (5e-2),
                                 Optimisers.ClipGrad(1))
opt_state = Optimisers.setup(rule, model);

train!(model, centers, contexts, sampler)



U = model.U.weight
V = model.V.weight


root = pwd() * "/Documents/github/tokenized-embeddings/embeds/"
params = Dict("U" => U, "V" => V)
f = root * "word2vec_adam.safetensors"
SafeTensors.serialize(f, params)


# for loading 

loaded = SafeTensors.deserialize(f)

# foo deneme