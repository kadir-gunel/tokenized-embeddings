using Flux, Random, Statistics, Unicode

# Get character n-grams (e.g., <lo, lov, ove, ve>, with < and > as boundaries)
function get_ngrams(word::String, n_min::Int, n_max::Int)
    word = "<" * word * ">"
    ngrams = Set{String}()
    for n in n_min:n_max
        for i in 1:(lastindex(word) - n + 1)
            push!(ngrams, word[i:i+n-1])
        end
    end
    return collect(ngrams)
end

# Prepare vocabulary and n-gram mapping
function build_vocab_and_ngrams(corpus, n_min=3, n_max=6)
    words = split(lowercase(corpus))
    vocab = unique(words)
    word2idx = Dict(word => i for (i, word) in enumerate(vocab))

    all_ngrams = Set{String}()
    word_ngrams = Dict{String, Vector{String}}()

    for word in vocab
        ngrams = get_ngrams(word, n_min, n_max)
        word_ngrams[word] = ngrams
        union!(all_ngrams, ngrams)
    end

    ngram2idx = Dict(ng => i for (i, ng) in enumerate(all_ngrams))
    return vocab, word2idx, word_ngrams, ngram2idx
end

# Create skip-gram pairs
function generate_pairs(words, word2idx, window=2)
    idx_seq = [word2idx[w] for w in words if w in word2idx]
    pairs = []
    for i in 1:length(idx_seq)
        for j in max(i-window, 1):min(i+window, length(idx_seq))
            if i != j
                push!(pairs, (idx_seq[i], idx_seq[j]))
            end
        end
    end
    return pairs
end

# FastText model
mutable struct FastText
    ngram_embeddings::Matrix{Float32}  # (embed_dim, num_ngrams)
    output_embeddings::Matrix{Float32} # (embed_dim, vocab_size)
    word_ngrams::Dict{String, Vector{String}}
    ngram2idx::Dict{String, Int}
end

function FastText(embed_dim::Int, vocab, word_ngrams, ngram2idx)
    num_ngrams = length(ngram2idx)
    vocab_size = length(vocab)

    input_embed = randn(Float32, embed_dim, num_ngrams) .* 0.01
    output_embed = randn(Float32, embed_dim, vocab_size) .* 0.01
    return FastText(input_embed, output_embed, word_ngrams, ngram2idx)
end

# Build word embedding from its n-grams
function get_word_vector(model::FastText, word::String)
    ngrams = get(model.word_ngrams, word, String[])
    if isempty(ngrams)
        return zeros(Float32, size(model.ngram_embeddings, 1))
    end
    vectors = [model.ngram_embeddings[:, model.ngram2idx[ng]] for ng in ngrams if haskey(model.ngram2idx, ng)]
    return sum(vectors) / length(vectors)
end

# Loss function with negative sampling
function loss_fn(model::FastText, center_word, context_word_idx, negative_indices)
    v_c = get_word_vector(model, center_word)
    v_o = model.output_embeddings[:, context_word_idx]
    pos_score = log(sigmoid(dot(v_c, v_o)))

    neg_score = 0.0
    for neg in negative_indices
        v_n = model.output_embeddings[:, neg]
        neg_score += log(sigmoid(-dot(v_c, v_n)))
    end
    return - (pos_score + neg_score)
end

# Training loop
function train!(model::FastText, pairs, vocab, word2idx; epochs=10, lr=0.05, neg_samples=5)
    opt = Descent(lr)
    vocab_size = length(vocab)

    for epoch in 1:epochs
        total_loss = 0.0
        for (center_idx, context_idx) in pairs
            center_word = vocab[center_idx]
            negative_samples = rand(setdiff(1:vocab_size, [context_idx]), neg_samples)

            grads = Flux.gradient(() -> loss_fn(model, center_word, context_idx, negative_samples),
                Flux.params(model.ngram_embeddings, model.output_embeddings))

            Flux.Optimise.update!(opt, Flux.params(model.ngram_embeddings, model.output_embeddings), grads)
            total_loss += loss_fn(model, center_word, context_idx, negative_samples)
        end
        println("Epoch $epoch, Loss: $(total_loss / length(pairs))")
    end
end

# Example usage
corpus = "the quick brown fox jumps over the lazy dog"
words = split(lowercase(corpus))
vocab, word2idx, word_ngrams, ngram2idx = build_vocab_and_ngrams(corpus)
pairs = generate_pairs(words, word2idx, 2)

embed_dim = 50
model = FastText(embed_dim, vocab, word_ngrams, ngram2idx)
train!(model, pairs, vocab, word2idx, epochs=50, lr=0.05)
