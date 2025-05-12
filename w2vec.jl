using Flux, Flux.Zygote, Flux.Optimise
using LinearAlgebra
using Flux: @layer

struct Embedding{T}
    W::T
end

Embedding(vocab_size::Integer, embedding_size::Integer) = Embedding(randn(Float32, embedding_size, vocab_size))
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



using Flux
using Random
using Statistics

# Tokenize and prepare data
function preprocess(corpus)
    words = split(lowercase(corpus))
    vocab = Dict(word => i for (i, word) in enumerate(unique(words)))
    idx_to_word = [word for (word, _) in sort(collect(vocab), by = x -> x[2])]
    word_indices = [vocab[word] for word in words]
    return word_indices, vocab, idx_to_word
end

# Generate skip-gram pairs
function generate_pairs(word_indices, window_size=2)
    pairs = []
    for i in 1:length(word_indices)
        for j in max(i - window_size, 1):min(i + window_size, length(word_indices))
            if i != j
                push!(pairs, (word_indices[i], word_indices[j]))
            end
        end
    end
    return pairs
end

# Negative sampling: generate n negative samples not equal to the positive context word
function get_negative_samples(target, vocab_size, n)
    samples = Int[]
    while length(samples) < n
        neg = rand(1:vocab_size)
        if neg != target
            push!(samples, neg)
        end
    end
    return samples
end

# Word2Vec model
struct Word2Vec
    input_embeddings::Matrix{Float32}
    output_embeddings::Matrix{Float32}
end

function Word2Vec(vocab_size::Int, embed_dim::Int)
    input_embeddings = rand(Float32, embed_dim, vocab_size)
    output_embeddings = rand(Float32, embed_dim, vocab_size)
    return Word2Vec(input_embeddings, output_embeddings)
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
function train!(model::Word2Vec, pairs, vocab_size, epochs=10, lr=0.01, neg_samples=5)
    opt = Descent(lr)
    for epoch in 1:epochs
        total_loss = 0.0
        for (center, context) in pairs
            negative_samples = get_negative_samples(context, vocab_size, neg_samples)
            gs = Flux.gradient(() -> loss_fn(model, center, context, negative_samples),
                               Flux.params(model.input_embeddings, model.output_embeddings))
            Flux.Optimise.update!(opt, Flux.params(model.input_embeddings, model.output_embeddings), gs)
            total_loss += loss_fn(model, center, context, negative_samples)
        end
        println("Epoch $epoch, Loss = $(total_loss / length(pairs))")
    end
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
