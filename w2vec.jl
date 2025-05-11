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

