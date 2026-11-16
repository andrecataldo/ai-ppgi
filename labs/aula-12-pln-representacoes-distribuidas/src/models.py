"""Baselines n-grama e modelo neural inspirado em Bengio et al. (2003)."""
from __future__ import annotations
from collections import Counter, defaultdict
import math
import torch
from torch import nn


class SmoothedNGram:
    """P(next_word|context) por contagens com suavizacao aditiva."""

    def __init__(self, vocab_size: int, alpha: float = 0.5):
        if vocab_size < 2 or alpha <= 0:
            raise ValueError("Parametros invalidos")
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.counts = defaultdict(Counter)
        self.totals = Counter()

    def fit(self, examples):
        for context, target in examples:
            context = tuple(context)
            self.counts[context][int(target)] += 1
            self.totals[context] += 1
        return self

    def probability(self, context, target: int) -> float:
        context = tuple(context)
        if not 0 <= target < self.vocab_size:
            raise ValueError("target fora do vocabulario")
        return (self.counts[context][target] + self.alpha) / (
            self.totals[context] + self.alpha * self.vocab_size
        )

    def perplexity(self, examples) -> float:
        return math.exp(-sum(math.log(self.probability(c, t)) for c, t in examples) / len(examples))


class NeuralLanguageModel(nn.Module):
    """C -> concatenacao -> tanh -> softmax (via CrossEntropyLoss).

    Equacao do artigo: y = b + Wx + U tanh(d + Hx)
    """

    def __init__(self, vocab_size: int, context_size: int = 3,
                 embedding_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.context_size = context_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        input_dim = context_size * embedding_dim
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.direct = nn.Linear(input_dim, vocab_size, bias=False)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[1] != self.context_size:
            raise ValueError(f"Esperado tensor [batch, {self.context_size}]")
        x = self.embedding(context).reshape(context.shape[0], -1)
        return self.output(torch.tanh(self.hidden(x))) + self.direct(x)
