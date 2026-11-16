"""Treino e avaliacao em corpus sintetico, com sementes fixas."""
from __future__ import annotations
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data import (CONTEXT_SIZE, EOS, UNK, assert_no_oov,
                      build_vocab, encode_examples, split_sentences)
from src.models import NeuralLanguageModel, SmoothedNGram


def train_neural(train_examples, vocab_size: int, *, seed: int = 42,
                 epochs: int = 25, batch_size: int = 128,
                 embedding_dim: int = 16, hidden_dim: int = 32, lr: float = 0.025):
    """Uma rede pequena, CPU, SGD/Adam e avaliacao apenas apos o treino."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs e batch_size devem ser positivos")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = NeuralLanguageModel(vocab_size, CONTEXT_SIZE, embedding_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    contexts = torch.tensor([c for c, _ in train_examples], dtype=torch.long)
    targets = torch.tensor([t for _, t in train_examples], dtype=torch.long)
    losses = []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(contexts))
        total_loss = 0.0
        for start in range(0, len(contexts), batch_size):
            indexes = permutation[start:start+batch_size]
            optimizer.zero_grad()
            logits = model(contexts[indexes])
            loss = F.cross_entropy(logits, targets[indexes])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indexes)
        losses.append(total_loss / len(contexts))
    return model, losses


@torch.no_grad()
def evaluate_neural(model: NeuralLanguageModel, examples, batch_size: int = 512) -> float:
    model.eval()
    nll = 0.0
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start+batch_size]
        contexts = torch.tensor([c for c, _ in batch], dtype=torch.long)
        targets = torch.tensor([t for _, t in batch], dtype=torch.long)
        logits = model(contexts)
        nll += F.cross_entropy(logits, targets, reduction="sum").item()
    return math.exp(nll / len(examples))


@torch.no_grad()
def nearest_neighbors(model: NeuralLanguageModel, vocab: dict[str, int], word: str, k: int = 4):
    if word not in vocab:
        raise ValueError(f"Palavra desconhecida: {word}")
    matrix = F.normalize(model.embedding.weight.detach(), dim=1)
    similarities = (matrix @ matrix[vocab[word]]).tolist()
    results = []
    for token, idx in sorted(vocab.items(), key=lambda item: -similarities[item[1]]):
        if token != word and token not in {"<bos>", "<eos>", "<unk>"}:
            results.append({"palavra": token, "cosseno": round(similarities[idx], 4)})
        if len(results) == k:
            break
    return results


@torch.no_grad()
def predict_next(model: NeuralLanguageModel, vocab: dict[str, int], text: str, k: int = 5):
    tokens = (["<bos>"] * CONTEXT_SIZE + text.split())[-CONTEXT_SIZE:]
    ids = [vocab.get(word, vocab[UNK]) for word in tokens]
    model.eval()
    probabilities = torch.softmax(model(torch.tensor([ids], dtype=torch.long)), dim=-1)[0]
    inverse = {idx: token for token, idx in vocab.items()}
    valid_indices = [idx for token, idx in vocab.items() if token not in {"<bos>", "<unk>"}]
    sorted_ids = sorted(valid_indices, key=lambda idx: -probabilities[idx].item())[:k]
    return [{"palavra": inverse[idx], "probabilidade": round(probabilities[idx].item(), 4)} for idx in sorted_ids]


def run(seed=42, epochs=25, output_dir="results"):
    train_sentences, test_sentences = split_sentences(seed)
    vocab = build_vocab(train_sentences)
    assert_no_oov(test_sentences, vocab)
    train_examples = encode_examples(train_sentences, vocab)
    test_examples = encode_examples(test_sentences, vocab)

    baseline = SmoothedNGram(len(vocab), alpha=0.5).fit(train_examples)
    model, losses = train_neural(train_examples, len(vocab), seed=seed, epochs=epochs)
    ppl_ngram = baseline.perplexity(test_examples)
    ppl_neural = evaluate_neural(model, test_examples)
    neighbors = {w: nearest_neighbors(model, vocab, w) for w in ("gato", "cachorro", "menino", "pássaro")}
    prediction = predict_next(model, vocab, "o gato dorme no")
    metrics = {
        "semente": seed, "epocas": epochs, "n_frases_treino": len(train_sentences),
        "n_frases_teste": len(test_sentences), "n_exemplos_treino": len(train_examples),
        "n_exemplos_teste": len(test_examples), "tamanho_vocabulario": len(vocab),
        "contexto_palavras": CONTEXT_SIZE, "embedding_dim": 16, "hidden_dim": 32,
        "perplexidade_ngram_teste": round(ppl_ngram, 4),
        "perplexidade_neural_teste": round(ppl_neural, 4),
        "loss_treino_inicio": round(losses[0], 4),
        "loss_treino_final": round(losses[-1], 4),
        "vizinhos_embeddings": neighbors,
        "previsao_exemplo": prediction,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metricas.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "RELATORIO_RESULTADOS.md").write_text(
        "# Relatório de resultados — PLN e representações distribuídas\n\n"
        "**Experimento didático com corpus sintético.** Não reproduz os números de Bengio et al. (2003).\n\n"
        f"- Semente: {seed}; épocas: {epochs}; frases treino/teste: {len(train_sentences)}/{len(test_sentences)}.\n"
        f"- Vocabulário: {len(vocab)} tokens (incluindo símbolos especiais); contexto: {CONTEXT_SIZE} palavras.\n"
        f"- Perplexidade no teste — n-grama aditivo: **{ppl_ngram:.3f}**.\n"
        f"- Perplexidade no teste — rede neural: **{ppl_neural:.3f}**.\n"
        f"- Loss média no treino (primeira/última época): **{losses[0]:.3f} / {losses[-1]:.3f}**.\n\n"
        "## Vizinhos no espaço de embeddings\n\n"
        + "\n".join(f"- **{w}**: " + ", ".join(f"{p['palavra']} (cos={p['cosseno']})" for p in vs) for w, vs in neighbors.items())
        + "\n\n## Próxima palavra — entrada: `o gato dorme no`\n\n"
        + "\n".join(f"- {p['palavra']}: {p['probabilidade']:.4f}" for p in prediction)
        + "\n\n## Interpretação e limites\n\n"
        "A diferença entre perplexidades não permite afirmar compreensão semântica.\n"
        "O corpus é artificial, pequeno e repetitivo; os modelos recebem os mesmos exemplos de treino e teste,\n"
        "mas arquiteturas, regularização e esforço computacional não são equiparados.\n"
        "A ordem e conteúdo do conjunto de teste dependem da semente. Os vizinhos vetoriais são indicativos,\n"
        "não uma avaliação semântica independente. Não há alegação de superioridade geral.\n",
        encoding="utf-8",
    )
    return metrics
