"""Corpus didatico fechado: nao representa amostra natural da lingua portuguesa."""
from __future__ import annotations
import random

BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
CONTEXT_SIZE = 3

# Pequenos grupos construidos para ilustrar similaridade de distribuicao/contexto.
GROUPS = (
    (("gato", "cachorro"), ("dorme", "descansa", "brinca", "corre"),
     ("no sofá", "no tapete", "na sala")),
    (("menino", "menina"), ("estuda", "lê", "escreve", "aprende"),
     ("na escola", "na biblioteca", "em casa")),
    (("pássaro", "canário"), ("canta", "voa", "pousa", "descansa"),
     ("na árvore", "no jardim", "no quintal")),
)
PREFIXES = ("", "hoje ", "agora ", "de manhã ")


def make_sentences() -> list[str]:
    """Produz 288 frases distintas, sem uso de arquivos ou internet."""
    sentences = []
    for subjects, verbs, locations in GROUPS:
        for prefix in PREFIXES:
            for subject in subjects:
                for verb in verbs:
                    for location in locations:
                        sentences.append(f"{prefix}o {subject} {verb} {location}")
    assert len(sentences) == len(set(sentences)) == 288
    return sentences


def split_sentences(seed: int = 42, test_fraction: float = 0.2):
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction deve estar entre 0 e 1")
    sentences = make_sentences()
    random.Random(seed).shuffle(sentences)
    cut = round(len(sentences) * (1 - test_fraction))
    return sentences[:cut], sentences[cut:]


def build_vocab(train_sentences: list[str]) -> dict[str, int]:
    vocab_words = sorted({token for sentence in train_sentences for token in sentence.split()})
    return {word: i for i, word in enumerate((BOS, EOS, UNK, *vocab_words))}


def encode_examples(sentences: list[str], vocab: dict[str, int], context_size: int = CONTEXT_SIZE):
    """Gera pares (contexto, proxima palavra), incluindo fim de frase."""
    examples = []
    for sentence in sentences:
        tokens = [BOS] * context_size + sentence.split() + [EOS]
        ids = [vocab.get(token, vocab[UNK]) for token in tokens]
        for pos in range(context_size, len(ids)):
            examples.append((tuple(ids[pos-context_size:pos]), ids[pos]))
    return examples


def assert_no_oov(sentences: list[str], vocab: dict[str, int]):
    unseen = {w for sentence in sentences for w in sentence.split() if w not in vocab}
    if unseen:
        raise ValueError(f"Palavras do teste nao vistas no treino: {sorted(unseen)}")
