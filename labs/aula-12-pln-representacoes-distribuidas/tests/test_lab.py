"""Testes de corretude do lab; executar `python -m unittest discover -s tests -v`."""
import math
import tempfile
import unittest

import torch

from src.data import (BOS, CONTEXT_SIZE, assert_no_oov, build_vocab,
                      encode_examples, make_sentences, split_sentences)
from src.models import NeuralLanguageModel, SmoothedNGram
from src.experiment import evaluate_neural, nearest_neighbors, run


class TestData(unittest.TestCase):
    def test_corpus_distinct(self):
        sentences = make_sentences()
        self.assertEqual(len(sentences), 288)
        self.assertEqual(len(set(sentences)), 288)

    def test_split_has_no_duplicate_sentences(self):
        train, test = split_sentences(seed=42)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(len(train) + len(test), 288)
        assert_no_oov(test, build_vocab(train))

    def test_examples_include_eos(self):
        train, _ = split_sentences()
        vocab = build_vocab(train)
        examples = encode_examples(["o gato dorme no sofá"], vocab)
        self.assertEqual(len(examples), 6)  # cinco palavras + <eos>
        self.assertEqual(examples[0][0], (vocab[BOS],) * CONTEXT_SIZE)
        self.assertEqual(examples[-1][1], vocab["<eos>"])


class TestModels(unittest.TestCase):
    def test_ngram_smoothing_is_normalized(self):
        vocab_size = 7
        model = SmoothedNGram(vocab_size, 0.5).fit([((0, 1, 2), 3)])
        self.assertAlmostEqual(sum(model.probability((0, 1, 2), t) for t in range(vocab_size)), 1)
        self.assertGreater(model.probability((1, 2, 3), 4), 0)

    def test_neural_shape_and_probabilities(self):
        torch.manual_seed(42)
        model = NeuralLanguageModel(10, 3, 4, 6)
        logits = model(torch.tensor([[0, 1, 2], [1, 2, 3]]))
        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertTrue(torch.allclose(logits.softmax(-1).sum(-1), torch.ones(2)))

    def test_embedding_has_gradients(self):
        torch.manual_seed(42)
        model = NeuralLanguageModel(12, 3, 4, 6)
        loss = torch.nn.functional.cross_entropy(model(torch.tensor([[1, 2, 3]])), torch.tensor([4]))
        loss.backward()
        self.assertGreater(model.embedding.weight.grad[1].abs().sum().item(), 0)

    def test_evaluation_and_similarity_are_finite(self):
        train, _ = split_sentences()
        vocab = build_vocab(train)
        examples = encode_examples(train[:2], vocab)
        model = NeuralLanguageModel(len(vocab), CONTEXT_SIZE, 8, 8)
        self.assertTrue(math.isfinite(evaluate_neural(model, examples)))
        neighbors = nearest_neighbors(model, vocab, "gato")
        self.assertTrue(all(-1 <= n["cosseno"] <= 1 for n in neighbors))

    def test_run_writes_results(self):
        with tempfile.TemporaryDirectory() as temp:
            metrics = run(42, 1, temp)
            self.assertTrue(math.isfinite(metrics["perplexidade_neural_teste"]))
            from pathlib import Path
            self.assertTrue((Path(temp) / "metricas.json").exists())
            self.assertTrue((Path(temp) / "RELATORIO_RESULTADOS.md").exists())


if __name__ == "__main__":
    unittest.main()
