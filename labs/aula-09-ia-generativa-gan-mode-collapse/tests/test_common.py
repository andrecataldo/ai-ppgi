"""Testes executáveis sem Streamlit e sem downloads de dados."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (NUM_MODES, Discriminator, Generator, centers_numpy,
                    evaluate_modes, reference_samples, sample_real)
from lab import gradient_penalty


class TestMetrics(unittest.TestCase):
    def test_real_data_shape(self):
        arr = sample_real(100, np.random.default_rng(3))
        self.assertEqual(arr.shape, (100, 2))
        self.assertEqual(arr.dtype, np.float32)

    def test_reproducible_data(self):
        a = sample_real(200, np.random.default_rng(10))
        b = sample_real(200, np.random.default_rng(10))
        np.testing.assert_array_equal(a, b)

    def test_centers_on_circle(self):
        c = centers_numpy()
        self.assertEqual(c.shape, (NUM_MODES, 2))
        np.testing.assert_allclose(np.linalg.norm(c, axis=1), 2, rtol=0, atol=1e-6)

    def test_full_reference(self):
        data = reference_samples(16000, np.random.default_rng(5), tuple(range(8)))
        m = evaluate_modes(data)
        self.assertEqual(m.coverage, 8)
        self.assertGreater(m.valid_fraction, 0.99)
        self.assertLess(m.js_divergence_bits, 0.01)

    def test_collapsed_reference(self):
        data = reference_samples(16000, np.random.default_rng(5), (1, 5))
        m = evaluate_modes(data)
        self.assertEqual(m.coverage, 2)
        self.assertGreater(m.valid_fraction, 0.99)
        self.assertGreater(m.js_divergence_bits, 0.50)
        self.assertLess(m.js_divergence_bits, 1.0)

    def test_invalid_data(self):
        m = evaluate_modes(np.full((200, 2), 500, dtype=np.float32))
        self.assertEqual(m.coverage, 0)
        self.assertEqual(m.valid_fraction, 0.0)
        self.assertEqual(m.js_divergence_bits, 1.0)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            evaluate_modes(np.array([1, 2]))
        with self.assertRaises(ValueError):
            sample_real(0, np.random.default_rng(3))


class TestModels(unittest.TestCase):
    def test_shapes_and_backward(self):
        g, d = Generator(), Discriminator()
        z = torch.randn(16, 2)
        fake = g(z)
        scores = d(fake)
        self.assertEqual(tuple(fake.shape), (16, 2))
        self.assertEqual(tuple(scores.shape), (16,))
        scores.mean().backward()
        self.assertIsNotNone(g.net[0].weight.grad)

    def test_gp_finite(self):
        d = Discriminator()
        real = torch.randn(10, 2)
        fake = torch.randn(10, 2)
        penalty = gradient_penalty(d, real, fake)
        self.assertTrue(torch.isfinite(penalty).item())
        self.assertGreaterEqual(penalty.item(), 0)


if __name__ == "__main__":
    unittest.main()
