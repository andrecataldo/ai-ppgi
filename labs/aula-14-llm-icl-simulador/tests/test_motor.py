import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from motor import caracteristicas, demonstracoes, prever, softmax


class TesteSimulador(unittest.TestCase):
    def test_features(self):
        self.assertEqual(caracteristicas(" TOV-42 "), {"prefixo": "tov", "sufixo": "42"})

    def test_zero_shot_is_indeterminate(self):
        r = prever("TOV-42", demonstracoes(0))
        self.assertEqual(r["predicao"], "INDETERMINADO")
        self.assertEqual(r["pesos_por_classe"], {"LUM": 0.5, "NEX": 0.5})

    def test_four_shot_prefix(self):
        r = prever("TOV-42", demonstracoes(4))
        self.assertEqual(r["predicao"], "NEX")
        self.assertAlmostEqual(sum(r["pesos_por_classe"].values()), 1)
        self.assertAlmostEqual(sum(p["atencao"] for p in r["passos"]), 1)

    def test_flip_labels_changes_prediction_without_changing_scores(self):
        a = prever("TOV-42", demonstracoes(4))
        b = prever("TOV-42", demonstracoes(4, invertidos=True))
        self.assertEqual(a["predicao"], "NEX")
        self.assertEqual(b["predicao"], "LUM")
        self.assertEqual([x["pontuacao"] for x in a["passos"]],
                         [x["pontuacao"] for x in b["passos"]])

    def test_order_invariance_without_position_bias(self):
        a = prever("TOV-42", demonstracoes(4), vies_posicao=0)
        b = prever("TOV-42", demonstracoes(4, ordem_inversa=True), vies_posicao=0)
        self.assertAlmostEqual(a["pesos_por_classe"]["NEX"], b["pesos_por_classe"]["NEX"])

    def test_position_bias_can_change_result(self):
        a = prever("TOV-42", demonstracoes(4), vies_posicao=2)
        b = prever("TOV-42", demonstracoes(4, ordem_inversa=True), vies_posicao=2)
        self.assertGreater(abs(a["pesos_por_classe"]["NEX"] - b["pesos_por_classe"]["NEX"]), 0.01)

    def test_softmax_stability(self):
        self.assertAlmostEqual(sum(softmax([1000, 1001, 1002])), 1)

    def test_invalid_shot_count(self):
        with self.assertRaises(ValueError):
            demonstracoes(3)


if __name__ == "__main__":
    unittest.main()
