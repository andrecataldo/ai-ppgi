"""Entrada do lab. Rode a partir da pasta do laboratorio."""
import argparse
from src.experiment import run


def main():
    parser = argparse.ArgumentParser(description="Lab PLN: embeddings versus n-grama")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    data = run(seed=args.seed, epochs=args.epochs, output_dir=args.output_dir)
    print("\nLAB — PLN e representacoes distribuidas")
    print(f"Frases treino/teste: {data['n_frases_treino']} / {data['n_frases_teste']}")
    print(f"Vocabulario: {data['tamanho_vocabulario']} | Contexto: {data['contexto_palavras']}")
    print(f"Perplexidade n-grama: {data['perplexidade_ngram_teste']:.3f}")
    print(f"Perplexidade neural: {data['perplexidade_neural_teste']:.3f}")
    print("\nVizinhos vetoriais de gato:", data['vizinhos_embeddings']['gato'])
    print("Previsoes de 'o gato dorme no':", data['previsao_exemplo'])
    print("\nResultados em: results/ (ou --output-dir)")


if __name__ == "__main__":
    main()
