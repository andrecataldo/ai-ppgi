# ============================================================
# TESTES DO TASTELENS
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Permite executar pytest a partir da pasta tastelens.
PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


from recommender import (
    matriz_similaridade,
    pearson_pair,
    prever_nota,
    recomendar_para,
)

# ------------------------------------------------------------
# TESTE 1
# Pearson reconhece padrões semelhantes e opostos
# ------------------------------------------------------------


def test_pearson_semelhante_e_oposto():

    matriz = pd.DataFrame(
        {
            "Pizza": [1, 1, 3],
            "Sushi": [2, 2, 2],
            "Feijoada": [3, 3, 1],
        },
        index=[
            "Andre",
            "Ana",
            "Bruno",
        ],
    )

    sim_ana = pearson_pair(
        matriz,
        "Andre",
        "Ana",
    )

    sim_bruno = pearson_pair(
        matriz,
        "Andre",
        "Bruno",
    )

    assert np.isclose(
        sim_ana,
        1.0,
    )

    assert np.isclose(
        sim_bruno,
        -1.0,
    )


# ------------------------------------------------------------
# TESTE 2
# Correlação negativa participa da previsão
# ------------------------------------------------------------


def test_correlacao_negativa_influencia_previsao():

    matriz = pd.DataFrame(
        {
            "Pizza": [
                1,
                3,
            ],
            "Sushi": [
                2,
                2,
            ],
            "Feijoada": [
                3,
                1,
            ],
            "Moqueca": [
                np.nan,
                1,
            ],
        },
        index=[
            "Andre",
            "Bruno",
        ],
    )

    previsao, influencias = prever_nota(
        matriz,
        "Andre",
        "Moqueca",
    )

    assert not np.isnan(previsao)

    # Bruno possui comportamento inverso ao de André.
    assert np.isclose(
        influencias[0][1],
        -1.0,
    )

    # Bruno deu uma avaliação baixa.
    # Como a correlação é negativa, isso contribui para
    # aumentar a previsão para André.
    assert previsao > matriz.loc["Andre"].mean()


# ------------------------------------------------------------
# TESTE 3
# Matriz de similaridade
# ------------------------------------------------------------


def test_matriz_similaridade():

    matriz = pd.DataFrame(
        {
            "Pizza": [1, 1, 3],
            "Sushi": [2, 2, 2],
            "Feijoada": [3, 3, 1],
        },
        index=[
            "Andre",
            "Ana",
            "Bruno",
        ],
    )

    sim = matriz_similaridade(matriz)

    assert np.isclose(
        sim.loc["Andre", "Andre"],
        1.0,
    )

    assert np.isclose(
        sim.loc["Andre", "Ana"],
        1.0,
    )

    assert np.isclose(
        sim.loc["Andre", "Bruno"],
        -1.0,
    )

    assert np.isclose(
        sim.loc["Bruno", "Andre"],
        -1.0,
    )


# ------------------------------------------------------------
# TESTE 4
# Recomendação escolhe o item com maior previsão
# ------------------------------------------------------------


def test_recomendar_melhor_item():

    matriz = pd.DataFrame(
        {
            "Pizza": [
                5,
                5,
                1,
            ],
            "Sushi": [
                4,
                4,
                2,
            ],
            "Feijoada": [
                1,
                1,
                5,
            ],
            "Moqueca": [
                np.nan,
                5,
                1,
            ],
            "Risoto": [
                np.nan,
                2,
                4,
            ],
        },
        index=[
            "Andre",
            "Ana",
            "Bruno",
        ],
    )

    recomendacao = recomendar_para(
        matriz,
        "Andre",
        [
            "Moqueca",
            "Risoto",
        ],
    )

    assert recomendacao is not None

    item, previsao, influencias = recomendacao

    assert item == "Moqueca"

    assert previsao > 4.0

    assert len(influencias) == 2
