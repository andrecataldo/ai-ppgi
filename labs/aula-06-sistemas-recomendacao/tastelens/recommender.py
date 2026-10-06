# ============================================================
# TASTELENS
# Filtragem colaborativa baseada em usuários
# ============================================================

"""
Módulo responsável pelo núcleo do sistema de recomendação.

Implementa uma abordagem de filtragem colaborativa baseada
em usuários, inspirada no princípio apresentado pelo GroupLens:

    usuários que apresentaram padrões semelhantes de avaliação
    no passado podem fornecer informação útil para prever
    preferências futuras.

Etapas principais:

1. calcular similaridade entre usuários com Pearson;
2. construir a matriz de similaridade;
3. prever avaliações ainda não observadas;
4. recomendar o item com maior nota prevista.

O módulo é independente do domínio utilizado pela aplicação.
Ele trabalha apenas com usuários, itens e avaliações.
"""

import numpy as np
import pandas as pd
from catalog import (
    MIN_AVALIACOES_COMUNS,
    NOTA_MAXIMA,
    NOTA_MINIMA,
)

# ------------------------------------------------------------
# 1. SIMILARIDADE DE PEARSON
# ------------------------------------------------------------


def pearson_pair(
    matriz: pd.DataFrame,
    usuario_1: str,
    usuario_2: str,
) -> float:
    """
    Calcula a correlação de Pearson entre dois usuários.

    A comparação considera somente os itens avaliados por
    ambos os usuários.

    Retorna
    -------
    float
        Valor entre -1 e +1 quando a correlação pode ser
        calculada.

        np.nan quando:
        - algum usuário não existe;
        - existem poucas avaliações em comum;
        - algum dos usuários não apresenta variação nas notas.
    """

    if usuario_1 not in matriz.index:
        return np.nan

    if usuario_2 not in matriz.index:
        return np.nan

    notas_1 = matriz.loc[usuario_1]
    notas_2 = matriz.loc[usuario_2]

    # Somente posições em que ambos possuem avaliação.
    comum = notas_1.notna() & notas_2.notna()

    if comum.sum() < MIN_AVALIACOES_COMUNS:
        return np.nan

    x = notas_1[comum].astype(float)
    y = notas_2[comum].astype(float)

    # Pearson não é definido quando uma das séries
    # não possui variância.
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan

    correlacao = x.corr(y)

    if pd.isna(correlacao):
        return np.nan

    return float(correlacao)


# ------------------------------------------------------------
# 2. MATRIZ DE SIMILARIDADE
# ------------------------------------------------------------


def matriz_similaridade(
    matriz: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula a similaridade entre todos os pares de usuários.

    O resultado é uma matriz quadrada:

                    Ana     Bruno    Carlos
        Ana         1.00     0.82     -0.75
        Bruno       0.82     1.00     -0.60
        Carlos     -0.75    -0.60      1.00

    A diagonal é igual a 1 porque representa a relação de
    cada usuário consigo mesmo.
    """

    usuarios = matriz.index.tolist()

    similaridades = pd.DataFrame(
        index=usuarios,
        columns=usuarios,
        dtype=float,
    )

    for usuario_1 in usuarios:
        for usuario_2 in usuarios:
            if usuario_1 == usuario_2:
                similaridades.loc[
                    usuario_1,
                    usuario_2,
                ] = 1.0
                continue

            similaridades.loc[
                usuario_1,
                usuario_2,
            ] = pearson_pair(
                matriz,
                usuario_1,
                usuario_2,
            )

    return similaridades


# ------------------------------------------------------------
# 3. PREVISÃO DE UMA AVALIAÇÃO
# ------------------------------------------------------------


def prever_nota(
    matriz: pd.DataFrame,
    usuario: str,
    item: str,
) -> tuple[float, list]:
    """
    Estima a avaliação de um usuário para um item ainda
    não avaliado.

    Utiliza média centrada e similaridade de Pearson:

                            Σ sim(u,v) * (r(v,i) - média(v))
    r^(u,i) = média(u) + -----------------------------------
                                   Σ |sim(u,v)|

    Diferentemente de algumas implementações simplificadas,
    correlações negativas também são utilizadas.

    Isso significa que um usuário com padrão consistentemente
    oposto também pode fornecer informação útil.

    Retorna
    -------
    tuple
        (
            previsão,
            lista_de_influências
        )

    Cada influência possui:

        (
            vizinho,
            similaridade,
            nota_do_vizinho
        )
    """

    if usuario not in matriz.index:
        return np.nan, []

    if item not in matriz.columns:
        return np.nan, []

    # Não faz sentido prever um item que o usuário
    # já avaliou.
    if pd.notna(matriz.loc[usuario, item]):
        return float(matriz.loc[usuario, item]), []

    media_usuario = matriz.loc[usuario].mean()

    if pd.isna(media_usuario):
        return np.nan, []

    numerador = 0.0
    denominador = 0.0

    influencias = []

    for vizinho in matriz.index:
        if vizinho == usuario:
            continue

        nota_item = matriz.loc[vizinho, item]

        # Vizinho não avaliou o item.
        if pd.isna(nota_item):
            continue

        similaridade = pearson_pair(
            matriz,
            usuario,
            vizinho,
        )

        # Sem evidência suficiente para comparar.
        if pd.isna(similaridade):
            continue

        # Similaridade exatamente zero não acrescenta
        # informação à previsão.
        if np.isclose(similaridade, 0.0):
            continue

        media_vizinho = matriz.loc[vizinho].mean()

        if pd.isna(media_vizinho):
            continue

        desvio_vizinho = float(nota_item) - float(media_vizinho)

        contribuicao = float(similaridade) * desvio_vizinho

        numerador += contribuicao
        denominador += abs(float(similaridade))

        influencias.append(
            (
                vizinho,
                float(similaridade),
                float(nota_item),
            )
        )

    # Nenhum vizinho forneceu evidência utilizável.
    if np.isclose(denominador, 0.0):
        return np.nan, []

    previsao = float(media_usuario) + numerador / denominador

    # Mantemos a previsão dentro da escala de avaliação.
    previsao = max(
        float(NOTA_MINIMA),
        min(
            float(NOTA_MAXIMA),
            previsao,
        ),
    )

    # Ordena pela força absoluta da relação.
    influencias.sort(
        key=lambda registro: abs(registro[1]),
        reverse=True,
    )

    return float(previsao), influencias


# ------------------------------------------------------------
# 4. RECOMENDAÇÃO
# ------------------------------------------------------------


def recomendar_para(
    matriz: pd.DataFrame,
    usuario: str,
    candidatos: list[str],
):
    """
    Calcula previsões para os itens candidatos ainda não
    avaliados e retorna aquele com maior nota estimada.

    Retorna
    -------
    tuple | None

        (
            item,
            nota_prevista,
            influencias
        )

    ou None quando nenhuma previsão pode ser calculada.
    """

    if usuario not in matriz.index:
        return None

    previsoes = []

    for item in candidatos:
        # Se o item nem aparece na matriz, ninguém o avaliou
        # e não existe informação colaborativa para utilizá-lo.
        if item not in matriz.columns:
            continue

        # Nunca recomendamos algo já avaliado pelo usuário.
        if pd.notna(matriz.loc[usuario, item]):
            continue

        previsao, influencias = prever_nota(
            matriz,
            usuario,
            item,
        )

        if pd.isna(previsao):
            continue

        previsoes.append(
            (
                item,
                float(previsao),
                influencias,
            )
        )

    if not previsoes:
        return None

    # Maior previsão primeiro.
    previsoes.sort(
        key=lambda registro: registro[1],
        reverse=True,
    )

    return previsoes[0]
