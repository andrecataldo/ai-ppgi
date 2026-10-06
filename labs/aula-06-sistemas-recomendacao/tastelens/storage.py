# ============================================================
# TASTELENS
# Persistência de dados com SQLite
# ============================================================

"""
Módulo responsável pela persistência da aplicação.

Armazena:
- participantes;
- itens extras atribuídos;
- avaliações;
- estado de liberação das recomendações.

O restante da aplicação não precisa conhecer detalhes
do SQLite.
"""

import hashlib
import random
import sqlite3
import uuid
from pathlib import Path

import pandas as pd
from catalog import (
    DB_FILE,
    ITENS_CANDIDATOS,
    QUANTIDADE_EXTRAS,
)

# ------------------------------------------------------------
# BANCO
# ------------------------------------------------------------


def get_conn():
    """
    Abre uma conexão com o banco SQLite.

    A pasta do banco é criada automaticamente caso
    ainda não exista.
    """

    db_path = Path(DB_FILE)

    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conn = sqlite3.connect(
        db_path,
        check_same_thread=False,
    )

    conn.execute("PRAGMA foreign_keys = ON")

    criar_tabelas(conn)

    return conn


def criar_tabelas(conn):
    """
    Cria as tabelas utilizadas pela aplicação.
    """

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS participants (
            id TEXT PRIMARY KEY,
            pseudonym TEXT NOT NULL UNIQUE,
            assigned_items TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_id TEXT NOT NULL,
            item TEXT NOT NULL,
            rating INTEGER NOT NULL
                CHECK (rating BETWEEN 1 AND 5),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(participant_id, item),

            FOREIGN KEY(participant_id)
                REFERENCES participants(id)
                ON DELETE CASCADE
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY,
            recommendations_released INTEGER
                NOT NULL DEFAULT 0
        )
        """
    )

    conn.execute(
        """
        INSERT OR IGNORE INTO settings (
            id,
            recommendations_released
        )
        VALUES (1, 0)
        """
    )

    conn.commit()


# ------------------------------------------------------------
# PARTICIPANTES
# ------------------------------------------------------------


def buscar_participante(
    pseudonimo: str,
):
    """
    Recupera um participante pelo pseudônimo.
    """

    conn = get_conn()

    row = conn.execute(
        """
        SELECT
            id,
            pseudonym,
            assigned_items
        FROM participants
        WHERE pseudonym = ?
        """,
        (pseudonimo,),
    ).fetchone()

    conn.close()

    if row is None:
        return None

    return {
        "id": row[0],
        "pseudonym": row[1],
        "assigned_items": (row[2].split("|") if row[2] else []),
    }


def criar_participante(
    pseudonimo: str,
):
    """
    Cria um novo participante.

    O pseudônimo é usado como semente determinística
    para atribuir itens extras.

    Assim, se o mesmo pseudônimo fosse recriado em
    condições equivalentes, receberia o mesmo conjunto.
    """

    seed = int(
        hashlib.sha256(pseudonimo.encode("utf-8")).hexdigest()[:8],
        16,
    )

    rng = random.Random(seed)

    extras = rng.sample(
        ITENS_CANDIDATOS,
        QUANTIDADE_EXTRAS,
    )

    participante = {
        "id": str(uuid.uuid4()),
        "pseudonym": pseudonimo,
        "assigned_items": extras,
    }

    conn = get_conn()

    conn.execute(
        """
        INSERT INTO participants (
            id,
            pseudonym,
            assigned_items
        )
        VALUES (?, ?, ?)
        """,
        (
            participante["id"],
            participante["pseudonym"],
            "|".join(extras),
        ),
    )

    conn.commit()
    conn.close()

    return participante


def get_or_create_participante(
    pseudonimo: str,
):
    """
    Recupera o participante existente ou cria um novo.
    """

    participante = buscar_participante(pseudonimo)

    if participante is not None:
        return participante

    return criar_participante(pseudonimo)


# ------------------------------------------------------------
# AVALIAÇÕES
# ------------------------------------------------------------


def salvar_avaliacoes(
    participant_id: str,
    ratings: dict,
):
    """
    Salva ou atualiza avaliações de um participante.

    ratings:
        {
            "Pizza": 5,
            "Sushi": 4,
            ...
        }
    """

    conn = get_conn()

    for item, nota in ratings.items():
        conn.execute(
            """
            INSERT INTO ratings (
                participant_id,
                item,
                rating
            )
            VALUES (?, ?, ?)

            ON CONFLICT(participant_id, item)
            DO UPDATE SET
                rating = excluded.rating
            """,
            (
                participant_id,
                item,
                int(nota),
            ),
        )

    conn.commit()
    conn.close()


# ------------------------------------------------------------
# MATRIZ USUÁRIO × ITEM
# ------------------------------------------------------------


def carregar_dados():
    """
    Retorna:
    - dataframe de participantes;
    - matriz usuário × item.
    """

    conn = get_conn()

    participantes = pd.read_sql_query(
        """
        SELECT
            id,
            pseudonym,
            assigned_items
        FROM participants
        """,
        conn,
    )

    avaliacoes = pd.read_sql_query(
        """
        SELECT
            participant_id,
            item,
            rating
        FROM ratings
        """,
        conn,
    )

    conn.close()

    if participantes.empty:
        return (
            participantes,
            pd.DataFrame(),
        )

    if avaliacoes.empty:
        return (
            participantes,
            pd.DataFrame(),
        )

    merged = avaliacoes.merge(
        participantes[
            [
                "id",
                "pseudonym",
            ]
        ],
        left_on="participant_id",
        right_on="id",
        how="left",
    )

    matriz = merged.pivot_table(
        index="pseudonym",
        columns="item",
        values="rating",
        aggfunc="last",
    )

    return participantes, matriz


# ------------------------------------------------------------
# ESTADO DAS RECOMENDAÇÕES
# ------------------------------------------------------------


def recommendations_released() -> bool:
    """
    Informa se as recomendações estão liberadas.
    """

    conn = get_conn()

    row = conn.execute(
        """
        SELECT recommendations_released
        FROM settings
        WHERE id = 1
        """
    ).fetchone()

    conn.close()

    return bool(row[0])


def set_recommendations_released(
    value: bool,
):
    """
    Libera ou bloqueia a visualização das recomendações.
    """

    conn = get_conn()

    conn.execute(
        """
        UPDATE settings
        SET recommendations_released = ?
        WHERE id = 1
        """,
        (1 if value else 0,),
    )

    conn.commit()
    conn.close()


# ------------------------------------------------------------
# LIMPEZA DA DEMONSTRAÇÃO
# ------------------------------------------------------------


def limpar_tudo():
    """
    Remove participantes e avaliações.

    Utilizado para reiniciar uma nova demonstração.
    """

    conn = get_conn()

    conn.execute("DELETE FROM ratings")

    conn.execute("DELETE FROM participants")

    conn.execute(
        """
        UPDATE settings
        SET recommendations_released = 0
        WHERE id = 1
        """
    )

    conn.commit()
    conn.close()
