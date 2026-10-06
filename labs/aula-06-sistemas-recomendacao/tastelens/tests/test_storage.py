import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


import storage


def test_fluxo_storage(tmp_path, monkeypatch):

    db_teste = tmp_path / "teste.db"

    monkeypatch.setattr(
        storage,
        "DB_FILE",
        str(db_teste),
    )

    participante = storage.get_or_create_participante("TesteUser")

    assert participante is not None

    assert participante["pseudonym"] == "TesteUser"

    assert len(participante["assigned_items"]) == 3

    storage.salvar_avaliacoes(
        participante["id"],
        {
            "Pizza": 5,
            "Sushi": 4,
            "Feijoada": 2,
        },
    )

    participantes, matriz = storage.carregar_dados()

    assert len(participantes) == 1

    assert (
        matriz.loc[
            "TesteUser",
            "Pizza",
        ]
        == 5
    )

    assert (
        matriz.loc[
            "TesteUser",
            "Sushi",
        ]
        == 4
    )

    assert (
        matriz.loc[
            "TesteUser",
            "Feijoada",
        ]
        == 2
    )


def test_recommendations_release(
    tmp_path,
    monkeypatch,
):

    db_teste = tmp_path / "settings.db"

    monkeypatch.setattr(
        storage,
        "DB_FILE",
        str(db_teste),
    )

    assert storage.recommendations_released() is False

    storage.set_recommendations_released(True)

    assert storage.recommendations_released() is True

    storage.set_recommendations_released(False)

    assert storage.recommendations_released() is False
