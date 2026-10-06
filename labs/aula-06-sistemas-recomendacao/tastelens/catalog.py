# ============================================================
# TASTELENS
# Catálogo e parâmetros da demonstração
# ============================================================

"""
Configurações do domínio do TasteLens.

Este módulo concentra os itens avaliados pelos participantes
e os principais parâmetros utilizados pela aplicação.

O algoritmo de filtragem colaborativa permanece separado
em recommender.py, evitando dependência direta do domínio.
"""

# ------------------------------------------------------------
# ITENS-ÂNCORA
# ------------------------------------------------------------
# Todos os participantes avaliam estes itens.
#
# Eles criam uma base comum de avaliações que permite
# comparar os padrões de preferência entre os usuários.

ITENS_ANCORA = [
    "Pizza",
    "Sushi",
    "Feijoada",
    "Hambúrguer",
]


# ------------------------------------------------------------
# ITENS CANDIDATOS
# ------------------------------------------------------------
# Cada participante receberá apenas uma parte destes itens
# para avaliar.
#
# As células ausentes resultantes serão utilizadas pelo
# algoritmo para gerar previsões e recomendações.

ITENS_CANDIDATOS = [
    "Moqueca",
    "Lasanha",
    "Churrasco",
    "Yakisoba",
    "Risoto",
    "Acarajé",
    "Tacos",
    "Poke",
    "Curry indiano",
    "Escondidinho",
]


# ------------------------------------------------------------
# PARÂMETROS DA DINÂMICA
# ------------------------------------------------------------

# Quantidade de itens candidatos atribuídos aleatoriamente
# a cada participante.
QUANTIDADE_EXTRAS = 3

# Quantidade mínima de avaliações necessárias
# para considerar a etapa do participante concluída.
MIN_ANCORAS_RESPONDIDAS = 3
MIN_EXTRAS_RESPONDIDOS = 2


# Quantidade mínima de avaliações em comum necessária para
# calcular a correlação de Pearson entre dois participantes.
#
# Quanto maior esse valor, maior a evidência disponível para
# avaliar a similaridade, mas também maior a chance de não
# conseguirmos calcular a correlação em grupos pequenos.
MIN_AVALIACOES_COMUNS = 3


# Valor mínimo, em módulo, para mostrar uma relação na
# visualização da rede de similaridade.
LIMIAR_ARESTA_REDE = 0.45


# Escala utilizada nas avaliações.
NOTA_MINIMA = 1
NOTA_MAXIMA = 5


# ------------------------------------------------------------
# CONFIGURAÇÕES DA APLICAÇÃO
# ------------------------------------------------------------

APP_NAME = "TasteLens"

APP_SUBTITLE = (
    "Sistema interativo de recomendação baseado "
    "em filtragem colaborativa entre usuários"
)

DB_FILE = "data/tastelens.db"

PORT = 8501

ADMIN_PASSWORD = "tastelens2026"
