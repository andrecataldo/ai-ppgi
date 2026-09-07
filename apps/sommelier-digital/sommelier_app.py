# -*- coding: utf-8 -*-
"""
Sommelier Digital: sistema especialista de harmonizacao de jantar
Curso de Introducao a IA (UNIRIO) - demo em aula
Executar:  streamlit run sommelier_app.py
Requisitos: pip install experta streamlit && pip install --upgrade frozendict
"""
import streamlit as st
from experta import (Fact, KnowledgeEngine, Rule, DefFacts,
                     AS, MATCH, TEST, L)

# =================================================================
# PARTE 1: O SISTEMA ESPECIALISTA (independente da interface)
# =================================================================

class Prato(Fact):
    """proteina, preparo, picancia"""

class Preferencia(Fact):
    """bebida_alcoolica: sim|nao"""

class Ocasiao(Fact):
    """tipo: casual|formal|celebracao"""

class Recomendacao(Fact):
    """bebida, confianca"""


class SommelierDigital(KnowledgeEngine):
    """Base de conhecimento: 8 regras de harmonizacao."""

    def __init__(self):
        super().__init__()
        self.trilha = []   # modulo de explicacao

    def registrar(self, regra, texto):
        self.trilha.append((regra, texto))

    @DefFacts()
    def contexto(self):
        yield Fact(sistema="sommelier", versao="1.0")

    # ---- Regras de proteina x preparo ----
    @Rule(Prato(proteina='peixe', preparo=L('grelhado') | L('molho_leve')),
          Preferencia(bebida_alcoolica='sim'))
    def r1_peixe_leve(self):
        self.declare(Recomendacao(bebida='Vinho branco leve (Sauvignon Blanc)',
                                  confianca=0.90))
        self.registrar('R1', 'Peixe em preparo leve pede acidez e corpo leve; '
                             'o Sauvignon Blanc realca o frescor sem cobrir o sabor.')

    @Rule(Prato(proteina='peixe', preparo='molho_intenso'),
          Preferencia(bebida_alcoolica='sim'))
    def r2_peixe_intenso(self):
        self.declare(Recomendacao(bebida='Branco encorpado (Chardonnay com madeira)',
                                  confianca=0.85))
        self.registrar('R2', 'Molho intenso sobre peixe exige um branco com mais '
                             'estrutura; o Chardonnay barricado equilibra a untuosidade.')

    @Rule(Prato(proteina='carne_vermelha', preparo=L('assado') | L('grelhado')),
          Preferencia(bebida_alcoolica='sim'))
    def r3_carne(self):
        self.declare(Recomendacao(bebida='Tinto encorpado (Malbec)',
                                  confianca=0.90))
        self.registrar('R3', 'Gordura e proteina da carne suavizam os taninos; '
                             'o Malbec tem estrutura para acompanhar.')

    @Rule(Prato(proteina='frango', picancia=L('baixa') | L('media')),
          Preferencia(bebida_alcoolica='sim'))
    def r4_frango(self):
        self.declare(Recomendacao(bebida='Tinto leve (Pinot Noir) ou rose seco',
                                  confianca=0.80))
        self.registrar('R4', 'Aves de sabor delicado combinam com tintos de corpo '
                             'leve e taninos macios.')

    @Rule(Prato(proteina='massa', preparo='molho_intenso'),
          Preferencia(bebida_alcoolica='sim'))
    def r5_massa(self):
        self.declare(Recomendacao(bebida='Tinto italiano de boa acidez (Chianti)',
                                  confianca=0.85))
        self.registrar('R5', 'Molhos de tomate tem alta acidez; o vinho precisa de '
                             'acidez equivalente para nao parecer apagado.')

    @Rule(Prato(proteina='vegetariano'),
          Preferencia(bebida_alcoolica='sim'))
    def r6_vegetariano(self):
        self.declare(Recomendacao(bebida='Branco aromatico (Torrontes) ou rose',
                                  confianca=0.75))
        self.registrar('R6', 'Pratos vegetais valorizam brancos aromaticos e frescos.')

    # ---- Regra prioritaria: picancia domina (saliencia 10) ----
    @Rule(Prato(picancia='alta'),
          Preferencia(bebida_alcoolica='sim'),
          salience=10)
    def r7_picante(self):
        self.declare(Recomendacao(bebida='Riesling off-dry ou espumante demi-sec',
                                  confianca=0.85))
        self.registrar('R7', 'PRIORITARIA (saliencia 10): pimenta amplifica alcool e '
                             'taninos; um toque de doçura refresca o paladar. '
                             'A picancia domina a escolha da proteina.')

    # ---- Restricao do usuario: prioridade maxima (saliencia 20) ----
    @Rule(Preferencia(bebida_alcoolica='nao'), salience=20)
    def r8_sem_alcool(self):
        self.declare(Recomendacao(bebida='Agua com gas com limao siciliano ou '
                                         'cha gelado de hibisco',
                                  confianca=0.80))
        self.registrar('R8', 'PRIORITARIA (saliencia 20): a restricao do usuario '
                             'prevalece sobre qualquer harmonizacao.')

    # ---- Encadeamento: celebracao refina a recomendacao ----
    @Rule(AS.rec << Recomendacao(bebida=MATCH.b),
          Ocasiao(tipo='celebracao'),
          Preferencia(bebida_alcoolica='sim'),
          TEST(lambda b: 'espumante' not in b.lower()))
    def r9_celebracao(self, rec, b):
        self.retract(rec)
        self.declare(Recomendacao(bebida=f'{b} + espumante brut para o brinde',
                                  confianca=0.95))
        self.registrar('R9', 'ENCADEAMENTO: a regra casou com a recomendacao '
                             'declarada por outra regra, retraiu o fato e o '
                             'substituiu por uma versao refinada para a celebracao.')


def consultar(proteina, preparo, picancia, alcool, ocasiao):
    """Um ciclo completo: reset -> declare -> run -> coleta."""
    engine = SommelierDigital()
    engine.reset()
    engine.declare(Prato(proteina=proteina, preparo=preparo, picancia=picancia))
    engine.declare(Preferencia(bebida_alcoolica=alcool))
    engine.declare(Ocasiao(tipo=ocasiao))
    engine.run()
    recs = [f for f in engine.facts.values() if isinstance(f, Recomendacao)]
    memoria = [dict(f) for f in engine.facts.values()]
    return recs, engine.trilha, memoria


# =================================================================
# PARTE 2: INTERFACE (Streamlit)
# =================================================================

st.set_page_config(page_title="Sommelier Digital", page_icon="🍷", layout="centered")

st.title("🍷 Sommelier Digital")
st.caption("Sistema especialista de harmonizacao de jantar. "
           "Introducao a IA, UNIRIO. Motor: experta (RETE, encadeamento progressivo).")

col1, col2 = st.columns(2)
with col1:
    proteina = st.selectbox("Proteina principal",
        ['peixe', 'frango', 'carne_vermelha', 'massa', 'vegetariano'])
    preparo = st.selectbox("Preparo",
        ['grelhado', 'assado', 'molho_leve', 'molho_intenso', 'frito'])
    picancia = st.select_slider("Picancia", ['baixa', 'media', 'alta'])
with col2:
    alcool = st.radio("O convidado bebe alcool?", ['sim', 'nao'], horizontal=True)
    ocasiao = st.selectbox("Ocasiao", ['casual', 'formal', 'celebracao'])

if st.button("Harmonizar", type="primary", use_container_width=True):
    recs, trilha, memoria = consultar(proteina, preparo, picancia, alcool, ocasiao)

    if recs:
        for r in recs:
            st.success(f"**{r['bebida']}**")
            st.progress(r['confianca'], text=f"Confianca: {r['confianca']:.0%}")
    else:
        st.warning("Nenhuma regra da base cobre esta combinacao. "
                   "Eis a fragilidade dos SEs: conhecimento fora da base nao existe. "
                   "Que regra voce escreveria?")

    with st.expander("Por que esta recomendacao? (modulo de explicacao)",
                     expanded=True):
        if trilha:
            for regra, texto in trilha:
                st.markdown(f"**[{regra}]** {texto}")
        else:
            st.markdown("Nenhuma regra disparou.")

    with st.expander("Memoria de trabalho ao final da inferencia"):
        st.json(memoria)
