# -*- coding: utf-8 -*-
"""
Incident Triage Expert

Sistema especialista para diagnóstico inicial de incidentes
em aplicações Web.

Disciplina de Inteligência Artificial - PPGI/UNIRIO - 2026.2

Executar:
    streamlit run apps/incident-triage-expert/incident_triage_app.py

Requisitos:
    experta
    streamlit
"""

import streamlit as st

from experta import (
    AS,
    MATCH,
    TEST,
    Fact,
    KnowledgeEngine,
    Rule,
)


# =================================================================
# PARTE 1: SISTEMA ESPECIALISTA
# =================================================================


class Aplicacao(Fact):
    """
    taxa_erros: normal | alta
    http_5xx: normal | alto
    latencia: normal | alta
    reinicios: nao | sim
    """


class Infraestrutura(Fact):
    """
    cpu: normal | alta
    memoria: normal | alta
    disco: normal | cheio
    """


class BancoDados(Fact):
    """
    latencia: normal | alta
    conexoes: normal | esgotadas
    """


class Dependencia(Fact):
    """
    status: ok | degradada | fora
    """


class Contexto(Fact):
    """
    deploy_recente: nao | sim
    trafego: normal | pico
    """


class Diagnostico(Fact):
    """
    tipo: identificador do diagnóstico
    descricao: texto explicativo
    confianca: 0.0 .. 1.0
    """


class Acao(Fact):
    """
    acao: ação operacional recomendada
    prioridade: 1 (maior) .. 3 (menor)
    confianca: 0.0 .. 1.0
    origem: diagnóstico que originou a ação
    """


class IncidentTriageExpert(KnowledgeEngine):
    """
    Base de conhecimento com 18 regras.

    R1-R9   : fatos -> diagnósticos
    R10-R18 : diagnósticos -> ações
    """

    def __init__(self):
        super().__init__()
        self.trilha = []

    def registrar(self, regra, texto):
        """Registra a trilha utilizada pelo módulo de explicação."""
        self.trilha.append((regra, texto))

    # =============================================================
    # REGRAS DE DIAGNÓSTICO
    # =============================================================

    # -------------------------------------------------------------
    # R1 - Regressão após deploy
    # -------------------------------------------------------------

    @Rule(
        Contexto(deploy_recente="sim"),
        Aplicacao(taxa_erros="alta"),
        salience=10,
    )
    def r1_regressao_deploy(self):
        self.declare(
            Diagnostico(
                tipo="regressao_deploy",
                descricao="Possível regressão introduzida pelo último deploy.",
                confianca=0.90,
            )
        )

        self.registrar(
            "R1",
            "Deploy recente combinado com aumento da taxa de erros "
            "é um forte indicador de regressão introduzida pela nova versão.",
        )

    # -------------------------------------------------------------
    # R2 - Dependência externa
    # -------------------------------------------------------------

    @Rule(
        Dependencia(status=MATCH.status),
        TEST(lambda status: status in ("degradada", "fora")),
        salience=20,
    )
    def r2_dependencia_externa(self, status):
        confianca = 0.98 if status == "fora" else 0.85

        self.declare(
            Diagnostico(
                tipo="dependencia_externa",
                descricao=(
                    f"Dependência externa está {status}."
                ),
                confianca=confianca,
            )
        )

        self.registrar(
            "R2",
            f"A dependência externa foi reportada como '{status}'. "
            "Falhas externas podem se propagar para a aplicação.",
        )

    # -------------------------------------------------------------
    # R3 - Gargalo de banco
    # -------------------------------------------------------------

    @Rule(
        Aplicacao(latencia="alta"),
        BancoDados(latencia="alta"),
    )
    def r3_gargalo_banco(self):
        self.declare(
            Diagnostico(
                tipo="gargalo_banco",
                descricao="Banco de dados apresenta sinais de gargalo.",
                confianca=0.88,
            )
        )

        self.registrar(
            "R3",
            "Latência elevada simultaneamente na aplicação e no banco "
            "sugere que o banco pode estar no caminho crítico.",
        )

    # -------------------------------------------------------------
    # R4 - Pool de conexões esgotado
    # -------------------------------------------------------------

    @Rule(
        BancoDados(conexoes="esgotadas"),
        Aplicacao(http_5xx="alto"),
        salience=15,
    )
    def r4_pool_conexoes(self):
        self.declare(
            Diagnostico(
                tipo="pool_conexoes_esgotado",
                descricao="Pool de conexões com o banco pode estar esgotado.",
                confianca=0.95,
            )
        )

        self.registrar(
            "R4",
            "Esgotamento das conexões do banco associado a HTTP 5xx "
            "é compatível com falhas de aquisição de conexão.",
        )

    # -------------------------------------------------------------
    # R5 - Saturação por pico de tráfego
    # -------------------------------------------------------------

    @Rule(
        Infraestrutura(cpu="alta"),
        Contexto(trafego="pico"),
    )
    def r5_saturacao_capacidade(self):
        self.declare(
            Diagnostico(
                tipo="saturacao_capacidade",
                descricao="Possível saturação de capacidade por pico de tráfego.",
                confianca=0.90,
            )
        )

        self.registrar(
            "R5",
            "CPU elevada durante pico de tráfego é compatível com "
            "saturação de capacidade.",
        )

    # -------------------------------------------------------------
    # R6 - Regressão de consumo de CPU
    # -------------------------------------------------------------

    @Rule(
        Infraestrutura(cpu="alta"),
        Contexto(
            trafego="normal",
            deploy_recente="sim",
        ),
    )
    def r6_regressao_cpu(self):
        self.declare(
            Diagnostico(
                tipo="regressao_cpu",
                descricao="Possível regressão de consumo de CPU.",
                confianca=0.82,
            )
        )

        self.registrar(
            "R6",
            "CPU elevada sem crescimento de tráfego, logo após um deploy, "
            "sugere regressão de desempenho.",
        )

    # -------------------------------------------------------------
    # R7 - Pressão de memória / OOM
    # -------------------------------------------------------------

    @Rule(
        Infraestrutura(memoria="alta"),
        Aplicacao(reinicios="sim"),
        salience=25,
    )
    def r7_memoria_oom(self):
        self.declare(
            Diagnostico(
                tipo="memoria_oom",
                descricao="Pressão de memória ou possível condição de OOM.",
                confianca=0.95,
            )
        )

        self.registrar(
            "R7",
            "Memória elevada combinada com reinícios da aplicação "
            "é compatível com pressão de memória ou OOM.",
        )

    # -------------------------------------------------------------
    # R8 - Esgotamento de disco
    # -------------------------------------------------------------

    @Rule(
        Infraestrutura(disco="cheio"),
        salience=30,
    )
    def r8_disco_cheio(self):
        self.declare(
            Diagnostico(
                tipo="disco_cheio",
                descricao="Espaço em disco esgotado.",
                confianca=0.99,
            )
        )

        self.registrar(
            "R8",
            "Disco cheio é uma condição crítica e pode impedir logs, "
            "arquivos temporários, persistência e funcionamento da aplicação.",
        )

    # -------------------------------------------------------------
    # R9 - Falha isolada da aplicação
    # -------------------------------------------------------------

    @Rule(
        Aplicacao(
            taxa_erros="alta",
            http_5xx="alto",
        ),
        Infraestrutura(
            cpu="normal",
            memoria="normal",
            disco="normal",
        ),
        BancoDados(
            latencia="normal",
            conexoes="normal",
        ),
        Dependencia(status="ok"),
        Contexto(deploy_recente="nao"),
    )
    def r9_falha_aplicacao(self):
        self.declare(
            Diagnostico(
                tipo="falha_aplicacao",
                descricao="Falha provavelmente localizada na aplicação.",
                confianca=0.80,
            )
        )

        self.registrar(
            "R9",
            "A aplicação apresenta erros e HTTP 5xx, enquanto infraestrutura, "
            "banco e dependências não apresentam sinais de falha.",
        )

    # =============================================================
    # REGRAS DE AÇÃO
    #
    # salience=-10:
    # prioriza a geração dos diagnósticos antes das ações.
    # =============================================================

    # -------------------------------------------------------------
    # R10 - Ação: regressão de deploy
    # -------------------------------------------------------------

    @Rule(
        AS.diag
        << Diagnostico(
            tipo="regressao_deploy",
            confianca=MATCH.c,
        ),
        salience=-10,
    )
    def r10_acao_regressao_deploy(self, diag, c):
        self.declare(
            Acao(
                acao=(
                    "Comparar métricas antes/depois do deploy e considerar "
                    "rollback ou desativação progressiva da nova versão."
                ),
                prioridade=1,
                confianca=c,
                origem="regressao_deploy",
            )
        )

        self.registrar(
            "R10",
            "ENCADEAMENTO: o diagnóstico de regressão de deploy "
            "gerado por outra regra ativou uma ação de mitigação.",
        )

    # -------------------------------------------------------------
    # R11 - Ação: dependência externa
    # -------------------------------------------------------------

    @Rule(
        Diagnostico(
            tipo="dependencia_externa",
            confianca=MATCH.c,
        ),
        salience=-10,
    )
    def r11_acao_dependencia(self, c):
        self.declare(
            Acao(
                acao=(
                    "Verificar a saúde do fornecedor, ativar fallback/failover "
                    "quando disponível e avaliar circuit breaker."
                ),
                prioridade=1,
                confianca=c,
                origem="dependencia_externa",
            )
        )

        self.registrar(
            "R11",
            "ENCADEAMENTO: diagnóstico de dependência externa "
            "gerou uma recomendação de isolamento e contingência.",
        )

    # -------------------------------------------------------------
    # R12 - Ação: gargalo de banco
    # -------------------------------------------------------------

    @Rule(
        Diagnostico(
            tipo="gargalo_banco",
            confianca=MATCH.c,
        ),
        salience=-10,
    )
    def r12_acao_banco(self, c):
        self.declare(
            Acao(
                acao=(
                    "Investigar consultas lentas, locks, índices, I/O e "
                    "capacidade do banco antes de alterar recursos."
                ),
                prioridade=2,
                confianca=c,
                origem="gargalo_banco",
            )
        )

        self.registrar(
            "R12",
            "ENCADEAMENTO: o diagnóstico de gargalo de banco "
            "ativou ações de investigação específicas.",
        )

    # -------------------------------------------------------------
    # R13 - Ação: pool de conexões
    # -------------------------------------------------------------

    @Rule(
        Diagnostico(
            tipo="pool_conexoes_esgotado",
            confianca=MATCH.c,
        ),
        salience=-10,
    )
    def r13_acao_pool(self, c):
        self.declare(
            Acao(
                acao=(
                    "Verificar vazamento de conexões, limites do banco e "
                    "configuração do pool antes de simplesmente aumentá-lo."
                ),
                prioridade=1,
                confianca=c,
                origem="pool_conexoes_esgotado",
            )
        )

        self.registrar(
            "R13",
            "ENCADEAMENTO: o esgotamento do pool gerou investigação "
            "sobre conexões, limites e possíveis leaks.",
        )

    # -------------------------------------------------------------
    # R14 - Ação: saturação de capacidade
    # -------------------------------------------------------------

    @Rule(
        Diagnostico(
            tipo="saturacao_capacidade",
            confianca=MATCH.c,
        ),
        salience=-10,
    )
    def r14_acao_capacidade(self, c):
        self.declare(
            Acao(
                acao=(
                    "Avaliar scale-out emergencial e verificar limites, "
                    "autoscaling e comportamento sob carga."
                ),
                prioridade=2,
                confianca=c,
                origem="saturacao_capacidade",
            )
        )

        self.registrar(
            "R14",
            "ENCADEAMENTO: saturação de capacidade gerou uma ação "
            "de escalabilidade e análise de carga.",
        )

    # -------------------------------------------------------------
    # R15 - Ação: regressão de CPU
    # -------------------------------------------------------------

    @Rule(
        Diagnostico(
            tipo="regressao_cpu",
            confianca=MATCH.c,
        ),
        salience=-10,
    )
    def r15_acao_regressao_cpu(self, c):
        self.declare(
            Acao(
                acao=(
                    "Comparar profiling e métricas da versão atual com a "
                    "anterior e avaliar rollback se houver correlação."
                ),
                prioridade=2,
                confianca=c,
                origem="regressao_cpu",
            )
        )

        self.registrar(
            "R15",
            "ENCADEAMENTO: o diagnóstico de regressão de CPU "
            "gerou recomendação de profiling e comparação de versões.",
        )

    # -------------------------------------------------------------
    # R16 - Ação: memória / OOM
    # -------------------------------------------------------------

    @Rule(
        Diagnostico(
            tipo="memoria_oom",
            confianca=MATCH.c,
        ),
        salience=-10,
    )
    def r16_acao_memoria(self, c):
        self.declare(
            Acao(
                acao=(
                    "Mitigar a indisponibilidade e investigar heap, GC, "
                    "memory leaks, limites do container e eventos de OOM."
                ),
                prioridade=1,
                confianca=c,
                origem="memoria_oom",
            )
        )

        self.registrar(
            "R16",
            "ENCADEAMENTO: pressão de memória/reinícios gerou "
            "investigação de OOM, heap e vazamento de memória.",
        )

    # -------------------------------------------------------------
    # R17 - Ação: disco cheio
    # -------------------------------------------------------------

    @Rule(
        Diagnostico(
            tipo="disco_cheio",
            confianca=MATCH.c,
        ),
        salience=-10,
    )
    def r17_acao_disco(self, c):
        self.declare(
            Acao(
                acao=(
                    "Liberar ou expandir espaço de forma controlada e "
                    "verificar logs, retenção, temporários e crescimento anormal."
                ),
                prioridade=1,
                confianca=c,
                origem="disco_cheio",
            )
        )

        self.registrar(
            "R17",
            "ENCADEAMENTO: o diagnóstico de disco cheio ativou "
            "uma ação operacional de alta prioridade.",
        )

    # -------------------------------------------------------------
    # R18 - Ação: falha da aplicação
    # -------------------------------------------------------------

    @Rule(
        Diagnostico(
            tipo="falha_aplicacao",
            confianca=MATCH.c,
        ),
        salience=-10,
    )
    def r18_acao_aplicacao(self, c):
        self.declare(
            Acao(
                acao=(
                    "Correlacionar logs, traces e endpoints afetados; "
                    "investigar exceções e alterações funcionais da aplicação."
                ),
                prioridade=2,
                confianca=c,
                origem="falha_aplicacao",
            )
        )

        self.registrar(
            "R18",
            "ENCADEAMENTO: após excluir sinais de infraestrutura, banco "
            "e dependências, a investigação foi direcionada à aplicação.",
        )


def consultar(
    taxa_erros,
    http_5xx,
    latencia_app,
    reinicios,
    cpu,
    memoria,
    disco,
    latencia_banco,
    conexoes_banco,
    dependencia,
    deploy_recente,
    trafego,
):
    """
    Executa um ciclo completo de inferência.
    """

    engine = IncidentTriageExpert()

    engine.reset()

    engine.declare(
        Aplicacao(
            taxa_erros=taxa_erros,
            http_5xx=http_5xx,
            latencia=latencia_app,
            reinicios=reinicios,
        )
    )

    engine.declare(
        Infraestrutura(
            cpu=cpu,
            memoria=memoria,
            disco=disco,
        )
    )

    engine.declare(
        BancoDados(
            latencia=latencia_banco,
            conexoes=conexoes_banco,
        )
    )

    engine.declare(
        Dependencia(
            status=dependencia,
        )
    )

    engine.declare(
        Contexto(
            deploy_recente=deploy_recente,
            trafego=trafego,
        )
    )

    engine.run()

    diagnosticos = [
        fact
        for fact in engine.facts.values()
        if isinstance(fact, Diagnostico)
    ]

    acoes = [
        fact
        for fact in engine.facts.values()
        if isinstance(fact, Acao)
    ]

    diagnosticos = sorted(
        diagnosticos,
        key=lambda fact: fact["confianca"],
        reverse=True,
    )

    acoes = sorted(
        acoes,
        key=lambda fact: (
            fact["prioridade"],
            -fact["confianca"],
        ),
    )

    memoria = [
        dict(fact)
        for fact in engine.facts.values()
    ]

    return diagnosticos, acoes, engine.trilha, memoria


# =================================================================
# PARTE 2: INTERFACE STREAMLIT
# =================================================================


st.set_page_config(
    page_title="Incident Triage Expert",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 Incident Triage Expert")

st.caption(
    "Sistema especialista para diagnóstico inicial de incidentes "
    "em aplicações Web. "
    "Introdução à IA - PPGI/UNIRIO."
)

st.info(
    "Este protótipo é didático. As recomendações representam regras "
    "simplificadas de triagem e não substituem análise operacional."
)


# -----------------------------------------------------------------
# Entradas
# -----------------------------------------------------------------

st.subheader("1. Sinais da aplicação")

col1, col2, col3, col4 = st.columns(4)

with col1:
    taxa_erros = st.selectbox(
        "Taxa de erros",
        ["normal", "alta"],
    )

with col2:
    http_5xx = st.selectbox(
        "HTTP 5xx",
        ["normal", "alto"],
    )

with col3:
    latencia_app = st.selectbox(
        "Latência da aplicação",
        ["normal", "alta"],
    )

with col4:
    reinicios = st.selectbox(
        "Reinícios da aplicação",
        ["nao", "sim"],
    )


st.subheader("2. Infraestrutura")

col1, col2, col3 = st.columns(3)

with col1:
    cpu = st.selectbox(
        "CPU",
        ["normal", "alta"],
    )

with col2:
    memoria = st.selectbox(
        "Memória",
        ["normal", "alta"],
    )

with col3:
    disco = st.selectbox(
        "Disco",
        ["normal", "cheio"],
    )


st.subheader("3. Banco e dependências")

col1, col2, col3 = st.columns(3)

with col1:
    latencia_banco = st.selectbox(
        "Latência do banco",
        ["normal", "alta"],
    )

with col2:
    conexoes_banco = st.selectbox(
        "Conexões do banco",
        ["normal", "esgotadas"],
    )

with col3:
    dependencia = st.selectbox(
        "Dependência externa",
        ["ok", "degradada", "fora"],
    )


st.subheader("4. Contexto")

col1, col2 = st.columns(2)

with col1:
    deploy_recente = st.radio(
        "Houve deploy recente?",
        ["nao", "sim"],
        horizontal=True,
    )

with col2:
    trafego = st.radio(
        "Tráfego",
        ["normal", "pico"],
        horizontal=True,
    )


# -----------------------------------------------------------------
# Inferência
# -----------------------------------------------------------------

if st.button(
    "Analisar incidente",
    type="primary",
    use_container_width=True,
):

    diagnosticos, acoes, trilha, memoria = consultar(
        taxa_erros=taxa_erros,
        http_5xx=http_5xx,
        latencia_app=latencia_app,
        reinicios=reinicios,
        cpu=cpu,
        memoria=memoria,
        disco=disco,
        latencia_banco=latencia_banco,
        conexoes_banco=conexoes_banco,
        dependencia=dependencia,
        deploy_recente=deploy_recente,
        trafego=trafego,
    )

    st.divider()

    # -------------------------------------------------------------
    # Diagnósticos
    # -------------------------------------------------------------

    st.subheader("Diagnósticos")

    if diagnosticos:

        for diagnostico in diagnosticos:

            st.warning(
                f"**{diagnostico['descricao']}**"
            )

            st.progress(
                diagnostico["confianca"],
                text=(
                    "Confiança heurística: "
                    f"{diagnostico['confianca']:.0%}"
                ),
            )

    else:

        st.info(
            "Nenhuma regra de diagnóstico cobre esta combinação "
            "de sinais."
        )

        st.markdown(
            "Isso demonstra uma limitação típica dos sistemas "
            "especialistas: **conhecimento fora da base não existe "
            "para o motor de inferência**."
        )

    # -------------------------------------------------------------
    # Ações
    # -------------------------------------------------------------

    st.subheader("Ações recomendadas")

    if acoes:

        for acao in acoes:

            prioridade = acao["prioridade"]

            if prioridade == 1:
                label = "P1 — alta prioridade"
            elif prioridade == 2:
                label = "P2 — prioridade moderada"
            else:
                label = "P3 — baixa prioridade"

            st.success(
                f"**{label}** — {acao['acao']}"
            )

            st.caption(
                f"Origem: {acao['origem']} | "
                f"Confiança: {acao['confianca']:.0%}"
            )

    else:

        st.write(
            "Nenhuma ação foi inferida."
        )

    # -------------------------------------------------------------
    # Explicação
    # -------------------------------------------------------------

    with st.expander(
        "Por que o sistema chegou a essas conclusões?",
        expanded=True,
    ):

        if trilha:

            for regra, texto in trilha:
                st.markdown(
                    f"**[{regra}]** {texto}"
                )

        else:

            st.markdown(
                "Nenhuma regra disparou."
            )

    # -------------------------------------------------------------
    # Memória de trabalho
    # -------------------------------------------------------------

    with st.expander(
        "Memória de trabalho ao final da inferência"
    ):
        st.json(memoria)
