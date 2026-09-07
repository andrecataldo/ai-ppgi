# AI PPGI

Laboratório pessoal da disciplina de Inteligência Artificial
do PPGI/UNIRIO - 2026.2.

## Objetivo

Reunir implementações, experimentos, aplicações e estudos
desenvolvidos ao longo da disciplina de Inteligência Artificial.

## Estrutura

- `apps/` - aplicações demonstráveis
- `labs/` - experimentos relacionados às aulas
- `notebooks/` - análises e experimentos Jupyter
- `expert-systems/` - estudos de motores de inferência e regras
- `scripts/` - utilitários
- `src/` - código Python reutilizável
- `tests/` - testes automatizados
- `docs/` - documentação técnica e acadêmica
- `data/` - dados utilizados nos experimentos

## Conteúdo

### Sistemas Especialistas

- Sommelier Digital
- Expert SINTA
- Experta
- CLIPS
- Prolog
- Incident Triage Expert

### Busca e resolução de problemas

- Breadth-First Search
- Depth-First Search
- Uniform Cost Search
- Greedy Best-First Search
- A*
- Tree of Thoughts

## Implementações

Este sistema especialista possui duas implementações da mesma base de conhecimento.

### Expert SINTA

Implementação principal da atividade da disciplina.

Arquivo:

```text
expert-sinta/incident-triage.bcm
```

### Python + Experta + Streamlit

Implementação complementar utilizando:

```text
incident_triage_app.py
```

Essa versão oferece uma interface interativa e permite visualizar com maior clareza diagnósticos, ações inferidas e a trilha de regras disparadas.

As duas versões utilizam o mesmo domínio e procuram representar a mesma base de conhecimento.

## Status das implementações

### Expert SINTA

Status: **concluída e validada**.

A base está disponível em:

```text
expert-sinta/incident-triage.bcm
```

A implementação possui:

```text
14 variáveis
12 perguntas de entrada
2 variáveis-objetivo
18 regras conceituais
19 regras físicas no Expert SINTA
```

A diferença entre o número de regras ocorre porque a regra conceitual de dependência externa foi desdobrada em duas regras no Expert SINTA para preservar fatores de confiança distintos.

Foram validados cenários de:

- regressão após deploy;
- saturação de capacidade por pico de tráfego;
- falha da aplicação.

Os testes confirmaram também a propagação dos fatores de confiança entre diagnóstico e ação.

### Python + Experta + Streamlit

Status: **concluída e validada**.

A implementação complementar está disponível em:

```text
incident_triage_app.py
```

Ela utiliza o mesmo domínio e a mesma estrutura conceitual da base de conhecimento.

## Disciplina

PPGI / UNIRIO  
Inteligência Artificial - 2026.2  
Prof. Ana Cristina Bicharra Garcia