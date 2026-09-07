# Sommelier Digital

Demo de sistema especialista utilizado na disciplina de Inteligência Artificial
do PPGI/UNIRIO.

## Executar

A partir da raiz do projeto:

```bash
source .venv/bin/activate
streamlit run apps/sommelier-digital/sommelier_app.py
```

A aplicação ficará disponível em:

```text
http://localhost:8501
```

## Conceitos demonstrados

- fatos;
- regras de produção;
- encadeamento progressivo;
- memória de trabalho;
- saliência e prioridade;
- módulo de explicação;
- ausência de cobertura da base de conhecimento.

## Roteiro de demonstração

### 1. Regra simples

Configuração:

- Proteína: peixe
- Preparo: grelhado
- Álcool: sim

Resultado:

- R1 é disparada;
- recomendação: Sauvignon Blanc.

### 2. Encadeamento progressivo

Configuração:

- Proteína: carne vermelha
- Preparo: assado
- Álcool: sim
- Ocasião: celebração

Fluxo:

```text
Fatos iniciais
    ↓
R3
    ↓
Recomendacao(Malbec)
    ↓
R9
    ↓
Recomendação refinada para celebração
```

Esse cenário demonstra que uma regra pode gerar um novo fato que ativa outra regra.

### 3. Prioridade por saliência

Configuração:

- qualquer proteína;
- picância alta;
- álcool: sim.

A regra R7 possui:

```python
salience=10
```

e recebe prioridade sobre regras com saliência padrão.

### 4. Restrição do usuário

Configuração:

- Álcool: não.

A regra R8 possui:

```python
salience=20
```

e representa uma restrição de maior prioridade.

> Observação: saliência determina a prioridade de disparo das regras. Ela não implica, por si só, exclusividade de execução.

### 5. Ausência de cobertura

Configuração:

- Proteína: carne vermelha
- Preparo: frito

Nenhuma regra da base cobre essa combinação.

O exemplo evidencia uma fragilidade típica dos sistemas especialistas:

> conhecimento que não foi representado na base não pode ser inferido pelo sistema.

## Arquitetura

```text
Interface Streamlit
       ↓
Fatos do usuário
       ↓
Memória de trabalho
       ↓
Motor de inferência
       ↓
Regras Experta
       ↓
Recomendação
       ↓
Módulo de explicação
```

## Motor de inferência

A aplicação utiliza a biblioteca Python `experta`.

Principais elementos utilizados:

- `Fact`
- `KnowledgeEngine`
- `Rule`
- `DefFacts`
- `declare`
- `retract`
- `MATCH`
- `TEST`
- `salience`

O mecanismo utiliza encadeamento progressivo (*forward chaining*).

## Compatibilidade

O `experta==1.9.4` declara formalmente dependência de:

```text
frozendict==1.2
```

Em versões modernas do Python, o material da disciplina utiliza o seguinte workaround:

```bash
pip install experta streamlit
pip install --upgrade frozendict
```

O ambiente deste projeto foi validado com Python 3.11 e uma versão atualizada de `frozendict`.

Essa configuração pode gerar um aviso de conflito de dependência no `pip`, pois o Experta mantém a restrição antiga em seus metadados, embora o demo tenha sido validado funcionalmente neste ambiente.

## Origem

Material utilizado como demonstração na disciplina de Inteligência Artificial do PPGI/UNIRIO - 2026.2.
