# Aula 14 - LLMs e In-Context Learning

**Disciplina:** Inteligência Artificial  
**Programa:** PPGI - UNIRIO  
**Data:** 24/11/2026

## Objetivo

Demonstrar, de maneira transparente e reproduzível, como diferentes exemplos fornecidos no contexto podem influenciar a classificação produzida por um simulador didático.

A aplicação permite explorar cenários zero-shot, one-shot e few-shot, além de investigar o efeito da inversão de rótulos, da ordem das demonstrações e dos pesos atribuídos às características.

## Limitações

**Este projeto não implementa um LLM real.**

O simulador utiliza:

- Características definidas manualmente: prefixo e sufixo.
- Pontuações explícitas de similaridade.
- Normalização por softmax.
- Agregação dos pesos por classe.

Não possui pré-treinamento, embeddings aprendidos ou camadas Transformer.

Os pesos normalizados apresentados pela aplicação não representam probabilidades calibradas de acerto.

## Estrutura

```text
src/
  app.py
  demo_terminal.py
  motor.py
  requirements.txt

tests/
  test_motor.py
```

## Como executar

### 1. Criar o ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar as dependências

```bash
python -m pip install -r src/requirements.txt
```

### 3. Executar a interface visual

```bash
python -m streamlit run src/app.py
```

### 4. Executar a demonstração no terminal

```bash
python src/demo_terminal.py
```

### 5. Executar os testes

```bash
python -m unittest discover -s tests -v
```

## Experimentos disponíveis

- K=0: ausência de demonstrações.
- K=1: uma demonstração.
- K=2 e K=4: múltiplas demonstrações.
- Inversão de rótulos.
- Inversão da ordem dos exemplos.
- Alteração dos pesos de prefixo, sufixo e posição.

## Fundamentação acadêmica

- Brown, T. et al. (2020). *Language Models are Few-Shot Learners*. NeurIPS.
- Bender, E. M. et al. (2021). *On the Dangers of Stochastic Parrots*. FAccT.
- Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models*.

O estudo de Brown fundamenta a discussão sobre o comportamento few-shot. Bender contribui com a análise crítica das capacidades atribuídas aos modelos de linguagem. Hoffmann discute escolhas de pré-treinamento.

O simulador não reproduz os experimentos nem os mecanismos internos investigados nesses trabalhos.