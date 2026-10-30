# GAN Mode Collapse Lab - IA Generativa (Aula 09)

**Disciplina:** Inteligência Artificial · PPGI/UNIRIO · 2026.2  
**Aula de referência:** 20/10/2026 · **Tema:** modelos generativos adversariais

> **Pergunta-guia:** uma GAN que gera amostras próximas de diferentes modos aprendeu a distribuição completa dos dados?

Laboratório com duas redes generativas (Vanilla GAN e WGAN-GP), uma mistura de **oito gaussianas bidimensionais** e métricas para investigar **cobertura, aderência e equilíbrio dos modos**. Foi inspirado na demonstração de *mode collapse* da aula, mas estende a atividade com treinamento real, comparação entre três sementes e análise de limitações. Não reproduz o treinamento de imagens ou os valores numéricos do artigo seminal.

## Por que este laboratório?

O artigo de Goodfellow et al. (2014) estabelece o aprendizado adversarial; a revisão de Bond-Taylor et al. (2022) discute os compromissos e a avaliação de modelos generativos. A WGAN-GP (Gulrajani et al., 2017) entra como **extensão externa às leituras obrigatórias**: é uma alternativa para investigar estabilidade, **não** uma solução garantida para colapso. O trabalho de difusão latente (Rombach et al., 2022) é contexto, **não implementado** neste laboratório.

## O que é executado

| ID | Condição | Finalidade |
|---|---|---|
| E0 | Distribuições artificiais com 8 modos e com 2 modos | Referências para conferir as métricas; **não foram geradas por uma GAN treinada** |
| E1 | Vanilla GAN com perda não saturante | Baseline de treinamento adversarial |
| E2 | WGAN-GP, com crítico e penalização de gradiente | Examinar comportamento de outra estratégia de treinamento |

O dataset é uma mistura uniforme de oito gaussianas 2D, com centros em círculo de raio 2 e desvio-padrão 0,1. O código gera os dados, sem downloads. Modelos compactos em PyTorch tornam a execução viável em CPU.

## Instalação - Ubuntu / Linux

A partir da raiz de `ai-ppgi`:

```bash
cd labs/aula-09-ia-generativa-gan-mode-collapse
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Recomenda-se Python 3.11+. Caso necessite de PyTorch específico para CPU, consulte as instruções da instalação oficial e instale a distribuição correspondente antes de `requirements.txt`. O dispositivo padrão é CPU; `--device cuda` é opcional quando disponível.

## Executar

```bash
# Verificação rápida: seed 0, 250 passos por modelo (não avalia convergência)
python lab.py --quick

# Experimento principal: 1.500 passos do gerador, três seeds, duas arquiteturas
python lab.py --steps 1500 --seeds 0 1 2

# Refazer treinos mesmo quando já existirem resultados com a mesma configuração
python lab.py --steps 1500 --seeds 0 1 2 --force

# Somente um dos modelos
python lab.py --steps 1500 --seeds 1 --experiments vanilla
python lab.py --steps 1500 --seeds 1 --experiments wgan_gp

# Dashboard para explorar os resultados salvos (não executa treinamento)
streamlit run app.py

# Testes automatizados
python -m unittest discover -s tests -v
```

O programa salva saídas em `results/`: `summary.json`, `history.csv`, `samples.npz`, `amostras.png`, `amostras_full.png` e `progresso.png`, além da consolidação `comparison.csv/json`. `results/` está no `.gitignore` porque pode ser regenerado. **Figuras e números representativos, destinados ao repositório, estão em [`evidencias/`](evidencias/).**

## Como ler as métricas

| Indicador | Definição no experimento | O que NÃO permite concluir sozinho |
|---|---|---|
| **Cobertura** | Número de clusters com ≥ 1% de **todas as amostras** a até 0,45 de um centro | Que a distribuição gerada seja fiel aos dados |
| **Fração válida** | Proporção de pontos gerados a até 0,45 de algum centro real | Que os oito modos estejam representados |
| **JS (bits)** | Divergência entre a ocupação dos **pontos válidos** e a mistura uniforme de oito modos | Que pontos inválidos sejam raros; eles são excluídos dessa JS |

Por exemplo, uma rede pode ter cobertura **8/8** e uma fração válida inferior a 40%. Isso não é contradição: ela alcança os modos, mas gera muitos pontos nas regiões intermediárias. A marcação “fora da área” nos gráficos é somente o percentual fora do **zoom visual**, e não a fração de amostras inválidas.

## Resultados resumidos - três sementes

Resultados registrados para **1.500 atualizações do gerador**, `batch_size=256`, 2.048 amostras de avaliação, sementes 0, 1 e 2:

| Métrica (média ± desvio-padrão amostral) | Vanilla GAN | WGAN-GP |
|---|---:|---:|
| Modos cobertos (de 8) | 6,33 ± 2,89 | 5,00 ± 2,00 |
| Fração válida | 22,6% ± 11,4 p.p. | 14,0% ± 3,2 p.p. |
| JS entre pontos válidos (bits) | 0,157 ± 0,247 | 0,205 ± 0,179 |

**As duas arquiteturas apresentam variação considerável entre sementes.** Na seed 1, a Vanilla cobre 8/8 e a WGAN-GP cobre 3/8; na seed 2, os resultados se invertem (3/8 e 7/8). Esses dados são **descritivos**, não um teste de hipótese ou evidência de superioridade universal. O custo não é equiparado: a WGAN-GP usa três atualizações do crítico por atualização do gerador, contra uma atualização do discriminador na Vanilla GAN, além de função objetivo e hiperparâmetros diferentes.

→ **Análise científica completa:** [`RELATORIO_LAB.md`](RELATORIO_LAB.md).  
→ **Arquiteturas, funções de perda e limitações das métricas:** [`MANUAL.md`](MANUAL.md).

## Estrutura

```text
labs/aula-09-ia-generativa-gan-mode-collapse/
├── README.md                 # Escopo, instalação, execução e síntese
├── MANUAL.md                 # Fundamentos, implementação e interpretação
├── RELATORIO_LAB.md          # Problema, hipótese, método, resultados e discussão
├── common.py                 # Dataset, redes e métricas
├── lab.py                    # Treino, avaliação e gráficos
├── app.py                    # Interface de leitura Streamlit
├── requirements.txt
├── .gitignore
├── tests/
│   └── test_common.py
├── evidencias/               # Pequeno conjunto versionável de resultados
│   ├── metricas_1500.csv
│   ├── vanilla_seed1_amostras.png
│   ├── vanilla_seed2_amostras.png
│   └── wgan_gp_seed2_amostras.png
└── results/                  # Arquivos completos, regeneráveis (ignorado no Git)
```

## Referências

- GOODFELLOW, I. et al. *Generative Adversarial Nets*. NeurIPS, 2014.
- BOND-TAYLOR, S. et al. *Deep Generative Modelling: A Comparative Review of VAEs, GANs, Normalizing Flows, Energy-Based and Autoregressive Models*. IEEE TPAMI, 2022.
- ROMBACH, R. et al. *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR, 2022 (contexto; não implementado).
- ARJOVSKY, M.; CHINTALA, S.; BOTTOU, L. *Wasserstein Generative Adversarial Networks*. ICML, 2017 (referência adicional).
- GULRAJANI, I. et al. *Improved Training of Wasserstein GANs*. NeurIPS, 2017 (referência adicional para WGAN-GP).
