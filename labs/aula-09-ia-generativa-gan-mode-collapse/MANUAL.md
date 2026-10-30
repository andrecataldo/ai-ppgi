# Manual técnico - GAN Mode Collapse Lab

Este documento explica a implementação e as decisões experimentais. Para instalar e executar, use o [`README.md`](README.md); para evidências e conclusões, consulte o [`RELATORIO_LAB.md`](RELATORIO_LAB.md).

## 1. Delimitação científica

O objetivo é entender como um modelo adversarial **gera amostras** e por que isso não assegura **cobertura adequada da distribuição**. Goodfellow et al. (2014) fundamenta a GAN original; WGAN-GP (Gulrajani et al., 2017) é uma extensão posterior. Não tentamos reproduzir MNIST, CIFAR-10, as arquiteturas históricas nem imagens de alta resolução.

A demonstração da aula mostrou oito gaussianas e um exemplo de colapso. Aqui existem **duas referências amostradas artificialmente**, com oito ou dois modos (E0), e **duas redes de fato treinadas** (E1 e E2). Referências artificiais não são resultados de treinamento.

## 2. Dados sintéticos

A distribuição de referência é uma **mistura uniforme de oito gaussianas bidimensionais**:

```text
p_data(x) = (1/8) · Σ_{k=0}^{7} N(x; μ_k, 0,1² · I)
μ_k       = 2 · (cos(2πk/8), sin(2πk/8))
```

Nesta notação, `N` é a distribuição normal, `μ_k` é o centro do modo `k` e `I` é a matriz identidade. As fórmulas aparecem em blocos de texto para manter compatibilidade com visualizadores Markdown que **não** suportam LaTeX.

- Cada amostra é um ponto `x = (x₁, x₂)`; não são imagens.
- Os oito modos têm a mesma massa de probabilidade de origem.
- `sample_real()` gera pontos sob demanda, sem leitura de dataset externo.
- `reference_samples()` cria pontos sob distribuições escolhidas (8 ou 2 modos) para testar a interpretação das métricas.

A vantagem didática é conhecer a localização dos oito centros verdadeiros. Essa facilidade não costuma existir em dados de produção.

## 3. Redes neurais e implementação

| Componente | Arquitetura | Papel |
|---|---|---|
| Gerador `Generator` | 2 → 64 → 64 → 2; ReLU nas camadas ocultas | Mapeia `z ~ N(0, I)` para ponto 2D |
| `Discriminator` | 2 → 64 → 64 → 1; LeakyReLU nas ocultas | Na Vanilla: logit; na WGAN-GP: escore de crítico |

A saída do discriminador/crítico **não** possui sigmoid explícito. Na Vanilla GAN, o logit é passado a funções `softplus` para estabilidade numérica; na WGAN-GP, o crítico fornece um escore **não probabilístico**.

### 3.1. Vanilla GAN - E1

O artigo original formula um jogo **minimax** entre o gerador (`G`) e o discriminador (`D`):

```text
min_G max_D V(D,G)
  = E_{x ~ p_data}[ log D(x) ]
  + E_{z ~ p_z}[ log(1 - D(G(z))) ]
```

Na implementação, `d(x)` é o **logit** (saída antes da sigmoid). Ambas as perdas abaixo são **minimizadas**:

```text
L_D = E_x[ softplus(-d(x)) ] + E_z[ softplus(d(G(z))) ]
L_G = E_z[ softplus(-d(G(z))) ]
```

Aqui, `E` representa esperança matemática; `softplus(t) = log(1 + exp(t))`. A probabilidade utilizada na formulação original é `D(x) = sigmoid(d(x))`.

A perda do gerador é a variante **não saturante**, discutida por Goodfellow et al. em 2014. Cada passo executa **uma** atualização do discriminador e **uma** do gerador; Adam com `lr=2e-4` e betas `(0.5, 0.999)`.

### 3.2. WGAN-GP - E2

A WGAN-GP substitui o discriminador binário por um **crítico de escore contínuo**, com uma penalização na norma do gradiente do crítico em pontos interpolados entre dados reais e sintéticos:

```text
L_D = E_z[D(G(z))] - E_x[D(x)]
      + λ · E_x̂[(||∇_x̂ D(x̂)||₂ - 1)²]

L_G = -E_z[D(G(z))]

λ = 10
x̂ = α · x + (1 - α) · G(z), com α ~ Uniforme(0, 1)
```

O símbolo `||·||₂` indica norma euclidiana. O termo de penalização incentiva gradientes de norma próxima a 1 nos pontos interpolados. O crítico **não** estima uma probabilidade de real/falso.

Cada passo executa **três** atualizações do crítico e **uma** do gerador; Adam com `lr=1e-4` e betas `(0.5, 0.9)`. As diferenças de função objetivo, taxa de aprendizagem e número de atualizações impedem interpretar esta comparação como ablação causal perfeita.

### 3.3. Procedimento de treino

1. Fixar a semente e inicializar modelos e otimizadores.
2. Amostrar `batch_size=256` pontos reais e ruído, a cada atualização do discriminador/crítico.
3. Atualizar discriminador/crítico conforme a função de perda escolhida.
4. Amostrar novo ruído e atualizar o gerador.
5. A cada `eval_every=50` passos (além do primeiro e do último), avaliar 2.048 amostras geradas com o mesmo ruído de avaliação **dentro de cada seed**.
6. Salvar histórico, amostras, gráficos e resumo de métricas.

A função `run_experiment()` reutiliza `summary.json` quando encontra configuração idêntica; `--force` ignora esse cache. O dashboard `app.py` **não executa treinamento**; apenas lê dados salvos em `results/`.

## 4. Métricas: definições e interpretações

Considere `N` pontos gerados, centros `μ_k`, raio de validade `r = 0,45` e uma atribuição ao **centro mais próximo**. Uma amostra é válida quando a distância mínima é menor ou igual a `r`.

| Métrica | Cálculo | Risco de interpretação |
|---|---|---|
| Fração válida | `N_válidas / N` | Não descreve quantos modos foram cobertos |
| Cobertura | Número de centros com `N_k / N ≥ 0,01`, onde `N_k` conta apenas pontos válidos atribuídos ao centro `k` | Um modo pouco representado pode não atingir o limiar; 8/8 não implica qualidade |
| Ocupação condicional | `p_k = N_k / N_válidas` | Ignora todas as amostras inválidas |
| JS (bits) | `JSD(p || u)`, onde `u_k = 1/8` e `p` é a ocupação condicional | Pode ser baixa mesmo com poucos pontos válidos |

A JS é calculada em bits (`log2`) e mede diferença entre **frequências de modos válidos**, não entre as distribuições contínuas originais. Quando não há amostras válidas, a implementação retorna `JS=1` como valor de **sentinela didático**, não como uma estimativa estatística convencional.

**Exemplo prático:** a execução Vanilla seed 1 alcançou 8/8 e JS≈0,018, mas apenas 35,4% das amostras foram válidas. Assim, cerca de 64,6% ficaram fora do raio de validade, apesar da cobertura ampla.

**Não confunda:** “% fora da área”, no título de `amostras.png`, significa pontos fora do quadro de zoom ±3,4, **não** pontos inválidos conforme o raio de 0,45.

## 5. Como reconhecer os fenômenos

- **Possível mode collapse / mode dropping:** poucos modos cobertos, embora a distribuição real tenha oito.
- **Baixa fidelidade ou massificação entre modos:** muitos pontos longe de todos os centros, mesmo quando vários modos estão cobertos.
- **Desequilíbrio:** ocupação condicional concentrada em poucos centros (JS elevada).
- **Resultado enganoso:** boa cobertura e JS baixa com fração válida baixa.

As métricas são **proxies didáticas**, não uma identificação automática e inequívoca do mecanismo causal de colapso. Nem o desempenho visual nem uma só métrica comprovam convergência.

## 6. Reprodutibilidade e validação

```bash
python -m unittest discover -s tests -v
python lab.py --steps 1500 --seeds 0 1 2
streamlit run app.py
```

Os nove testes automatizados cobrem geração e reprodutibilidade dos dados, referências completas e colapsadas, parâmetros inválidos, propagação de gradientes e penalização de gradiente finita. **Eles não constituem teste de qualidade da convergência.**

Artefatos por execução: `summary.json`, `history.csv`, `samples.npz`, `amostras.png`, `amostras_full.png` e `progresso.png`; consolidação em `comparison.csv/json`. Os arquivos completos são regeneráveis em `results/`. O recorte versionável está em `evidencias/`.

**Não compare numericamente perdas (`g_loss`, `d_loss`) de Vanilla GAN e WGAN-GP:** seus objetivos e escalas são distintos. Compare suas métricas comuns e explicite a diferença de custo computacional.

## Referências metodológicas

Goodfellow et al. (2014); Bond-Taylor et al. (2022); Arjovsky et al. (2017); Gulrajani et al. (2017). Referências completas no [`README.md`](README.md).
