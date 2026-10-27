"""Interface Streamlit V4 do Deep Learning Ablation Lab.

A V2 preserva os checkpoints já treinados e acrescenta:
- leitura explícita de execução QUICK;
- nomenclatura amigável e eixos em porcentagem;
- confiança Top-1 e margem Top-1/Top-2;
- comparação dos filtros antes/depois do treinamento;
- Purity@k por consulta e avaliação amostral da qualidade da representação;
- compatibilidade com dois esquemas de nomes de experimentos.
"""
import glob
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import streamlit as st
import torch
from PIL import Image

from common import (
    CLASSES,
    RESULTS,
    build_model,
    get_device,
    load_checkpoint,
    load_cifar10_cpu,
    normalize,
)
from lab import fig_activation, fig_architecture, fig_optimizers, fig_regularization, load

st.set_page_config(page_title="Deep Learning Ablation Lab", page_icon="🧠", layout="wide")
dev = get_device()

# Compatibilidade entre a versão que usa reg_* / opt_adam e a que usa cnn_*.
ALIASES = {
    "mlp_relu": ["mlp_relu"],
    "cnn_relu": ["cnn_relu", "reg_base"],
    "cnn_tanh": ["cnn_tanh"],
    "reg_base": ["reg_base", "cnn_relu"],
    "reg_aug": ["reg_aug", "cnn_aug"],
    "reg_dropout": ["reg_dropout", "cnn_dropout"],
    "reg_all": ["reg_all", "cnn_full"],
    "opt_adam": ["opt_adam", "cnn_adam"],
}

FRIENDLY = {
    "mlp_relu": "MLP + ReLU",
    "cnn_relu": "CNN + ReLU",
    "cnn_tanh": "CNN + tanh",
    "reg_base": "Baseline + weight decay",
    "reg_aug": "+ Data augmentation",
    "reg_dropout": "+ Dropout",
    "reg_all": "+ Augmentation + dropout",
    "opt_adam": "Adam + regularização",
    "cnn_aug": "+ Data augmentation",
    "cnn_dropout": "+ Dropout",
    "cnn_full": "+ Augmentation + dropout",
    "cnn_adam": "Adam + regularização",
}


def first_existing(canonical):
    """Resolve um nome canônico para um resultado existente."""
    for name in ALIASES.get(canonical, [canonical]):
        if os.path.exists(f"{RESULTS}/{name}.json") or os.path.exists(f"{RESULTS}/{name}.pt"):
            return name
    return None


def label(name):
    return FRIENDLY.get(name, name)


def result_metrics(canonical):
    """Retorna métricas comparáveis para um experimento canônico existente."""
    name = first_existing(canonical)
    if not name:
        return None
    h = load(name)
    if not h:
        return None
    train_acc = 1 - h["train_err"][-1]
    test_acc = 1 - h["test_err"][-1]
    return {
        "canonical": canonical,
        "name": name,
        "history": h,
        "best_test": 1 - min(h["test_err"]),
        "final_train": train_acc,
        "final_test": test_acc,
        "gap": train_acc - test_acc,
        "time": sum(h["seconds"]),
        "config": h.get("config", {}),
    }


def config_signature(cfg):
    return (
        cfg.get("epochs"),
        cfg.get("train_size"),
        cfg.get("test_size"),
        cfg.get("seed", 0),
    )


@st.cache_resource
def data():
    xtr, ytr, xte, yte = load_cifar10_cpu()
    return xtr[:20000], ytr[:20000], xte, yte


@st.cache_resource
def model(name):
    return load_checkpoint(name, dev)[0]


@st.cache_resource
def train_features(name):
    """Features das primeiras 20 mil imagens de treino."""
    xtr = data()[0]
    m = model(name)
    chunks = []
    with torch.no_grad():
        for i in range(0, len(xtr), 500):
            _, h = m(normalize(xtr[i:i + 500].to(dev)), return_features=True)
            chunks.append(h.cpu())
    return torch.cat(chunks)


def enlarge(x, side=256):
    img = Image.fromarray(x.permute(1, 2, 0).numpy())
    return img.resize((side, side), Image.Resampling.NEAREST)


def percent_axis(fig):
    if fig is None:
        return None
    for ax in fig.axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    return fig


def filter_grid(weights, title):
    """Visualiza 64 filtros RGB da primeira convolução."""
    w = weights.detach().cpu().numpy().transpose(0, 2, 3, 1)
    w = (w - w.min((1, 2, 3), keepdims=True)) / (
        np.ptp(w, (1, 2, 3), keepdims=True) + 1e-8
    )
    fig, axs = plt.subplots(8, 8, figsize=(6.4, 6.4))
    for ax, filt in zip(axs.flat, w):
        ax.imshow(filt, interpolation="nearest")
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def recreate_initial_model(cfg):
    """Recria a inicialização determinística usada antes do treino."""
    seed = int(cfg.get("seed", 0))
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        m = build_model(cfg.get("model", "cnn"), cfg["activation"], cfg["dropout"])
    return m.eval()


def feature_batch(m, x, batch=250):
    chunks = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            _, h = m(normalize(x[i:i + batch].to(dev)), return_features=True)
            chunks.append(h.cpu())
    return torch.cat(chunks)


@st.cache_data(show_spinner=False)
def representation_quality(name, checkpoint_mtime, n_ref=1000, n_query=30, k=5):
    """Compara Purity@k no espaço aprendido e no espaço bruto de pixels.

    Usa uma amostra pequena para manter a análise viável em CPU.
    """
    del checkpoint_mtime  # entra na chave do cache para invalidar após novo treino
    xtr, ytr, xte, yte = data()
    xref, yref = xtr[:n_ref], ytr[:n_ref]
    xq, yq = xte[:n_query], yte[:n_query]
    m = model(name)

    ref_feat = feature_batch(m, xref)
    q_feat = feature_batch(m, xq)
    d_feat = torch.cdist(q_feat, ref_feat)
    nn_feat = d_feat.topk(k, largest=False).indices
    purity_feat = (yref[nn_feat] == yq[:, None]).float().mean().item()

    # Pixel-space: trabalha em blocos para controlar memória.
    ref_pix = xref.flatten(1).float()
    pix_hits = 0.0
    for i in range(0, n_query, 10):
        q_pix = xq[i:i + 10].flatten(1).float()
        d_pix = torch.cdist(q_pix, ref_pix)
        nn_pix = d_pix.topk(k, largest=False).indices
        pix_hits += (yref[nn_pix] == yq[i:i + 10, None]).float().sum().item()
    purity_pix = pix_hits / (n_query * k)

    return {
        "learned": purity_feat,
        "pixels": purity_pix,
        "n_ref": n_ref,
        "n_query": n_query,
        "k": k,
    }


st.title("🧠 Deep Learning Ablation Lab")
st.caption("CIFAR-10 · representação aprendida · ativação · regularização · otimização")
st.markdown(
    "**Pergunta-guia:** o salto do Deep Learning vem de uma técnica isolada ou da combinação "
    "entre arquitetura, ativação, regularização, dados e otimização?"
)

models = sorted(os.path.basename(p)[:-3] for p in glob.glob(f"{RESULTS}/*.pt"))
selected = None
if models:
    preferred = first_existing("cnn_tanh") or first_existing("opt_adam") or first_existing("cnn_relu") or models[0]
    selected = st.sidebar.selectbox(
        "Modelo treinado",
        models,
        index=models.index(preferred),
        format_func=label,
    )
else:
    st.sidebar.warning("Nenhum checkpoint encontrado. Rode `python lab.py --quick` ou `python lab.py`.")

_, _, xte, yte = data()
if "idx" not in st.session_state:
    st.session_state.idx = 0
if st.sidebar.button("Sortear imagem CIFAR-10"):
    st.session_state.idx = int(np.random.randint(len(xte)))
st.sidebar.number_input("Índice", 0, len(xte) - 1, key="idx")
uploaded = st.sidebar.file_uploader("Ou envie uma imagem", type=["png", "jpg", "jpeg"])

if uploaded:
    im = Image.open(uploaded).convert("RGB")
    scale = 32 / min(im.size)
    im = im.resize(
        (max(32, round(im.width * scale)), max(32, round(im.height * scale))),
        Image.Resampling.BILINEAR,
    )
    left, top = (im.width - 32) // 2, (im.height - 32) // 2
    im = im.crop((left, top, left + 32, top + 32))
    img = torch.from_numpy(np.array(im)).permute(2, 0, 1).contiguous()
    truth = None
else:
    img = xte[st.session_state.idx]
    truth = int(yte[st.session_state.idx])

st.sidebar.image(
    enlarge(img, 160),
    caption=(f"Real: {CLASSES[truth]}" if truth is not None else "Imagem enviada"),
)

aba1, aba2, aba3, aba4 = st.tabs(["Ablations", "Classificação", "Filtros", "Representações"])

with aba1:
    st.header("Experimentos controlados")

    # Distingue smoke test de execução experimental reduzida e detecta mistura de configs.
    visible_hist = []
    for n in models:
        h = load(n)
        if h:
            visible_hist.append(h)

    signatures = {config_signature(h.get("config", {})) for h in visible_hist}
    smoke_test = any(h.get("config", {}).get("epochs", 99) <= 2 for h in visible_hist)
    reduced_run = any(
        h.get("config", {}).get("train_size") is not None
        or h.get("config", {}).get("test_size") is not None
        for h in visible_hist
    )

    if len(signatures) > 1:
        st.warning(
            "**Resultados com configurações diferentes detectados.** Alguns checkpoints foram "
            "treinados com números de épocas, tamanhos de amostra ou seeds diferentes. Compare as "
            "ablação apenas quando as configurações forem equivalentes."
        )
    elif smoke_test:
        st.warning(
            "**Smoke test / QUICK.** Os resultados usam apenas 2 épocas e servem para validar o "
            "pipeline. Não devem ser usados para concluir superioridade entre técnicas."
        )
    elif reduced_run and visible_hist:
        cfg = visible_hist[0].get("config", {})
        st.info(
            f"**Execução experimental reduzida.** Resultados com "
            f"{cfg.get('train_size') or 'dataset completo'} imagens de treino, "
            f"{cfg.get('test_size') or 'dataset completo'} de teste, "
            f"{cfg.get('epochs', '?')} épocas e seed {cfg.get('seed', 0)}. "
            "Adequada para comparação didática controlada, mas não para benchmark de estado da arte."
        )

    items = [
        (fig_architecture, "**1. MLP × CNN:** compara uma rede totalmente conectada com uma arquitetura que incorpora localidade espacial e compartilhamento de pesos."),
        (fig_activation, "**2. ReLU × tanh:** compara a dinâmica inicial de duas funções de ativação na mesma família arquitetural."),
        (fig_regularization, "**3. Regularização:** mantém a receita-base e acrescenta augmentation, dropout ou ambos."),
        (fig_optimizers, "**4. SGD × Adam:** compara duas estratégias de otimização sob uma configuração semelhante de arquitetura/regularização."),
    ]
    missing = False
    for fn, description in items:
        st.markdown(description)
        fig = percent_axis(fn())
        if fig is None:
            missing = True
        else:
            st.pyplot(fig)
            plt.close(fig)
    if missing:
        st.warning("Alguns gráficos dependem de resultados ainda não treinados.")

    canonical_order = [
        "mlp_relu", "cnn_relu", "cnn_tanh", "reg_base",
        "reg_aug", "reg_dropout", "reg_all", "opt_adam",
    ]
    rows = []
    seen = set()
    metrics = {}
    for canonical in canonical_order:
        m = result_metrics(canonical)
        if not m or m["name"] in seen:
            continue
        seen.add(m["name"])
        metrics[canonical] = m
        rows.append({
            "Experimento": label(canonical),
            "Melhor acurácia de teste": f"{m['best_test']:.1%}",
            "Acurácia treino final": f"{m['final_train']:.1%}",
            "Acurácia teste final": f"{m['final_test']:.1%}",
            "Gap de generalização": f"{m['gap']:+.1%}",
            "Tempo (s)": f"{m['time']:.1f}",
        })
    if rows:
        st.subheader("Placar das ablações")
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # Parâmetros: baseline MLP vs CNN quando ambos estão disponíveis.
    mlp_name, cnn_name = first_existing("mlp_relu"), first_existing("cnn_relu")
    if mlp_name and cnn_name:
        mlp_params = sum(p.numel() for p in model(mlp_name).parameters())
        cnn_params = sum(p.numel() for p in model(cnn_name).parameters())
        c1, c2 = st.columns(2)
        c1.metric("Parâmetros · MLP", f"{mlp_params:,}".replace(",", "."))
        c2.metric("Parâmetros · CNN", f"{cnn_params:,}".replace(",", "."))
        st.caption(
            "A MLP possui mais parâmetros que a CNN. A comparação, portanto, ajuda a discutir "
            "estrutura arquitetural — não apenas capacidade bruta."
        )

    st.divider()
    st.subheader("Síntese dos experimentos")

    required = [
        "mlp_relu", "cnn_relu", "cnn_tanh", "reg_base",
        "reg_aug", "reg_dropout", "reg_all", "opt_adam",
    ]
    complete = all(k in metrics for k in required)
    comparable = complete and len({config_signature(metrics[k]["config"]) for k in required}) == 1

    if not complete:
        st.info(
            "A síntese final aparece quando as oito configurações estiverem treinadas. "
            "Os gráficos acima continuam úteis para acompanhar as ablações já disponíveis."
        )
    elif not comparable:
        st.warning(
            "Os oito resultados existem, mas não compartilham a mesma configuração de épocas, "
            "amostra e seed. A síntese automática foi suspensa para evitar comparações inválidas."
        )
    else:
        sig = config_signature(metrics[required[0]]["config"])
        epochs, train_size, test_size, seed = sig
        best_key = max(required, key=lambda k: metrics[k]["best_test"])
        largest_gap_key = max(required, key=lambda k: metrics[k]["gap"])

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Melhor acurácia observada",
            f"{metrics[best_key]['best_test']:.1%}",
            label(best_key),
        )
        c2.metric(
            "Maior gap de generalização",
            f"{metrics[largest_gap_key]['gap']:+.1%}",
            label(largest_gap_key),
        )
        c3.metric(
            "Regime experimental",
            f"{epochs} épocas",
            f"seed {seed} · {train_size or 'full'} treino / {test_size or 'full'} teste",
        )

        # 1) Arquitetura
        mlp, cnn = metrics["mlp_relu"], metrics["cnn_relu"]
        arch_delta = (mlp["best_test"] - cnn["best_test"]) * 100
        st.markdown(
            f"**1. Arquitetura — MLP × CNN.** A MLP atingiu {mlp['best_test']:.1%} de melhor "
            f"acurácia de teste e a CNN+ReLU {cnn['best_test']:.1%} ({arch_delta:+.1f} p.p. para a MLP), "
            f"mas o gap final foi {mlp['gap']:+.1%} na MLP contra {cnn['gap']:+.1%} na CNN. "
            "A leitura principal é separar capacidade de ajuste do treino de capacidade de generalização."
        )

        # 2) Ativação
        relu, tanh = metrics["cnn_relu"], metrics["cnn_tanh"]
        act_winner = tanh if tanh["best_test"] >= relu["best_test"] else relu
        act_loser = relu if act_winner is tanh else tanh
        act_delta = (act_winner["best_test"] - act_loser["best_test"]) * 100
        st.markdown(
            f"**2. Ativação — ReLU × tanh.** {label(act_winner['canonical'])} obteve "
            f"{act_winner['best_test']:.1%}, superando {label(act_loser['canonical'])} em "
            f"{act_delta:.1f} p.p. neste regime. O resultado deve ser interpretado como dependente "
            "de arquitetura, dataset e hiperparâmetros — não como superioridade universal de uma ativação."
        )

        # 3) Regularização
        reg_keys = ["reg_base", "reg_aug", "reg_dropout", "reg_all"]
        reg_winner_key = max(reg_keys, key=lambda k: metrics[k]["best_test"])
        reg_base_m = metrics["reg_base"]
        reg_all_m = metrics["reg_all"]
        st.markdown(
            f"**3. Regularização.** A melhor configuração do bloco foi **{label(reg_winner_key)}** "
            f"com {metrics[reg_winner_key]['best_test']:.1%}. A baseline encerrou com gap "
            f"{reg_base_m['gap']:+.1%}; como não havia overfitting relevante nesse regime, "
            f"augmentation + dropout com SGD ficou em {reg_all_m['best_test']:.1%}. Isso sugere "
            "que regularização adicional pode retardar o ajuste quando o modelo já está subajustado."
        )

        # 4) Otimização
        sgd, adam = metrics["reg_all"], metrics["opt_adam"]
        opt_delta = (adam["best_test"] - sgd["best_test"]) * 100
        st.markdown(
            f"**4. Otimização — SGD+momentum × Adam.** Sob a configuração regularizada, Adam "
            f"atingiu {adam['best_test']:.1%} contra {sgd['best_test']:.1%} do SGD+momentum "
            f"({opt_delta:+.1f} p.p.). A conclusão é sobre convergência neste orçamento de 10 épocas, "
            "não sobre superioridade universal do Adam."
        )

        st.markdown("**5. Qualidade da representação — accuracy × Purity@5.**")
        st.caption(
            "A avaliação usa as mesmas 30 imagens de teste, 1.000 referências de treino e k=5 "
            "para CNN+ReLU, CNN+tanh e Adam+regularização. Purity@5 mede organização local por classe; "
            "não mede 'compreensão' semântica."
        )
        if st.button("Calcular síntese de Purity@5", key="summary_purity"):
            purity_keys = ["cnn_relu", "cnn_tanh", "opt_adam"]
            purity_rows = []
            with st.spinner("Calculando Purity@5 nos três modelos..."):
                for pk in purity_keys:
                    pname = first_existing(pk)
                    stamp = os.path.getmtime(f"{RESULTS}/{pname}.pt")
                    qres = representation_quality(pname, stamp, n_ref=1000, n_query=30, k=5)
                    purity_rows.append({
                        "Modelo": label(pk),
                        "Melhor acurácia de teste": metrics[pk]["best_test"],
                        "Purity@5 · representação": qres["learned"],
                        "Purity@5 · pixels": qres["pixels"],
                        "Ganho sobre pixels": qres["learned"] - qres["pixels"],
                    })

            st.dataframe(
                [{
                    "Modelo": r["Modelo"],
                    "Melhor acurácia de teste": f"{r['Melhor acurácia de teste']:.1%}",
                    "Purity@5 · representação": f"{r['Purity@5 · representação']:.1%}",
                    "Purity@5 · pixels": f"{r['Purity@5 · pixels']:.1%}",
                    "Ganho sobre pixels": f"{r['Ganho sobre pixels']:+.1%}",
                } for r in purity_rows],
                use_container_width=True,
                hide_index=True,
            )

            best_acc = max(purity_rows, key=lambda r: r["Melhor acurácia de teste"])
            best_purity = max(purity_rows, key=lambda r: r["Purity@5 · representação"])
            if best_acc["Modelo"] != best_purity["Modelo"]:
                st.info(
                    f"**Accuracy e organização local não produziram o mesmo ranking.** "
                    f"{best_acc['Modelo']} obteve a maior acurácia ({best_acc['Melhor acurácia de teste']:.1%}), "
                    f"enquanto {best_purity['Modelo']} obteve a maior Purity@5 "
                    f"({best_purity['Purity@5 · representação']:.1%}). "
                    "As métricas capturam propriedades diferentes do modelo."
                )
            else:
                st.info(
                    f"{best_acc['Modelo']} liderou tanto em acurácia quanto em Purity@5 nesta amostra. "
                    "Ainda assim, as métricas representam propriedades distintas."
                )

        st.success(
            "**Mensagem central:** as técnicas não atuaram de forma independente. Arquitetura, ativação, "
            "regularização e otimização interagiram; por isso, atribuir o desempenho a um único componente "
            "seria uma simplificação excessiva."
        )
        st.caption(
            "Limitações: CIFAR-10 reduzido, uma única seed, 10 épocas e arquitetura pequena. "
            "Os resultados sustentam a discussão didática do laboratório, não um benchmark geral."
        )

with aba2:
    st.header("Classificação, ranking e confiança")
    st.caption(
        "No CIFAR-10 há apenas 10 classes; o Top-5 é apenas uma referência didática e não deve "
        "ser comparado diretamente ao Top-5 do ImageNet/AlexNet."
    )
    if not selected:
        st.info("Treine ao menos um modelo.")
    else:
        m = model(selected)
        with torch.no_grad():
            probs = torch.softmax(m(normalize(img[None].to(dev))), dim=1)[0].cpu().numpy()
        order = np.argsort(-probs)
        top = order[:5]
        top1, top2 = int(order[0]), int(order[1])
        confidence = float(probs[top1])
        margin = float(probs[top1] - probs[top2])

        c1, c2 = st.columns([1, 2])
        c1.image(enlarge(img), caption="Entrada 32×32 ampliada")
        m1, m2, m3 = c2.columns(3)
        m1.metric("Predição Top-1", CLASSES[top1])
        m2.metric("Softmax Top-1", f"{confidence:.1%}")
        m3.metric("Margem Top-1/Top-2", f"{margin:.1%}")

        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.barh([CLASSES[k] for k in top][::-1], probs[top][::-1])
        ax.set(xlim=(0, 1), xlabel="probabilidade (softmax)", title=f"Top-5 · {label(selected)}")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        fig.tight_layout()
        c2.pyplot(fig)
        plt.close(fig)

        if truth is not None:
            if top1 == truth:
                c2.success(f"Top-1 correto: {CLASSES[truth]}.")
            else:
                c2.error(f"Top-1 incorreto. Real: {CLASSES[truth]} · Predito: {CLASSES[top1]}.")
            rank = np.where(order == truth)[0]
            if len(rank):
                c2.caption(f"A classe correta está na posição {int(rank[0]) + 1} do ranking de 10 classes.")
        else:
            c2.warning(
                "Imagem externa ao CIFAR-10: o classificador é obrigado a escolher entre as 10 "
                "classes conhecidas. Softmax alto não garante que a imagem pertença à distribuição de treino."
            )

        if margin < 0.05:
            c2.info(
                "**Baixa separação Top-1/Top-2 nesta entrada.** O modelo escolheu uma classe, mas as "
                "duas melhores pontuações estão próximas; isso é compatível com uma decisão pouco discriminativa."
            )

with aba3:
    st.header("Como o treinamento transforma os filtros?")
    if not selected:
        st.info("Treine ao menos um modelo.")
    else:
        trained, cfg = load_checkpoint(selected, dev)
        if cfg.get("model", "cnn") != "cnn":
            st.warning("A MLP não possui filtros convolucionais. Selecione um modelo CNN.")
        else:
            initial = recreate_initial_model(cfg)
            w0 = initial.features[0].weight.detach().cpu()
            w1 = trained.features[0].weight.detach().cpu()
            rel_change = (w1 - w0).norm().item() / (w0.norm().item() + 1e-12)
            epochs = cfg.get("epochs", "?")

            st.caption(
                f"Modelo: **{label(selected)}** · camada: **Conv1** · filtros: **{w1.shape[0]}** · "
                f"kernel: **{w1.shape[2]}×{w1.shape[3]}×{w1.shape[1]}** · treinamento: **{epochs} épocas**"
            )
            c1, c2 = st.columns(2)
            fig0 = filter_grid(w0, "Antes do treinamento")
            fig1 = filter_grid(w1, f"Após {epochs} épocas")
            c1.pyplot(fig0)
            c2.pyplot(fig1)
            plt.close(fig0)
            plt.close(fig1)

            st.metric("Mudança relativa dos pesos da Conv1 (norma L2)", f"{rel_change:.2f}×")
            st.latex(r"\Delta_W = \frac{\lVert W_{final}-W_{inicial}\rVert_2}{\lVert W_{inicial}\rVert_2}")
            st.caption(
                f"Interpretação: a distância entre os pesos finais e iniciais equivale a "
                f"aproximadamente {rel_change:.0%} da norma dos pesos iniciais."
            )
            if isinstance(epochs, int) and epochs <= 2:
                st.warning(
                    "Com apenas 2 épocas, os filtros ainda podem parecer ruidosos. A comparação acima "
                    "mostra que os pesos já mudaram, mas não permite afirmar que detectores bem estruturados emergiram."
                )
            st.caption(
                "A primeira convolução não contém detectores programados manualmente: seus kernels começam "
                "com pesos aleatórios e são ajustados por backpropagation durante o treinamento."
            )

with aba4:
    st.header("Como o modelo organiza as imagens?")
    if not selected:
        st.info("Treine ao menos um modelo.")
    else:
        xtr, ytr = data()[0], data()[1]
        m = model(selected)
        with torch.no_grad():
            _, q = m(normalize(img[None].to(dev)), return_features=True)
            q = q.cpu()
        d_feat = torch.cdist(q, train_features(selected))[0]
        qp = img.flatten().float()
        d_pix = torch.cat([
            (xtr[i:i + 2000].flatten(1).float() - qp).norm(dim=1)
            for i in range(0, len(xtr), 2000)
        ])

        st.image(enlarge(img, 128), caption="Imagem de consulta")
        k_view = 6
        learned_idx = torch.topk(d_feat, k_view, largest=False).indices.tolist()
        pixel_idx = torch.topk(d_pix, k_view, largest=False).indices.tolist()

        for title, subtitle, ids in [
            (
                "Espaço aprendido",
                "Vizinhos segundo a última camada oculta do modelo.",
                learned_idx,
            ),
            (
                "Espaço de pixels",
                "Vizinhos segundo a distância Euclidiana entre valores RGB.",
                pixel_idx,
            ),
        ]:
            st.subheader(title)
            st.caption(subtitle)
            cols = st.columns(k_view)
            for col, idx in zip(cols, ids):
                col.image(enlarge(xtr[idx], 128), caption=CLASSES[int(ytr[idx])])

        if truth is not None:
            learned_purity = np.mean([int(ytr[i]) == truth for i in learned_idx])
            pixel_purity = np.mean([int(ytr[i]) == truth for i in pixel_idx])
            c1, c2 = st.columns(2)
            c1.metric(f"Purity@{k_view} · espaço aprendido", f"{learned_purity:.1%}")
            c2.metric(f"Purity@{k_view} · pixels", f"{pixel_purity:.1%}")
            st.caption(
                "Purity@k = fração dos k vizinhos que pertencem à mesma classe da imagem de consulta. "
                "Uma única consulta é apenas ilustrativa; a avaliação amostral abaixo é mais informativa."
            )

        st.divider()
        st.subheader("Qualidade da representação em uma amostra")
        st.caption(
            "Avaliação leve para CPU: 30 imagens de teste, 1.000 referências de treino e k=5. "
            "O objetivo é comparar a organização do espaço aprendido com a similaridade bruta entre pixels."
        )
        if st.button("Calcular Purity@5 da amostra", type="primary"):
            with st.spinner("Calculando vizinhos no espaço aprendido e no espaço de pixels..."):
                stamp = os.path.getmtime(f"{RESULTS}/{selected}.pt")
                qres = representation_quality(selected, stamp, n_ref=1000, n_query=30, k=5)
            c1, c2 = st.columns(2)
            c1.metric("Purity@5 · representação aprendida", f"{qres['learned']:.1%}")
            c2.metric("Purity@5 · pixels", f"{qres['pixels']:.1%}")
            delta = qres["learned"] - qres["pixels"]
            if delta > 0:
                st.success(f"Nesta amostra, a representação aprendida supera pixels em {delta:.1%} de Purity@5.")
            elif delta < 0:
                st.warning(f"Nesta amostra, pixels superam a representação aprendida em {-delta:.1%} de Purity@5.")
            else:
                st.info("Nesta amostra, os dois espaços obtiveram a mesma Purity@5.")

        st.caption(
            "Uma boa representação tende a aproximar exemplos relevantes para a tarefa mesmo quando seus "
            "pixels diferem. A métrica mede organização de vizinhança, não 'compreensão' semântica humana."
        )
