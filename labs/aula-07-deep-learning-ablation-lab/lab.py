"""Treino e ablações de Deep Learning no CIFAR-10.

Exemplos:
  python lab.py --quick
  python lab.py --quick --epochs 10
  python lab.py --quick --epochs 10 --only mlp_relu cnn_relu
  python lab.py --epochs 15
  python lab.py --train-size 10000 --test-size 2000 --epochs 10
  python lab.py --force
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import torch
import torch.nn.functional as F

from common import RESULTS, augment_batch, build_model, get_device, load_cifar10_cpu, normalize

# nome: model, activation, dropout, augment, optimizer, weight_decay, lr
EXPERIMENTS = {
    # Hipótese 1: arquitetura
    "mlp_relu":    ("mlp", "relu", 0.0, False, "sgd", 0.0, 0.01),
    "cnn_relu":    ("cnn", "relu", 0.0, False, "sgd", 0.0, 0.01),

    # Hipótese 2: ativação
    "cnn_tanh":    ("cnn", "tanh", 0.0, False, "sgd", 0.0, 0.01),

    # Hipótese 3: regularização — weight decay fixo
    "reg_base":    ("cnn", "relu", 0.0, False, "sgd", 5e-4, 0.01),
    "reg_aug":     ("cnn", "relu", 0.0, True,  "sgd", 5e-4, 0.01),
    "reg_dropout": ("cnn", "relu", 0.5, False, "sgd", 5e-4, 0.01),
    "reg_all":     ("cnn", "relu", 0.5, True,  "sgd", 5e-4, 0.01),

    # Hipótese 4: otimização — mesma arquitetura/regularização
    "opt_adam":    ("cnn", "relu", 0.5, True,  "adam", 5e-4, 1e-3),
}


def evaluate(model, x, y, batch_size=1000):
    """Avalia com model.eval(): sem dropout e sem data augmentation."""
    model.eval()
    wrong = 0
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb, yb = x[i:i + batch_size], y[i:i + batch_size]
            out = model(normalize(xb))
            total_loss += F.cross_entropy(out, yb, reduction="sum").item()
            wrong += (out.argmax(1) != yb).sum().item()
    return total_loss / len(x), wrong / len(x)


def run(name, model_kind, activation, dropout, augment, optimizer,
        weight_decay, lr, epochs, batch_size=128, seed=0,
        train_size=None, test_size=None, force=False):
    cfg = dict(
        name=name, model=model_kind, activation=activation, dropout=dropout,
        augment=augment, optimizer=optimizer, weight_decay=weight_decay,
        lr=lr, epochs=epochs, batch_size=batch_size, seed=seed,
        train_size=train_size, test_size=test_size,
    )

    os.makedirs(RESULTS, exist_ok=True)
    json_path = f"{RESULTS}/{name}.json"

    if os.path.exists(json_path) and not force:
        with open(json_path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("config") == cfg:
            print(f"[{name}] configuração já treinada; pulando")
            return old

    torch.manual_seed(seed)
    dev = get_device()
    xtr, ytr, xte, yte = [t.to(dev) for t in load_cifar10_cpu()]

    if train_size:
        xtr, ytr = xtr[:train_size], ytr[:train_size]
    if test_size:
        xte, yte = xte[:test_size], yte[:test_size]

    model = build_model(model_kind, activation, dropout).to(dev)

    if optimizer == "sgd":
        opt = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
        milestones = sorted(set([
            max(1, int(0.6 * epochs)),
            max(1, int(0.85 * epochs)),
        ]))
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones, gamma=0.1)
    else:
        opt = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        sched = None

    hist = dict(
        epoch=[],
        train_loss=[],
        test_loss=[],
        train_err=[],
        test_err=[],
        seconds=[],
    )

    n = len(xtr)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, device=dev)

        # Treino propriamente dito
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = xtr[idx], ytr[idx]

            if augment:
                xb = augment_batch(xb)

            out = model(normalize(xb))
            loss = F.cross_entropy(out, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if sched:
            sched.step()

        # Métricas comparáveis: treino e teste avaliados em eval mode.
        train_loss, train_err = evaluate(model, xtr, ytr)
        test_loss, test_err = evaluate(model, xte, yte)

        sec = time.time() - t0
        hist["epoch"].append(ep)
        hist["train_loss"].append(train_loss)
        hist["test_loss"].append(test_loss)
        hist["train_err"].append(train_err)
        hist["test_err"].append(test_err)
        hist["seconds"].append(sec)

        gap = (1 - train_err) - (1 - test_err)
        print(
            f"[{name}] época {ep:02d}/{epochs}  "
            f"treino={1-train_err:.1%}  teste={1-test_err:.1%}  "
            f"gap={gap:+.1%}  {sec:.1f}s",
            flush=True,
        )

    res = dict(config=cfg, **hist)

    torch.save(
        dict(
            config=cfg,
            state_dict={k: v.cpu() for k, v in model.state_dict().items()},
        ),
        f"{RESULTS}/{name}.pt",
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    return res


def load(name):
    path = f"{RESULTS}/{name}.json"
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _curves(names, labels, title):
    hs = [load(n) for n in names]
    if any(h is None for h in hs):
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for h, label in zip(hs, labels):
        ax.plot(
            h["epoch"],
            [1 - v for v in h["test_err"]],
            marker="o",
            markersize=3,
            label=label,
        )

    ax.set(
        xlabel="época",
        ylabel="acurácia de teste",
        title=title,
        ylim=(0, 1),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_architecture():
    return _curves(
        ["mlp_relu", "cnn_relu"],
        ["MLP + ReLU", "CNN + ReLU"],
        "Ablação 1 — Arquitetura: MLP × CNN",
    )


def fig_activation():
    return _curves(
        ["cnn_relu", "cnn_tanh"],
        ["ReLU", "tanh"],
        "Ablação 2 — Ativação: ReLU × tanh",
    )


def fig_regularization():
    return _curves(
        ["reg_base", "reg_aug", "reg_dropout", "reg_all"],
        [
            "baseline + weight decay",
            "+ augmentation",
            "+ dropout",
            "+ augmentation + dropout",
        ],
        "Ablação 3 — Regularização",
    )


def fig_optimizers():
    return _curves(
        ["reg_all", "opt_adam"],
        ["SGD + momentum", "Adam"],
        "Ablação 4 — Otimização",
    )


def save_all():
    figs = [
        ("architecture", fig_architecture),
        ("activation", fig_activation),
        ("regularization", fig_regularization),
        ("optimizers", fig_optimizers),
    ]
    os.makedirs(RESULTS, exist_ok=True)

    for name, fn in figs:
        fig = fn()
        if fig is None:
            print(f"sem resultados suficientes para {name}.png")
            continue
        fig.savefig(f"{RESULTS}/{name}.png", dpi=150)
        plt.close(fig)
        print(f"salvo {RESULTS}/{name}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Deep Learning Ablation Lab / CIFAR-10"
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="número de épocas; padrão: 2 com --quick, 15 sem --quick",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="usa subconjunto padrão de 10.000 treino / 2.000 teste",
    )
    p.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="limita o número de imagens de treino",
    )
    p.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="limita o número de imagens de teste",
    )
    p.add_argument(
        "--only",
        nargs="+",
        choices=list(EXPERIMENTS),
        help="roda apenas os experimentos informados",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    epochs = a.epochs if a.epochs is not None else (2 if a.quick else 15)
    train_size = a.train_size if a.train_size is not None else (10000 if a.quick else None)
    test_size = a.test_size if a.test_size is not None else (2000 if a.quick else None)

    print(
        f"Configuração: epochs={epochs}, train_size={train_size or 'full'}, "
        f"test_size={test_size or 'full'}, seed={a.seed}"
    )

    for name in a.only or EXPERIMENTS:
        model_kind, act, dropout, aug, opt, wd, lr = EXPERIMENTS[name]
        run(
            name, model_kind, act, dropout, aug, opt, wd, lr, epochs,
            seed=a.seed,
            train_size=train_size,
            test_size=test_size,
            force=a.force,
        )

    save_all()
