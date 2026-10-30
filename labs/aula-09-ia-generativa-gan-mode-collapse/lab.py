"""GAN Mode Collapse Lab - treinamento, medição e gráficos reproducíveis.

Exemplos:
  python lab.py --quick
  python lab.py --steps 1500 --seeds 0 1 2
  python lab.py --steps 300 --experiments vanilla --force
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import (Discriminator, Generator, RESULTS, centers_numpy,
                    evaluate_modes, reference_samples, sample_real)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def sample_generator(model: Generator, n: int, noise_dim: int,
                     seed: int, device: str) -> np.ndarray:
    """Usa gerador independente para avaliar com o mesmo z por seed/modelo."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(n, noise_dim, generator=generator).to(device)
    model.eval()
    with torch.no_grad():
        out = model(noise).cpu().numpy()
    model.train()
    return out


def gradient_penalty(critic: Discriminator, real: torch.Tensor,
                     fake: torch.Tensor) -> torch.Tensor:
    alpha = torch.rand(real.shape[0], 1, device=real.device)
    interpolated = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    scores = critic(interpolated)
    grad = torch.autograd.grad(outputs=scores, inputs=interpolated,
                               grad_outputs=torch.ones_like(scores),
                               create_graph=True, only_inputs=True)[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def plot_scatter(real: np.ndarray, fake: np.ndarray, title: str, output: Path,
                 full_extent: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(real[:, 0], real[:, 1], s=5, alpha=0.2, label="Reais")
    ax.scatter(fake[:, 0], fake[:, 1], s=8, alpha=0.5, label="Geradas")
    centers = centers_numpy()
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=85,
               label="Centros dos modos")
    for i, (x, y) in enumerate(centers):
        ax.annotate(str(i), (x, y), xytext=(5, 4), textcoords="offset points")
    limit = float(max(3.4, np.max(np.abs(fake)) * 1.06)) if full_extent else 3.4
    outside_fraction = float(np.mean(np.max(np.abs(fake), axis=1) > limit))
    title_suffix = "(todos os pontos)" if full_extent else f"(zoom ±3.4; {outside_fraction:.1%} fora da área)"
    ax.set(xlabel="x1", ylabel="x2", title=f"{title}\n{title_suffix}",
           xlim=(-limit, limit), ylim=(-limit, limit))
    ax.set_aspect("equal")
    ax.grid(alpha=0.17)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def plot_progress(history: list[dict], output: Path) -> None:
    if not history:
        return
    steps = [h["step"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(steps, [h["coverage"] for h in history], marker="o", markersize=3)
    axes[0].set(ylabel="Modos com >= 1% das amostras", ylim=(-0.5, 8.5), title="Cobertura (0-8)")
    axes[1].plot(steps, [h["valid_fraction"] for h in history], marker="o", markersize=3)
    axes[1].set(ylabel="Fração válida", ylim=(-0.03, 1.03), title="Aderência aos clusters")
    axes[2].plot(steps, [h["js_divergence_bits"] for h in history], marker="o", markersize=3)
    axes[2].set(ylabel="JS em bits (menor = melhor)", ylim=(-0.03, 1.03), title="Equilíbrio entre modos")
    for ax in axes:
        ax.set_xlabel("Passos de atualização do gerador")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_experiment(name: str, steps: int, seed: int, batch_size: int = 256,
                   eval_every: int = 50, n_eval: int = 2048,
                   noise_dim: int = 2, device: str = "cpu", force: bool = False,
                   out_dir: Path = RESULTS) -> dict:
    if name not in {"vanilla", "wgan_gp"}:
        raise ValueError(f"experimento desconhecido: {name}")
    if min(steps, batch_size, eval_every, n_eval, noise_dim) <= 0:
        raise ValueError("passos, tamanhos e dimensões devem ser positivos")
    cfg = dict(name=name, steps=steps, seed=seed, batch_size=batch_size,
               eval_every=eval_every, n_eval=n_eval, noise_dim=noise_dim,
               device=device, hidden=64, n_critic=1 if name == "vanilla" else 3,
               lr=2e-4 if name == "vanilla" else 1e-4, gp_lambda=10.0 if name == "wgan_gp" else 0.0)
    run_dir = Path(out_dir) / f"{name}_seed{seed}_steps{steps}"
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not force:
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        if old.get("config") == cfg:
            print(f"[cache] {run_dir.name}")
            return old
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    rng_train = np.random.default_rng(seed + 109)
    rng_real_eval = np.random.default_rng(999)
    real_eval = sample_real(n_eval, rng_real_eval)
    g = Generator(noise_dim=noise_dim).to(device)
    d = Discriminator().to(device)
    opt_g = torch.optim.Adam(g.parameters(), lr=cfg["lr"], betas=(0.5, 0.9 if name == "wgan_gp" else 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=cfg["lr"], betas=(0.5, 0.9 if name == "wgan_gp" else 0.999))
    history = []
    start = time.perf_counter()
    last_g_loss, last_d_loss = 0.0, 0.0
    for step in range(1, steps + 1):
        for _ in range(cfg["n_critic"]):
            x = torch.from_numpy(sample_real(batch_size, rng_train)).to(device)
            z = torch.randn(batch_size, noise_dim, device=device)
            with torch.no_grad():
                fake = g(z)
            opt_d.zero_grad(set_to_none=True)
            d_real = d(x)
            d_fake = d(fake)
            if name == "vanilla":
                d_loss = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()
            else:
                d_loss = d_fake.mean() - d_real.mean() + cfg["gp_lambda"] * gradient_penalty(d, x, fake)
            d_loss.backward()
            opt_d.step()
            last_d_loss = float(d_loss.detach().cpu())
        opt_g.zero_grad(set_to_none=True)
        z = torch.randn(batch_size, noise_dim, device=device)
        fake = g(z)
        if name == "vanilla":
            g_loss = F.softplus(-d(fake)).mean()  # objetivo non-saturating
        else:
            g_loss = -d(fake).mean()
        g_loss.backward()
        opt_g.step()
        last_g_loss = float(g_loss.detach().cpu())
        if step == 1 or step % eval_every == 0 or step == steps:
            generated = sample_generator(g, n_eval, noise_dim, seed=50000 + seed, device=device)
            met = evaluate_modes(generated)
            history.append({"step": step, "g_loss": last_g_loss,
                            "d_loss": last_d_loss, "time_sec": round(time.perf_counter()-start, 3),
                            "coverage": met.coverage, "valid_fraction": met.valid_fraction,
                            "js_divergence_bits": met.js_divergence_bits})
            print(f"[{name} s={seed} {step:4d}/{steps}] modos={met.coverage}/8 "
                  f"válidas={met.valid_fraction:.1%} JS={met.js_divergence_bits:.3f}", flush=True)
    fake_eval = sample_generator(g, n_eval, noise_dim, seed=50000 + seed, device=device)
    metrics = evaluate_modes(fake_eval)
    np.savez_compressed(run_dir / "samples.npz", real=real_eval, fake=fake_eval,
                        centers=centers_numpy())
    plot_scatter(real_eval, fake_eval, f"{name} | seed={seed} | steps={steps}", run_dir / "amostras.png")
    plot_scatter(real_eval, fake_eval, f"{name} | seed={seed} | steps={steps}",
                 run_dir / "amostras_full.png", full_extent=True)
    plot_progress(history, run_dir / "progresso.png")
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    seconds = round(time.perf_counter()-start, 3)
    summary = {"config": cfg, "metrics": metrics.as_dict(), "duration_seconds": seconds,
               "notes": "JS calculada apenas entre amostras válidas; interpretar junto com valid_fraction. "
                        "As perdas de Vanilla e WGAN-GP possuem escalas e significados diferentes."}
    save_json(summary_path, summary)
    return summary


def run_references(out_dir: Path, n_eval: int = 2048) -> dict:
    """Dois exemplos didáticos construídos à mão: NÃO são resultados de treinamento."""
    out_dir = Path(out_dir) / "referencias_didaticas"
    out_dir.mkdir(parents=True, exist_ok=True)
    real = sample_real(n_eval, np.random.default_rng(999))
    full = reference_samples(n_eval, np.random.default_rng(456), tuple(range(8)))
    collapsed = reference_samples(n_eval, np.random.default_rng(456), (1, 5))
    m_full, m_collapse = evaluate_modes(full), evaluate_modes(collapsed)
    plot_scatter(real, full, "Referência didática: 8 modos (NÃO é GAN)", out_dir / "referencia_completa.png")
    plot_scatter(real, collapsed, "Referência didática: 2 modos (NÃO é GAN)", out_dir / "referencia_colapsada.png")
    save_json(out_dir / "summary.json", {"reference_complete": m_full.as_dict(),
                                          "reference_collapsed": m_collapse.as_dict(),
                                          "warning": "Referências artificiais, não geradas por redes treinadas."})
    return {"complete": m_full.as_dict(), "collapsed": m_collapse.as_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="execução rápida: 250 passos, uma seed")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--experiments", nargs="+", choices=["vanilla", "wgan_gp"], default=["vanilla", "wgan_gp"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--n-eval", type=int, default=2048)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA indisponível; execute --device cpu")
    steps = args.steps if args.steps is not None else (250 if args.quick else 1500)
    RESULTS.mkdir(parents=True, exist_ok=True)
    references = run_references(RESULTS, args.n_eval)
    all_results = []
    for seed in args.seeds:
        for name in args.experiments:
            result = run_experiment(name, steps, seed, args.batch_size, args.eval_every,
                                    args.n_eval, device=args.device, force=args.force)
            all_results.append(result)
    summary = {"comparison": [dict(**r["config"], **r["metrics"],
                                     duration_seconds=r["duration_seconds"]) for r in all_results],
               "references": references}
    save_json(RESULTS / "comparison.json", summary)
    cols = ["name", "seed", "steps", "n_critic", "coverage", "valid_fraction", "js_divergence_bits", "duration_seconds"]
    with (RESULTS / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary["comparison"])
    print("\n=== SUMÁRIO (resultado desta execução; não é garantia universal) ===")
    for r in summary["comparison"]:
        print(f"{r['name']:8s} seed={r['seed']} coverage={r['coverage']}/8 "
              f"valid={r['valid_fraction']:.1%} JS={r['js_divergence_bits']:.3f} "
              f"tempo={r['duration_seconds']:.1f}s")
    print(f"Arquivos gerados em: {RESULTS}")


if __name__ == "__main__":
    main()
