"""Componentes comuns do laboratório de Deep Learning.

Inclui dispositivo, CIFAR-10, normalização, augmentation e dois modelos:
uma MLP baseline e uma CNN pequena inspirada na lógica da AlexNet.
"""
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torchvision

CLASSES = ["avião", "automóvel", "pássaro", "gato", "veado",
           "cachorro", "sapo", "cavalo", "navio", "caminhão"]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
RESULTS = "results"


def get_device():
    """DEVICE=mps|cuda|cpu força o dispositivo; senão MPS > CUDA > CPU."""
    forced = os.environ.get("DEVICE")
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_cifar10_cpu(root="data"):
    """Retorna imagens CIFAR-10 em uint8 N×3×32×32 e rótulos int64."""
    out = []
    for train in (True, False):
        ds = torchvision.datasets.CIFAR10(root, train=train, download=True)
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
        y = torch.tensor(ds.targets, dtype=torch.long)
        out += [x, y]
    return tuple(out)


def normalize(x):
    """Converte uint8 em float e normaliza por canal."""
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x.float() / 255.0 - mean) / std


def augment_batch(x, pad=4):
    """Random crop 32×32 após padding + flip horizontal por amostra."""
    n, _, h, w = x.shape
    xp = torch.nn.functional.pad(x, (pad,) * 4)
    dev = x.device
    oy = torch.randint(0, 2 * pad + 1, (n,), device=dev)
    ox = torch.randint(0, 2 * pad + 1, (n,), device=dev)
    flip = torch.rand(n, device=dev) < 0.5
    base = torch.arange(w, device=dev)
    rows = oy[:, None] + torch.arange(h, device=dev)
    cols = torch.where(flip[:, None], ox[:, None] + w - 1 - base,
                       ox[:, None] + base)
    out = xp[
        torch.arange(n, device=dev)[:, None, None],
        :,
        rows[:, :, None],
        cols[:, None, :],
    ]
    return out.permute(0, 3, 1, 2).contiguous()


class MLPBaseline(nn.Module):
    """Baseline sem convolução: pixels achatados -> camadas totalmente conectadas."""
    def __init__(self, activation="relu", dropout=0.0):
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        self.hidden = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 1024),
            act(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            act(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(512, 10)

    def forward(self, x, return_features=False):
        h = self.hidden(x)
        logits = self.out(h)
        return (logits, h) if return_features else logits


class MiniCNN(nn.Module):
    """CNN pequena para CIFAR-10, com cinco convoluções e pooling sobreposto."""
    def __init__(self, activation="relu", dropout=0.0):
        super().__init__()
        act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        pool = lambda: nn.MaxPool2d(3, stride=2)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2), act(), pool(),      # 32 -> 15
            nn.Conv2d(64, 192, 3, padding=1), act(), pool(),   # 15 -> 7
            nn.Conv2d(192, 256, 3, padding=1), act(),
            nn.Conv2d(256, 192, 3, padding=1), act(),
            nn.Conv2d(192, 128, 3, padding=1), act(), pool(),  # 7 -> 3
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 3 * 3, 512),
            act(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            act(),
        )
        self.out = nn.Linear(512, 10)

    def forward(self, x, return_features=False):
        h = self.fc(self.features(x))
        logits = self.out(h)
        return (logits, h) if return_features else logits


def build_model(kind="cnn", activation="relu", dropout=0.0):
    if kind == "mlp":
        return MLPBaseline(activation, dropout)
    if kind == "cnn":
        return MiniCNN(activation, dropout)
    raise ValueError(f"Modelo desconhecido: {kind}")


def load_checkpoint(name, device):
    """Carrega results/<name>.pt e devolve (modelo em eval, config)."""
    ck = torch.load(f"{RESULTS}/{name}.pt", map_location=device, weights_only=True)
    cfg = ck["config"]
    model = build_model(cfg["model"], cfg["activation"], cfg["dropout"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), cfg
