"""
engineering_template.py - engineering template for small PyTorch experiments

Features:
- Reproducible runs (seed + deterministic mode)
- Run configuration and results tracked as JSON
- Timestamped run directory per experiment
- Checkpoint saving (best + last) with resume support
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 1. Dataclasses: config & run record
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    seed: int = 123
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 64
    model_name: str = "TinyMLP"
    comment: str = ""  # free-text note: "xor baseline", "try higher lr", etc.


@dataclass
class RunRecord:
    config: RunConfig
    created_at: str
    run_dir: str
    final_epoch: int = 0
    final_train_loss: float = float("nan")
    final_val_loss: float = float("nan")
    final_val_acc: float = float("nan")
    best_val_loss: float = float("nan")
    best_val_epoch: int = 0


# ---------------------------------------------------------------------------
# 2. Utilities: seeding, directories, JSON logging
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def make_run_dir(base_dir: Path, cfg: RunConfig) -> Path:
    """Create a timestamped run directory."""
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    name = f"{stamp}_seed{cfg.seed}_{cfg.model_name}"
    run_dir = base_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 3. Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    run_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loss: float,
    val_loss: float,
    is_best: bool = False,
) -> Path:
    """Save a checkpoint and optionally mark it as 'best'."""
    ckpt_dir = run_dir / "checkpoints"
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
    torch.save(ckpt, ckpt_path)

    # Always update "last.pt"
    last_path = ckpt_dir / "last.pt"
    torch.save(ckpt, last_path)

    # Optionally update "best.pt"
    if is_best:
        best_path = ckpt_dir / "best.pt"
        torch.save(ckpt, best_path)

    return ckpt_path


def load_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, float, float]:
    """Load model/optimizer from checkpoint. Returns (epoch, train_loss, val_loss)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    epoch = ckpt.get("epoch", 0)
    train_loss = ckpt.get("train_loss", float("nan"))
    val_loss = ckpt.get("val_loss", float("nan"))
    return epoch, train_loss, val_loss


# ---------------------------------------------------------------------------
# 4. Example model & data – replace with your own
#    (This is a tiny 2D -> 2 hidden -> 2 output MLP for XOR-like tasks.)
# ---------------------------------------------------------------------------

class TinyMLP(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 16, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_toy_xor_data(n_samples: int = 1000) -> Tuple[DataLoader, DataLoader]:
    """Create a simple XOR-style dataset for demo purposes."""
    # Uniform samples in [-1, 1]^2
    x = torch.rand(n_samples, 2) * 2 - 1  # shape (N, 2)

    # Label by sign of product x1 * x2: same sign -> class 0, different -> class 1
    y = (x[:, 0] * x[:, 1] < 0).long()  # shape (N,)

    # Train/val split
    n_train = int(0.8 * n_samples)
    x_train, x_val = x[:n_train], x[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# 5. Training & evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item()
            total_batches += 1

            preds = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_samples += y.numel()

    avg_loss = total_loss / max(total_batches, 1)
    acc = total_correct / max(total_samples, 1)
    return avg_loss, acc


# ---------------------------------------------------------------------------
# 6. Main training orchestration
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: RunConfig,
    base_run_dir: Path = Path("runs"),
    resume_from: Optional[Path] = None,
) -> RunRecord:
    """End-to-end training with tracking and checkpoints."""

    # 1. Seed & device
    set_seed(cfg.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Create run dir and initial record
    run_dir = make_run_dir(base_run_dir, cfg)
    created_at = datetime.now().isoformat(timespec="seconds")

    record = RunRecord(
        config=cfg,
        created_at=created_at,
        run_dir=str(run_dir),
    )

    # Save initial config/record
    save_json(run_dir / "config.json", asdict(cfg))
    save_json(run_dir / "run_record.json", asdict(record))

    # 3. Build model, data, optimizer, loss
    model = TinyMLP(in_dim=2, hidden_dim=16, out_dim=2).to(device)
    train_loader, val_loader = make_toy_xor_data(n_samples=2000)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_epoch = 0

    # 4. Optionally resume from checkpoint
    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}")
        start_epoch, train_loss, val_loss = load_checkpoint(
            resume_from, model, optimizer
        )
        start_epoch = start_epoch + 1
        best_val_loss = val_loss
        best_val_epoch = start_epoch - 1

    # 5. Training loop
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_epoch = epoch

        save_checkpoint(
            run_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_loss=train_loss,
            val_loss=val_loss,
            is_best=is_best,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.3f} | "
            f"{'(best)' if is_best else ''}"
        )

        # Update record after each epoch (optional but nice)
        record.final_epoch = epoch
        record.final_train_loss = float(train_loss)
        record.final_val_loss = float(val_loss)
        record.final_val_acc = float(val_acc)
        record.best_val_loss = float(best_val_loss)
        record.best_val_epoch = best_val_epoch
        save_json(run_dir / "run_record.json", asdict(record))

    print(f"\nRun completed. Artifacts stored in: {run_dir}")
    return record


# ---------------------------------------------------------------------------
# 7. CLI-style entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example: tweak this or wire up argparse/typer if you want real CLI
    cfg = RunConfig(
        seed=123,
        epochs=30,
        lr=1e-3,
        batch_size=64,
        model_name="TinyMLP",
        comment="xor demo with tracking/checkpoints",
    )

    base_dir = Path("runs")
    record = run_experiment(cfg, base_run_dir=base_dir)

    print("\nFinal run record:")
    print(json.dumps(asdict(record), indent=2))
