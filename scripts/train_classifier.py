"""
Train a time-dependent classifier on noisy images for CelebA attributes.
"""

# ruff: noqa: I001

import argparse
import os
from datetime import datetime

import yaml
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.data import create_dataloader_from_config
from src.models import create_classifier_from_config


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train time-dependent classifier")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num_iterations", type=int, default=None, help="Override iterations")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.num_iterations is not None:
        config["training"]["num_iterations"] = args.num_iterations
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloader = create_dataloader_from_config(config, split="train")
    if isinstance(dataloader, DataLoader):
        dataset = dataloader.dataset
    else:
        dataset = dataloader.dataset

    if getattr(dataset, "attr_dim", None) is None:
        raise ValueError("Dataset does not provide attr_dim. Check attributes loading.")

    num_classes = dataset.attr_dim
    classifier = create_classifier_from_config(config, num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=config["training"]["learning_rate"],
        betas=tuple(config["training"]["betas"]),
        weight_decay=config["training"]["weight_decay"],
    )
    scaler = GradScaler(
        "cuda" if device.type == "cuda" else "cpu",
        enabled=config["infrastructure"]["mixed_precision"],
    )
    criterion = nn.BCEWithLogitsLoss()

    num_iterations = config["training"]["num_iterations"]
    log_every = config["training"]["log_every"]
    save_every = config["training"].get("save_every", 5000)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config["logging"]["dir"], f"classifier_{timestamp}")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    data_iter = iter(dataloader)
    classifier.train()

    print(f"Training classifier for {num_iterations} iterations (num_classes={num_classes})")
    for step in range(num_iterations):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        if isinstance(batch, dict):
            x_0 = batch["image"]
            attr = batch["attr"]
        elif isinstance(batch, (tuple, list)):
            x_0, attr = batch
        else:
            raise ValueError("Unexpected batch format")

        x_0 = x_0.to(device)
        attr = attr.to(device)

        batch_size = x_0.shape[0]
        t = torch.rand(batch_size, device=device)
        t_broadcast = t.view(batch_size, *([1] * (x_0.ndim - 1)))
        x_1 = torch.randn_like(x_0)
        x_t = (1.0 - t_broadcast) * x_0 + t_broadcast * x_1

        optimizer.zero_grad()
        with autocast(device.type, enabled=config["infrastructure"]["mixed_precision"]):
            logits = classifier(x_t, t)
            loss = criterion(logits, attr)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (step + 1) % log_every == 0:
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc = (preds == attr).float().mean().item()
            print(f"[{step+1}/{num_iterations}] loss={loss.item():.4f} acc={acc:.4f}")

        if (step + 1) % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"classifier_{step+1:07d}.pt")
            torch.save(
                {
                    "model": classifier.state_dict(),
                    "config": config,
                    "attr_columns": getattr(dataset, "attr_columns", None),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

    final_path = os.path.join(ckpt_dir, "classifier_final.pt")
    torch.save(
        {
            "model": classifier.state_dict(),
            "config": config,
            "attr_columns": getattr(dataset, "attr_columns", None),
        },
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
