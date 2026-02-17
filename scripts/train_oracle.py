"""
Train a clean-image oracle classifier on CelebA attributes.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser(description="Train oracle attribute classifier")
    parser.add_argument("--dataset-path", type=str, default="/data/celeba")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-path", type=str, default="/data/logs/oracle/resnet18_oracle.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = load_from_disk(args.dataset_path)["train"]
    attr_columns = [c for c in dataset.column_names if c not in ["image", "image_id"]]

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    def collate_fn(items):
        images = [transform(item["image"]) for item in items]
        attrs = [[item[c] for c in attr_columns] for item in items]
        images = torch.stack(images, dim=0)
        attrs = torch.tensor(attrs, dtype=torch.float32)
        attrs = (attrs > 0).float()
        return images, attrs

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn
    )

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(attr_columns))
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(args.epochs):
        for images, attrs in loader:
            images = images.to(device)
            attrs = attrs.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, attrs)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{args.epochs} loss={loss.item():.4f}")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "attr_columns": attr_columns},
        save_path,
    )
    print(f"Saved oracle checkpoint to {save_path}")


if __name__ == "__main__":
    main()
