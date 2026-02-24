"""
Evaluate guided vs unguided generation with classifier and dataset ground truth.
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import load_from_disk

from src.methods import FlowMatching
from src.models import create_classifier_from_config, create_model_from_config
from src.utils import EMA


def load_flow_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = create_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model"])

    ema = EMA(model, decay=config["training"]["ema_decay"])
    ema.load_state_dict(checkpoint["ema"])
    return model, config, ema


def load_classifier_checkpoint(checkpoint_path: str, device: torch.device, fallback_config: dict):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", fallback_config)
    attr_columns = checkpoint.get("attr_columns")
    if attr_columns is None:
        raise ValueError("attr_columns missing from classifier checkpoint.")

    classifier = create_classifier_from_config(config, num_classes=len(attr_columns)).to(device)
    classifier.load_state_dict(checkpoint["model"])
    return classifier, attr_columns


def iter_dataset_batches(dataset, attr_columns, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def collate_fn(items):
        images = []
        attrs = []
        for item in items:
            images.append(transform(item["image"]))
            attrs.append([item[c] for c in attr_columns])
        images = torch.stack(images, dim=0)
        attrs = torch.tensor(attrs, dtype=torch.float32)
        attrs = (attrs > 0).float()
        return images, attrs

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    for images, attrs in loader:
        yield images, attrs


def compute_dataset_stats(
    dataset, classifier, attr_columns, target_idx, device, batch_size, max_items
):
    total = 0
    attr_sum = torch.zeros(len(attr_columns), device=device)
    correct_sum = torch.zeros(len(attr_columns), device=device)

    classifier.eval()
    for images, attrs in iter_dataset_batches(dataset, attr_columns, batch_size):
        if max_items is not None and total >= max_items:
            break
        target_mask = attrs[:, target_idx] > 0.5
        if target_mask.sum() == 0:
            continue
        images = images[target_mask]
        attrs = attrs[target_mask]
        if max_items is not None:
            keep = min(images.shape[0], max_items - total)
            images = images[:keep]
            attrs = attrs[:keep]

        images = images.to(device)
        attrs = attrs.to(device)
        t_eval = torch.zeros(images.shape[0], device=device)
        with torch.no_grad():
            logits = classifier(images, t_eval)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

        attr_sum += attrs.sum(dim=0)
        correct_sum += (preds == attrs).float().sum(dim=0)
        total += images.shape[0]

    if total == 0:
        raise ValueError("No samples matched the target attribute in dataset.")
    attr_freq = (attr_sum / float(total)).detach().cpu()
    acc = (correct_sum / float(total)).detach().cpu()
    return attr_freq, acc, total


def compute_generated_stats(samples, classifier, device, threshold):
    t_eval = torch.zeros(samples.shape[0], device=device)
    with torch.no_grad():
        logits = classifier(samples, t_eval)
        probs = torch.sigmoid(logits)
    pos_rates = (probs >= threshold).float().mean(dim=0).detach().cpu()
    mean_probs = probs.mean(dim=0).detach().cpu()
    return pos_rates, mean_probs


def main():
    parser = argparse.ArgumentParser(description="Overall guided vs unguided evaluation")
    parser.add_argument("--flow-checkpoint", type=str, required=True)
    parser.add_argument("--classifier-checkpoint", type=str, required=True)
    parser.add_argument("--attr-name", type=str, default="Smiling")
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument(
        "--guidance-mode",
        type=str,
        default="fmps",
        choices=["logit", "logprob", "fmps", "orthogonal", "parallel", "pcgrad", "rescaling", "sequential", "manifold", "alternating"],
    )
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dataset-path", type=str, default="/data/celeba")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    flow_model, flow_config, ema = load_flow_checkpoint(args.flow_checkpoint, device)
    classifier, attr_columns = load_classifier_checkpoint(
        args.classifier_checkpoint, device, flow_config
    )

    if args.attr_name not in attr_columns:
        raise ValueError(f"Unknown attr-name: {args.attr_name}")
    target_idx = attr_columns.index(args.attr_name)

    dataset = load_from_disk(args.dataset_path)["train"]
    attr_freq, acc, total = compute_dataset_stats(
        dataset,
        classifier,
        attr_columns,
        target_idx,
        device,
        args.batch_size,
        args.max_items,
    )

    method = FlowMatching.from_config(flow_model, flow_config, device)
    ema.apply_shadow()
    method.eval_mode()

    data_config = flow_config["data"]
    image_shape = (
        data_config["channels"],
        data_config["image_size"],
        data_config["image_size"],
    )

    guided = method.sample_guided(
        batch_size=args.num_samples,
        image_shape=image_shape,
        classifier=classifier,
        target_class_idx=target_idx,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        guidance_mode=args.guidance_mode,
    )

    guided_pos, guided_prob = compute_generated_stats(
        guided, classifier, device, args.threshold
    )

    mae = (guided_pos - attr_freq).abs().mean().item()
    target_pos = guided_pos[target_idx].item()
    target_prob = guided_prob[target_idx].item()

    print(f"Dataset ground truth (train, {args.attr_name}=1 subset):")
    print(f"  total: {total}")
    print(f"  mean attr frequency: {attr_freq.mean().item():.4f}")
    print(f"  mean classifier accuracy: {acc.mean().item():.4f}")
    print("")

    print("guided overall:")
    print(f"  target '{args.attr_name}' pos_rate: {target_pos:.4f}")
    print(f"  target '{args.attr_name}' mean_prob: {target_prob:.4f}")
    print(f"  mean abs error to GT freqs: {mae:.4f}")

    ema.restore()


if __name__ == "__main__":
    main()
