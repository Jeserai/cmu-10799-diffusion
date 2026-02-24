"""
Evaluate guidance scale tradeoff using an oracle classifier and torch-fidelity.
"""

import argparse
from datetime import datetime
from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

from src.methods import FlowMatching
from src.models import create_model_from_config, create_classifier_from_config
from src.utils import EMA
from src.data import unnormalize, save_image


def load_flow_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = create_model_from_config(config).to(device)
    model.load_state_dict(checkpoint["model"])

    ema = EMA(model, decay=config["training"]["ema_decay"])
    ema.load_state_dict(checkpoint["ema"])

    return model, config, ema


def load_oracle(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    attr_columns = checkpoint.get("attr_columns")
    if attr_columns is None:
        raise ValueError("attr_columns missing from oracle checkpoint.")

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(attr_columns))
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    return model, attr_columns


def load_guidance_classifier(checkpoint_path: str, flow_config: dict, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    attr_columns = checkpoint.get("attr_columns")
    if attr_columns is None:
        raise ValueError("attr_columns missing from guidance classifier checkpoint.")
    config = checkpoint.get("config", flow_config)
    classifier = create_classifier_from_config(config, num_classes=len(attr_columns)).to(device)
    classifier.load_state_dict(checkpoint["model"])
    classifier.eval()
    return classifier, attr_columns


def compute_oracle_pos_rate(images, oracle, target_idx):
    # images in [-1, 1], resize to 224 and normalize for ResNet
    x = unnormalize(images).clamp(0.0, 1.0)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        logits = oracle(x)
        probs = torch.sigmoid(logits[:, target_idx])
    return (probs >= 0.5).float().mean().item()


def parse_fidelity(output: str):
    result = {}
    fid_match = re.search(r"frechet_inception_distance:\s*([0-9.]+)", output)
    kid_match = re.search(r"kernel_inception_distance_mean:\s*([0-9.eE+-]+)", output)
    if fid_match:
        result["fid"] = float(fid_match.group(1))
    if kid_match:
        result["kid"] = float(kid_match.group(1))
    return result


def main():
    parser = argparse.ArgumentParser(description="Guidance tradeoff evaluation")
    parser.add_argument("--flow-checkpoint", type=str, required=True)
    parser.add_argument("--guidance-classifier-checkpoint", type=str, required=True)
    parser.add_argument("--oracle-checkpoint", type=str, required=True)
    parser.add_argument("--attr-name", type=str, default=None)
    parser.add_argument("--attr-names", type=str, default=None)
    parser.add_argument("--scales", type=str, default="0,1.5,3.0,5.0,7.5")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--guidance-mode",
        type=str,
        default="fmps",
        choices=["logit", "logprob", "fmps", "orthogonal", "parallel", "pcgrad", "rescaling", "sequential", "manifold", "alternating"],
    )
    parser.add_argument("--dataset-images", type=str, default="/data/celeba_images")
    parser.add_argument("--dataset-path", type=str, default="/data/celeba")
    parser.add_argument("--cache-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scales = [float(s) for s in args.scales.split(",") if s.strip()]

    flow_model, flow_config, ema = load_flow_checkpoint(args.flow_checkpoint, device)
    guidance_classifier, _ = load_guidance_classifier(
        args.guidance_classifier_checkpoint, flow_config, device
    )
    oracle, attr_columns = load_oracle(args.oracle_checkpoint, device)
    
    # Handle both single and multiple attribute names
    target_indices = None
    if args.attr_names is not None:
        names = [n.strip() for n in args.attr_names.split(",") if n.strip()]
        missing = [n for n in names if n not in attr_columns]
        if missing:
            raise ValueError(f"Unknown attr-names: {missing}")
        target_indices = [attr_columns.index(n) for n in names]
        print(f"Using attr-names {names} at indices {target_indices}")
    elif args.attr_name is not None:
        if args.attr_name not in attr_columns:
            raise ValueError(f"Unknown attr-name: {args.attr_name}")
        target_indices = [attr_columns.index(args.attr_name)]
        print(f"Using attr-name '{args.attr_name}' at index {target_indices[0]}")
    else:
        raise ValueError("Provide --attr-name or --attr-names")

    method = FlowMatching.from_config(flow_model, flow_config, device)
    ema.apply_shadow()
    method.eval_mode()

    data_config = flow_config["data"]
    image_shape = (data_config["channels"], data_config["image_size"], data_config["image_size"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(args.output_dir or f"/data/logs/guided_tradeoff/{timestamp}")
    base_dir.mkdir(parents=True, exist_ok=True)

    dataset_images = Path(args.dataset_images)
    if not dataset_images.exists():
        from datasets import load_from_disk

        dataset = load_from_disk(args.dataset_path)["train"]
        dataset_images.mkdir(parents=True, exist_ok=True)
        for idx, item in enumerate(dataset):
            img_path = dataset_images / f"{idx:06d}.png"
            item["image"].save(img_path)

    # Determine dataset size (number of images) for KID subset sizing
    if dataset_images.exists():
        dataset_len = sum(1 for _ in dataset_images.glob("*.png"))
    else:
        # fallback: try to load dataset to count
        from datasets import load_from_disk

        dataset_len = len(load_from_disk(args.dataset_path)["train"])

    results = []
    for w in scales:
        run_dir = base_dir / f"w_{w}"
        gen_dir = run_dir / "generated"
        gen_dir.mkdir(parents=True, exist_ok=True)

        remaining = args.num_samples
        sample_idx = 0
        all_samples = []
        while remaining > 0:
            batch = min(args.batch_size, remaining)
            if w == 0:
                samples = method.sample(batch_size=batch, image_shape=image_shape, num_steps=args.num_steps)
            else:
                sec_idx = target_indices[1] if len(target_indices) > 1 else None
                sec_scale = w if sec_idx is not None else 0.0
                samples = method.sample_guided(
                    batch_size=batch,
                    image_shape=image_shape,
                    classifier=guidance_classifier,
                    target_class_idx=target_indices[0],
                    secondary_target_class_idx=sec_idx,
                    guidance_scale=w,
                    secondary_guidance_scale=sec_scale,
                    num_steps=args.num_steps,
                    guidance_mode=args.guidance_mode,
                )
            all_samples.append(samples)
            images = unnormalize(samples.detach().cpu()).clamp(0.0, 1.0)
            for i in range(images.shape[0]):
                img_path = gen_dir / f"{sample_idx:06d}.png"
                save_image(images[i : i + 1], str(img_path), nrow=1)
                sample_idx += 1
            remaining -= batch

        samples = torch.cat(all_samples, dim=0)[: args.num_samples]
        # Compute accuracy for the first attribute (or average if multiple)
        acc = compute_oracle_pos_rate(samples.to(device), oracle, target_indices[0])

        cache_root = args.cache_root or str(run_dir / "cache")
        # KID subset size must be at most the size of the smaller input set
        kid_subset_size = min(args.num_samples, dataset_len)
        fidelity_cmd = [
            "fidelity",
            "--gpu", "0",
            "--batch-size", str(args.batch_size),
            "--cache-root", cache_root,
            "--input1", str(gen_dir),
            "--input2", args.dataset_images,
            "--fid",
            "--kid",
            "--kid-subset-size",
            str(kid_subset_size),
        ]
        import subprocess
        try:
            proc = subprocess.run(fidelity_cmd, check=True, capture_output=True, text=True)
            fid_kid = parse_fidelity(proc.stdout)
        except subprocess.CalledProcessError as e:
            print("Error running fidelity:")
            if hasattr(e, 'stdout') and e.stdout:
                print("Stdout:", e.stdout)
            if hasattr(e, 'stderr') and e.stderr:
                print("Stderr:", e.stderr)
            fid_kid = {}

        results.append((w, acc, fid_kid.get("fid"), fid_kid.get("kid")))
        print(f"w={w} oracle_pos_rate={acc:.4f} fid={fid_kid.get('fid')} kid={fid_kid.get('kid')}")

    print("\nResults (w, oracle_pos_rate, fid, kid):")
    for row in results:
        print(row)

    ema.restore()


if __name__ == "__main__":
    main()
