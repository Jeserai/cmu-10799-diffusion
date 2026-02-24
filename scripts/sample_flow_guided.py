"""
Classifier-guided sampling for Flow Matching.

Example:
  python scripts/sample_flow_guided.py \
    --flow-checkpoint /data/logs/flow_matching/checkpoints/flow_matching_final.pt \
    --classifier-checkpoint /data/logs/classifier/checkpoints/classifier_final.pt \
    --attr-name Smiling \
    --guidance-scale 2.0 \
    --num-steps 200 \
    --num-samples 64 \
    --output guided_grid.png
"""

import argparse
import os
from datetime import datetime

import torch
from tqdm import tqdm

from src.data import save_image, unnormalize
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


def load_classifier_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    fallback_config: dict,
    num_classes: int | None,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", fallback_config)
    attr_columns = checkpoint.get("attr_columns")

    if num_classes is None:
        if attr_columns is not None:
            num_classes = len(attr_columns)
        else:
            raise ValueError(
                "num_classes not provided and attr_columns missing from classifier checkpoint."
            )

    classifier = create_classifier_from_config(config, num_classes=num_classes).to(device)
    classifier.load_state_dict(checkpoint["model"])

    return classifier, attr_columns


def save_grid(samples: torch.Tensor, output_path: str, num_samples: int) -> None:
    import math

    nrow = max(1, int(math.sqrt(num_samples)))
    images = unnormalize(samples.detach().cpu()).clamp(0.0, 1.0)
    save_image(images, output_path, nrow=nrow)


def save_individual_samples(
    samples: torch.Tensor,
    output_dir: str,
    start_idx: int,
) -> int:
    os.makedirs(output_dir, exist_ok=True)
    images = unnormalize(samples.detach().cpu()).clamp(0.0, 1.0)
    for i in range(images.shape[0]):
        img_path = os.path.join(output_dir, f"{start_idx:06d}.png")
        save_image(images[i : i + 1], img_path, nrow=1)
        start_idx += 1
    return start_idx


def main():
    parser = argparse.ArgumentParser(description="Classifier-guided Flow Matching sampling")
    parser.add_argument("--flow-checkpoint", type=str, required=True)
    parser.add_argument("--classifier-checkpoint", type=str, required=True)
    parser.add_argument("--target-class-idx", type=int, default=None)
    parser.add_argument("--target-class-indices", type=str, default=None)
    parser.add_argument("--attr-name", type=str, default=None)
    parser.add_argument("--attr-names", type=str, default=None)
    parser.add_argument("--secondary-target-class-idx", type=int, default=None)
    parser.add_argument("--secondary-target-class-indices", type=str, default=None)
    parser.add_argument("--secondary-attr-name", type=str, default=None)
    parser.add_argument("--secondary-attr-names", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--secondary-guidance-scale", type=float, default=0.0)
    parser.add_argument(
        "--guidance-mode",
        type=str,
        default="fmps",
        choices=["logit", "logprob", "fmps", "orthogonal", "parallel", "pcgrad", "rescaling", "sequential"],
    )
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--report-classifier", action="store_true")
    parser.add_argument("--classifier-threshold", type=float, default=0.5)
    parser.add_argument("--report-all-attributes", action="store_true")
    parser.add_argument("--report-output", type=str, default=None)
    parser.add_argument("--print-attr-columns", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    print(f"Loading flow checkpoint from {args.flow_checkpoint}...")
    flow_model, flow_config, ema = load_flow_checkpoint(args.flow_checkpoint, device)

    print(f"Loading classifier checkpoint from {args.classifier_checkpoint}...")
    classifier, attr_columns = load_classifier_checkpoint(
        args.classifier_checkpoint, device, flow_config, args.num_classes
    )

    target_indices = None
    secondary_indices = None
    if args.attr_names is not None:
        if attr_columns is None:
            raise ValueError("attr_columns missing from classifier checkpoint.")
        names = [n.strip() for n in args.attr_names.split(",") if n.strip()]
        missing = [n for n in names if n not in attr_columns]
        if missing:
            raise ValueError(f"Unknown attr-names: {missing}")
        target_indices = [attr_columns.index(n) for n in names]
        print(f"Using attr-names {names} at indices {target_indices}")
        if args.print_attr_columns:
            print(f"attr_columns: {attr_columns}")
    elif args.attr_name is not None:
        if attr_columns is None:
            raise ValueError("attr_columns missing from classifier checkpoint.")
        if args.attr_name not in attr_columns:
            raise ValueError(f"Unknown attr-name: {args.attr_name}")
        target_indices = [attr_columns.index(args.attr_name)]
        print(f"Using attr-name '{args.attr_name}' at index {target_indices[0]}")
        if args.print_attr_columns:
            print(f"attr_columns: {attr_columns}")
    elif args.target_class_indices is not None:
        target_indices = [int(x) for x in args.target_class_indices.split(",") if x.strip()]
    else:
        if args.target_class_idx is None:
            raise ValueError(
                "Provide --target-class-idx/--target-class-indices "
                "or --attr-name/--attr-names."
            )
        target_indices = [args.target_class_idx]

    if args.secondary_attr_names is not None:
        if attr_columns is None:
            raise ValueError("attr_columns missing from classifier checkpoint.")
        names = [n.strip() for n in args.secondary_attr_names.split(",") if n.strip()]
        missing = [n for n in names if n not in attr_columns]
        if missing:
            raise ValueError(f"Unknown secondary attr-names: {missing}")
        secondary_indices = [attr_columns.index(n) for n in names]
        print(f"Using secondary attr-names {names} at indices {secondary_indices}")
    elif args.secondary_attr_name is not None:
        if attr_columns is None:
            raise ValueError("attr_columns missing from classifier checkpoint.")
        if args.secondary_attr_name not in attr_columns:
            raise ValueError(f"Unknown secondary attr-name: {args.secondary_attr_name}")
        secondary_indices = [attr_columns.index(args.secondary_attr_name)]
        print(
            f"Using secondary attr-name '{args.secondary_attr_name}' at index "
            f"{secondary_indices[0]}"
        )
    elif args.secondary_target_class_indices is not None:
        secondary_indices = [
            int(x) for x in args.secondary_target_class_indices.split(",") if x.strip()
        ]
    elif args.secondary_target_class_idx is not None:
        secondary_indices = [args.secondary_target_class_idx]

    method = FlowMatching.from_config(flow_model, flow_config, device)

    if not args.no_ema:
        print("Using EMA weights")
        ema.apply_shadow()
    else:
        print("Using training weights (no EMA)")

    method.eval_mode()
    classifier.eval()

    data_config = flow_config["data"]
    image_shape = (
        data_config["channels"],
        data_config["image_size"],
        data_config["image_size"],
    )

    print(f"Generating {args.num_samples} guided samples...")
    all_samples = []
    remaining = args.num_samples
    sample_idx = 0
    pos_count = 0
    prob_sum = 0.0
    total_count = 0
    pos_count_all = None
    prob_sum_all = None
    with tqdm(total=args.num_samples, desc="Guided sampling") as pbar:
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)
            samples = method.sample_guided(
                batch_size=batch_size,
                image_shape=image_shape,
                classifier=classifier,
                target_class_idx=target_indices,
                secondary_target_class_idx=secondary_indices,
                guidance_scale=args.guidance_scale,
                secondary_guidance_scale=args.secondary_guidance_scale,
                guidance_mode=args.guidance_mode,
                num_steps=args.num_steps,
            )
            if args.output_dir:
                sample_idx = save_individual_samples(samples, args.output_dir, sample_idx)
            if not args.no_grid:
                all_samples.append(samples)
            if args.report_classifier or args.report_all_attributes:
                t_eval = torch.zeros(batch_size, device=device)
                with torch.no_grad():
                    logits = classifier(samples, t_eval)
                    probs_all = torch.sigmoid(logits)
                if args.report_classifier:
                    probs = probs_all[:, target_indices]
                    pos_count += (probs >= args.classifier_threshold).all(dim=1).sum().item()
                    prob_sum += probs.mean(dim=1).sum().item()
                    total_count += batch_size
                if args.report_all_attributes:
                    if attr_columns is None:
                        raise ValueError("attr_columns missing from classifier checkpoint.")
                    if pos_count_all is None:
                        num_classes = probs_all.shape[1]
                        pos_count_all = torch.zeros(num_classes, device=device)
                        prob_sum_all = torch.zeros(num_classes, device=device)
                    pos_count_all += (probs_all >= args.classifier_threshold).sum(dim=0)
                    prob_sum_all += probs_all.sum(dim=0)
            remaining -= batch_size
            pbar.update(batch_size)

    if not args.no_grid:
        all_samples = torch.cat(all_samples, dim=0)[: args.num_samples]
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"guided_samples_{timestamp}.png"

        save_grid(all_samples, args.output, args.num_samples)
        print(f"Saved guided grid to {args.output}")
    if args.output_dir:
        print(f"Saved {args.num_samples} individual images to {args.output_dir}")
    if args.report_classifier and total_count > 0:
        pos_rate = pos_count / float(total_count)
        mean_prob = prob_sum / float(total_count)
        print(
            f"Classifier positive rate (threshold={args.classifier_threshold}): {pos_rate:.4f}"
        )
        print(f"Classifier mean probability: {mean_prob:.4f}")
    if args.report_all_attributes and pos_count_all is not None:
        total_count = float(args.num_samples)
        pos_rates = (pos_count_all / total_count).detach().cpu().tolist()
        mean_probs = (prob_sum_all / total_count).detach().cpu().tolist()
        for name, pos_rate, mean_prob in zip(attr_columns, pos_rates, mean_probs):
            print(f"[{name}] pos_rate={pos_rate:.4f} mean_prob={mean_prob:.4f}")
        if args.report_output:
            import csv

            with open(args.report_output, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["attr", "pos_rate", "mean_prob"])
                for name, pos_rate, mean_prob in zip(attr_columns, pos_rates, mean_probs):
                    writer.writerow([name, f"{pos_rate:.6f}", f"{mean_prob:.6f}"])
            print(f"Saved attribute report to {args.report_output}")

    if not args.no_ema:
        ema.restore()


if __name__ == "__main__":
    main()
