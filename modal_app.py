"""
Modal Configuration for CMU 10799 Diffusion Homework

Defines Modal environment and training functions for cloud GPU training.

See docs/QUICKSTART-MODAL.md for setup and usage instructions.

All parameters are read from config YAML files first, then overridden by command-line arguments.
"""

import modal

# =============================================================================
# Modal App Definition
# =============================================================================

# Create the Modal app
app = modal.App("cmu-10799-diffusion")

# Define the container image with all dependencies
# This mirrors the CPU-only environment (environments/environment-cpu.yml)
# but installs GPU-enabled PyTorch automatically on Modal's GPU machines
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "pillow>=9.0.0",
        "pyyaml>=6.0",
        "einops>=0.6.0",
        "tqdm>=4.64.0",
        "scipy>=1.9.0",
        "wandb>=0.15.0",
        "datasets>=2.0.0",  # For HuggingFace Hub dataset loading
        "torch-fidelity>=0.3.0",  # Comprehensive evaluation metrics
    )
    # Copy the local project directory into the image
    .add_local_dir(".", "/root", ignore=[".git", ".venv*", "venv", "__pycache__", "logs", "checkpoints", "*.md", "docs", "environments", "notebooks"])
)

# Create a persistent volume for checkpoints and data
volume = modal.Volume.from_name("cmu-10799-diffusion-data", create_if_missing=True)

# =============================================================================
# Training Function
# =============================================================================

def _train_impl(
    method: str,
    config_path: str,
    resume_from: str,
    num_iterations: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    overfit_single_batch: bool = False,
):
    """
    Internal training implementation.

    Reads config from YAML file, applies command-line overrides.
    """
    import os
    import sys
    import yaml
    import tempfile
    import subprocess

    sys.path.insert(0, "/root")

    # Load config
    config_tag = method
    if config_path is None:
        config_path = f"/root/configs/{method}.yaml"
    else:
        config_path = f"/root/{config_path}"
        config_tag = os.path.splitext(os.path.basename(config_path))[0]

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get num_gpus from config
    config_device = config['infrastructure'].get('device', 'cuda')
    num_gpus = config['infrastructure'].get('num_gpus', 1 if config_device == 'cuda' else 0)
    if num_gpus is None:
        num_gpus = 1 if config_device == 'cuda' else 0

    # Apply command-line overrides if provided
    if num_iterations is not None:
        config['training']['num_iterations'] = num_iterations
    if batch_size is not None:
        config['training']['batch_size'] = batch_size
    if learning_rate is not None:
        config['training']['learning_rate'] = learning_rate

    # Set Modal-specific paths
    config['data']['repo_name'] = "electronickale/cmu-10799-celeba64-subset"
    # Set root path for both modes:
    # - from_hub=true: checks for cached Arrow format first, then downloads from HF
    # - from_hub=false: looks for traditional folder structure (train/images/)
    config['data']['root'] = "/data/celeba"
    config['checkpoint']['dir'] = f"/data/checkpoints/{config_tag}"
    config['logging']['dir'] = f"/data/logs/{config_tag}"

    # Create directories
    os.makedirs(config['checkpoint']['dir'], exist_ok=True)
    os.makedirs(config['logging']['dir'], exist_ok=True)

    resume_path = f"/data/{resume_from}" if resume_from else None

    # Use torchrun for multi-GPU, direct import for single GPU
    if num_gpus > 1:
        temp_config_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
                yaml.safe_dump(config, temp_file)
                temp_config_path = temp_file.name

            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={num_gpus}",
                "/root/train.py",
                "--method", method,
                "--config", temp_config_path,
            ]
            if resume_path:
                cmd.extend(["--resume", resume_path])
            if overfit_single_batch:
                cmd.append("--overfit-single-batch")

            subprocess.run(cmd, check=True)
        finally:
            if temp_config_path and os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    else:
        from train import train as run_training
        run_training(method_name=method, config=config, resume_path=resume_path, overfit_single_batch=overfit_single_batch)

    volume.commit()
    return f"Training complete! Checkpoints saved to /data/checkpoints/{method}"


# Create training functions for different GPU counts
@app.function(image=image, gpu="L40S:1", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_1gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:2", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_2gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:3", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_3gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:4", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_4gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:5", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_5gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:6", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_6gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:7", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_7gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:8", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_8gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)


@app.function(image=image, gpu="L40S:1", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_classifier(config_path: str = None, num_iterations: int = None, batch_size: int = None):
    import os
    import subprocess
    import yaml
    import tempfile

    local_config_path = config_path or "configs/classifier_modal.yaml"
    with open(local_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Ensure Modal volume paths so checkpoints persist
    config["data"]["root"] = "/data/celeba"
    config["checkpoint"]["dir"] = "/data/checkpoints/classifier"
    config["logging"]["dir"] = "/data/logs/classifier"
    os.makedirs(config["checkpoint"]["dir"], exist_ok=True)
    os.makedirs(config["logging"]["dir"], exist_ok=True)

    if num_iterations is not None:
        config["training"]["num_iterations"] = num_iterations
    if batch_size is not None:
        config["training"]["batch_size"] = batch_size

    temp_config_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config, tmp)
            temp_config_path = tmp.name

        cmd = [
            "python", "/root/scripts/train_classifier.py",
            "--config", temp_config_path,
        ]
        subprocess.run(cmd, check=True)
    finally:
        if temp_config_path and os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    volume.commit()
    return "Classifier training complete!"


@app.function(image=image, gpu="L40S", timeout=60 * 60 * 2, volumes={"/data": volume})
def sample_flow_guided(
    flow_checkpoint: str,
    classifier_checkpoint: str,
    attr_name: str = None,
    attr_names: str = None,
    target_class_idx: int = None,
    target_class_indices: str = None,
    guidance_scale: float = None,
    guidance_mode: str = None,
    num_steps: int = None,
    num_samples: int = None,
    batch_size: int = None,
    output: str = None,
    output_dir: str = None,
    no_grid: bool = False,
    report_classifier: bool = False,
    classifier_threshold: float = 0.5,
    report_all_attributes: bool = False,
    report_output: str = None,
    no_ema: bool = False,
):
    import os
    import subprocess
    from datetime import datetime

    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"/data/logs/guided_samples_{timestamp}.png"
    elif not output.startswith("/data/"):
        output = f"/data/{output.lstrip('./')}"

    os.makedirs(os.path.dirname(output), exist_ok=True)

    cmd = [
        "python", "/root/scripts/sample_flow_guided.py",
        "--flow-checkpoint", flow_checkpoint,
        "--classifier-checkpoint", classifier_checkpoint,
        "--output", output,
    ]

    if attr_name is not None:
        cmd.extend(["--attr-name", attr_name])
    if attr_names is not None:
        cmd.extend(["--attr-names", attr_names])
    if target_class_idx is not None:
        cmd.extend(["--target-class-idx", str(target_class_idx)])
    if target_class_indices is not None:
        cmd.extend(["--target-class-indices", target_class_indices])
    if guidance_scale is not None:
        cmd.extend(["--guidance-scale", str(guidance_scale)])
    if guidance_mode is not None:
        cmd.extend(["--guidance-mode", guidance_mode])
    if num_steps is not None:
        cmd.extend(["--num-steps", str(num_steps)])
    if num_samples is not None:
        cmd.extend(["--num-samples", str(num_samples)])
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    if output_dir is not None:
        cmd.extend(["--output-dir", output_dir])
    if no_grid:
        cmd.append("--no-grid")
    if report_classifier:
        cmd.append("--report-classifier")
        cmd.extend(["--classifier-threshold", str(classifier_threshold)])
    if report_all_attributes:
        cmd.append("--report-all-attributes")
        cmd.extend(["--classifier-threshold", str(classifier_threshold)])
        if report_output is not None:
            cmd.extend(["--report-output", report_output])
    if no_ema:
        cmd.append("--no-ema")

    subprocess.run(cmd, check=True)
    volume.commit()
    return f"Guided samples saved to {output}"


@app.function(image=image, gpu="L40S", timeout=60 * 60 * 2, volumes={"/data": volume})
def evaluate_guided_torch_fidelity(
    flow_checkpoint: str,
    classifier_checkpoint: str,
    attr_name: str = None,
    attr_names: str = None,
    target_class_idx: int = None,
    target_class_indices: str = None,
    guidance_scale: float = None,
    guidance_mode: str = None,
    num_steps: int = None,
    num_samples: int = 1000,
    batch_size: int = 128,
    metrics: str = "kid",
    output_dir: str = None,
    report_classifier: bool = False,
    classifier_threshold: float = 0.5,
    report_all_attributes: bool = False,
    report_output: str = None,
    override: bool = False,
):
    import os
    import subprocess
    import glob
    import shutil
    from datetime import datetime
    from pathlib import Path
    from datasets import load_from_disk

    flow_ckpt_path = flow_checkpoint
    if not flow_ckpt_path.startswith("/data/"):
        flow_ckpt_path = f"/data/{flow_ckpt_path.lstrip('./')}"
    checkpoint_dir = Path(flow_ckpt_path).parent

    run_tag = "guided"
    if attr_name:
        run_tag = f"{run_tag}_{attr_name}"
    if guidance_scale is not None:
        run_tag = f"{run_tag}_g{guidance_scale}"
    if num_steps is not None:
        run_tag = f"{run_tag}_s{num_steps}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = output_dir or str(checkpoint_dir / "guided_samples" / run_tag / timestamp)
    gen_dir = os.path.join(base_dir, "generated")
    grid_path = os.path.join(base_dir, "grid.png")
    cache_root = os.path.join(base_dir, "cache")

    os.makedirs(gen_dir, exist_ok=True)

    # Prepare dataset path for torch-fidelity
    dataset_arrow_path = "/data/celeba"
    dataset_images_path = "/data/celeba_images"
    if not os.path.exists(dataset_images_path):
        print("=" * 60)
        print("Extracting dataset images for torch-fidelity...")
        print("=" * 60)

        dataset = load_from_disk(dataset_arrow_path)
        train_data = dataset["train"]
        os.makedirs(dataset_images_path, exist_ok=True)

        print(f"Extracting {len(train_data)} images...")
        for idx, item in enumerate(train_data):
            img = item["image"]
            img_path = os.path.join(dataset_images_path, f"{idx:06d}.png")
            img.save(img_path)
            if (idx + 1) % 1000 == 0:
                print(f"  Extracted {idx + 1}/{len(train_data)} images")

        volume.commit()
        print(f"Dataset images saved to {dataset_images_path}")
    else:
        print(f"Using cached dataset images at {dataset_images_path}")

    # Step 1: Generate samples (with override handling)
    print("=" * 60)
    print("Step 1/2: Generating guided samples...")
    print("=" * 60)

    need_generation = True
    if os.path.exists(gen_dir) and not override:
        existing_samples = (
            glob.glob(os.path.join(gen_dir, "*.png")) +
            glob.glob(os.path.join(gen_dir, "*.jpg")) +
            glob.glob(os.path.join(gen_dir, "*.jpeg"))
        )
        num_existing = len(existing_samples)
        if num_existing >= num_samples:
            print(f"Found {num_existing} existing samples (need {num_samples})")
            print("Skipping sample generation (use --override to force regeneration)")
            need_generation = False
        else:
            print(f"Found {num_existing} existing samples but need {num_samples}")
            print("Regenerating samples...")
            shutil.rmtree(gen_dir)
    elif os.path.exists(gen_dir) and override:
        print("Override flag set, regenerating samples...")
        shutil.rmtree(gen_dir)

    if need_generation:
        os.makedirs(gen_dir, exist_ok=True)
        cmd = [
            "python", "/root/scripts/sample_flow_guided.py",
            "--flow-checkpoint", flow_checkpoint,
            "--classifier-checkpoint", classifier_checkpoint,
            "--output-dir", gen_dir,
            "--no-grid",
            "--num-samples", str(num_samples),
            "--batch-size", str(batch_size),
        ]

        if attr_name is not None:
            cmd.extend(["--attr-name", attr_name])
        if attr_names is not None:
            cmd.extend(["--attr-names", attr_names])
        if target_class_idx is not None:
            cmd.extend(["--target-class-idx", str(target_class_idx)])
        if target_class_indices is not None:
            cmd.extend(["--target-class-indices", target_class_indices])
        if guidance_scale is not None:
            cmd.extend(["--guidance-scale", str(guidance_scale)])
        if guidance_mode is not None:
            cmd.extend(["--guidance-mode", guidance_mode])
        if num_steps is not None:
            cmd.extend(["--num-steps", str(num_steps)])
        if report_classifier:
            cmd.append("--report-classifier")
            cmd.extend(["--classifier-threshold", str(classifier_threshold)])
        if report_all_attributes:
            cmd.append("--report-all-attributes")
            cmd.extend(["--classifier-threshold", str(classifier_threshold)])
            if report_output is not None:
                cmd.extend(["--report-output", report_output])

        subprocess.run(cmd, check=True)
    else:
        print(f"Using existing samples from {gen_dir}")

    # Save a 4x4 grid preview for quick inspection
    try:
        from PIL import Image as PILImage
        from torchvision import transforms
        from torchvision.utils import make_grid as torch_make_grid

        preview_paths = sorted(glob.glob(os.path.join(gen_dir, "*.png")))[:16]
        if preview_paths:
            images = [transforms.ToTensor()(PILImage.open(p).convert("RGB")) for p in preview_paths]
            grid = torch_make_grid(images, nrow=4)
            torch_save_image = __import__("torchvision.utils", fromlist=["save_image"]).save_image
            torch_save_image(grid, grid_path)
            print(f"Saved grid preview to {grid_path}")
    except Exception as e:
        print(f"Warning: Failed to save grid preview: {e}")

    # Step 2: Run fidelity
    print("\n" + "=" * 60)
    print("Step 2/2: Running torch-fidelity...")
    print("=" * 60)

    os.makedirs(cache_root, exist_ok=True)
    cmd = [
        "fidelity",
        "--gpu", "0",
        "--batch-size", str(batch_size),
        "--cache-root", cache_root,
        "--input1", gen_dir,
        "--input2", dataset_images_path,
    ]
    if "fid" in metrics:
        cmd.append("--fid")
    if "kid" in metrics:
        cmd.append("--kid")
    if "is" in metrics or "isc" in metrics:
        cmd.append("--isc")

    print(f"\nRunning command: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)
    volume.commit()
    return f"Guided eval complete. Outputs in {base_dir}"


@app.function(image=image, volumes={"/data": volume})
def inspect_classifier_checkpoint(checkpoint: str, attr_name: str = None):
    import subprocess

    cmd = [
        "python", "/root/scripts/inspect_classifier_checkpoint.py",
        "--checkpoint", checkpoint,
    ]
    if attr_name is not None:
        cmd.extend(["--attr-name", attr_name])
    subprocess.run(cmd, check=True)
    return "Inspection complete"


@app.function(image=image, gpu="L40S", timeout=60 * 60 * 2, volumes={"/data": volume})
def evaluate_guidance_overall(
    flow_checkpoint: str,
    classifier_checkpoint: str,
    attr_name: str = "Smiling",
    guidance_scale: float = 2.0,
    guidance_mode: str = "fmps",
    num_steps: int = 200,
    num_samples: int = 1000,
    batch_size: int = 128,
    max_items: int = None,
):
    import subprocess

    cmd = [
        "python", "/root/scripts/evaluate_guidance_overall.py",
        "--flow-checkpoint", flow_checkpoint,
        "--classifier-checkpoint", classifier_checkpoint,
        "--attr-name", attr_name,
        "--guidance-scale", str(guidance_scale),
        "--guidance-mode", guidance_mode,
        "--num-steps", str(num_steps),
        "--num-samples", str(num_samples),
        "--batch-size", str(batch_size),
    ]
    if max_items is not None:
        cmd.extend(["--max-items", str(max_items)])

    subprocess.run(cmd, check=True)
    return "Overall guidance evaluation complete"


@app.function(image=image, gpu="L40S", timeout=60 * 60 * 6, volumes={"/data": volume})
def train_oracle(
    dataset_path: str = "/data/celeba",
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-4,
    save_path: str = "/data/logs/oracle/resnet50_oracle.pt",
):
    import subprocess

    cmd = [
        "python", "/root/scripts/train_oracle.py",
        "--dataset-path", dataset_path,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--save-path", save_path,
    ]
    subprocess.run(cmd, check=True)
    volume.commit()
    return f"Oracle trained and saved to {save_path}"


@app.function(image=image, gpu="L40S", timeout=60 * 60 * 6, volumes={"/data": volume})
def evaluate_guidance_tradeoff(
    flow_checkpoint: str,
    guidance_classifier_checkpoint: str,
    oracle_checkpoint: str,
    attr_name: str = None,
    attr_names: str = None,
    scales: str = "0,1.5,3.0,5.0,7.5",
    guidance_mode: str = "fmps",
    num_steps: int = 200,
    num_samples: int = 1000,
    batch_size: int = 128,
):
    import subprocess

    cmd = [
        "python", "/root/scripts/evaluate_guidance_tradeoff.py",
        "--flow-checkpoint", flow_checkpoint,
        "--guidance-classifier-checkpoint", guidance_classifier_checkpoint,
        "--oracle-checkpoint", oracle_checkpoint,
        "--scales", scales,
        "--guidance-mode", guidance_mode,
        "--num-steps", str(num_steps),
        "--num-samples", str(num_samples),
        "--batch-size", str(batch_size),
    ]
    if attr_name is not None:
        cmd.extend(["--attr-name", attr_name])
    if attr_names is not None:
        cmd.extend(["--attr-names", attr_names])
    subprocess.run(cmd, check=True)
    volume.commit()
    return "Guidance tradeoff evaluation complete"

# Map GPU counts to functions
TRAIN_FUNCTIONS = {
    1: train_1gpu,
    2: train_2gpu,
    3: train_3gpu,
    4: train_4gpu,
    5: train_5gpu,
    6: train_6gpu,
    7: train_7gpu,
    8: train_8gpu,
}


# =============================================================================
# Sampling Function
# =============================================================================

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 3,  # 3 hours
    volumes={"/data": volume},
)
def sample(
    method: str = "ddpm",
    checkpoint: str = "checkpoints/ddpm/ddpm_final.pt",
    num_samples: int = None,
    num_steps: int = None,
):
    """
    Generate samples from a trained model.

    Uses sample.py via subprocess, similar to how training uses train.py.
    """
    import os
    import subprocess
    from datetime import datetime

    # Set up paths
    checkpoint_path = f"/data/{checkpoint}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/data/samples/{method}_{timestamp}.png"

    os.makedirs("/data/samples", exist_ok=True)

    # Build command to run sample.py
    cmd = [
        "python", "/root/sample.py",
        "--checkpoint", checkpoint_path,
        "--method", method,
        "--grid",
        "--output", output_path,
    ]

    if num_samples is not None:
        cmd.extend(["--num_samples", str(num_samples)])
    if num_steps is not None:
        cmd.extend(["--num_steps", str(num_steps)])

    subprocess.run(cmd, check=True)
    volume.commit()

    return f"Samples saved to {output_path}"


# =============================================================================
# Dataset Download Function
# =============================================================================

@app.function(
    image=image,
    timeout=60 * 60,  # 1 hour
    volumes={"/data": volume},
)
def download_dataset():
    """
    Download the dataset from HuggingFace Hub to Modal volume.

    Caches the dataset in Arrow format at /data/celeba. After downloading,
    training with from_hub=true will automatically use this cached version
    instead of redownloading.
    """
    import sys
    sys.path.insert(0, "/root")

    from datasets import load_dataset
    import os

    print("Downloading dataset from HuggingFace Hub...")
    dataset = load_dataset("electronickale/cmu-10799-celeba64-subset")

    # Save to volume in Arrow format
    os.makedirs("/data/celeba", exist_ok=True)
    dataset.save_to_disk("/data/celeba")

    volume.commit()

    print(f"Dataset cached to /data/celeba")
    print(f"Train set size: {len(dataset['train'])}")
    return "Dataset download complete! Training with from_hub=true with root = '/data/celeba' will now use this cached version."


# =============================================================================
# Evaluation Function (using torch-fidelity)
# =============================================================================

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 8,  # 8 hours
    volumes={"/data": volume},
)
def evaluate_torch_fidelity(
    method: str = "ddpm",
    checkpoint: str = "checkpoints/ddpm/ddpm_final.pt",
    metrics: str = "fid,kid",
    num_samples: int = 5000,
    batch_size: int = 128,
    num_steps: int = None,
    sampler: str = None,
    override: bool = False,
):
    """
    Evaluate using torch-fidelity CLI.

    Uses the fidelity command to compute metrics directly.

    Args:
        method: 'ddpm'
        checkpoint: Path to checkpoint (relative to /data)
        metrics: Comma-separated: 'fid', 'kid', 'is' (default: 'fid,kid')
        num_samples: Number of samples to generate
        batch_size: Batch size
        num_steps: Sampling steps (optional)
        override: Force regenerate samples even if they exist
    """
    import sys
    import subprocess
    from pathlib import Path
    sys.path.insert(0, "/root")

    checkpoint_path = f"/data/{checkpoint}"

    # Put samples in same parent dir as checkpoint under samples/
    checkpoint_dir = Path(checkpoint_path).parent
    generated_dir = str(checkpoint_dir / "samples" / "generated")
    cache_dir = str(checkpoint_dir / "samples" / "cache")

    # Prepare dataset path for torch-fidelity
    # torch-fidelity needs actual image files, not Arrow format
    dataset_arrow_path = "/data/celeba"
    dataset_images_path = "/data/celeba_images"

    # Extract images from Arrow format if not already done
    import os
    if not os.path.exists(dataset_images_path):
        print("=" * 60)
        print("Extracting dataset images for torch-fidelity...")
        print("=" * 60)

        from datasets import load_from_disk

        dataset = load_from_disk(dataset_arrow_path)
        train_data = dataset['train']

        os.makedirs(dataset_images_path, exist_ok=True)

        print(f"Extracting {len(train_data)} images...")
        for idx, item in enumerate(train_data):
            img = item['image']
            img_path = os.path.join(dataset_images_path, f"{idx:06d}.png")
            img.save(img_path)

            if (idx + 1) % 1000 == 0:
                print(f"  Extracted {idx + 1}/{len(train_data)} images")

        volume.commit()
        print(f"Dataset images saved to {dataset_images_path}")
    else:
        print(f"Using cached dataset images at {dataset_images_path}")

    dataset_path = dataset_images_path

    # Step 1: Generate samples
    print("=" * 60)
    print("Step 1/2: Generating samples...")
    print("=" * 60)

    import os
    import shutil
    import glob

    # Check if samples already exist
    need_generation = True
    if os.path.exists(generated_dir) and not override:
        # Check for both png and jpg files
        existing_samples = (
            glob.glob(os.path.join(generated_dir, "*.png")) +
            glob.glob(os.path.join(generated_dir, "*.jpg")) + 
            glob.glob(os.path.join(generated_dir, "*.jpeg"))
        )
        num_existing = len(existing_samples)

        if num_existing >= num_samples:
            print(f"Found {num_existing} existing samples (need {num_samples})")
            print("Skipping sample generation (use --override to force regeneration)")
            need_generation = False
        else:
            print(f"Found {num_existing} existing samples but need {num_samples}")
            print("Regenerating samples...")
            shutil.rmtree(generated_dir)
    elif os.path.exists(generated_dir) and override:
        print("Override flag set, regenerating samples...")
        shutil.rmtree(generated_dir)

    if need_generation:
        sample_cmd = [
            "python", "/root/sample.py",
            "--checkpoint", checkpoint_path,
            "--method", method,
            "--output_dir", generated_dir,
            "--num_samples", str(num_samples),
            "--batch_size", str(batch_size),
        ]

        if num_steps:
            sample_cmd.extend(["--num_steps", str(num_steps)])
        if sampler:
            sample_cmd.extend(["--sampler", sampler])

        subprocess.run(sample_cmd, check=True)
        print(f"Generated {num_samples} samples to {generated_dir}")
    else:
        print(f"Using existing samples from {generated_dir}")

    # Save a 4x4 grid preview for quick inspection
    try:
        from PIL import Image as PILImage
        from torchvision import transforms
        from torchvision.utils import make_grid as torch_make_grid

        preview_paths = sorted(glob.glob(os.path.join(generated_dir, "*.png")))[:16]
        if preview_paths:
            images = [transforms.ToTensor()(PILImage.open(p).convert("RGB")) for p in preview_paths]
            grid = torch_make_grid(images, nrow=4)
            grid_path = str(Path(generated_dir).parent / "grid.png")
            torch_save_image = __import__("torchvision.utils", fromlist=["save_image"]).save_image
            torch_save_image(grid, grid_path)
            print(f"Saved grid preview to {grid_path}")
    except Exception as e:
        print(f"Warning: Failed to save grid preview: {e}")

    # Step 2: Run fidelity
    print("\n" + "=" * 60)
    print("Step 2/2: Running torch-fidelity...")
    print("=" * 60)

    os.makedirs(cache_dir, exist_ok=True)

    fidelity_cmd = [
        "fidelity",
        "--gpu", "0",
        "--batch-size", str(batch_size),
        "--cache-root", cache_dir,
        "--input1", generated_dir,
        "--input2", dataset_path,
    ]

    if "fid" in metrics:
        fidelity_cmd.append("--fid")
    if "kid" in metrics:
        fidelity_cmd.append("--kid")
    if "is" in metrics or "isc" in metrics:
        fidelity_cmd.append("--isc")

    print(f"\nRunning command: {' '.join(fidelity_cmd)}\n")

    try:
        result = subprocess.run(fidelity_cmd, check=True, capture_output=True, text=True)
        volume.commit()
        return result.stdout
    except subprocess.CalledProcessError as e:
        # Print the error output to help debug
        print(f"\nError running fidelity command:")
        print(f"Command: {' '.join(fidelity_cmd)}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"\nStdout:\n{e.stdout}")
        if e.stderr:
            print(f"\nStderr:\n{e.stderr}")
        raise


# =============================================================================
# CLI Entry Points
# =============================================================================

@app.local_entrypoint()
def main(
    action: str = "train",
    method: str = "ddpm",
    config: str = None,
    checkpoint: str = None,
    classifier_checkpoint: str = None,
    oracle_checkpoint: str = None,
    iterations: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    num_samples: int = None,
    num_steps: int = None,
    sampler: str = None,
    metrics: str = None,
    scales: str = None,
    attr_path: str = None,
    attr_name: str = None,
    attr_names: str = None,
    target_class_idx: int = None,
    target_class_indices: str = None,
    guidance_scale: float = None,
    guidance_mode: str = None,
    output: str = None,
    output_dir: str = None,
    no_grid: bool = False,
    report_classifier: bool = False,
    classifier_threshold: float = 0.5,
    report_all_attributes: bool = False,
    report_output: str = None,
    no_ema: bool = False,
    overfit_single_batch: bool = False,
    override: bool = False,
    max_items: int = None,
):
    """
    Main entry point for Modal CLI.

    See docs/QUICKSTART-MODAL.md for usage instructions.

    All parameters are read from config YAML files first, then overridden by command-line arguments.
    """
    if action == "download":
        result = download_dataset.remote()
        print(result)
    elif action == "train":
        # Read config to determine GPU count
        import yaml

        local_config_path = config or f"configs/{method}.yaml"
        with open(local_config_path, 'r') as f:
            local_config = yaml.safe_load(f)

        # Get num_gpus from config
        config_device = local_config['infrastructure'].get('device', 'cuda')
        num_gpus = local_config['infrastructure'].get('num_gpus', 1 if config_device == 'cuda' else 0)
        if num_gpus is None:
            num_gpus = 1 if config_device == 'cuda' else 0

        # Get the appropriate training function
        train_fn = TRAIN_FUNCTIONS.get(num_gpus)
        if train_fn is None:
            raise ValueError(
                f"Unsupported num_gpus={num_gpus} in config. "
                f"Supported: 1-8"
            )

        result = train_fn.remote(
            method=method,
            config_path=config,
            num_iterations=iterations,
            batch_size=batch_size,
            learning_rate=learning_rate,
            overfit_single_batch=overfit_single_batch,
        )
        print(result)
    elif action == "train_classifier":
        result = train_classifier.remote(
            config_path=config,
            num_iterations=iterations,
            batch_size=batch_size,
        )
        print(result)
    elif action == "sample_flow_guided":
        if checkpoint is None or classifier_checkpoint is None:
            raise ValueError("Both --checkpoint and --classifier-checkpoint are required.")
        result = sample_flow_guided.remote(
            flow_checkpoint=checkpoint,
            classifier_checkpoint=classifier_checkpoint,
            attr_name=attr_name,
            attr_names=attr_names,
            target_class_idx=target_class_idx,
            target_class_indices=target_class_indices,
            guidance_scale=guidance_scale,
            guidance_mode=guidance_mode,
            num_steps=num_steps,
            num_samples=num_samples,
            batch_size=batch_size,
            output=output,
            output_dir=output_dir,
            no_grid=no_grid,
            report_classifier=report_classifier,
            classifier_threshold=classifier_threshold,
            report_all_attributes=report_all_attributes,
            report_output=report_output,
            no_ema=no_ema,
        )
        print(result)
    elif action == "evaluate_guided_torch_fidelity":
        if checkpoint is None or classifier_checkpoint is None:
            raise ValueError("Both --checkpoint and --classifier-checkpoint are required.")
        result = evaluate_guided_torch_fidelity.remote(
            flow_checkpoint=checkpoint,
            classifier_checkpoint=classifier_checkpoint,
            attr_name=attr_name,
            attr_names=attr_names,
            target_class_idx=target_class_idx,
            target_class_indices=target_class_indices,
            guidance_scale=guidance_scale,
            guidance_mode=guidance_mode,
            num_steps=num_steps,
            num_samples=num_samples,
            batch_size=batch_size,
            metrics=metrics or "kid",
            output_dir=output_dir,
            report_classifier=report_classifier,
            classifier_threshold=classifier_threshold,
            report_all_attributes=report_all_attributes,
            report_output=report_output,
            override=override,
        )
        print(result)
    elif action == "inspect_classifier_checkpoint":
        if checkpoint is None:
            raise ValueError("--checkpoint is required.")
        result = inspect_classifier_checkpoint.remote(
            checkpoint=checkpoint,
            attr_name=attr_name,
        )
        print(result)
    elif action == "evaluate_guidance_overall":
        if checkpoint is None or classifier_checkpoint is None:
            raise ValueError("Both --checkpoint and --classifier-checkpoint are required.")
        result = evaluate_guidance_overall.remote(
            flow_checkpoint=checkpoint,
            classifier_checkpoint=classifier_checkpoint,
            attr_name=attr_name or "Smiling",
            guidance_scale=guidance_scale or 2.0,
            guidance_mode=guidance_mode or "fmps",
            num_steps=num_steps or 200,
            num_samples=num_samples or 1000,
            batch_size=batch_size or 128,
            max_items=max_items,
        )
        print(result)
    elif action == "train_oracle":
        result = train_oracle.remote(
            dataset_path=attr_path or "/data/celeba",
            epochs=iterations or 3,
            batch_size=batch_size or 128,
            lr=learning_rate or 1e-4,
            save_path=output or "/data/logs/oracle/resnet50_oracle.pt",
        )
        print(result)
    elif action == "evaluate_guidance_tradeoff":
        if checkpoint is None or classifier_checkpoint is None:
            raise ValueError("Both --checkpoint and --classifier-checkpoint are required.")
        result = evaluate_guidance_tradeoff.remote(
            flow_checkpoint=checkpoint,
            guidance_classifier_checkpoint=classifier_checkpoint,
            oracle_checkpoint=oracle_checkpoint or "/data/logs/oracle/resnet50_oracle.pt",
            attr_name=attr_name,
            attr_names=attr_names,
            scales=scales or "0,1.5,3.0,5.0,7.5",
            guidance_mode=guidance_mode or "fmps",
            num_steps=num_steps or 200,
            num_samples=num_samples or 1000,
            batch_size=batch_size or 128,
        )
        print(result)
    elif action == "sample":
        if checkpoint is None:
            checkpoint = f"checkpoints/{method}/{method}_final.pt"
        result = sample.remote(
            method=method,
            checkpoint=checkpoint,
            num_samples=num_samples,
            num_steps=num_steps,
        )
        print(result)
    elif action == "evaluate" or action == "evaluate_torch_fidelity":
        if checkpoint is None:
            checkpoint = f"checkpoints/{method}/{method}_final.pt"

        eval_kwargs = {
            'method': method,
            'checkpoint': checkpoint,
            'override': override,
        }
        if metrics is not None:
            eval_kwargs['metrics'] = metrics
        if num_samples is not None:
            eval_kwargs['num_samples'] = num_samples
        if batch_size is not None:
            eval_kwargs['batch_size'] = batch_size
        if num_steps is not None:
            eval_kwargs['num_steps'] = num_steps
        if sampler is not None:
            eval_kwargs['sampler'] = sampler

        result = evaluate_torch_fidelity.remote(**eval_kwargs)
        print(result)
    else:
        print(f"Unknown action: {action}")
        print("Valid actions: download, train, sample, evaluate")
