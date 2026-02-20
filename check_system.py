#!/usr/bin/env python3
"""
BBScore System Diagnostic Tool

This script checks if your machine can run BBScore benchmarks and provides
recommendations based on your hardware and software configuration.

Usage:
    python check_system.py                      # Full system check
    python check_system.py --model resnet50     # Check specific model
    python check_system.py --benchmark OnlineTVSDV1  # Check specific benchmark
    python check_system.py --quick              # Quick check (no model loading)
"""

import argparse
import sys
import os
import platform
import psutil
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

warnings.filterwarnings('ignore')


@dataclass
class SystemRequirements:
    """Minimum and recommended system requirements."""
    # Memory in GB
    min_ram: float = 8.0
    recommended_ram: float = 16.0

    # GPU memory in GB
    min_vram: float = 4.0
    recommended_vram: float = 8.0

    # Disk space in GB
    min_disk: float = 50.0
    recommended_disk: float = 100.0


# Model size estimates (parameters in millions, VRAM in GB for batch_size=4)
MODEL_SPECS = {
    # ResNet family
    "resnet18": {"params": 11.7, "vram": 2.0, "type": "image"},
    "resnet34": {"params": 21.8, "vram": 2.5, "type": "image"},
    "resnet50": {"params": 25.6, "vram": 3.0, "type": "image"},
    "resnet101": {"params": 44.5, "vram": 4.0, "type": "image"},
    "resnet152": {"params": 60.2, "vram": 5.0, "type": "image"},

    # ViT family
    "vit_base": {"params": 86.0, "vram": 4.0, "type": "image"},
    "vit_large": {"params": 304.0, "vram": 8.0, "type": "image"},
    "vit_huge": {"params": 632.0, "vram": 16.0, "type": "image"},

    # DINO family
    "dinov2_small": {"params": 22.0, "vram": 2.5, "type": "image"},
    "dinov2_base": {"params": 86.0, "vram": 4.0, "type": "image"},
    "dinov2_large": {"params": 304.0, "vram": 8.0, "type": "image"},
    "dinov2_giant": {"params": 1100.0, "vram": 24.0, "type": "image"},

    # CLIP family
    "clip_vit_b32": {"params": 151.0, "vram": 4.0, "type": "image"},
    "clip_vit_b16": {"params": 149.0, "vram": 5.0, "type": "image"},
    "clip_vit_l14": {"params": 428.0, "vram": 10.0, "type": "image"},

    # Video models
    "videomae_base": {"params": 87.0, "vram": 8.0, "type": "video"},
    "videomae_large": {"params": 305.0, "vram": 16.0, "type": "video"},
    "timesformer_base": {"params": 121.0, "vram": 10.0, "type": "video"},
    "vivit_base": {"params": 89.0, "vram": 8.0, "type": "video"},

    # 3D CNNs
    "r3d_18": {"params": 33.0, "vram": 4.0, "type": "video"},

    # MAE family
    "mae_base": {"params": 86.0, "vram": 4.0, "type": "image"},
    "mae_large": {"params": 304.0, "vram": 8.0, "type": "image"},

    # ConvNeXt
    "convnext_tiny": {"params": 29.0, "vram": 2.5, "type": "image"},
    "convnext_small": {"params": 50.0, "vram": 3.0, "type": "image"},
    "convnext_base": {"params": 89.0, "vram": 4.0, "type": "image"},
    "convnext_large": {"params": 198.0, "vram": 6.0, "type": "image"},

    # EfficientNet
    "efficientnet_b0": {"params": 5.3, "vram": 1.5, "type": "image"},
    "efficientnet_b4": {"params": 19.0, "vram": 2.5, "type": "image"},
    "efficientnet_b7": {"params": 66.0, "vram": 4.0, "type": "image"},

    # Default for unknown models
    "_default_image": {"params": 100.0, "vram": 6.0, "type": "image"},
    "_default_video": {"params": 100.0, "vram": 10.0, "type": "video"},
}

# Benchmark requirements (RAM in GB, disk in GB, samples)
BENCHMARK_SPECS = {
    # NSD benchmarks
    "NSDShared": {"ram": 16.0, "disk": 30.0, "samples": 10000, "type": "image", "online": False},
    "NSDV1Shared": {"ram": 16.0, "disk": 30.0, "samples": 10000, "type": "image", "online": False},
    "NSDV4Shared": {"ram": 16.0, "disk": 30.0, "samples": 10000, "type": "image", "online": False},

    # TVSD benchmarks (video)
    "TVSDFull": {"ram": 32.0, "disk": 50.0, "samples": 5000, "type": "video", "online": False},
    "TVSDV1": {"ram": 16.0, "disk": 50.0, "samples": 5000, "type": "video", "online": False},
    "TVSDV4": {"ram": 16.0, "disk": 50.0, "samples": 5000, "type": "video", "online": False},
    "TVSDIT": {"ram": 16.0, "disk": 50.0, "samples": 5000, "type": "video", "online": False},

    # Online TVSD
    "OnlineTVSDFull": {"ram": 8.0, "disk": 50.0, "samples": 5000, "type": "video", "online": True},
    "OnlineTVSDV1": {"ram": 8.0, "disk": 50.0, "samples": 5000, "type": "video", "online": True},
    "OnlineTVSDV4": {"ram": 8.0, "disk": 50.0, "samples": 5000, "type": "video", "online": True},
    "OnlineTVSDIT": {"ram": 8.0, "disk": 50.0, "samples": 5000, "type": "video", "online": True},

    # Physion
    "PhysionContact": {"ram": 8.0, "disk": 20.0, "samples": 2000, "type": "video", "online": False},
    "PhysionPlacement": {"ram": 8.0, "disk": 20.0, "samples": 2000, "type": "video", "online": False},
    "OnlinePhysionContactDetection": {"ram": 4.0, "disk": 20.0, "samples": 2000, "type": "video", "online": True},
    "OnlinePhysionPlacementDetection": {"ram": 4.0, "disk": 20.0, "samples": 2000, "type": "video", "online": True},

    # SSV2
    "SSV2Benchmark": {"ram": 16.0, "disk": 100.0, "samples": 20000, "type": "video", "online": True},

    # V1 Gratings
    "V1SineGratingsBenchmark": {"ram": 4.0, "disk": 1.0, "samples": 500, "type": "image", "online": False},

    # LeBel (language/fMRI)
    "LeBel2023": {"ram": 8.0, "disk": 10.0, "samples": 84, "type": "text", "online": False},

    # Default
    "_default": {"ram": 16.0, "disk": 50.0, "samples": 5000, "type": "image", "online": False},
}

# Metric requirements
METRIC_SPECS = {
    "ridge": {"ram_multiplier": 2.0, "gpu_required": False, "description": "Ridge regression (sklearn)"},
    "torch_ridge": {"ram_multiplier": 1.5, "gpu_required": True, "description": "Ridge regression (PyTorch)"},
    "pls": {"ram_multiplier": 2.5, "gpu_required": False, "description": "Partial Least Squares"},
    "rsa": {"ram_multiplier": 3.0, "gpu_required": False, "description": "Representational Similarity Analysis"},
    "online_linear_regressor": {"ram_multiplier": 1.0, "gpu_required": True, "description": "Online ridge with SGD"},
    "online_linear_classifier": {"ram_multiplier": 1.0, "gpu_required": True, "description": "Online linear classifier"},
    "online_transformer_classifier": {"ram_multiplier": 1.5, "gpu_required": True, "description": "Transformer classifier"},
    "_default": {"ram_multiplier": 2.0, "gpu_required": False, "description": "Unknown metric"},
}


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

    @classmethod
    def disable(cls):
        cls.GREEN = cls.YELLOW = cls.RED = cls.BLUE = cls.BOLD = cls.END = ''


def get_gpu_info() -> List[Dict]:
    """Get GPU information using PyTorch."""
    gpus = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / (1024**3)
                # Get free memory
                torch.cuda.set_device(i)
                free_mem = (props.total_memory -
                            torch.cuda.memory_allocated(i)) / (1024**3)
                gpus.append({
                    "name": props.name,
                    "total_memory_gb": round(total_mem, 2),
                    "free_memory_gb": round(free_mem, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                })
    except Exception:
        pass
    return gpus


def get_system_info() -> Dict:
    """Collect system information."""
    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
    }

    # Get disk space for SCIKIT_LEARN_DATA or current directory
    data_dir = os.environ.get('SCIKIT_LEARN_DATA', os.getcwd())
    try:
        disk = psutil.disk_usage(data_dir)
        info["disk_total_gb"] = round(disk.total / (1024**3), 2)
        info["disk_free_gb"] = round(disk.free / (1024**3), 2)
        info["data_directory"] = data_dir
    except Exception:
        info["disk_total_gb"] = 0
        info["disk_free_gb"] = 0
        info["data_directory"] = "unknown"

    # GPU info
    info["gpus"] = get_gpu_info()

    return info


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are installed."""
    deps = {}

    required = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("tqdm", "tqdm"),
        ("h5py", "h5py"),
        ("transformers", "HuggingFace Transformers"),
        ("timm", "PyTorch Image Models"),
    ]

    optional = [
        ("cuml", "RAPIDS cuML (GPU acceleration)"),
        ("wandb", "Weights & Biases"),
        ("boto3", "AWS SDK (for S3 datasets)"),
    ]

    for module, name in required:
        try:
            __import__(module)
            deps[name] = True
        except ImportError:
            deps[name] = False

    deps["_optional"] = {}
    for module, name in optional:
        try:
            __import__(module)
            deps["_optional"][name] = True
        except ImportError:
            deps["_optional"][name] = False

    return deps


def get_model_spec(model_name: str) -> Dict:
    """Get model specifications."""
    # Try exact match
    if model_name in MODEL_SPECS:
        return MODEL_SPECS[model_name]

    # Try partial match
    model_lower = model_name.lower()
    for key, spec in MODEL_SPECS.items():
        if key in model_lower or model_lower in key:
            return spec

    # Determine if video model
    video_keywords = ["video", "3d", "slowfast", "i3d",
                      "vivit", "timesformer", "r3d", "mae_video"]
    if any(kw in model_lower for kw in video_keywords):
        return MODEL_SPECS["_default_video"]

    return MODEL_SPECS["_default_image"]


def get_benchmark_spec(benchmark_name: str) -> Dict:
    """Get benchmark specifications."""
    if benchmark_name in BENCHMARK_SPECS:
        return BENCHMARK_SPECS[benchmark_name]

    # Try partial match
    for key, spec in BENCHMARK_SPECS.items():
        if key.lower() in benchmark_name.lower() or benchmark_name.lower() in key.lower():
            return spec

    return BENCHMARK_SPECS["_default"]


def get_metric_spec(metric_name: str) -> Dict:
    """Get metric specifications."""
    if metric_name in METRIC_SPECS:
        return METRIC_SPECS[metric_name]
    return METRIC_SPECS["_default"]


def estimate_memory_requirements(
    model_name: str,
    benchmark_name: str,
    metric_name: str,
    batch_size: int = 4
) -> Dict:
    """Estimate memory requirements for a specific configuration."""
    model_spec = get_model_spec(model_name)
    benchmark_spec = get_benchmark_spec(benchmark_name)
    metric_spec = get_metric_spec(metric_name)

    # VRAM estimate (scales with batch size)
    base_vram = model_spec["vram"]
    vram_estimate = base_vram * (batch_size / 4)

    # RAM estimate
    ram_for_data = benchmark_spec["ram"]
    ram_multiplier = metric_spec["ram_multiplier"]
    ram_estimate = ram_for_data * ram_multiplier

    # Feature storage estimate (rough)
    # Assume ~1KB per sample per 1000 features
    feature_dim = model_spec["params"] * 10  # Rough estimate
    feature_storage_mb = (
        benchmark_spec["samples"] * feature_dim * 4) / (1024 * 1024)

    return {
        "vram_gb": round(vram_estimate, 2),
        "ram_gb": round(ram_estimate, 2),
        "disk_gb": benchmark_spec["disk"],
        "feature_storage_mb": round(feature_storage_mb, 2),
        "gpu_required": metric_spec["gpu_required"] or model_spec["type"] == "video",
        "model_type": model_spec["type"],
        "benchmark_type": benchmark_spec["type"],
        "is_online": benchmark_spec["online"],
    }


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_status(name: str, status: bool, details: str = ""):
    """Print a status line."""
    icon = f"{Colors.GREEN}✓{Colors.END}" if status else f"{Colors.RED}✗{Colors.END}"
    detail_str = f" ({details})" if details else ""
    print(f"  {icon} {name}{detail_str}")


def print_warning(text: str):
    """Print a warning message."""
    print(f"  {Colors.YELLOW}⚠ {text}{Colors.END}")


def print_recommendation(text: str):
    """Print a recommendation."""
    print(f"  {Colors.BLUE}→ {text}{Colors.END}")


def generate_report(
    system_info: Dict,
    deps: Dict,
    model_name: Optional[str] = None,
    benchmark_name: Optional[str] = None,
    metric_name: str = "ridge",
    batch_size: int = 4,
) -> Dict:
    """Generate a comprehensive compatibility report."""

    report = {
        "can_run": True,
        "warnings": [],
        "recommendations": [],
        "details": {},
    }

    reqs = SystemRequirements()

    # Check RAM
    ram_available = system_info["ram_available_gb"]
    if ram_available < reqs.min_ram:
        report["can_run"] = False
        report["warnings"].append(
            f"Insufficient RAM: {ram_available:.1f}GB available, {reqs.min_ram}GB minimum required")
    elif ram_available < reqs.recommended_ram:
        report["warnings"].append(
            f"Low RAM: {ram_available:.1f}GB available, {reqs.recommended_ram}GB recommended")

    # Check GPU
    gpus = system_info["gpus"]
    has_gpu = len(gpus) > 0
    max_vram = max([g["total_memory_gb"] for g in gpus]) if gpus else 0

    report["details"]["has_gpu"] = has_gpu
    report["details"]["max_vram_gb"] = max_vram

    # Check dependencies
    missing_deps = [k for k, v in deps.items() if k != "_optional" and not v]
    if missing_deps:
        report["can_run"] = False
        report["warnings"].append(
            f"Missing required dependencies: {', '.join(missing_deps)}")

    # Check disk space
    disk_free = system_info["disk_free_gb"]
    if disk_free < reqs.min_disk:
        report["warnings"].append(
            f"Low disk space: {disk_free:.1f}GB free, {reqs.min_disk}GB recommended")

    # Check specific configuration if provided
    if model_name and benchmark_name:
        mem_reqs = estimate_memory_requirements(
            model_name, benchmark_name, metric_name, batch_size)
        report["details"]["estimated_requirements"] = mem_reqs

        # Check VRAM
        if mem_reqs["gpu_required"] and not has_gpu:
            report["can_run"] = False
            report["warnings"].append("GPU required but no GPU detected")
        elif mem_reqs["gpu_required"] and max_vram < mem_reqs["vram_gb"]:
            report["warnings"].append(
                f"GPU memory may be insufficient: {max_vram:.1f}GB available, "
                f"{mem_reqs['vram_gb']:.1f}GB estimated for batch_size={batch_size}"
            )
            report["recommendations"].append(
                f"Try reducing batch_size (current: {batch_size})")

        # Check RAM
        if ram_available < mem_reqs["ram_gb"]:
            report["warnings"].append(
                f"RAM may be insufficient: {ram_available:.1f}GB available, "
                f"{mem_reqs['ram_gb']:.1f}GB estimated"
            )

        # Check disk
        if disk_free < mem_reqs["disk_gb"]:
            report["warnings"].append(
                f"Disk space may be insufficient: {disk_free:.1f}GB free, "
                f"{mem_reqs['disk_gb']:.1f}GB required for dataset"
            )

        # Online benchmark recommendation
        if not mem_reqs["is_online"] and ram_available < 16:
            report["recommendations"].append(
                "Consider using Online* benchmarks which have lower memory requirements"
            )

    # General recommendations
    if not has_gpu:
        report["recommendations"].append(
            "GPU highly recommended for faster training")
        report["recommendations"].append(
            "Use 'ridge' metric instead of 'online_linear_regressor' for CPU-only")

    if not deps.get("_optional", {}).get("RAPIDS cuML (GPU acceleration)", False):
        if has_gpu:
            report["recommendations"].append(
                "Install cuML for 50x faster ridge regression on GPU")

    return report


def print_full_report(
    system_info: Dict,
    deps: Dict,
    report: Dict,
    model_name: Optional[str] = None,
    benchmark_name: Optional[str] = None,
    metric_name: str = "ridge",
):
    """Print the full diagnostic report."""

    print_header("BBScore System Diagnostic Report")

    # System Info
    print(f"{Colors.BOLD}System Information:{Colors.END}")
    print(f"  OS: {system_info['platform']} ({system_info['architecture']})")
    print(f"  Python: {system_info['python_version']}")
    print(
        f"  CPU: {system_info['cpu_count']} cores ({system_info['cpu_count_logical']} threads)")
    print(
        f"  RAM: {system_info['ram_available_gb']:.1f}GB available / {system_info['ram_total_gb']:.1f}GB total")
    print(
        f"  Disk: {system_info['disk_free_gb']:.1f}GB free / {system_info['disk_total_gb']:.1f}GB total")
    print(f"  Data Directory: {system_info['data_directory']}")

    # GPU Info
    print(f"\n{Colors.BOLD}GPU Information:{Colors.END}")
    if system_info["gpus"]:
        for i, gpu in enumerate(system_info["gpus"]):
            print(f"  GPU {i}: {gpu['name']}")
            print(
                f"    Memory: {gpu['free_memory_gb']:.1f}GB free / {gpu['total_memory_gb']:.1f}GB total")
            print(f"    Compute Capability: {gpu['compute_capability']}")
    else:
        print(f"  {Colors.YELLOW}No GPU detected{Colors.END}")

    # Dependencies
    print(f"\n{Colors.BOLD}Required Dependencies:{Colors.END}")
    for name, installed in deps.items():
        if name == "_optional":
            continue
        print_status(name, installed)

    print(f"\n{Colors.BOLD}Optional Dependencies:{Colors.END}")
    for name, installed in deps.get("_optional", {}).items():
        print_status(name, installed, "optional")

    # Configuration-specific analysis
    if model_name and benchmark_name:
        print(f"\n{Colors.BOLD}Configuration Analysis:{Colors.END}")
        print(f"  Model: {model_name}")
        print(f"  Benchmark: {benchmark_name}")
        print(f"  Metric: {metric_name}")

        if "estimated_requirements" in report["details"]:
            reqs = report["details"]["estimated_requirements"]
            print(f"\n{Colors.BOLD}Estimated Requirements:{Colors.END}")
            print(f"  VRAM: ~{reqs['vram_gb']:.1f}GB")
            print(f"  RAM: ~{reqs['ram_gb']:.1f}GB")
            print(f"  Disk: ~{reqs['disk_gb']:.1f}GB")
            print(f"  GPU Required: {'Yes' if reqs['gpu_required'] else 'No'}")

    # Warnings
    if report["warnings"]:
        print(f"\n{Colors.BOLD}Warnings:{Colors.END}")
        for warning in report["warnings"]:
            print_warning(warning)

    # Recommendations
    if report["recommendations"]:
        print(f"\n{Colors.BOLD}Recommendations:{Colors.END}")
        for rec in report["recommendations"]:
            print_recommendation(rec)

    # Final verdict
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    if report["can_run"]:
        print(
            f"{Colors.GREEN}{Colors.BOLD}✓ Your system can run BBScore benchmarks{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Your system may not be able to run BBScore benchmarks{Colors.END}")
        print(f"  Please address the warnings above before proceeding.")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}\n")


def list_available_options():
    """List available models, benchmarks, and metrics."""
    print_header("Available Options")

    # Try to load registries
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from benchmarks import BENCHMARK_REGISTRY
        from models import MODEL_REGISTRY
        from metrics import METRICS

        print(f"{Colors.BOLD}Benchmarks ({len(BENCHMARK_REGISTRY)}):{Colors.END}")
        # Group by type
        online = [k for k in sorted(
            BENCHMARK_REGISTRY.keys()) if "Online" in k]
        offline = [k for k in sorted(
            BENCHMARK_REGISTRY.keys()) if "Online" not in k]

        print(
            f"  Online (lower memory): {', '.join(online[:10])}{'...' if len(online) > 10 else ''}")
        print(
            f"  Standard: {', '.join(offline[:10])}{'...' if len(offline) > 10 else ''}")

        print(f"\n{Colors.BOLD}Models ({len(MODEL_REGISTRY)}):{Colors.END}")
        models = sorted(MODEL_REGISTRY.keys())
        print(f"  {', '.join(models[:15])}{'...' if len(models) > 15 else ''}")

        print(f"\n{Colors.BOLD}Metrics ({len(METRICS)}):{Colors.END}")
        for name in sorted(METRICS.keys()):
            spec = get_metric_spec(name)
            gpu_tag = "[GPU]" if spec["gpu_required"] else "[CPU]"
            print(f"  {name}: {spec['description']} {gpu_tag}")

    except Exception as e:
        print(f"{Colors.RED}Could not load registries: {e}{Colors.END}")
        print("Run from the bbscore_public directory.")


def quick_check():
    """Perform a quick system check without loading models."""
    print_header("Quick System Check")

    system_info = get_system_info()

    # Quick RAM check
    ram = system_info["ram_available_gb"]
    if ram >= 16:
        print_status(
            "RAM", True, f"{ram:.1f}GB available - Good for most benchmarks")
    elif ram >= 8:
        print_status(
            "RAM", True, f"{ram:.1f}GB available - OK for Online benchmarks")
    else:
        print_status(
            "RAM", False, f"{ram:.1f}GB available - May be insufficient")

    # Quick GPU check
    gpus = system_info["gpus"]
    if gpus:
        max_vram = max([g["total_memory_gb"] for g in gpus])
        if max_vram >= 8:
            print_status(
                "GPU", True, f"{gpus[0]['name']} ({max_vram:.1f}GB) - Good for most models")
        elif max_vram >= 4:
            print_status(
                "GPU", True, f"{gpus[0]['name']} ({max_vram:.1f}GB) - OK for smaller models")
        else:
            print_status(
                "GPU", True, f"{gpus[0]['name']} ({max_vram:.1f}GB) - Limited, use small batch sizes")
    else:
        print_status("GPU", False, "No GPU - Will use CPU (slower)")

    # Quick disk check
    disk = system_info["disk_free_gb"]
    if disk >= 100:
        print_status(
            "Disk", True, f"{disk:.1f}GB free - Good for all datasets")
    elif disk >= 50:
        print_status("Disk", True, f"{disk:.1f}GB free - OK for most datasets")
    else:
        print_status("Disk", False, f"{disk:.1f}GB free - May need more space")

    # Recommendations
    print(f"\n{Colors.BOLD}Quick Recommendations:{Colors.END}")

    if not gpus:
        print_recommendation(
            "Use 'ridge' metric instead of online metrics (no GPU required)")
        print_recommendation(
            "Consider using Google Colab or cloud compute for GPU access")

    if ram < 16:
        print_recommendation(
            "Use Online* benchmarks (e.g., OnlineTVSDV1) for lower memory usage")
        print_recommendation(
            "Use smaller models (e.g., resnet18, efficientnet_b0)")

    print_recommendation(
        "Set SCIKIT_LEARN_DATA environment variable to a directory with enough space")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="BBScore System Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_system.py                              # Full system check
  python check_system.py --quick                      # Quick check
  python check_system.py --model resnet50 --benchmark OnlineTVSDV1
  python check_system.py --list                       # List available options
  python check_system.py --json                       # Output as JSON
        """
    )

    parser.add_argument("--model", "-m", type=str,
                        help="Model to check (e.g., resnet50, dinov2_base)")
    parser.add_argument("--benchmark", "-b", type=str,
                        help="Benchmark to check (e.g., OnlineTVSDV1)")
    parser.add_argument("--metric", type=str, default="ridge",
                        help="Metric to use (default: ridge)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick check without detailed analysis")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available models/benchmarks/metrics")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output report as JSON")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")

    args = parser.parse_args()

    if args.no_color:
        Colors.disable()

    if args.list:
        list_available_options()
        return

    if args.quick:
        quick_check()
        return

    # Full check
    print("Collecting system information...")
    system_info = get_system_info()

    print("Checking dependencies...")
    deps = check_dependencies()

    print("Generating report...")
    report = generate_report(
        system_info,
        deps,
        model_name=args.model,
        benchmark_name=args.benchmark,
        metric_name=args.metric,
        batch_size=args.batch_size,
    )

    if args.json:
        output = {
            "system_info": system_info,
            "dependencies": deps,
            "report": report,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_full_report(
            system_info,
            deps,
            report,
            model_name=args.model,
            benchmark_name=args.benchmark,
            metric_name=args.metric,
        )


if __name__ == "__main__":
    main()
