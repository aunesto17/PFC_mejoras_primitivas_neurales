"""Gaussian-activated NeRF training with preconditioner optimizers.

Trains a Gaussian-activated NeRF (8 layers, 256 wide, sigma=0.1, no PE)
on LLFF or NeRF Synthetic datasets using Adam, ESGD, ESGD_Max, or AdaHessian.

Based on: "Preconditioners for the Stochastic Training of Neural Fields"
(Chng et al., 2024)

ENTRENAMIENTO DE PRECONDICIONADORES EN GAUSSIAN NERF
=====================================================
Script de entrenamiento para la tecnica de Preconditioners.
Entrena un NeRF basado en activaciones Gaussianas (sin codificacion posicional)
con optimizadores conscientes de curvatura.

ARQUITECTURA: MLP de 8 capas, 256 neuronas por capa, activacion Gaussiana
(exp(-0.5*x^2/sigma^2) con sigma=0.1), skip connection en capa 4.
Usa OccGridEstimator (grilla de ocupacion, resolucion 64) para muestreo eficiente.

OPTIMIZADORES SOPORTADOS:
- adam:       Adam estandar (LR=1e-4, step scheduler). Baseline para comparacion.
- esgd:       Equilibrated SGD con precondicionador diagonal basado en productos
              Hessiano-vector (HVP) via traza de Hutchinson.
- esgd_max:   Variante de ESGD que usa maximo historico en lugar de EMA para
              la estimacion de la diagonal Hessiana. Mejor convergencia en practica.
- adahessian: AdaHessian con precondicionador Jacobi. Mayor consumo de memoria
              (retain_graph=True), no utilizable en GPU de 8GB.

MEMORIA: ESGD/ESGD_Max requieren create_graph=True para calcular HVP, lo que
duplica el consumo de VRAM (~2x vs Adam). Batch size limitado a 1024 en lugar de 2048.
Se ejecuta torch.cuda.empty_cache() cada 10 pasos.

En nuestro trabajo, este script se uso para entrenar Gaussian NeRF en la escena
fern con Adam y ESGD_Max (100K iteraciones). NOTA: ESGD_Max no converge en esta
implementacion (PSNR ~12), posiblemente por diferencias en el scheduler/hiperparametros.
Los resultados funcionales se obtuvieron con la implementacion GARF (directorio
garf_preconditioners/).
"""
import argparse
import json
import pathlib
import time
import sys
import os

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from radiance_fields.gaussian_nerf import GaussianNeRFRadianceField

from examples.utils import (
    NERF_SYNTHETIC_SCENES,
    LLFF_NDC_SCENES,
    render_image_with_occgrid,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator

# Optimizer imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "optimizers"))
import esgd as esgd_module
import esgd_max as esgd_max_module
from precond_adahessian import adahessian as adahessian_module

# Scheduler import
from schedulers import create_scheduler

# SSIM import
try:
    from pytorch_msssim import ssim as compute_ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    print("WARNING: pytorch_msssim not available, SSIM will not be computed")

# ---------- CLI arguments ----------
parser = argparse.ArgumentParser(
    description="Gaussian-activated NeRF training with preconditioner optimizers."
)
parser.add_argument("--data_root", type=str, default=None,
                    help="Root dir of the dataset (auto-detected if not set)")
parser.add_argument("--train_split", type=str, default="train",
                    choices=["train", "trainval"])
parser.add_argument("--scene", type=str, default="fern",
                    choices=NERF_SYNTHETIC_SCENES + LLFF_NDC_SCENES)
parser.add_argument("--test_chunk_size", type=int, default=4096)
parser.add_argument("--i_print", type=int, default=1000,
                    help="Log training metrics every N steps")
parser.add_argument("--i_eval", type=str, default="1000,5000,10000,20000,50000",
                    help="Comma-separated list of steps at which to run full evaluation")
parser.add_argument("--scheduler", type=str, default="step",
                    choices=["cosineannealing", "multistep", "chain", "exponential", "step", "none"])
parser.add_argument("--end_lr", type=float, default=None,
                    help="Final LR for exponential scheduler (auto-set if not given)")
parser.add_argument("--optimizer", type=str, default="adam",
                    choices=["adam", "esgd", "esgd_max", "adahessian"],
                    help="Optimizer to use")
parser.add_argument("--lr", type=float, default=None,
                    help="Learning rate (overrides default for optimizer)")
parser.add_argument("--batch_size", type=int, default=None,
                    help="Rays per training step (default: auto)")
parser.add_argument("--max_steps", type=int, default=None,
                    help="Max training steps (default: 200000 for LLFF, 50000 for synthetic)")
parser.add_argument("--esgd_update_every", type=int, default=100,
                    help="ESGD: update Hessian diagonal every N steps")
parser.add_argument("--esgd_d_warmup", type=int, default=50,
                    help="ESGD: always compute Hessian for first N steps")
parser.add_argument("--weight_decay", type=float, default=1e-8,
                    help="Weight decay for optimizer")
parser.add_argument("--result_file", type=str, default=None,
                    help="File to append final results (for batch scripts)")
args = parser.parse_args()

# Parse i_eval into a set of checkpoint steps
eval_checkpoints = sorted(set(int(x.strip()) for x in args.i_eval.split(",")))

# ---------- Scene detection ----------
device = "cuda:0"
set_random_seed(42)

is_llff = args.scene in LLFF_NDC_SCENES
is_synthetic = args.scene in NERF_SYNTHETIC_SCENES

# Default data root
if args.data_root is None:
    if is_llff:
        args.data_root = str(pathlib.Path(__file__).resolve().parent.parent.parent / "nerf_llff_data")
    else:
        args.data_root = str(pathlib.Path(__file__).resolve().parent.parent / "data" / "nerf_synthetic")

# ---------- Scene configs ----------
# Default batch size: 2048 for Adam, 1024 for preconditioner optimizers (memory)
default_batch = 1024 if args.optimizer in ("esgd", "esgd_max", "adahessian") else 2048

if is_llff:
    from datasets.nerf_llff import SubjectLoader
    max_steps = args.max_steps or 200000
    init_batch_size = args.batch_size or default_batch
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 0.0
    far_plane = 1.0e10
    render_step_size = 5e-3
    color_bkgd_aug = "gray"
    opaque_bkgd = False
    train_dataset_kwargs = {"factor": 4}
    test_dataset_kwargs = {"factor": 4}
elif is_synthetic:
    from datasets.nerf_synthetic import SubjectLoader
    max_steps = args.max_steps or 50000
    init_batch_size = args.batch_size or default_batch
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 2.0
    far_plane = 6.0
    render_step_size = 5e-3
    color_bkgd_aug = "white"
    opaque_bkgd = False
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
else:
    raise ValueError(f"Unknown scene: {args.scene}")

# Target sample batch size for dynamic batching
# Reduced from 65536 to fit in GPU memory with the 8-layer Gaussian MLP
target_sample_batch_size = 1 << 14  # 16384

# ---------- Dataset ----------
train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    color_bkgd_aug=color_bkgd_aug,
    device=device,
    **train_dataset_kwargs,
)
test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    color_bkgd_aug=color_bkgd_aug,
    device=device,
    **test_dataset_kwargs,
)

# ---------- Occupancy grid ----------
# Resolution 64 instead of 128 — Gaussian MLP uses more memory per point
# than ReLU MLP, and 128^3 points causes OOM on 7.78GB GPU
estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=64, levels=1
).to(device)

def occ_eval_fn(x):
    density = radiance_field.query_density(x)
    return density * render_step_size

# ---------- Radiance field ----------
radiance_field = GaussianNeRFRadianceField().to(device)
num_params = sum(p.numel() for p in radiance_field.parameters())
print(f"GaussianNeRF params: {num_params:,}")

# ---------- Optimizer ----------
# Default LRs from the paper (Section 4.3.1 for LLFF, 4.3.2 for Blender)
# Adam: 1e-4 (LLFF: step schedule; Blender: exponential 1e-4 -> 1e-6)
# ESGD: 1.0 (LLFF: exponential 1 -> 0.01; Blender: exponential 1 -> 0.1)
lr_map = {"adam": 1e-4, "esgd": 1.0, "esgd_max": 1.0, "adahessian": 0.15}
lr = args.lr or lr_map[args.optimizer]

if args.optimizer == "adam":
    optimizer = torch.optim.Adam(
        radiance_field.parameters(), lr=lr, eps=1e-15,
        weight_decay=args.weight_decay,
    )
elif args.optimizer == "esgd":
    optimizer = esgd_module.ESGD(
        radiance_field.parameters(), lr=lr, eps=1e-4,
        weight_decay=args.weight_decay,
        update_d_every=args.esgd_update_every,
        d_warmup=args.esgd_d_warmup,
        preconditioner_type="equilbrated",
    )
elif args.optimizer == "esgd_max":
    optimizer = esgd_max_module.ESGD_Max(
        radiance_field.parameters(), lr=lr, eps=1e-4,
        weight_decay=args.weight_decay,
        update_d_every=args.esgd_update_every,
        d_warmup=args.esgd_d_warmup,
        preconditioner_type="equilbrated",
    )
elif args.optimizer == "adahessian":
    optimizer = adahessian_module.Adahessian(
        radiance_field.parameters(), lr=lr, eps=1e-4,
        weight_decay=args.weight_decay,
    )
else:
    raise ValueError(f"Unknown optimizer: {args.optimizer}")

# ---------- Scheduler ----------
# Auto-set end_lr for exponential scheduler based on paper defaults
if args.end_lr is not None:
    end_lr = args.end_lr
elif args.scheduler == "exponential":
    # Paper defaults: ESGD 1->0.01 (LLFF) or 1->0.1 (Blender)
    if is_llff:
        end_lr_map = {"adam": 1e-6, "esgd": 0.01, "esgd_max": 0.01, "adahessian": 0.01}
    else:
        end_lr_map = {"adam": 1e-6, "esgd": 0.1, "esgd_max": 0.1, "adahessian": 0.1}
    end_lr = end_lr_map[args.optimizer]
else:
    end_lr = None

scheduler = create_scheduler(optimizer, args.scheduler, max_steps, lr, end_lr=end_lr)

# ---------- LPIPS ----------
lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

# ---------- Helpers ----------
def compute_psnr(mse):
    return -10.0 * torch.log(mse) / np.log(10.0)

def compute_metrics_for_view(rgb, pixels, h=None, w=None):
    """Compute PSNR, SSIM, LPIPS for a single view."""
    mse_val = F.mse_loss(rgb, pixels)
    psnr = compute_psnr(mse_val).item()
    lpips_val = lpips_fn(rgb, pixels).item()
    ssim_val = None
    if HAS_SSIM:
        if h is None or w is None:
            h, w = pixels.shape[:2] if pixels.dim() == 3 else (int(pixels.shape[0]**0.5), int(pixels.shape[0]**0.5))
        rgb_img = rgb.view(1, -1, 3).permute(0, 2, 1).view(1, 3, h, w).clamp(0, 1)
        pix_img = pixels.view(1, -1, 3).permute(0, 2, 1).view(1, 3, h, w).clamp(0, 1)
        ssim_val = compute_ssim(rgb_img, pix_img, data_range=1.0).item()
    return psnr, ssim_val, lpips_val

def evaluate_on_test_set():
    """Run full evaluation on test set. Returns avg psnr, ssim, lpips."""
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    radiance_field.eval()
    estimator.eval()
    psnrs, ssims, lpips_vals = [], [], []
    with torch.no_grad():
        for j in tqdm.tqdm(range(len(test_dataset)), desc=f"eval", leave=False):
            data = test_dataset[j]
            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]
            rgb, acc, depth, _ = render_image_with_occgrid(
                radiance_field, estimator, rays,
                near_plane=near_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                test_chunk_size=args.test_chunk_size,
            )
            h, w = pixels.shape[:2] if pixels.dim() == 3 else (int(pixels.shape[0]**0.5), int(pixels.shape[0]**0.5))
            psnr_v, ssim_v, lpips_v = compute_metrics_for_view(rgb, pixels, h, w)
            psnrs.append(psnr_v)
            lpips_vals.append(lpips_v)
            if ssim_v is not None:
                ssims.append(ssim_v)
    result = {
        "psnr": sum(psnrs) / len(psnrs),
        "lpips": sum(lpips_vals) / len(lpips_vals),
    }
    if ssims:
        result["ssim"] = sum(ssims) / len(ssims)
    return result

is_precond = args.optimizer in ("esgd", "esgd_max", "adahessian")

print(f"\nTraining: scene={args.scene}, optimizer={args.optimizer}, lr={lr}, "
      f"max_steps={max_steps}, dataset={'LLFF' if is_llff else 'Synthetic'}")
print(f"Evaluation checkpoints: {eval_checkpoints}\n")

# ---------- Results log ----------
output_dir = pathlib.Path(__file__).resolve().parent.parent / "outputs" / args.scene
output_dir.mkdir(parents=True, exist_ok=True)
results_log = output_dir / f"gaussian_nerf_{args.optimizer}_results.txt"

# ---------- Metrics storage ----------
train_curve = []           # [{step, loss, psnr, time_s}, ...]
checkpoint_metrics = {}    # {step_str: {psnr, ssim, lpips, time_s}, ...}

# ---------- Training loop ----------
tic = time.time()

for step in range(max_steps + 1):
    radiance_field.train()
    estimator.train()

    # Sample a random training image
    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]

    # Update occupancy grid (no_grad)
    estimator.update_every_n_steps(
        step=step,
        occ_eval_fn=occ_eval_fn,
        occ_thre=1e-2,
    )

    # Render
    rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
        radiance_field,
        estimator,
        rays,
        near_plane=near_plane,
        render_step_size=render_step_size,
        render_bkgd=render_bkgd,
    )
    if n_rendering_samples == 0:
        continue

    # Dynamic batch size
    if target_sample_batch_size > 0:
        num_rays = len(pixels)
        num_rays = int(num_rays * (target_sample_batch_size / float(n_rendering_samples)))
        train_dataset.update_num_rays(num_rays)

    # Loss
    loss = F.mse_loss(rgb, pixels)

    # Backward + optimizer step
    optimizer.zero_grad(set_to_none=True)

    if is_precond:
        if args.optimizer == "adahessian":
            loss.backward(create_graph=True)
        else:
            should_create = optimizer.should_create_graph()
            loss.backward(create_graph=should_create)
    else:
        loss.backward()

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    # Free computation graph memory (especially important for AdaHessian)
    if is_precond and step % 10 == 0:
        torch.cuda.empty_cache()

    # Logging (every i_print steps)
    if step % args.i_print == 0:
        elapsed = time.time() - tic
        mse_val = F.mse_loss(rgb, pixels)
        psnr_val = compute_psnr(mse_val)
        print(
            f"[step {step:>6d}/{max_steps}] elapsed={elapsed:.1f}s | "
            f"loss={mse_val:.6f} | psnr={psnr_val:.2f} | "
            f"samples={n_rendering_samples} | rays={len(pixels)}"
        )
        train_curve.append({
            "step": step,
            "loss": mse_val.item(),
            "psnr": psnr_val.item(),
            "time_s": elapsed,
        })

    # Evaluation at checkpoint steps
    if step > 0 and step in eval_checkpoints:
        elapsed = time.time() - tic
        print(f"  Running evaluation at step {step}...")
        metrics = evaluate_on_test_set()
        metrics["time_s"] = elapsed
        checkpoint_metrics[str(step)] = metrics
        psnr_str = f"{metrics['psnr']:.2f}"
        ssim_str = f"{metrics.get('ssim', 0):.4f}" if HAS_SSIM else "N/A"
        lpips_str = f"{metrics['lpips']:.4f}"
        print(f"  EVAL @ step {step}: psnr={psnr_str}, ssim={ssim_str}, lpips={lpips_str} ({elapsed:.1f}s)")

# ---------- Final evaluation ----------
# Reuse checkpoint result if we already evaluated at max_steps
if str(max_steps) in checkpoint_metrics:
    print("\nReusing checkpoint evaluation at max_steps.")
    final_metrics = checkpoint_metrics[str(max_steps)]
    final_metrics["time_s"] = time.time() - tic
else:
    print("\nRunning final evaluation...")
    final_metrics = evaluate_on_test_set()
    final_metrics["time_s"] = time.time() - tic

print(f"\nFinal results for {args.scene} ({args.optimizer}):")
print(f"  Test PSNR:  {final_metrics['psnr']:.2f} dB")
print(f"  Test LPIPS: {final_metrics['lpips']:.4f}")
if "ssim" in final_metrics:
    print(f"  Test SSIM:  {final_metrics['ssim']:.4f}")
print(f"  Total time: {final_metrics['time_s']:.1f}s")
print(f"  Total steps: {max_steps}")

# ---------- Write results (txt, backward compatible) ----------
with open(results_log, "a") as f:
    header = f"# scene={args.scene} optimizer={args.optimizer} lr={lr} steps={max_steps}\n"
    f.write(header)
    line = (f"psnr={final_metrics['psnr']:.2f} lpips={final_metrics['lpips']:.4f} "
            f"time={final_metrics['time_s']:.1f}s steps={max_steps}")
    if "ssim" in final_metrics:
        line += f" ssim={final_metrics['ssim']:.4f}"
    f.write(line + "\n")
print(f"Results saved to {results_log}")

# ---------- Write JSON metrics ----------
dataset_name = "llff" if is_llff else "synthetic"
json_metrics = {
    "paper": "preconditioners",
    "task": "nerf",
    "dataset": dataset_name,
    "scene": args.scene,
    "method": args.optimizer,
    "hyperparams": {
        "lr": lr,
        "scheduler": args.scheduler,
        "end_lr": end_lr,
        "batch_size": init_batch_size,
        "max_steps": max_steps,
        "esgd_update_every": args.esgd_update_every,
        "weight_decay": args.weight_decay,
    },
    "eval_checkpoints": eval_checkpoints,
    "checkpoints": checkpoint_metrics,
    "train_curve": train_curve,
    "final": final_metrics,
}
json_path = output_dir / f"gaussian_nerf_{args.optimizer}_metrics.json"
with open(json_path, "w") as f:
    json.dump(json_metrics, f, indent=2)
print(f"JSON metrics saved to {json_path}")

# ---------- Save model ----------
model_path = output_dir / f"gaussian_nerf_{args.optimizer}_final.pt"
torch.save({
    "step": max_steps,
    "radiance_field_state_dict": radiance_field.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "estimator_state_dict": estimator.state_dict(),
}, model_path)
print(f"Model saved to {model_path}")

# ---------- Append to batch results file if requested ----------
if args.result_file:
    with open(args.result_file, "a") as f:
        line = (f"{args.scene} {args.optimizer} lr={lr} "
                f"psnr={final_metrics['psnr']:.2f} lpips={final_metrics['lpips']:.4f} "
                f"time={final_metrics['time_s']:.1f}s steps={max_steps}")
        if "ssim" in final_metrics:
            line += f" ssim={final_metrics['ssim']:.4f}"
        f.write(line + "\n")
