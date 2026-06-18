"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
Modified to support checkpoint evaluation and JSON metrics output.

ENTRENAMIENTO DE SOFT MINING EN NGP NERF CON PROPNET
======================================================
Script principal de entrenamiento para la tecnica Soft Mining.
Entrena un NeRF basado en Instant-NGP (HashGrid + tinycudann) con PropNet
(proposal network) como estimador de muestreo.

SOPORTA DOS TIPOS DE MUESTREO:
- --sampling_type uniform: Muestreo uniforme tradicional de rayos (baseline)
- --sampling_type lmc:     Muestreo Langevin Monte Carlo para Soft Mining

FLUJO DE SOFT MINING (cuando sampling_type="lmc"):
1. En cada iteracion, el dataset llama a LMC.forward() que actualiza las posiciones
   de las particulas usando la gradiente del error (net_grad) obtenida del paso anterior.
2. Las nuevas coordenadas 2D se usan para muestrear rayos de la imagen.
3. Se aplica una correccion por importance sampling: la perdida se escala por
   Q(x)^{-alpha} donde alpha crece linealmente de 0 a su valor objetivo (0.8 para LLFF)
   durante las primeras 1000 iteraciones (warm-up del MCMC).
4. Se guardan checkpoints cada N pasos y metricas (PSNR, SSIM, LPIPS) en JSON.

ARQUITECTURA: Instant-NGP con codificacion hash grid multiresolucion + Spherical
Harmonics para view-dependence, optimizada con tinycudann (CUDA).

En nuestro trabajo, este script se usa para todos los experimentos de Soft Mining
sobre las 8 escenas LLFF (20K iteraciones, batch_size=4096, alpha=0.8).
"""
import argparse
import itertools
import json
import pathlib
import time
from typing import Callable

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from radiance_fields.ngp import NGPDensityField, NGPRadianceField
from losses import NeRFLoss
from schedulers import create_scheduler
torch.autograd.set_detect_anomaly(True)

from examples.utils import (
    LLFF_NDC_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_with_propnet,
    set_random_seed,
)
from nerfacc.estimators.prop_net import (
    PropNetEstimator,
    get_proposal_requires_grad_fn,
)

# SSIM import
try:
    from pytorch_msssim import ssim as compute_ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    print("WARNING: pytorch_msssim not available, SSIM will not be computed")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default=str("../../data/nerf_synthetic"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--scene",
    type=str,
    default="mic",
    choices=NERF_SYNTHETIC_SCENES + LLFF_NDC_SCENES,
    help="which scene to use",
)
parser.add_argument(
    "--test_chunk_size",
    type=int,
    default=8192,
)
parser.add_argument(
    "--sampling_type",
    type=str,
    choices=["uniform", "lmc"],
    default="lmc",
)
parser.add_argument(
    "--i_print",
    type=int,
    default=1000,
    help="Log training metrics every N steps",
)
parser.add_argument(
    "--i_eval",
    type=str,
    default="1000,5000,10000,20000",
    help="Comma-separated list of steps at which to run full evaluation",
)
parser.add_argument(
    "--scheduler",
    type=str,
    default="cosineannealing",
)
parser.add_argument(
    "--minpct",
    type=float,
    default=0.1,
)
parser.add_argument(
    "--lossminpc",
    type=float,
    default=0.1,
)
args = parser.parse_args()

# Parse i_eval into a set of checkpoint steps
eval_checkpoints = sorted(set(int(x.strip()) for x in args.i_eval.split(",")))

device = "cuda:0"
set_random_seed(42)

if args.scene in LLFF_NDC_SCENES:
    from datasets.nerf_llff import SubjectLoader
     # training parameters
    max_steps = 20000
    init_batch_size = 4096
    unbounded = 2
    weight_decay = 1e-5 
    # scene parameters
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 0
    far_plane = 1
    # dataset parameters
    train_dataset_kwargs = {"sampling_type": args.sampling_type, 
                            "minpct": args.minpct, "lossminpc": args.lossminpc}
    test_dataset_kwargs = {}
    # model parameters
    proposal_networks = [
        NGPDensityField(
            aabb=aabb,
            unbounded=unbounded,
            n_levels=5,
            max_resolution=128,
        ).to(device),
    ]
    # render parameters
    num_samples = 64
    num_samples_per_prop = [128]
    prop_sampling_type = "uniform"
    opaque_bkgd = False

    # lmc
    alpha = 0.8
else:
    from datasets.nerf_synthetic import SubjectLoader

    # training parameters
    max_steps = 20000
    init_batch_size = 4096
    weight_decay = (
        1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
    )
    # scene parameters
    unbounded = False
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 2.0
    far_plane = 6.0
    # dataset parameters
    train_dataset_kwargs = {"sampling_type": args.sampling_type, "minpct": args.minpct, "lossminpc": args.lossminpc}
    test_dataset_kwargs = {}
    # model parameters
    proposal_networks = [
        NGPDensityField(
            aabb=aabb,
            unbounded=unbounded,
            n_levels=5,
            max_resolution=128,
        ).to(device),
    ]
    # render parameters
    num_samples = 64
    num_samples_per_prop = [128]
    prop_sampling_type = "uniform"
    opaque_bkgd = False

    # lmc
    alpha = 0.6

train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    device=device,
    **train_dataset_kwargs,
)

test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    device=device,
    **test_dataset_kwargs,
)

# setup the radiance field we want to train.
prop_optimizer = torch.optim.Adam(
    itertools.chain(
        *[p.parameters() for p in proposal_networks],
    ),
    lr=1e-2,
    eps=1e-15,
    weight_decay=weight_decay,
)
prop_scheduler = create_scheduler(prop_optimizer, args.scheduler, max_steps, 1e-2)
estimator = PropNetEstimator(prop_optimizer, prop_scheduler).to(device)

grad_scaler = torch.cuda.amp.GradScaler(2**10)
radiance_field = NGPRadianceField(aabb=aabb, unbounded=unbounded).to(device)
optimizer = torch.optim.Adam(
    radiance_field.parameters(),
    lr=1e-2,
    eps=1e-15,
    weight_decay=weight_decay,
)
scheduler = create_scheduler(optimizer, args.scheduler, max_steps, 1e-2)
proposal_requires_grad_fn = get_proposal_requires_grad_fn()

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

loss_fn = NeRFLoss()

gradval = None
lossperpix_prev = None

# ---------- Output directory and metrics storage ----------
output_dir = pathlib.Path(__file__).resolve().parent.parent / "outputs" / args.scene
output_dir.mkdir(parents=True, exist_ok=True)

train_curve = []
checkpoint_metrics = {}

print(f"\nTraining: scene={args.scene}, sampling={args.sampling_type}, "
      f"max_steps={max_steps}, alpha={alpha}")
print(f"Evaluation checkpoints: {eval_checkpoints}\n")


def evaluate_on_test_set():
    """Run full evaluation on test set. Returns avg psnr, ssim, lpips."""
    torch.cuda.empty_cache()
    radiance_field.eval()
    for p in proposal_networks:
        p.eval()
    estimator.eval()
    psnrs, ssims, lpips_vals = [], [], []
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(test_dataset)), desc="eval", leave=False):
            data = test_dataset[i]
            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]
            rgb, acc, depth, _, _ = render_image_with_propnet(
                radiance_field,
                proposal_networks,
                estimator,
                rays,
                num_samples=num_samples,
                num_samples_per_prop=num_samples_per_prop,
                near_plane=near_plane,
                far_plane=far_plane,
                sampling_type=prop_sampling_type,
                opaque_bkgd=opaque_bkgd,
                render_bkgd=render_bkgd,
                test_chunk_size=args.test_chunk_size,
            )
            mse = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            psnrs.append(psnr.item())
            lpips_vals.append(lpips_fn(rgb, pixels).item())
            if HAS_SSIM:
                h, w = pixels.shape[:2] if pixels.dim() == 3 else (int(pixels.shape[0]**0.5), int(pixels.shape[0]**0.5))
                rgb_img = rgb.view(1, -1, 3).permute(0, 2, 1).view(1, 3, h, w).clamp(0, 1)
                pix_img = pixels.view(1, -1, 3).permute(0, 2, 1).view(1, 3, h, w).clamp(0, 1)
                ssim_val = compute_ssim(rgb_img, pix_img, data_range=1.0).item()
                ssims.append(ssim_val)
    result = {
        "psnr": sum(psnrs) / len(psnrs),
        "lpips": sum(lpips_vals) / len(lpips_vals),
    }
    if ssims:
        result["ssim"] = sum(ssims) / len(ssims)
    return result


# training
tic = time.time()
for step in range(max_steps + 1):
    radiance_field.train()
    for p in proposal_networks:
        p.train()
    estimator.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset.__getitem__(i, net_grad=gradval, loss_per_pix=lossperpix_prev)


    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]

    proposal_requires_grad = proposal_requires_grad_fn(step)
    # render
    rgb, acc, depth, extras, distkwargs = render_image_with_propnet(
        radiance_field,
        proposal_networks,
        estimator,
        rays,
        # rendering options
        num_samples=num_samples,
        num_samples_per_prop=num_samples_per_prop,
        near_plane=near_plane,
        far_plane=far_plane,
        sampling_type=prop_sampling_type,
        opaque_bkgd=opaque_bkgd,
        render_bkgd=render_bkgd,
        # train options
        proposal_requires_grad=proposal_requires_grad,
    )
    estimator.update_every_n_steps(
        extras["trans"], proposal_requires_grad, loss_scaler=1024
    )

    # compute loss
    loss_d = loss_fn(rgb, pixels, acc, distkwargs)
    loss_per_pix = loss_d['rgb'].mean(-1)
    if 'opacity' in loss_d:
        loss_per_pix = loss_per_pix + loss_d['opacity'].squeeze(-1)
    if 'distorion' in loss_d:
        loss_per_pix = loss_per_pix + loss_d['distortion']

    if args.sampling_type in ["lmc"]:
        imp_loss = torch.abs(rgb - pixels).mean(-1).detach()
        if 'opacity' in loss_d:
            imp_loss = (imp_loss + loss_d['opacity'].squeeze(-1)).detach()
        if 'distorion' in loss_d:
            imp_loss = imp_loss + loss_d['distortion'].detach()
        correction = 1.0 / torch.clip(imp_loss, min=torch.finfo(torch.float16).eps).detach()
        if alpha != 0:
            r = min((step/1000), alpha)
        else:
            r = alpha
        correction.pow_(r)
        correction.clamp_(min=0.2, max=correction.mean()+correction.std())
        loss_per_pix.mul_(correction) 
    loss = loss_per_pix.mean()

    optimizer.zero_grad()
    # do not unscale it because we are using Adam.
    grad_scaler.scale(loss).backward()
    optimizer.step()
    scheduler.step()

    if args.sampling_type == "lmc":
        with torch.no_grad():            
            net_grad = data['points_2d'].grad.detach()
            loss_per_pix = loss_per_pix.detach()
            net_grad = net_grad / ((grad_scaler._scale * (loss_per_pix).unsqueeze(1))+ torch.finfo(net_grad.dtype).eps)
            gradval = net_grad
            lossperpix_prev = loss_per_pix

    # Training logging
    if step % args.i_print == 0:
        elapsed_time = time.time() - tic
        train_loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(train_loss) / np.log(10.0)
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={train_loss:.5f} | psnr={psnr:.2f} | "
            f"num_rays={len(pixels):d} | "
            f"max_depth={depth.max():.3f} | "
        )
        train_curve.append({
            "step": step,
            "loss": train_loss.item(),
            "psnr": psnr.item(),
            "time_s": elapsed_time,
        })

    # Evaluation at checkpoint steps
    if step > 0 and step in eval_checkpoints:
        elapsed_time = time.time() - tic
        print(f"  Running evaluation at step {step}...")
        metrics = evaluate_on_test_set()
        metrics["time_s"] = elapsed_time
        checkpoint_metrics[str(step)] = metrics
        psnr_str = f"{metrics['psnr']:.2f}"
        ssim_str = f"{metrics.get('ssim', 0):.4f}" if HAS_SSIM else "N/A"
        lpips_str = f"{metrics['lpips']:.4f}"
        print(f"  EVAL @ step {step}: psnr={psnr_str}, ssim={ssim_str}, lpips={lpips_str} ({elapsed_time:.1f}s)")

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

print(f"\nFinal results for {args.scene} ({args.sampling_type}):")
print(f"  Test PSNR:  {final_metrics['psnr']:.2f} dB")
print(f"  Test LPIPS: {final_metrics['lpips']:.4f}")
if "ssim" in final_metrics:
    print(f"  Test SSIM:  {final_metrics['ssim']:.4f}")
print(f"  Total time: {final_metrics['time_s']:.1f}s")
print(f"  Total steps: {max_steps}")

# ---------- Write JSON metrics ----------
dataset_name = "llff" if args.scene in LLFF_NDC_SCENES else "synthetic"
json_metrics = {
    "paper": "softmining",
    "task": "nerf",
    "dataset": dataset_name,
    "scene": args.scene,
    "method": args.sampling_type,
    "hyperparams": {
        "alpha": alpha,
        "batch_size": init_batch_size,
        "max_steps": max_steps,
        "scheduler": args.scheduler,
        "minpct": args.minpct,
        "lossminpc": args.lossminpc,
        "weight_decay": weight_decay,
    },
    "eval_checkpoints": eval_checkpoints,
    "checkpoints": checkpoint_metrics,
    "train_curve": train_curve,
    "final": final_metrics,
}
json_path = output_dir / f"ngp_{args.sampling_type}_metrics.json"
with open(json_path, "w") as f:
    json.dump(json_metrics, f, indent=2)
print(f"JSON metrics saved to {json_path}")

# ---------- Save model checkpoint ----------
model_path = output_dir / f"ngp_{args.sampling_type}_final.pt"
torch.save({
    "step": max_steps,
    "radiance_field_state_dict": radiance_field.state_dict(),
    "proposal_networks_state_dicts": [p.state_dict() for p in proposal_networks],
    # Inference-critical metadata
    "unbounded": unbounded,
    "near_plane": near_plane,
    "far_plane": far_plane,
    "num_samples": num_samples,
    "num_samples_per_prop": num_samples_per_prop,
    "prop_sampling_type": prop_sampling_type,
    "opaque_bkgd": opaque_bkgd,
    "scene": args.scene,
    "aabb": aabb,
}, model_path)
print(f"Model saved to {model_path}")
