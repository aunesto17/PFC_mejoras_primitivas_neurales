#!/usr/bin/env python3
"""Compare evaluation results from two optimizer runs (e.g., Adam vs ESGD_Max).

Usage:
    python scripts/compare_results.py \
        --dirs output/fern_comparison/adam output/fern_comparison/esgd_max \
        --names Adam ESGD_Max \
        --output output/fern_comparison/comparison
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path


def load_results(results_dir):
    with open(os.path.join(results_dir, "results.json")) as f:
        return json.load(f)


def load_tensorboard_scalars(log_dir, tag="val/PSNR"):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    if not os.path.isdir(log_dir):
        return None
    acc = EventAccumulator(log_dir)
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if tag not in tags:
        return None
    events = acc.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return np.array(steps), np.array(values)


def generate_latex_table(results_list, names, output_path):
    r"""Generate a LaTeX table comparing PSNR/SSIM/LPIPS across optimizers."""
    latex = []
    latex.append(r"\begin{table}[ht]")
    latex.append(r"\centering")
    latex.append(r"\caption{Quantitative comparison on \texttt{" + results_list[0]['scene'] + r"} scene.}")
    latex.append(r"\label{tab:" + results_list[0]['scene'] + r"_comparison}")
    n_cols = len(names)
    col_spec = "l" + "c" * n_cols
    latex.append(r"\begin{tabular}{" + col_spec + "}")
    latex.append(r"\toprule")
    header = "Metric"
    for name in names:
        header += " & " + name
    header += r" \\"
    latex.append(header)
    latex.append(r"\midrule")
    for metric, fmt, higher_better in [("PSNR", "{:.2f}", True), ("SSIM", "{:.3f}", True), ("LPIPS", "{:.4f}", False)]:
        values = [r["average"][metric.lower()] for r in results_list]
        best_idx = int(np.argmax(values) if higher_better else np.argmin(values))
        row = metric.upper()
        for i, val in enumerate(values):
            if i == best_idx:
                row += r" & \textbf{" + fmt.format(val) + "}"
            else:
                row += " & " + fmt.format(val)
        row += r" \\"
        latex.append(row)
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    table_str = "\n".join(latex)
    with open(os.path.join(output_path, "comparison_table.tex"), "w") as f:
        f.write(table_str)
    print(table_str)
    return table_str


def generate_convergence_plot(dirs, names, output_path, window=200):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (d, name) in enumerate(zip(dirs, names)):
        data = load_tensorboard_scalars(d, "val/PSNR")
        if data is None:
            print(f"Warning: no val/PSNR data in {d}")
            continue
        steps, values = data
        ax.plot(steps, values, color=colors[i % len(colors)], alpha=0.5, linewidth=1)
        if len(values) >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(values, kernel, mode="valid")
            smooth_steps = steps[window - 1:]
            ax.plot(smooth_steps, smoothed, color=colors[i % len(colors)],
                    linewidth=2, label=name)
        else:
            ax.plot(steps, values, color=colors[i % len(colors)],
                    linewidth=2, label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation PSNR (dB)")
    ax.set_title("PSNR Convergence: " + results_list[0]["scene"].title())
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, "psnr_convergence.png"), dpi=150)
    plt.close(fig)
    print(f"Saved psnr_convergence.png")


def generate_comparison_figure(dirs, names, output_path, num_images=2):
    from PIL import Image
    test_paths = [os.path.join(d, "test_view") for d in dirs]
    for p in test_paths:
        if not os.path.isdir(p):
            print(f"Warning: test_view not found in {p}")
            return
    n_views = len([f for f in os.listdir(test_paths[0]) if f.startswith("rgb_") and f.endswith(".png")
                   and not f.startswith("rgb_GT_")])
    selected = list(range(min(num_images, n_views)))
    for idx in selected:
        panels = []
        for i, name in enumerate(names):
            rgb = Image.open(os.path.join(test_paths[i], f"rgb_{idx}.png"))
            panels.append((name, rgb))
        gt = Image.open(os.path.join(test_paths[0], f"rgb_GT_{idx}.png"))
        err_paths = []
        for i, name in enumerate(names):
            ep = os.path.join(test_paths[i], f"error_{idx}.png")
            if os.path.exists(ep):
                err_paths.append((name, Image.open(ep)))
        all_panels = [(n, im) for n, im in panels]
        if err_paths:
            all_panels += err_paths
        all_panels.append(("GT", gt))
        widths = [im.width for _, im in all_panels]
        heights = [im.height for _, im in all_panels]
        gap = 5
        total_w = sum(widths) + gap * (len(all_panels) - 1)
        total_h = max(heights) + 30
        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        x_offset = 0
        for name, im in all_panels:
            canvas.paste(im, (x_offset, 30))
            x_offset += im.width + gap
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        x_offset = 0
        for name, im in all_panels:
            w = im.width
            bbox = draw.textbbox((0, 0), name)
            tw = bbox[2] - bbox[0]
            draw.text((x_offset + (w - tw) // 2, 5), name, fill=(0, 0, 0))
            x_offset += w + gap
        canvas.save(os.path.join(output_path, f"view_{idx}_comparison.png"))
    print(f"Saved {len(selected)} comparison figures")


def main():
    parser = argparse.ArgumentParser(description="Compare optimizer results")
    parser.add_argument("--dirs", nargs="+", required=True, help="Result directories")
    parser.add_argument("--names", nargs="+", required=True, help="Optimizer names")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--num_images", type=int, default=2, help="Number of test views to compare")
    args = parser.parse_args()

    assert len(args.dirs) == len(args.names), "Must provide same number of dirs and names"
    os.makedirs(args.output, exist_ok=True)

    global results_list
    results_list = []
    for d in args.dirs:
        r = load_results(d)
        results_list.append(r)

    print("=== Quantitative Comparison ===")
    for r, name in zip(results_list, args.names):
        avg = r["average"]
        print(f"{name}: PSNR={avg['psnr']:.2f}  SSIM={avg['ssim']:.3f}  LPIPS={avg['lpips']:.4f}")

    print("\n=== LaTeX Table ===")
    generate_latex_table(results_list, args.names, args.output)

    print("\n=== Generating Convergence Plot ===")
    generate_convergence_plot(args.dirs, args.names, args.output)

    print("\n=== Generating Comparison Figures ===")
    generate_comparison_figure(args.dirs, args.names, args.output, args.num_images)

    print(f"\nAll outputs saved to: {args.output}")


if __name__ == "__main__":
    results_list = []
    main()
