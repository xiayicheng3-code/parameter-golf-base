#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for visualization. Install it in the plotting environment first."
        ) from exc
    return plt, TwoSlopeNorm


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_sidecar(path):
    data = torch.load(path, map_location="cpu")
    if "layers" not in data:
        raise SystemExit(f"{path} does not look like a KAN sidecar file.")
    return data


def _select_layer_entry(data, layer_idx, occurrence):
    matches = [
        entry
        for entry in data["layers"]
        if int(entry["layer_idx"]) == layer_idx
        and int(entry["occurrence"]) == occurrence
    ]
    if not matches:
        available = sorted(
            {
                (int(entry["layer_idx"]), int(entry["occurrence"]))
                for entry in data["layers"]
            }
        )
        raise SystemExit(
            f"Layer {layer_idx} occurrence {occurrence} not found. "
            f"Available entries: {available}"
        )
    return matches[0]


def _normalize_by_p75(values, quantile_names, eps=1e-8):
    p75_idx = list(quantile_names).index("p75")
    denom = values[:, p75_idx : p75_idx + 1]
    safe = np.where(np.abs(denom) >= eps, denom, 1.0)
    normalized = values / safe
    degenerate = np.where(np.abs(denom[:, 0]) < eps)[0].tolist()
    return normalized, degenerate


def _relative_x_positions(quantiles, quantile_names, eps=1e-8):
    p10_idx = list(quantile_names).index("p10")
    p90_idx = list(quantile_names).index("p90")
    p10 = quantiles[:, p10_idx : p10_idx + 1]
    p90 = quantiles[:, p90_idx : p90_idx + 1]
    denom = p90 - p10
    safe = np.where(np.abs(denom) >= eps, denom, 1.0)
    x_positions = (quantiles - p10) / safe
    degenerate = np.where(np.abs(denom[:, 0]) < eps)[0].tolist()
    return x_positions, degenerate


def _make_colors(group_count):
    plt, _ = _require_matplotlib()
    if group_count <= 20:
        return plt.cm.tab20(np.linspace(0.0, 1.0, group_count))
    return plt.cm.turbo(np.linspace(0.0, 1.0, group_count))


def _plot_overlay(ax, x_positions, values, quantile_names, title, colors, show_legend):
    tick_positions = np.median(x_positions, axis=0)
    for group_idx, color in enumerate(colors):
        ax.plot(
            x_positions[group_idx],
            values[group_idx],
            color=color,
            linewidth=1.5,
            alpha=0.95,
        )
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle=":", alpha=0.35)
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.35)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(quantile_names)
    ax.set_title(title)
    ax.set_xlabel("relative input position ((q - p10) / (p90 - p10))")
    ax.set_ylabel("normalized value")
    ax.grid(alpha=0.2)
    if show_legend:
        labels = [f"g{idx:02d}" for idx in range(values.shape[0])]
        ax.legend(
            labels,
            ncol=4,
            fontsize=7,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
        )


def _resample_rows(x_positions, values, grid_size=256):
    x_min = float(np.nanmin(x_positions))
    x_max = float(np.nanmax(x_positions))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise SystemExit(
            "Encountered non-finite x positions during heatmap resampling."
        )
    if abs(x_max - x_min) < 1e-8:
        x_max = x_min + 1.0
    grid = np.linspace(x_min, x_max, grid_size)
    resampled = np.empty((values.shape[0], grid_size), dtype=np.float64)
    for row_idx, (x_row, y_row) in enumerate(zip(x_positions, values)):
        uniq_x, uniq_idx = np.unique(x_row, return_index=True)
        uniq_y = y_row[uniq_idx]
        if uniq_x.size == 1:
            resampled[row_idx] = uniq_y[0]
        else:
            resampled[row_idx] = np.interp(grid, uniq_x, uniq_y)
    return grid, resampled


def _plot_heatmap(fig, ax, x_positions, values, title):
    plt, TwoSlopeNorm = _require_matplotlib()
    x_grid, resampled = _resample_rows(x_positions, values)
    vmin = float(np.nanmin(resampled))
    vmax = float(np.nanmax(resampled))
    if vmin <= 1.0 <= vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    else:
        norm = None
    im = ax.imshow(
        resampled,
        aspect="auto",
        cmap="coolwarm",
        norm=norm,
        extent=[x_grid[0], x_grid[-1], values.shape[0] - 0.5, -0.5],
    )
    ax.set_title(title)
    ax.set_xlabel("relative input position ((q - p10) / (p90 - p10))")
    ax.set_ylabel("group")
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels([f"g{idx:02d}" for idx in range(values.shape[0])])
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle=":", alpha=0.35)
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.35)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _build_figure(
    value_x,
    value_norm,
    gate_x,
    gate_norm,
    quantile_names,
    layer_idx,
    occurrence,
    output_path,
    mode,
    show_legend,
):
    plt, _ = _require_matplotlib()
    colors = _make_colors(value_norm.shape[0])
    if mode == "overlay":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        _plot_overlay(
            axes[0],
            value_x,
            value_norm,
            quantile_names,
            f"Layer {layer_idx} occ {occurrence} value phi(u)",
            colors,
            show_legend,
        )
        _plot_overlay(
            axes[1],
            gate_x,
            gate_norm,
            quantile_names,
            f"Layer {layer_idx} occ {occurrence} gate psi(g)",
            colors,
            show_legend,
        )
    elif mode == "heatmap":
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        _plot_heatmap(
            fig,
            axes[0],
            value_x,
            value_norm,
            f"Layer {layer_idx} occ {occurrence} value phi(u)",
        )
        _plot_heatmap(
            fig,
            axes[1],
            gate_x,
            gate_norm,
            f"Layer {layer_idx} occ {occurrence} gate psi(g)",
        )
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
        _plot_overlay(
            axes[0, 0],
            value_x,
            value_norm,
            quantile_names,
            f"Layer {layer_idx} occ {occurrence} value phi(u)",
            colors,
            show_legend,
        )
        _plot_overlay(
            axes[0, 1],
            gate_x,
            gate_norm,
            quantile_names,
            f"Layer {layer_idx} occ {occurrence} gate psi(g)",
            colors,
            show_legend,
        )
        _plot_heatmap(
            fig,
            axes[1, 0],
            value_x,
            value_norm,
            f"Layer {layer_idx} occ {occurrence} value heatmap",
        )
        _plot_heatmap(
            fig,
            axes[1, 1],
            gate_x,
            gate_norm,
            f"Layer {layer_idx} occ {occurrence} gate heatmap",
        )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_summary_text(
    output_path,
    layer_idx,
    occurrence,
    quantile_names,
    value_x,
    value_norm,
    gate_x,
    gate_norm,
    value_degenerate,
    gate_degenerate,
    value_x_degenerate,
    gate_x_degenerate,
):
    lines = [
        f"layer={layer_idx}",
        f"occurrence={occurrence}",
        f"quantiles={','.join(quantile_names)}",
        f"value_groups_with_small_f_p75={value_degenerate}",
        f"gate_groups_with_small_f_p75={gate_degenerate}",
        f"value_groups_with_small_p90_minus_p10={value_x_degenerate}",
        f"gate_groups_with_small_p90_minus_p10={gate_x_degenerate}",
        "",
        "value_normalized_rows:",
    ]
    for idx, row in enumerate(value_norm):
        lines.append(
            f"g{idx:02d} "
            + " ".join(
                f"{name}={value:.6f}" for name, value in zip(quantile_names, row)
            )
        )
    lines.append("")
    lines.append("value_relative_x_rows:")
    for idx, row in enumerate(value_x):
        lines.append(
            f"g{idx:02d} "
            + " ".join(
                f"{name}={value:.6f}" for name, value in zip(quantile_names, row)
            )
        )
    lines.append("")
    lines.append("gate_normalized_rows:")
    for idx, row in enumerate(gate_norm):
        lines.append(
            f"g{idx:02d} "
            + " ".join(
                f"{name}={value:.6f}" for name, value in zip(quantile_names, row)
            )
        )
    lines.append("")
    lines.append("gate_relative_x_rows:")
    for idx, row in enumerate(gate_x):
        lines.append(
            f"g{idx:02d} "
            + " ".join(
                f"{name}={value:.6f}" for name, value in zip(quantile_names, row)
            )
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize grouped swiglu_kan shapes from final_model.kan_shapes.pt"
    )
    parser.add_argument("sidecar", type=Path, help="Path to final_model.kan_shapes.pt")
    parser.add_argument("--layer", type=int, required=True, help="Physical layer index")
    parser.add_argument(
        "--occurrence",
        type=int,
        default=0,
        help="Occurrence index for repeated layers under looping/recurrence",
    )
    parser.add_argument(
        "--mode",
        choices=("overlay", "heatmap", "both"),
        default="both",
        help="Visualization layout",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots and summaries",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Hide overlay legend if the figure becomes too crowded",
    )
    args = parser.parse_args()

    data = _load_sidecar(args.sidecar)
    entry = _select_layer_entry(data, args.layer, args.occurrence)
    quantile_names = tuple(data["quantile_names"])
    value_quantiles = _to_numpy(entry["u_stats"]["quantiles"])
    gate_quantiles = _to_numpy(entry["g_stats"]["quantiles"])
    value_norm, value_degenerate = _normalize_by_p75(
        _to_numpy(entry["u_stats"]["phi_at_quantiles"]), quantile_names
    )
    gate_norm, gate_degenerate = _normalize_by_p75(
        _to_numpy(entry["g_stats"]["psi_at_quantiles"]), quantile_names
    )
    value_x, value_x_degenerate = _relative_x_positions(value_quantiles, quantile_names)
    gate_x, gate_x_degenerate = _relative_x_positions(gate_quantiles, quantile_names)

    output_dir = args.output_dir or args.sidecar.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"kan_layer{args.layer:02d}_occ{args.occurrence:02d}"
    figure_path = output_dir / f"{stem}_{args.mode}.png"
    summary_path = output_dir / f"{stem}_normalized.txt"

    _build_figure(
        value_x,
        value_norm,
        gate_x,
        gate_norm,
        quantile_names,
        args.layer,
        args.occurrence,
        figure_path,
        args.mode,
        show_legend=not args.no_legend,
    )
    _write_summary_text(
        summary_path,
        args.layer,
        args.occurrence,
        quantile_names,
        value_x,
        value_norm,
        gate_x,
        gate_norm,
        value_degenerate,
        gate_degenerate,
        value_x_degenerate,
        gate_x_degenerate,
    )
    print(f"wrote figure: {figure_path}")
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
