#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_sidecar(path):
    data = torch.load(path, map_location="cpu")
    if "layers" not in data:
        raise SystemExit(f"{path} does not look like a KAN sidecar file.")
    return data


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _phi(points, linear, bias, spline, knots):
    basis = np.maximum(points[:, None] - knots[None, :], 0.0)
    return linear * points + bias + basis @ spline


def _psi(points, linear, spline, knots):
    basis = np.maximum(points[:, None] - knots[None, :], 0.0)
    return _silu(points) + linear * points + basis @ spline


def _sample_interval(lo, hi, num_points):
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Non-finite interval boundary encountered.")
    if abs(hi - lo) < 1e-8:
        hi = lo + 1e-4
    return np.linspace(lo, hi, num_points)


def _fit_design(x, y, design_cols):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.column_stack([col(x) for col in design_cols])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    y_hat = design @ coeffs
    resid = y - y_hat
    ss_res = float(np.sum(resid**2))
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    rmse = float(np.sqrt(np.mean(resid**2)))
    std = float(np.std(y))
    value_range = float(np.max(y) - np.min(y))
    r2 = 1.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    nrmse_std = 0.0 if std < 1e-12 else rmse / std
    nrmse_range = 0.0 if value_range < 1e-12 else rmse / value_range
    return {
        "coeffs": coeffs,
        "y_hat": y_hat,
        "r2": r2,
        "rmse": rmse,
        "nrmse_std": nrmse_std,
        "nrmse_range": nrmse_range,
    }


def _fit_affine(x, y):
    return _fit_design(
        x,
        y,
        [
            lambda z: z,
            lambda z: np.ones_like(z),
        ],
    )


def _fit_leaky_like(x, y):
    result = _fit_design(
        x,
        y,
        [
            lambda z: z,
            lambda z: np.maximum(z, 0.0),
            lambda z: np.ones_like(z),
        ],
    )
    a, b, c = result["coeffs"]
    neg_slope = float(a)
    pos_slope = float(a + b)
    if abs(pos_slope) < 1e-12:
        slope_ratio = np.nan
    else:
        slope_ratio = neg_slope / pos_slope
    result.update(
        {
            "neg_slope": neg_slope,
            "pos_slope": pos_slope,
            "slope_ratio": float(slope_ratio) if np.isfinite(slope_ratio) else np.nan,
            "bias": float(c),
        }
    )
    return result


def _analyze_group(
    layer_idx,
    occurrence,
    group_idx,
    knots,
    value_linear,
    value_bias,
    value_spline,
    gate_linear,
    gate_spline,
    u_quantiles,
    g_quantiles,
    num_points,
):
    u_p01, u_p10, _, _, _, u_p90, u_p99 = u_quantiles
    g_p01, g_p10, _, _, _, g_p90, g_p99 = g_quantiles

    u_full_x = _sample_interval(u_p01, u_p99, num_points)
    u_core_x = _sample_interval(u_p10, u_p90, num_points)
    g_full_x = _sample_interval(g_p01, g_p99, num_points)
    g_core_x = _sample_interval(g_p10, g_p90, num_points)

    phi_full = _phi(u_full_x, value_linear, value_bias, value_spline, knots)
    phi_core = _phi(u_core_x, value_linear, value_bias, value_spline, knots)
    psi_full = _psi(g_full_x, gate_linear, gate_spline, knots)
    psi_core = _psi(g_core_x, gate_linear, gate_spline, knots)

    value_full = _fit_affine(u_full_x, phi_full)
    value_core = _fit_affine(u_core_x, phi_core)
    gate_full = _fit_leaky_like(g_full_x, psi_full)
    gate_core = _fit_leaky_like(g_core_x, psi_core)

    return {
        "layer_idx": int(layer_idx),
        "occurrence": int(occurrence),
        "group_idx": int(group_idx),
        "value_full_r2": value_full["r2"],
        "value_full_nrmse_std": value_full["nrmse_std"],
        "value_core_r2": value_core["r2"],
        "value_core_nrmse_std": value_core["nrmse_std"],
        "gate_full_r2": gate_full["r2"],
        "gate_full_nrmse_std": gate_full["nrmse_std"],
        "gate_full_slope_ratio": gate_full["slope_ratio"],
        "gate_full_bias": gate_full["bias"],
        "gate_core_r2": gate_core["r2"],
        "gate_core_nrmse_std": gate_core["nrmse_std"],
        "gate_core_slope_ratio": gate_core["slope_ratio"],
        "gate_core_neg_slope": gate_core["neg_slope"],
        "gate_core_pos_slope": gate_core["pos_slope"],
        "gate_core_bias": gate_core["bias"],
    }


def _summarize_entry(rows):
    arr = lambda key: np.asarray([row[key] for row in rows], dtype=np.float64)
    return {
        "groups": len(rows),
        "value_full_r2_median": float(np.median(arr("value_full_r2"))),
        "value_full_r2_min": float(np.min(arr("value_full_r2"))),
        "value_core_r2_median": float(np.median(arr("value_core_r2"))),
        "value_core_r2_min": float(np.min(arr("value_core_r2"))),
        "gate_full_r2_median": float(np.median(arr("gate_full_r2"))),
        "gate_full_r2_min": float(np.min(arr("gate_full_r2"))),
        "gate_core_r2_median": float(np.median(arr("gate_core_r2"))),
        "gate_core_r2_min": float(np.min(arr("gate_core_r2"))),
        "gate_core_slope_ratio_median": float(
            np.nanmedian(arr("gate_core_slope_ratio"))
        ),
        "gate_core_slope_ratio_p10": float(
            np.nanpercentile(arr("gate_core_slope_ratio"), 10)
        ),
        "gate_core_slope_ratio_p90": float(
            np.nanpercentile(arr("gate_core_slope_ratio"), 90)
        ),
    }


def _format_entry(layer_idx, occurrence, summary):
    return (
        f"layer={layer_idx:02d} occ={occurrence} groups={summary['groups']} "
        f"value_affine_r2 median/min(full)={summary['value_full_r2_median']:.5f}/{summary['value_full_r2_min']:.5f} "
        f"value_affine_r2 median/min(core)={summary['value_core_r2_median']:.5f}/{summary['value_core_r2_min']:.5f} "
        f"gate_leaky_r2 median/min(full)={summary['gate_full_r2_median']:.5f}/{summary['gate_full_r2_min']:.5f} "
        f"gate_leaky_r2 median/min(core)={summary['gate_core_r2_median']:.5f}/{summary['gate_core_r2_min']:.5f} "
        f"gate_core_slope_ratio median/p10/p90={summary['gate_core_slope_ratio_median']:.3f}/"
        f"{summary['gate_core_slope_ratio_p10']:.3f}/{summary['gate_core_slope_ratio_p90']:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Quantitatively test whether swiglu_kan value paths are affine-like and gate paths are leaky-ReLU-like."
    )
    parser.add_argument("sidecar", type=Path, help="Path to final_model.kan_shapes.pt")
    parser.add_argument(
        "--num-points",
        type=int,
        default=257,
        help="Dense sample count per fitted interval",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many worst-case exceptions to print per category",
    )
    args = parser.parse_args()

    data = _load_sidecar(args.sidecar)
    knots = _to_numpy(data["kan_knots"]).astype(np.float64)

    all_rows = []
    grouped_rows = {}
    for entry in data["layers"]:
        layer_idx = int(entry["layer_idx"])
        occurrence = int(entry["occurrence"])
        key = (layer_idx, occurrence)

        value_linear = _to_numpy(entry["value_linear"]).astype(np.float64)
        value_bias = _to_numpy(entry["value_bias"]).astype(np.float64)
        value_spline = _to_numpy(entry["value_spline"]).astype(np.float64)
        gate_linear = _to_numpy(entry["gate_linear"]).astype(np.float64)
        gate_spline = _to_numpy(entry["gate_spline"]).astype(np.float64)
        u_quantiles = _to_numpy(entry["u_stats"]["quantiles"]).astype(np.float64)
        g_quantiles = _to_numpy(entry["g_stats"]["quantiles"]).astype(np.float64)

        entry_rows = []
        for group_idx in range(value_linear.shape[0]):
            row = _analyze_group(
                layer_idx=layer_idx,
                occurrence=occurrence,
                group_idx=group_idx,
                knots=knots,
                value_linear=value_linear[group_idx],
                value_bias=value_bias[group_idx],
                value_spline=value_spline[group_idx],
                gate_linear=gate_linear[group_idx],
                gate_spline=gate_spline[group_idx],
                u_quantiles=u_quantiles[group_idx],
                g_quantiles=g_quantiles[group_idx],
                num_points=args.num_points,
            )
            entry_rows.append(row)
            all_rows.append(row)
        grouped_rows[key] = entry_rows

    print("Per-layer summary")
    print("=================")
    for key in sorted(grouped_rows):
        print(_format_entry(key[0], key[1], _summarize_entry(grouped_rows[key])))

    print("")
    print("Worst value-path affine exceptions (lowest full-range R^2)")
    print("=========================================================")
    for row in sorted(all_rows, key=lambda r: r["value_full_r2"])[: args.top_k]:
        print(
            f"layer={row['layer_idx']:02d} occ={row['occurrence']} g={row['group_idx']:02d} "
            f"value_r2 full/core={row['value_full_r2']:.5f}/{row['value_core_r2']:.5f} "
            f"value_nrmse_std full/core={row['value_full_nrmse_std']:.5f}/{row['value_core_nrmse_std']:.5f}"
        )

    print("")
    print("Worst gate-path leaky-like exceptions (lowest full-range R^2)")
    print("============================================================")
    for row in sorted(all_rows, key=lambda r: r["gate_full_r2"])[: args.top_k]:
        print(
            f"layer={row['layer_idx']:02d} occ={row['occurrence']} g={row['group_idx']:02d} "
            f"gate_r2 full/core={row['gate_full_r2']:.5f}/{row['gate_core_r2']:.5f} "
            f"gate_nrmse_std full/core={row['gate_full_nrmse_std']:.5f}/{row['gate_core_nrmse_std']:.5f} "
            f"gate_core_slope_ratio={row['gate_core_slope_ratio']:.3f} "
            f"gate_core_bias={row['gate_core_bias']:.5f}"
        )

    print("")
    print("Most atypical gate core slope ratios")
    print("====================================")
    rows_with_ratio = [
        row for row in all_rows if np.isfinite(row["gate_core_slope_ratio"])
    ]
    median_ratio = float(
        np.median([row["gate_core_slope_ratio"] for row in rows_with_ratio])
    )
    for row in sorted(
        rows_with_ratio,
        key=lambda r: abs(r["gate_core_slope_ratio"] - median_ratio),
        reverse=True,
    )[: args.top_k]:
        print(
            f"layer={row['layer_idx']:02d} occ={row['occurrence']} g={row['group_idx']:02d} "
            f"gate_core_slope_ratio={row['gate_core_slope_ratio']:.3f} "
            f"neg/pos={row['gate_core_neg_slope']:.5f}/{row['gate_core_pos_slope']:.5f} "
            f"gate_r2 core={row['gate_core_r2']:.5f}"
        )


if __name__ == "__main__":
    main()
