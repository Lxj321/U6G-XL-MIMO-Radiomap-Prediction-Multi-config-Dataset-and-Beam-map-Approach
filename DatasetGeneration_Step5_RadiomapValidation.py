#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-folder radiomap vs heightmap visualization.

What it does:
- Find one radiomap matrix (*.npy) and one height map (*height_matrix.npy) in current folder (or given paths)
- Build 3-class region mask:
    2 = building (from heightmap)
    0 = invalid (radiomap <= invalid_threshold; default targets RT sentinel -300 dB)
    1 = valid (everything else)
- Save:
    - labeled_radiomap.npy  (building cells replaced by building_fill_value for visualization)
    - region_mask.npy
    - radiomap.png / labeled_radiomap.png / region_mask.png
Optional:
- Compare two radiomaps (e.g., tx0 vs tx23) and output diff.png
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def find_one(patterns, root: Path):
    for pat in patterns:
        hits = sorted(root.glob(pat))
        if hits:
            return hits[0]
    return None


def build_building_mask(heightmap_256: np.ndarray, out_shape=(128, 128)) -> np.ndarray:
    """
    heightmap: (256,256)
    returns: building_mask (128,128) boolean
    rule: if any of the corresponding 2x2 block is non-zero => building
    """
    if heightmap_256.shape != (256, 256):
        raise ValueError(f"heightmap must be (256,256), got {heightmap_256.shape}")

    H, W = out_shape
    bm = np.zeros((H, W), dtype=bool)
    for i in range(H):
        p0, p1 = 2 * i, 2 * i + 2
        for j in range(W):
            q0, q1 = 2 * j, 2 * j + 2
            block = heightmap_256[p0:p1, q0:q1]
            bm[i, j] = np.any(block != 0)
    return bm


def build_region_mask(radiomap_128: np.ndarray,
                      building_mask: np.ndarray,
                      invalid_threshold: float) -> np.ndarray:
    """
    0=invalid, 1=valid, 2=building
    IMPORTANT: invalid should be tied to RT sentinel (e.g., -300 dB), NOT vmin/vmax of plotting.
    """
    if radiomap_128.shape != (128, 128):
        raise ValueError(f"radiomap must be (128,128), got {radiomap_128.shape}")
    if building_mask.shape != (128, 128):
        raise ValueError(f"building_mask must be (128,128), got {building_mask.shape}")

    invalid_mask = radiomap_128 <= invalid_threshold
    region = np.ones((128, 128), dtype=np.uint8)  # valid=1 by default
    region[invalid_mask] = 0
    region[building_mask] = 2  # building overrides invalid/valid for visualization
    return region


def labeled_radiomap(radiomap_128: np.ndarray,
                     region_mask: np.ndarray,
                     building_fill_value: float) -> np.ndarray:
    out = radiomap_128.astype(np.float32).copy()
    out[region_mask == 2] = building_fill_value
    return out


def save_png_radiomap(mat, out_path: Path, title: str, vmin=None, vmax=None, cmap=None):
    plt.figure(figsize=(9, 8))
    plt.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_png_mask(mask, out_path: Path, title: str):
    plt.figure(figsize=(9, 8))
    plt.imshow(mask)
    plt.colorbar()
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radiomap", type=str, default="", help="Path to radiomap (128x128) .npy. If empty, auto-find in cwd.")
    ap.add_argument("--heightmap", type=str, default="", help="Path to heightmap (256x256) .npy. If empty, auto-find in cwd.")
    ap.add_argument("--out_dir", type=str, default="viz_out", help="Output folder (relative or absolute).")

    # mask rules
    ap.add_argument("--invalid_threshold", type=float, default=-299.5,
                    help="Radiomap <= this is treated as invalid. Default targets RT sentinel -300 dB.")
    ap.add_argument("--building_fill_value", type=float, default=1000,
                    help="Value used to fill building cells in labeled radiomap (for visualization only).")

    # plotting
    ap.add_argument("--vmin", type=float, default=-120.0)
    ap.add_argument("--vmax", type=float, default=-60.0)

    # optional compare
    ap.add_argument("--compare", type=int, default=0, help="1 to compare two radiomaps")
    ap.add_argument("--radiomap_b", type=str, default="", help="Second radiomap for comparison (128x128 .npy)")

    args = ap.parse_args()

    cwd = Path(".").resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # locate files
    radiomap_path = Path(args.radiomap) if args.radiomap else find_one(["*matrix.npy", "*radiomap*.npy"], cwd)
    heightmap_path = Path(args.heightmap) if args.heightmap else find_one(["*height*matrix.npy", "*heightmap*.npy"], cwd)

    if radiomap_path is None or not radiomap_path.exists():
        raise FileNotFoundError("Cannot find radiomap .npy in current folder. Provide --radiomap explicitly.")
    if heightmap_path is None or not heightmap_path.exists():
        raise FileNotFoundError("Cannot find heightmap .npy in current folder. Provide --heightmap explicitly.")

    R = np.load(radiomap_path)
    H = np.load(heightmap_path)

    # build masks
    bmask = build_building_mask(H, out_shape=(128, 128))
    region = build_region_mask(R, bmask, invalid_threshold=args.invalid_threshold)
    L = labeled_radiomap(R, region, building_fill_value=args.building_fill_value)

    # save npy
    np.save(out_dir / "region_mask.npy", region)
    np.save(out_dir / "labeled_radiomap.npy", L)

    # save pngs
    save_png_radiomap(R, out_dir / "radiomap.png",
                      title=f"Radiomap (raw) | {radiomap_path.name}",
                      vmin=args.vmin, vmax=args.vmax)

    # labeled uses its own display range (so building fill value is visible)
    save_png_radiomap(L, out_dir / "labeled_radiomap.png",
                      title="Labeled radiomap (building filled for display)",
                      vmin=min(args.vmin, float(np.nanmin(R))),
                      vmax=max(args.vmax, float(args.building_fill_value)))

    save_png_mask(region, out_dir / "region_mask.png",
                  title="Region mask (0=invalid,1=valid,2=building)")

    # stats (helps you sanity-check correctness)
    n_total = region.size
    n_inv = int(np.sum(region == 0))
    n_val = int(np.sum(region == 1))
    n_bld = int(np.sum(region == 2))
    print("=== Stats ===")
    print(f"radiomap: {radiomap_path}")
    print(f"heightmap: {heightmap_path}")
    print(f"invalid_threshold: {args.invalid_threshold}")
    print(f"counts: invalid={n_inv} ({n_inv/n_total:.2%}), valid={n_val} ({n_val/n_total:.2%}), building={n_bld} ({n_bld/n_total:.2%})")
    print(f"radiomap min/max: {float(np.min(R)):.2f} / {float(np.max(R)):.2f}")

    # optional compare
    if int(args.compare) == 1:
        if not args.radiomap_b:
            raise ValueError("Use --radiomap_b when --compare=1")
        Rb = np.load(Path(args.radiomap_b))
        if Rb.shape != (128, 128):
            raise ValueError(f"radiomap_b must be (128,128), got {Rb.shape}")

        D = R - Rb  # A - B
        # IMPORTANT: keep invalid(-300) unchanged in diff viz (optional):
        # only show diff on points where both are valid (not -300) and not building
        valid_both = (R > args.invalid_threshold) & (Rb > args.invalid_threshold) & (~bmask)
        D_show = np.full_like(D, np.nan, dtype=np.float64)
        D_show[valid_both] = D[valid_both]

        plt.figure(figsize=(9, 8))
        plt.imshow(D_show)
        plt.colorbar(label="A - B (dB)")
        plt.title("Diff (only valid & non-building shown)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(out_dir / "diff_valid_only.png", dpi=180)
        plt.close()

        print(f"Saved comparison diff to: {out_dir / 'diff_valid_only.png'}")

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()