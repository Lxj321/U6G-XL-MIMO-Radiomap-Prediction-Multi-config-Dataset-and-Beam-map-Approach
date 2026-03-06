#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beam map generator (LOS-only geometry), supports:
- plane-wave approximation (far-field)
- spherical-wave model (near-/mid-field)

Removes redundant parts from your previous two scripts:
- single script with CLI switch instead of two files
- no unused plane-wave RT function, no unused subarray vars, no unused omega
- consistent TX power handling (uses --tx_power_dbm)

Output:
- per-beam .npy matrix
- per-beam .png plot
- one .npz containing all beams
- a beam_settings.txt

Notes:
- This script computes *beam map only*: no Sionna RT (no reflection/diffraction/occlusion).
- If you need RT mode later, you can add it as another flag and reuse your old process_receiver_config().
"""

import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Beamforming codebook
# -----------------------------
def generate_extended_codebook(
    num_rows: int,
    num_cols: int,
    theta_azim_deg: float,
    theta_elev_deg: float = 0.0,
    subarray_rows: int = 1,
    subarray_cols: int = 1,
    window_type: str = "rect",
    flip_azim_sign: bool = True,
) -> np.ndarray:
    """
    Subarray-DFT + optional amplitude windowing.

    flip_azim_sign:
      - True matches your previous change: use -theta_azim so that negative angles
        point to your intended visual direction.
    """
    theta_azim = -theta_azim_deg if flip_azim_sign else theta_azim_deg
    theta_azim_rad = np.radians(theta_azim)
    theta_elev_rad = np.radians(theta_elev_deg)
    d_lambda = 0.5  # spacing in wavelengths

    def window(n: int) -> np.ndarray:
        if window_type.lower() == "hann":
            return np.hanning(n)
        if window_type.lower() == "rect":
            return np.ones(n)
        raise ValueError(f"Unsupported window_type={window_type}. Use 'rect' or 'hann'.")

    if num_rows % subarray_rows != 0 or num_cols % subarray_cols != 0:
        raise ValueError(
            f"Array ({num_rows}x{num_cols}) not divisible by subarray "
            f"({subarray_rows}x{subarray_cols})"
        )

    n_sub_r = num_rows // subarray_rows
    n_sub_c = num_cols // subarray_cols

    sub_vecs = []
    for _sr in range(n_sub_r):
        for _sc in range(n_sub_c):
            elev_idx = np.arange(subarray_rows)
            azim_idx = np.arange(subarray_cols)

            phase_elev = 2 * np.pi * d_lambda * elev_idx * np.sin(theta_elev_rad)
            phase_azim = 2 * np.pi * d_lambda * azim_idx * np.sin(theta_azim_rad)

            v_elev = window(subarray_rows).reshape(-1, 1) * np.exp(1j * phase_elev).reshape(-1, 1)
            v_azim = window(subarray_cols).reshape(-1, 1) * np.exp(1j * phase_azim).reshape(-1, 1)

            # NOTE: keep same kron order as your original (v_elev, v_azim)
            sub_vec = np.kron(v_elev, v_azim)  # (subarray_rows*subarray_cols, 1)
            sub_vecs.append(sub_vec)

    w = np.concatenate(sub_vecs, axis=0).flatten()
    w = w / (np.linalg.norm(w) + 1e-30)
    return w


# -----------------------------
# Antenna element gain (optional)
# -----------------------------
def antenna_gain_3gpp_linear(
    theta_deg: float,
    phi_deg: float,
    G_max_dBi: float = 8.0,
    theta_3dB: float = 65.0,
    phi_3dB: float = 65.0,
    A_max: float = 30.0,
) -> float:
    """
    Rough 3GPP-style element pattern (TR 38.901-like), returns linear gain.
    theta_deg: azimuth angle (deg)
    phi_deg: elevation angle (deg), 90 is horizontal
    """
    A_H = -min(12 * (theta_deg / theta_3dB) ** 2, A_max)
    A_V = -min(12 * ((phi_deg - 90.0) / phi_3dB) ** 2, A_max)
    A_total = -min(-(A_H + A_V), A_max)
    G_dB = G_max_dBi + A_total
    return 10 ** (G_dB / 10.0)


# -----------------------------
# Geometry beam map (LOS-only)
# -----------------------------
def compute_beammap_los(
    rx_positions_xyz: np.ndarray,        # (N,3)
    tx_center_xyz: np.ndarray,           # (3,)
    antenna_offsets_xyz: np.ndarray,     # (M,3) offsets from tx_center
    w_precoder: np.ndarray,              # (M,)
    frequency_hz: float,
    plane_wave: bool,
    use_element_gain: bool,
    tx_power_dbm: float,
) -> np.ndarray:
    """
    Returns received power in dBm at each RX point, LOS only.

    plane_wave:
      - amplitude uses center distance r_ref for all elements
      - phase uses r_ref - projection(offset, direction)

    spherical_wave:
      - amplitude uses per-element distance
      - phase uses per-element distance

    Power scaling:
      - Uses Friis-like scaling via effective aperture A_e and 4π factor (kept close to your previous code)
      - tx_power_dbm is applied consistently.
    """
    c = 3e8
    wavelength = c / frequency_hz

    # Convert TX power
    P_t_watt = 10 ** ((tx_power_dbm - 30.0) / 10.0)

    # Effective aperture and extra 4π factor as you had (kept for consistency)
    A_e = wavelength**2 / (4 * np.pi)

    M = antenna_offsets_xyz.shape[0]
    w_conj = np.conj(w_precoder).astype(np.complex128)

    out_watt = np.zeros(rx_positions_xyz.shape[0], dtype=np.float64)

    for i, rx in enumerate(rx_positions_xyz):
        vec_to_rx = rx - tx_center_xyz
        r_ref = np.linalg.norm(vec_to_rx) + 1e-30
        direction = vec_to_rx / r_ref

        h = np.zeros(M, dtype=np.complex128)

        for m in range(M):
            ant_global = tx_center_xyz + antenna_offsets_xyz[m]
            vec_ant_to_rx = rx - ant_global

            if plane_wave:
                # amplitude shared
                distance_amp = r_ref
                projection = float(np.dot(antenna_offsets_xyz[m], direction))
                distance_phase = r_ref - projection
            else:
                d = np.linalg.norm(vec_ant_to_rx) + 1e-30
                distance_amp = d
                distance_phase = d

            if use_element_gain:
                theta_rad = np.arctan2(vec_ant_to_rx[1], vec_ant_to_rx[0])
                theta_deg = float(np.degrees(theta_rad))

                d_for_angle = np.linalg.norm(vec_ant_to_rx) + 1e-30
                # elevation: phi=90 is horizontal (your previous convention)
                phi_rad = np.arcsin(vec_ant_to_rx[2] / d_for_angle)
                phi_deg = float(90.0 - np.degrees(phi_rad))

                G_lin = antenna_gain_3gpp_linear(theta_deg, phi_deg)
            else:
                G_lin = 1.0

            h[m] = (
                np.sqrt(G_lin)
                * (1.0 / distance_amp)
                * np.exp(1j * 2 * np.pi * distance_phase / wavelength)
            )

        beam_resp = np.dot(h, w_conj)

        # Keep your previous scaling structure
        P_rx = A_e * P_t_watt * (np.abs(beam_resp) ** 2) / (4 * np.pi)
        out_watt[i] = float(P_rx)

    # W -> dBm
    out_dbm = 10.0 * np.log10(out_watt / 1e-3 + 1e-30)
    return out_dbm


# -----------------------------
# RX grid helpers (match your 8 blocks * 16x128)
# -----------------------------
def build_rx_positions(
    receiver_height: float,
    num_blocks: int = 8,
    block_rows: int = 16,
    block_cols: int = 128,
    step_m: float = 10.0,
    x_start: float = -635.0,
    y_start_list=None,
) -> np.ndarray:
    """
    Builds the same 2048 points as your code:
      - 8 blocks
      - each block is 16 x 128
      - x changes with col, y changes with row, z fixed = receiver_height
    """
    if y_start_list is None:
        y_start_list = [-635, -475, -315, -155, 5, 165, 325, 485]
    assert len(y_start_list) == num_blocks

    pts = []
    for b in range(num_blocks):
        y0 = y_start_list[b]
        for r in range(block_rows):
            for c in range(block_cols):
                x = x_start + c * step_m
                y = y0 + r * step_m
                pts.append([x, y, receiver_height])
    return np.asarray(pts, dtype=np.float64)


def build_tx_antenna_offsets(
    num_rows: int,
    num_cols: int,
    frequency_hz: float,
    d_lambda: float = 0.5,
) -> np.ndarray:
    """
    Antenna offsets in y-z plane (normal +x), same as your previous convention:
      offset = [0, y, z]
    """
    c = 3e8
    wavelength = c / frequency_hz
    offsets = []
    for row in range(num_rows):
        for col in range(num_cols):
            y = (col - num_cols / 2.0) * d_lambda * wavelength
            z = (row - num_rows / 2.0) * d_lambda * wavelength
            offsets.append([0.0, y, z])
    return np.asarray(offsets, dtype=np.float64)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default="u0", help="Scene name (used for folder naming only)")
    ap.add_argument("--out_root", type=str, default="BeamMaps", help="Output root directory")
    ap.add_argument("--frequency_hz", type=float, default=6.7e9)
    ap.add_argument("--num_rows", type=int, default=32)
    ap.add_argument("--num_cols", type=int, default=32)

    ap.add_argument("--num_beams", type=int, default=64)
    ap.add_argument("--start_angle_deg", type=float, default=-32.0)
    ap.add_argument("--beam_spacing_deg", type=float, default=1.0)

    ap.add_argument("--plane_wave", type=int, default=1, help="1=plane wave, 0=spherical wave")
    ap.add_argument("--use_element_gain", type=int, default=1, help="1=use 3GPP element gain, 0=unity gain")
    ap.add_argument("--tx_power_dbm", type=float, default=0.0, help="TX power used in BeamMapOnly power scaling")

    ap.add_argument("--window_type", type=str, default="rect", choices=["rect", "hann"])
    ap.add_argument("--subarray_rows", type=int, default=32, help="DFT subarray rows (your old row_array_num)")
    ap.add_argument("--subarray_cols", type=int, default=32, help="DFT subarray cols (your old col_array_num)")
    ap.add_argument("--flip_azim_sign", type=int, default=1, help="1=use -theta_azim in codebook (your previous behavior)")

    ap.add_argument("--receiver_height", type=float, default=1.5)
    ap.add_argument("--grid_step_m", type=float, default=10.0)
    ap.add_argument("--vmin", type=float, default=-120.0, help="Plot vmin")
    ap.add_argument("--vmax", type=float, default=-60.0, help="Plot vmax")
    args = ap.parse_args()

    plane_wave = bool(args.plane_wave)
    use_element_gain = bool(args.use_element_gain)
    flip_azim_sign = bool(args.flip_azim_sign)

    freq = args.frequency_hz
    tx_center = np.array([0.0, 0.0, 40.0], dtype=np.float64)

    # Build geometry
    rx_xyz = build_rx_positions(
        receiver_height=args.receiver_height,
        step_m=args.grid_step_m,
    )  # (2048, 3)
    ant_offsets = build_tx_antenna_offsets(args.num_rows, args.num_cols, freq)  # (M,3)
    M = ant_offsets.shape[0]

    # Output folder
    wave_tag = "PlaneWave" if plane_wave else "SphericalWave"
    gain_tag = "ElemGain" if use_element_gain else "NoElemGain"
    param_folder = (
        f"freq_{freq/1e9:.1f}GHz_{M}TR_{args.num_beams}beams_{wave_tag}_{gain_tag}_"
        f"sub_{args.subarray_rows}x{args.subarray_cols}_win_{args.window_type}"
    )
    save_dir = Path(args.out_root) / param_folder / args.scene
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save settings
    settings_path = save_dir / "beam_settings.txt"
    with open(settings_path, "w", encoding="utf-8") as f:
        f.write("=== Beam Map Only Settings ===\n\n")
        f.write(f"scene: {args.scene}\n")
        f.write(f"out_dir: {save_dir}\n")
        f.write(f"frequency_hz: {freq}\n")
        f.write(f"tx_center_xyz: {tx_center.tolist()}\n")
        f.write(f"array: {args.num_rows} x {args.num_cols} (M={M})\n")
        f.write(f"plane_wave: {plane_wave}\n")
        f.write(f"use_element_gain: {use_element_gain}\n")
        f.write(f"tx_power_dbm: {args.tx_power_dbm}\n")
        f.write(f"num_beams: {args.num_beams}\n")
        f.write(f"start_angle_deg: {args.start_angle_deg}\n")
        f.write(f"beam_spacing_deg: {args.beam_spacing_deg}\n")
        f.write(f"subarray_rows: {args.subarray_rows}\n")
        f.write(f"subarray_cols: {args.subarray_cols}\n")
        f.write(f"window_type: {args.window_type}\n")
        f.write(f"flip_azim_sign: {flip_azim_sign}\n")
        f.write(f"receiver_height: {args.receiver_height}\n")
        f.write(f"grid_step_m: {args.grid_step_m}\n")

    # Generate beams
    beam_keys = []
    all_beams = {}

    for b in range(args.num_beams):
        theta = args.start_angle_deg + b * args.beam_spacing_deg
        print(f"[{b+1:02d}/{args.num_beams}] beam angle = {theta:.1f} deg")

        w = generate_extended_codebook(
            args.num_rows,
            args.num_cols,
            theta_azim_deg=theta,
            theta_elev_deg=0.0,
            subarray_rows=args.subarray_rows,
            subarray_cols=args.subarray_cols,
            window_type=args.window_type,
            flip_azim_sign=flip_azim_sign,
        )
        if w.shape[0] != M:
            raise RuntimeError(f"Precoding length {w.shape[0]} != num antennas {M}")

        rx_dbm = compute_beammap_los(
            rx_positions_xyz=rx_xyz,
            tx_center_xyz=tx_center,
            antenna_offsets_xyz=ant_offsets,
            w_precoder=w,
            frequency_hz=freq,
            plane_wave=plane_wave,
            use_element_gain=use_element_gain,
            tx_power_dbm=args.tx_power_dbm,
        )

        # Reshape to your final map layout: 8 blocks stacked vertically => (128, 128)
        # Each block is 16x128; 8 blocks -> 128x128
        map_2d = rx_dbm.reshape(8, 16, 128).reshape(128, 128)

        beam_key = f"beam_{b:02d}_angle_{theta:.1f}"
        beam_keys.append(beam_key)
        all_beams[beam_key] = map_2d

        # Save matrix
        np.save(save_dir / f"{beam_key}_matrix.npy", map_2d)

        # Save plot
        plt.figure(figsize=(10, 9))
        plt.imshow(map_2d, vmin=args.vmin, vmax=args.vmax)
        plt.colorbar(label="Rx power (dBm)")
        plt.title(f"{args.scene} | {beam_key} | {freq/1e9:.1f} GHz | {wave_tag}")
        plt.xlabel("X index")
        plt.ylabel("Y index")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_dir / f"{beam_key}_plot.png", dpi=180)
        plt.close()

        print(f"    saved: {beam_key} (max={map_2d.max():.2f}, min={map_2d.min():.2f}) dBm")

    # Save .npz summary
    npz_path = save_dir / f"{args.scene}_all_beams.npz"
    np.savez(npz_path, **all_beams)
    print(f"\nSaved all beams: {npz_path} ({len(all_beams)} beams)")

    # Save comparison grid
    ncols = 8
    nrows = int(np.ceil(args.num_beams / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.2))
    if nrows == 1:
        axes = np.asarray(axes).reshape(1, -1)
    if ncols == 1:
        axes = np.asarray(axes).reshape(-1, 1)

    for idx, k in enumerate(beam_keys):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        ax.imshow(all_beams[k], vmin=args.vmin, vmax=args.vmax)
        ax.set_title(k.split("_angle_")[-1] + "°", fontsize=8)
        ax.axis("off")

    for idx in range(len(beam_keys), nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].axis("off")

    plt.tight_layout()
    comp_path = save_dir / f"{args.scene}_all_beams_comparison.png"
    plt.savefig(comp_path, dpi=160)
    plt.close()
    print(f"Saved comparison: {comp_path}")
    print(f"Output dir: {save_dir}\n")


if __name__ == "__main__":
    main()