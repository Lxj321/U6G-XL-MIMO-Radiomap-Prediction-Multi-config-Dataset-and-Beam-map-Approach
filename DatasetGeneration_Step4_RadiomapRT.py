#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

# 必须尽量早（在 TF 初始化 GPU 之后就不能改了）
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        # 已初始化就忽略
        pass
tf.get_logger().setLevel('ERROR')

# 现在才 import 其他
import sionna
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver

import argparse
import time
import gc
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Coverage grid
# -----------------------------
class CoverageGridConfig:
    def __init__(self, x_min, x_max, y_min, y_max, grid_spacing, receiver_height):
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.grid_spacing = float(grid_spacing)
        self.receiver_height = float(receiver_height)

        self.num_x = int(round((self.x_max - self.x_min) / self.grid_spacing)) + 1
        self.num_y = int(round((self.y_max - self.y_min) / self.grid_spacing)) + 1
        self.total_points = self.num_x * self.num_y

        print("CoverageGrid:")
        print(f"  X: {self.x_min} ~ {self.x_max} m  (num_x={self.num_x})")
        print(f"  Y: {self.y_min} ~ {self.y_max} m  (num_y={self.num_y})")
        print(f"  spacing: {self.grid_spacing} m, rx_height: {self.receiver_height} m")
        print(f"  total_points: {self.total_points}")

    def idx_to_pos(self, global_idx: int):
        x_idx = global_idx % self.num_x
        y_idx = global_idx // self.num_x
        x = self.x_min + x_idx * self.grid_spacing
        y = self.y_min + y_idx * self.grid_spacing
        return [float(x), float(y), float(self.receiver_height)]

    def reshape_results(self, flat_results: np.ndarray):
        return flat_results.reshape(self.num_y, self.num_x)


def generate_position_batches(cfg: CoverageGridConfig, num_rx_per_batch: int):
    if num_rx_per_batch < 1:
        raise ValueError("num_rx_per_batch must be >= 1")
    num_batches = int(np.ceil(cfg.total_points / num_rx_per_batch))
    batch_meta = []
    for b in range(num_batches):
        s = b * num_rx_per_batch
        e = min(s + num_rx_per_batch, cfg.total_points)
        batch_meta.append((b, s, e, e - s))
    return batch_meta


# -----------------------------
# tau align
# -----------------------------
def auto_align_tau(a, tau):
    if a.ndim != 7:
        raise ValueError(f"a must be 7D, got {a.ndim}D")

    _, _, _, _, _, _, num_time_steps = a.shape

    if tau.ndim == 6:
        tau_expanded = tf.expand_dims(tau, axis=-1)
        tau_aligned = tf.repeat(tau_expanded, repeats=num_time_steps, axis=-1)
        return tau_aligned

    if tau.ndim == 4:
        tau_with_rx_ant = tf.expand_dims(tau, axis=2)
        tau_with_ants = tf.expand_dims(tau_with_rx_ant, axis=4)
        tau_expanded = tf.expand_dims(tau_with_ants, axis=-1)
        tau_aligned = tf.repeat(tau_expanded, repeats=num_time_steps, axis=-1)
        return tau_aligned

    raise ValueError(f"tau must be 6D or 4D, got {tau.ndim}D")


# -----------------------------
# Codebook (RT version kron order)
# -----------------------------
def generate_extended_codebook(num_rows, num_cols, theta_azim, theta_elev=0,
                              subarray_rows=1, subarray_cols=1, window_type='rect'):
    theta_azim_rad = np.radians(theta_azim)
    theta_elev_rad = np.radians(theta_elev)
    d_lambda = 0.5

    def get_window(size):
        if window_type == 'hann':
            return np.hanning(size)
        return np.ones(size)

    num_subarrays_rows = num_rows // subarray_rows
    num_subarrays_cols = num_cols // subarray_cols

    subarray_list = []
    for _sr in range(num_subarrays_rows):
        for _sc in range(num_subarrays_cols):
            elev_indices = np.arange(subarray_rows)
            phase_elev = 2 * np.pi * d_lambda * elev_indices * np.sin(theta_elev_rad)
            v_elev = get_window(subarray_rows).reshape(-1, 1) * np.exp(1j * phase_elev).reshape(-1, 1)

            azim_indices = np.arange(subarray_cols)
            phase_azim = 2 * np.pi * d_lambda * azim_indices * np.sin(theta_azim_rad)
            v_azim = get_window(subarray_cols).reshape(-1, 1) * np.exp(1j * phase_azim).reshape(-1, 1)

            # RT: kron(v_azim, v_elev)
            subarray_vec = np.kron(v_azim, v_elev)
            subarray_list.append(subarray_vec)

    w = np.concatenate(subarray_list, axis=0).flatten()
    w = w / (np.linalg.norm(w) + 1e-30)
    return w


# -----------------------------
# One batch processing
# -----------------------------
def process_receiver_batch(scene,
                           precoding_w: np.ndarray,
                           cfg: CoverageGridConfig,
                           batch_start: int,
                           batch_count: int,
                           *,
                           max_depth: int,
                           num_samples: float,
                           enable_los: bool,
                           enable_reflection: bool,
                           enable_diffraction: bool,
                           synthetic_array: bool,
                           tx_power_dbm: float,
                           ref_tx_power_dbm: float,
                           floor_db: float = -300.0):
    """
    Returns: (batch_count,) array in dB

    tx_power behavior:
      - For points > floor_db (non-floor): add (tx_power_dbm - ref_tx_power_dbm)
      - For points at/under floor_db: keep floor_db unchanged
    """
    scene.synthetic_array = bool(synthetic_array)

    # remove old receivers
    for name in list(scene.receivers.keys()):
        if name.startswith("rx_"):
            scene.remove(name)

    # add receivers for this batch
    for i in range(batch_count):
        pos = cfg.idx_to_pos(batch_start + i)
        scene.add(Receiver(name=f"rx_{i}", position=pos, orientation=[0, 0, 0]))

    # trace
    paths = scene.compute_paths(
        max_depth=int(max_depth),
        num_samples=float(num_samples),
        los=bool(enable_los),
        reflection=bool(enable_reflection),
        diffraction=bool(enable_diffraction),
    )

    a, tau = paths.cir()
    tau = auto_align_tau(a, tau)

    # numpy post-processing
    a = a.numpy()
    tau = tau.numpy()
    omega = 2 * np.pi * float(scene.frequency.numpy())

    phase_shifts = np.exp(-1j * omega * tau)
    a_with_phase = a * phase_shifts
    a_multipath = np.sum(a_with_phase, axis=5)  # sum over paths

    a_precoded = np.tensordot(a_multipath, np.conj(precoding_w), axes=([4], [0]))
    power_synth = np.square(np.abs(a_precoded))

    # numeric floor at 1e-30 -> about -300 dB
    result_db = 10.0 * np.log10(power_synth + 1e-30)

    # extract [num_rx]
    out = result_db[0, :, 0, 0, 0].astype(np.float64)

    # --- enforce "tx_power shifts only non-floor points" ---
    delta_db = float(tx_power_dbm) - float(ref_tx_power_dbm)

    # Treat near-floor as floor (robust to tiny numerical noise)
    eps = 1e-6
    is_floor = out <= (floor_db + eps)

    # Shift non-floor points
    if abs(delta_db) > 0:
        out[~is_floor] = out[~is_floor] + delta_db

    # Keep floor invariant
    out[is_floor] = floor_db

    # cleanup
    del a, tau, phase_shifts, a_with_phase, a_multipath, a_precoded, power_synth, result_db, paths
    gc.collect()

    return out


# -----------------------------
# Scene processing (multi-beam)
# -----------------------------
def run_scene(args):
    sionna.config.seed = int(args.seed)
    np.random.seed(int(args.seed))

    # load scene
    if args.building_exist:
        scene = load_scene(str(Path(args.scene_root) / args.scene_name / f"{args.scene_name}.xml"))
    else:
        scene = load_scene()

    # arrays
    scene.tx_array = PlanarArray(
        num_rows=args.num_rows,
        num_cols=args.num_cols,
        pattern=args.tx_pattern,
        polarization=args.tx_polarization,
        vertical_spacing=args.vertical_spacing,
        horizontal_spacing=args.horizontal_spacing,
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        pattern=args.rx_pattern,
        polarization=args.rx_polarization,
        vertical_spacing=args.vertical_spacing,
        horizontal_spacing=args.horizontal_spacing,
    )

    # tx (note: in your current pipeline this may not affect CIR scaling, but we still set it)
    tx = Transmitter(
        name="tx",
        position=[args.tx_x, args.tx_y, args.tx_z],
        orientation=[args.tx_yaw, args.tx_pitch, args.tx_roll],
        power_dbm=args.tx_power_dbm,
    )
    scene.add(tx)

    # frequency
    scene.frequency = float(args.frequency_hz)

    # grid
    cfg = CoverageGridConfig(
        x_min=args.x_min, x_max=args.x_max,
        y_min=args.y_min, y_max=args.y_max,
        grid_spacing=args.grid_spacing,
        receiver_height=args.receiver_height,
    )
    batches = generate_position_batches(cfg, args.num_rx_per_batch)
    print(f"\nBatching: num_rx_per_batch={args.num_rx_per_batch}, num_batches={len(batches)}")

    # beams
    beam_angles = [args.start_angle_deg + i * args.beam_spacing_deg for i in range(args.num_beams)]
    print(f"\nBeams: num_beams={args.num_beams}, azim=[{beam_angles[0]}..{beam_angles[-1]}] deg, spacing={args.beam_spacing_deg}")

    # output dir (include tx_power to avoid overwriting)
    freq_tag = f"{args.frequency_hz/1e9:.1f}GHz"
    param_folder = (
        f"scene_{args.scene_name}_freq_{freq_tag}_{args.num_rows*args.num_cols}TR_"
        f"{args.num_beams}beams_pat_{args.tx_pattern}_syn_{int(args.synthetic_array)}_"
        f"txp{args.tx_power_dbm:.1f}dBm_ref{args.ref_tx_power_dbm:.1f}dBm_"
        f"RTdepth{args.max_depth}_samp{args.num_samples:g}_refl{int(args.reflection)}_diff{int(args.diffraction)}"
    )
    save_dir = Path(args.out_root) / param_folder
    save_dir.mkdir(parents=True, exist_ok=True)

    # save settings
    settings_path = save_dir / "beam_settings.txt"
    with open(settings_path, "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")
        f.write(f"grid_num_x: {cfg.num_x}\n")
        f.write(f"grid_num_y: {cfg.num_y}\n")
        f.write(f"grid_total_points: {cfg.total_points}\n")

    all_beams = {}

    for b, theta_azim in enumerate(beam_angles):
        print(f"\n[{b+1:02d}/{args.num_beams}] Beam azim={theta_azim:.2f} deg")
        beam_t0 = time.time()

        w = generate_extended_codebook(
            args.num_rows, args.num_cols,
            theta_azim=theta_azim,
            theta_elev=args.theta_elev_deg,
            subarray_rows=args.subarray_rows,
            subarray_cols=args.subarray_cols,
            window_type=args.window_type,
        )

        batch_outs = []
        for (bi, s, e, cnt) in batches:
            if bi % args.print_every_batches == 0 or bi == len(batches) - 1:
                prog = (bi + 1) / len(batches) * 100
                print(f"  batch {bi+1:4d}/{len(batches)} ({prog:5.1f}%)  rx={cnt}")

            out = process_receiver_batch(
                scene, w, cfg, s, cnt,
                max_depth=args.max_depth,
                num_samples=args.num_samples,
                enable_los=bool(args.los),
                enable_reflection=bool(args.reflection),
                enable_diffraction=bool(args.diffraction),
                synthetic_array=bool(args.synthetic_array),
                tx_power_dbm=float(args.tx_power_dbm),
                ref_tx_power_dbm=float(args.ref_tx_power_dbm),
                floor_db=float(args.floor_db),
            )
            batch_outs.append(out)

        flat = np.concatenate(batch_outs, axis=0)
        map2d = cfg.reshape_results(flat)
        key = f"beam_{b:02d}_angle_{theta_azim:.1f}"
        all_beams[key] = map2d

        if args.save_npy:
            np.save(save_dir / f"{key}_matrix.npy", map2d)

        if args.save_png:
            plt.figure(figsize=(10, 9))
            plt.imshow(map2d, vmin=args.vmin, vmax=args.vmax)
            plt.colorbar(label="Value (dB)")
            plt.title(f"{args.scene_name} | {key} | txp={args.tx_power_dbm:.1f} dBm")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(save_dir / f"{key}_plot.png", dpi=180)
            plt.close()

        print(f"  done. range=({map2d.max():.2f},{map2d.min():.2f}) dB, time={time.time()-beam_t0:.1f}s")

    if args.save_npz:
        np.savez(save_dir / f"{args.scene_name}_all_beams.npz", **all_beams)

    if args.save_png and args.save_comparison:
        ncols = min(8, args.num_beams)
        nrows = int(np.ceil(args.num_beams / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.2))
        axes = np.asarray(axes).reshape(nrows, ncols)
        keys = list(all_beams.keys())
        for i, k in enumerate(keys):
            r, c = divmod(i, ncols)
            axes[r, c].imshow(all_beams[k], vmin=args.vmin, vmax=args.vmax)
            axes[r, c].set_title(k.split("_angle_")[-1] + "°", fontsize=8)
            axes[r, c].axis("off")
        for i in range(len(keys), nrows * ncols):
            r, c = divmod(i, ncols)
            axes[r, c].axis("off")
        plt.tight_layout()
        plt.savefig(save_dir / f"{args.scene_name}_all_beams_comparison.png", dpi=160)
        plt.close()

    print(f"\nSaved to: {save_dir}")
    del scene
    gc.collect()
    tf.keras.backend.clear_session()


def build_argparser():
    p = argparse.ArgumentParser("RT Radiomap Multi-beam Generator")

    # scene
    p.add_argument("--scene_name", type=str, default="u1")
    p.add_argument("--scene_root", type=str, default="maps")
    p.add_argument("--building_exist", type=int, default=1)

    # seed
    p.add_argument("--seed", type=int, default=42)

    # frequency
    p.add_argument("--frequency_hz", type=float, default=6.7e9)

    # tx
    p.add_argument("--tx_x", type=float, default=0.0)
    p.add_argument("--tx_y", type=float, default=0.0)
    p.add_argument("--tx_z", type=float, default=40.0)
    p.add_argument("--tx_yaw", type=float, default=0.0)
    p.add_argument("--tx_pitch", type=float, default=0.0)
    p.add_argument("--tx_roll", type=float, default=0.0)

    # tx power behavior you want:
    p.add_argument("--tx_power_dbm", type=float, default=0.0, help="Shift applied to non-floor points (in dBm)")
    p.add_argument("--ref_tx_power_dbm", type=float, default=0.0, help="Reference power (default 0 dBm). Delta shifts non-floor points.")

    # arrays
    p.add_argument("--num_rows", type=int, default=32)
    p.add_argument("--num_cols", type=int, default=32)
    p.add_argument("--vertical_spacing", type=float, default=0.5)
    p.add_argument("--horizontal_spacing", type=float, default=0.5)
    p.add_argument("--tx_pattern", type=str, default="tr38901")
    p.add_argument("--tx_polarization", type=str, default="V")
    p.add_argument("--rx_pattern", type=str, default="iso")
    p.add_argument("--rx_polarization", type=str, default="V")

    # near/far-field toggle in RT
    p.add_argument("--synthetic_array", type=int, default=1,
                   help="1=use synthetic array (far-field array model); 0=use true array geometry (enables near-field effects).")

    # beams
    p.add_argument("--num_beams", type=int, default=5)
    p.add_argument("--start_angle_deg", type=float, default=-32.0)
    p.add_argument("--beam_spacing_deg", type=float, default=1.0)
    p.add_argument("--theta_elev_deg", type=float, default=0.0)
    p.add_argument("--window_type", type=str, default="rect", choices=["rect", "hann"])
    p.add_argument("--subarray_rows", type=int, default=32)
    p.add_argument("--subarray_cols", type=int, default=32)

    # coverage
    p.add_argument("--x_min", type=float, default=-635.0)
    p.add_argument("--x_max", type=float, default=635.0)
    p.add_argument("--y_min", type=float, default=-635.0)
    p.add_argument("--y_max", type=float, default=635.0)
    p.add_argument("--grid_spacing", type=float, default=10.0)
    p.add_argument("--receiver_height", type=float, default=1.5)

    # batching
    p.add_argument("--num_rx_per_batch", type=int, default=2048)
    p.add_argument("--print_every_batches", type=int, default=10)

    # RT params
    p.add_argument("--max_depth", type=int, default=3)
    p.add_argument("--num_samples", type=float, default=1e6)
    p.add_argument("--los", type=int, default=1)
    p.add_argument("--reflection", type=int, default=1)
    p.add_argument("--diffraction", type=int, default=1)

    # output
    p.add_argument("--out_root", type=str, default="simulation_results_multibeam_2026")
    p.add_argument("--save_png", type=int, default=1)
    p.add_argument("--save_npy", type=int, default=1)
    p.add_argument("--save_npz", type=int, default=1)
    p.add_argument("--save_comparison", type=int, default=1)
    p.add_argument("--vmin", type=float, default=-120.0)
    p.add_argument("--vmax", type=float, default=-60.0)

    # floor control (keep -300 invariant)
    p.add_argument("--floor_db", type=float, default=-300.0, help="Floor value to keep invariant (default -300 dB)")

    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    args.building_exist = bool(args.building_exist)
    run_scene(args)