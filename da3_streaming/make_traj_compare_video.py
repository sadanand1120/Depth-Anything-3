#!/usr/bin/env python3

import argparse
import math
from pathlib import Path

import cv2
import imageio
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a passive trajectory-comparison MP4 from DA3 streaming poses and GT poses."
    )
    parser.add_argument("da3_output_dir", type=Path, help="DA3 streaming output directory")
    parser.add_argument(
        "--pose-file",
        type=Path,
        default=None,
        help="DA3 camera_poses.txt path. Defaults to <da3_output_dir>/camera_poses.txt",
    )
    parser.add_argument(
        "--gt-pose-dir",
        type=Path,
        required=True,
        help="Directory containing per-frame GT pose txt files, e.g. ScanNet pose/*.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output mp4 path. Defaults to <da3_output_dir>/da3_aux_vis/traj.mp4",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS")
    parser.add_argument("--plane", choices=["xy", "xz", "yz"], default="xy", help="2D plane used for plotting")
    parser.add_argument(
        "--alignment",
        choices=["sim3", "se3", "first"],
        default="sim3",
        help="Alignment used to overlay prediction onto GT",
    )
    parser.add_argument("--size", type=int, default=900, help="Square canvas size in pixels")
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export aligned predicted poses as per-frame txt files in ScanNet GT pose format.",
    )
    return parser.parse_args()


def pose_sort_key(path: Path):
    return int(path.stem) if path.stem.isdigit() else path.name


def load_da3_pose_file(pose_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not pose_file.is_file():
        raise FileNotFoundError(f"Missing DA3 pose file: {pose_file}")

    frame_ids = []
    pose_mats = []
    xyz = []
    with pose_file.open() as f:
        for frame_id, line in enumerate(f):
            if not line.strip():
                continue
            vals = np.fromstring(line, sep=" ", dtype=np.float64)
            if vals.size != 16:
                continue
            pose = vals.reshape(4, 4)
            if not np.isfinite(pose).all():
                continue
            frame_ids.append(frame_id)
            pose_mats.append(pose)
            xyz.append(pose[:3, 3])

    if len(frame_ids) == 0:
        raise ValueError(f"No valid DA3 poses found in {pose_file}")

    return (
        np.asarray(frame_ids, dtype=np.int64),
        np.asarray(pose_mats, dtype=np.float64),
        np.asarray(xyz, dtype=np.float64),
    )


def load_gt_pose_dir(gt_pose_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose_files = sorted(gt_pose_dir.glob("*.txt"), key=pose_sort_key)
    if len(pose_files) == 0:
        raise FileNotFoundError(f"No GT pose txt files found under {gt_pose_dir}")

    frame_ids = []
    pose_mats = []
    xyz = []
    for pose_file in tqdm(pose_files, desc="Loading GT poses"):
        pose = np.loadtxt(pose_file, dtype=np.float64)
        if pose.shape != (4, 4):
            continue
        if not np.isfinite(pose).all():
            continue
        if not pose_file.stem.isdigit():
            raise ValueError(f"GT pose filenames must be numeric to match DA3 frame indices: {pose_file}")
        frame_ids.append(int(pose_file.stem))
        pose_mats.append(pose)
        xyz.append(pose[:3, 3])

    if len(frame_ids) == 0:
        raise ValueError(f"No valid GT poses found under {gt_pose_dir}")

    return (
        np.asarray(frame_ids, dtype=np.int64),
        np.asarray(pose_mats, dtype=np.float64),
        np.asarray(xyz, dtype=np.float64),
    )


def umeyama_alignment(src_xyz: np.ndarray, dst_xyz: np.ndarray, estimate_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    if src_xyz.shape != dst_xyz.shape or src_xyz.shape[0] < 2:
        raise ValueError("Need at least two matched points for alignment")

    src_mean = src_xyz.mean(axis=0)
    dst_mean = dst_xyz.mean(axis=0)
    src_centered = src_xyz - src_mean
    dst_centered = dst_xyz - dst_mean
    cov = (dst_centered.T @ src_centered) / src_xyz.shape[0]
    u, singular_vals, vh = np.linalg.svd(cov)

    sign = np.eye(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        sign[-1, -1] = -1.0

    rotation = u @ sign @ vh
    if estimate_scale:
        src_var = np.mean(np.sum(src_centered * src_centered, axis=1))
        scale = np.sum(singular_vals * np.diag(sign)) / src_var
    else:
        scale = 1.0

    translation = dst_mean - scale * (rotation @ src_mean)
    return float(scale), rotation, translation


def apply_alignment(xyz: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (xyz @ rotation.T) + translation


def apply_alignment_to_pose_mats(
    pose_mats: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    aligned = pose_mats.copy()
    aligned[:, :3, :3] = rotation[None] @ aligned[:, :3, :3]
    aligned[:, :3, 3] = apply_alignment(aligned[:, :3, 3], scale, rotation, translation)
    aligned[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return aligned


def apply_first_pose_alignment(
    pred_pose_mats: np.ndarray,
    pred_first_pose: np.ndarray,
    gt_first_pose: np.ndarray,
) -> np.ndarray:
    first_to_gt = gt_first_pose @ np.linalg.inv(pred_first_pose)
    aligned = first_to_gt[None] @ pred_pose_mats
    aligned[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return aligned


def export_aligned_pose_dir(export_dir: Path, frame_ids: np.ndarray, pose_mats: np.ndarray) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    for frame_id, pose_mat in tqdm(zip(frame_ids.tolist(), pose_mats), total=len(frame_ids), desc="Exporting aligned poses"):
        np.savetxt(export_dir / f"{frame_id}.txt", pose_mat, fmt="%.6f")


def plane_indices(plane: str) -> tuple[int, int]:
    return {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]


def nice_grid_step(extent: float) -> float:
    if extent <= 0:
        return 1.0
    raw = extent / 8.0
    power = 10 ** math.floor(math.log10(raw))
    for mult in (1.0, 2.0, 5.0, 10.0):
        step = power * mult
        if step >= raw:
            return step
    return power * 10.0


def make_world_to_image(all_xy: np.ndarray, size: int, margin: int = 70):
    xy_min = all_xy.min(axis=0)
    xy_max = all_xy.max(axis=0)
    center = 0.5 * (xy_min + xy_max)
    half_extent = 0.5 * np.max(xy_max - xy_min)
    half_extent = max(half_extent, 1e-3) * 1.1
    scale = (size - 2 * margin) / (2.0 * half_extent)

    def project(xy: np.ndarray) -> np.ndarray:
        proj = (xy - center) * scale
        proj[:, 0] += size / 2.0
        proj[:, 1] = size / 2.0 - proj[:, 1]
        return np.round(proj).astype(np.int32)

    return center, half_extent, scale, project


def draw_grid(canvas: np.ndarray, half_extent: float, scale: float, plane: str) -> None:
    grid_step = nice_grid_step(2.0 * half_extent)
    size = canvas.shape[0]
    origin_px = np.array([size / 2.0, size / 2.0])
    n_steps = int(math.ceil(half_extent / grid_step))
    grid_color = (225, 225, 225)
    axis_color = (170, 170, 170)

    for i in range(-n_steps, n_steps + 1):
        x_px = int(round(origin_px[0] + i * grid_step * scale))
        cv2.line(canvas, (x_px, 0), (x_px, size - 1), grid_color, 1, cv2.LINE_AA)
        y_px = int(round(origin_px[1] - i * grid_step * scale))
        cv2.line(canvas, (0, y_px), (size - 1, y_px), grid_color, 1, cv2.LINE_AA)

    cv2.line(canvas, (int(round(origin_px[0])), 0), (int(round(origin_px[0])), size - 1), axis_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (0, int(round(origin_px[1]))), (size - 1, int(round(origin_px[1]))), axis_color, 1, cv2.LINE_AA)

    axis_names = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}[plane]
    cv2.putText(canvas, axis_names[0], (size - 24, int(round(origin_px[1])) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, axis_names[1], (int(round(origin_px[0])) + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, axis_color, 1, cv2.LINE_AA)


def draw_polyline(canvas: np.ndarray, pts_px: np.ndarray, color: tuple[int, int, int]) -> None:
    if len(pts_px) >= 2:
        cv2.polylines(canvas, [pts_px.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
    if len(pts_px) >= 1:
        cv2.circle(canvas, tuple(pts_px[-1]), 5, color, -1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    pose_file = args.pose_file if args.pose_file is not None else args.da3_output_dir / "camera_poses.txt"
    pred_inds, pred_pose_mats, pred_xyz = load_da3_pose_file(pose_file)
    gt_inds, gt_pose_mats, gt_xyz = load_gt_pose_dir(args.gt_pose_dir)

    pred_map = {int(idx): xyz for idx, xyz in zip(pred_inds.tolist(), pred_xyz)}
    gt_map = {int(idx): xyz for idx, xyz in zip(gt_inds.tolist(), gt_xyz)}
    pred_pose_map = {int(idx): pose for idx, pose in zip(pred_inds.tolist(), pred_pose_mats)}
    gt_pose_map = {int(idx): pose for idx, pose in zip(gt_inds.tolist(), gt_pose_mats)}
    overlap_ids = np.asarray(sorted(set(pred_map) & set(gt_map)), dtype=np.int64)
    min_overlap = 1 if args.alignment == "first" else 2
    if overlap_ids.size < min_overlap:
        raise ValueError("Not enough overlapping GT/pred frames to align trajectories")

    if args.alignment == "first":
        first_id = int(overlap_ids[0])
        pred_pose_mats_aligned = apply_first_pose_alignment(
            pred_pose_mats,
            pred_pose_map[first_id],
            gt_pose_map[first_id],
        )
        pred_xyz_aligned = pred_pose_mats_aligned[:, :3, 3]
    else:
        pred_overlap = np.stack([pred_map[int(idx)] for idx in overlap_ids], axis=0)
        gt_overlap = np.stack([gt_map[int(idx)] for idx in overlap_ids], axis=0)
        scale, rotation, translation = umeyama_alignment(
            pred_overlap,
            gt_overlap,
            estimate_scale=args.alignment == "sim3",
        )
        pred_xyz_aligned = apply_alignment(pred_xyz, scale, rotation, translation)
        pred_pose_mats_aligned = apply_alignment_to_pose_mats(pred_pose_mats, scale, rotation, translation)

    ax0, ax1 = plane_indices(args.plane)
    gt_xy_px_src = gt_xyz[:, [ax0, ax1]]
    pred_xy_px_src = pred_xyz_aligned[:, [ax0, ax1]]
    _, half_extent, scale_px, project = make_world_to_image(
        np.concatenate([gt_xy_px_src, pred_xy_px_src], axis=0),
        size=args.size,
    )
    gt_xy_px = project(gt_xy_px_src)
    pred_xy_px = project(pred_xy_px_src)

    output_path = args.output if args.output is not None else args.da3_output_dir / "da3_aux_vis" / "traj.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_dir = args.da3_output_dir / "pose_aligned"
    if args.export:
        export_aligned_pose_dir(export_dir, pred_inds, pred_pose_mats_aligned)

    frame_ids = np.arange(0, int(max(gt_inds.max(), pred_inds.max())) + 1, dtype=np.int64)
    with imageio.get_writer(str(output_path), fps=args.fps, codec="libx264", macro_block_size=None) as writer:
        for frame_id in tqdm(frame_ids, desc="Writing trajectory video"):
            canvas = np.full((args.size, args.size, 3), 255, dtype=np.uint8)
            draw_grid(canvas, half_extent, scale_px, args.plane)

            gt_mask = gt_inds <= frame_id
            pred_mask = pred_inds <= frame_id
            draw_polyline(canvas, gt_xy_px[gt_mask], (60, 180, 75))
            draw_polyline(canvas, pred_xy_px[pred_mask], (215, 110, 30))

            pos_err_text = "Pos err N/A"
            if np.any(gt_mask) and np.any(pred_mask):
                gt_current = gt_xyz[np.flatnonzero(gt_mask)[-1]]
                pred_current = pred_xyz_aligned[np.flatnonzero(pred_mask)[-1]]
                pos_err_text = f"Pos err {np.linalg.norm(gt_current - pred_current):.3f} m"

            cv2.putText(canvas, f"Frame {frame_id}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"GT: green  Pred: blue  Align: {args.alignment}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{pos_err_text}  |  Metric scale: ScanNet GT", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
            writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    print(output_path)
    if args.export:
        print(export_dir)


if __name__ == "__main__":
    main()
