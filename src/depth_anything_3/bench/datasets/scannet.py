# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ScanNet benchmark dataset implementation for internal streaming evaluation.

This dataset uses:
- processed RGB/pose/intrinsic/depth frames from a ScanNet export
- ground-truth geometry from the raw ScanNet scan mesh
"""

import json
import gc
import os
import time
import zipfile

from io import BytesIO
from pathlib import Path
from typing import Dict as TDict

import cv2
import numpy as np
import open3d as o3d
import torch
from addict import Dict
from tqdm import tqdm

from depth_anything_3.bench.dataset import Dataset
from depth_anything_3.bench.registries import MONO_REGISTRY, MV_REGISTRY
from depth_anything_3.bench.utils import (
    create_tsdf_volume,
    evaluate_3d_reconstruction_l2,
    fuse_depth_to_tsdf,
    sample_points_from_mesh,
)
from depth_anything_3.utils.image_order import sort_image_sequence
from depth_anything_3.utils.pose_align import align_poses_umeyama


def _scannet_input_root() -> str:
    return os.environ.get(
        "DA3_SCANNET_INPUT_ROOT",
        "/robodata/smodak/repos/ovo/data/input/ScanNet",
    )


def _scannet_raw_root() -> str:
    return os.environ.get(
        "DA3_SCANNET_RAW_ROOT",
        "/robodata/smodak/datasets/scannet_v2/scans",
    )


def _scene_sort_key(name: str):
    prefix = name.replace("scene", "")
    major, minor = prefix.split("_")
    return int(major), int(minor)


def _load_scene_list():
    root = Path(_scannet_input_root())
    if not root.exists():
        return []
    scenes = [p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("scene")]
    return sorted(scenes, key=_scene_sort_key)


def _image_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def _image_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    scores = []
    for channel in range(3):
        x = pred[:, :, channel].astype(np.float32, copy=False)
        y = target[:, :, channel].astype(np.float32, copy=False)

        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
        sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy

        numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        scores.append(float(np.mean(numerator / np.maximum(denominator, 1e-12))))
    return float(np.mean(scores))


def _voxelized_render_arrays(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    render_pcd = pcd.voxel_down_sample(voxel_size)
    points = np.asarray(render_pcd.points).astype(np.float32, copy=False)
    colors = np.asarray(render_pcd.colors).astype(np.float32, copy=False)
    if len(colors) != len(points):
        colors = np.zeros((len(points), 3), dtype=np.float32)
    return points, np.clip(colors, 0.0, 1.0)


def _point_cloud_from_arrays(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return pcd


def _load_cached_point_cloud(cache_path: Path) -> o3d.geometry.PointCloud | None:
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as data:
        points = data["points"]
        colors = data["colors"] if "colors" in data else None
    return _point_cloud_from_arrays(points, colors)


def _write_cached_point_cloud(cache_path: Path, pcd: o3d.geometry.PointCloud) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(pcd.points).astype(np.float32, copy=False)
    colors = np.asarray(pcd.colors).astype(np.float32, copy=False)
    if len(colors) != len(points):
        colors = np.zeros((len(points), 3), dtype=np.float32)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez(f, points=points, colors=np.clip(colors, 0.0, 1.0))
    os.replace(tmp_path, cache_path)


def _load_cached_render_arrays(
    cache_path: Path,
    source_path: str,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not cache_path.exists():
        return None
    source_stat = os.stat(source_path)
    with np.load(cache_path, allow_pickle=False) as data:
        if int(data["source_mtime_ns"]) != source_stat.st_mtime_ns:
            return None
        if int(data["source_size"]) != source_stat.st_size:
            return None
        if float(data["voxel_size"]) != float(voxel_size):
            return None
        return data["points"], data["colors"]


def _write_cached_render_arrays(
    cache_path: Path,
    source_path: str,
    voxel_size: float,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    source_stat = os.stat(source_path)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez(
            f,
            points=points.astype(np.float32, copy=False),
            colors=colors.astype(np.float32, copy=False),
            source_mtime_ns=np.array(source_stat.st_mtime_ns, dtype=np.int64),
            source_size=np.array(source_stat.st_size, dtype=np.int64),
            voxel_size=np.array(voxel_size, dtype=np.float32),
        )
    os.replace(tmp_path, cache_path)


def _make_offset_shells(radius_cap: int, device: torch.device) -> list[tuple[int, torch.Tensor]]:
    shells = []
    for radius in range(radius_cap + 1):
        offsets = [
            (dx, dy)
            for dy in range(-radius_cap, radius_cap + 1)
            for dx in range(-radius_cap, radius_cap + 1)
            if max(abs(dx), abs(dy)) == radius
        ]
        shells.append((radius, torch.tensor(offsets, dtype=torch.int64, device=device)))
    return shells


@torch.no_grad()
def _render_voxel_cloud_cuda(
    points: torch.Tensor,
    colors: torch.Tensor,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
    voxel_size: float,
    chunk_size: int,
    radius_cap: int,
    offset_shells: list[tuple[int, torch.Tensor]],
) -> np.ndarray:
    device = points.device
    zbuf = torch.full((height * width,), float("inf"), dtype=torch.float32, device=device)
    image = torch.zeros((height * width, 3), dtype=torch.float32, device=device)

    r = torch.as_tensor(extrinsic[:3, :3], dtype=torch.float32, device=device)
    t = torch.as_tensor(extrinsic[:3, 3], dtype=torch.float32, device=device)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    max_focal = max(abs(fx), abs(fy))

    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        cam = points[start:end] @ r.T + t
        z = cam[:, 2]
        valid = z > 1e-4
        if not bool(valid.any().item()):
            continue

        cam = cam[valid]
        z = z[valid]
        cols = colors[start:end][valid]
        u = torch.round(fx * cam[:, 0] / z + cx).to(torch.int64)
        v = torch.round(fy * cam[:, 1] / z + cy).to(torch.int64)
        radii = torch.ceil(0.5 * voxel_size * max_focal / z).to(torch.int64).clamp_(0, radius_cap)

        near_image = (u >= -radius_cap) & (u < width + radius_cap) & (v >= -radius_cap) & (v < height + radius_cap)
        if not bool(near_image.any().item()):
            continue

        u = u[near_image]
        v = v[near_image]
        z = z[near_image]
        cols = cols[near_image]
        radii = radii[near_image]

        for radius, offsets in offset_shells:
            if radius > 0:
                active = radii >= radius
                if not bool(active.any().item()):
                    continue
                u0 = u[active]
                v0 = v[active]
                z0 = z[active]
                cols0 = cols[active]
            else:
                u0, v0, z0, cols0 = u, v, z, cols

            uu = u0[:, None] + offsets[:, 0][None, :]
            vv = v0[:, None] + offsets[:, 1][None, :]
            inside = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            if not bool(inside.any().item()):
                continue

            rows, cols_idx = inside.nonzero(as_tuple=True)
            pix = vv[rows, cols_idx] * width + uu[rows, cols_idx]
            pix_z = z0[rows]
            zbuf.scatter_reduce_(0, pix, pix_z, reduce="amin", include_self=True)
            winners = torch.abs(pix_z - zbuf[pix]) <= 1e-6
            if bool(winners.any().item()):
                image[pix[winners]] = cols0[rows[winners]]

    return image.reshape(height, width, 3).cpu().numpy()


@MV_REGISTRY.register(name="scannet")
@MONO_REGISTRY.register(name="scannet")
class ScanNetDataset(Dataset):
    """
    ScanNet benchmark wrapper for DA3 streaming evaluation.

    Metrics:
    - Pose: AUC on relative pose error, identical to the native evaluator
    - Recon: TSDF/backproject fusion + L2 nearest-neighbor geometry + RGB render PSNR/SSIM
    """

    data_root = _scannet_input_root()
    raw_root = _scannet_raw_root()
    SCENES = _load_scene_list()

    max_depth = 5.0
    sampling_number = 10_000_000
    voxel_length = 0.02
    sdf_trunc = 0.15
    render_voxel_size = 0.001
    render_num_images = 800
    render_chunk_size = 1_000_000
    render_radius_cap = 8

    def __init__(self):
        super().__init__()
        self._scene_cache = {}

    def _refresh_roots(self):
        self.data_root = _scannet_input_root()
        self.raw_root = _scannet_raw_root()

    def get_data(self, scene: str) -> Dict:
        self._refresh_roots()
        if scene in self._scene_cache:
            return self._scene_cache[scene]

        scene_dir = Path(self.data_root) / scene
        color_dir = scene_dir / "color"
        pose_dir = scene_dir / "pose"
        depth_dir = scene_dir / "depth"
        intrinsic_path = scene_dir / "intrinsic" / "intrinsic_color.txt"

        if not color_dir.is_dir():
            raise FileNotFoundError(f"Missing ScanNet color dir: {color_dir}")
        if not pose_dir.is_dir():
            raise FileNotFoundError(f"Missing ScanNet pose dir: {pose_dir}")
        if not intrinsic_path.is_file():
            raise FileNotFoundError(f"Missing ScanNet intrinsic file: {intrinsic_path}")

        raw_scene_dir = Path(self.raw_root) / scene
        gt_mesh_path = raw_scene_dir / f"{scene}_vh_clean_2.ply"
        if not gt_mesh_path.is_file():
            fallback = raw_scene_dir / f"{scene}_vh_clean.ply"
            if fallback.is_file():
                gt_mesh_path = fallback
            else:
                raise FileNotFoundError(
                    f"Missing ScanNet GT mesh: {gt_mesh_path} (and fallback {fallback})"
                )

        intrinsic_4x4 = np.loadtxt(intrinsic_path, dtype=np.float32)
        ixt_shared = intrinsic_4x4[:3, :3]

        image_files = sort_image_sequence(color_dir.glob("*.jpg"))

        out = Dict(
            {
                "image_files": [],
                "extrinsics": [],
                "intrinsics": [],
                "aux": Dict(
                    {
                        "gt_mesh_path": str(gt_mesh_path),
                        "gt_depth_files": [],
                    }
                ),
            }
        )

        for img_path in tqdm(image_files, desc=f"[ScanNet] {scene} load poses", leave=False):
            frame_id = img_path.stem
            pose_path = pose_dir / f"{frame_id}.txt"
            depth_path = depth_dir / f"{frame_id}.png"
            if not pose_path.is_file():
                continue

            c2w = np.loadtxt(pose_path, dtype=np.float32)
            if c2w.shape != (4, 4):
                continue

            out.image_files.append(str(img_path))
            out.extrinsics.append(np.linalg.inv(c2w).astype(np.float32))
            out.intrinsics.append(ixt_shared.copy())
            out.aux.gt_depth_files.append(str(depth_path))

        out.extrinsics = np.asarray(out.extrinsics, dtype=np.float32)
        out.intrinsics = np.asarray(out.intrinsics, dtype=np.float32)

        self._scene_cache[scene] = out
        tqdm.write(f"[ScanNet] {scene}: {len(out.image_files)} images")
        return out

    def eval3d(self, scene: str, fuse_path: str) -> TDict[str, float]:
        start_eval = time.perf_counter()
        full_gt_data = self.get_data(scene)
        tqdm.write(f"[ScanNet] eval3d start | {scene} | {fuse_path}")
        gt_cache_path = Path(fuse_path).parents[3] / "eval_cache" / f"gt_sample_{self.sampling_number}.npz"
        start_gt = time.perf_counter()
        gt_pcd = _load_cached_point_cloud(gt_cache_path)
        if gt_pcd is None:
            tqdm.write(f"[ScanNet] sample GT start | {scene} | points={self.sampling_number:,}")
            gt_mesh = o3d.io.read_triangle_mesh(full_gt_data.aux.gt_mesh_path)
            gt_pcd = sample_points_from_mesh(gt_mesh, self.sampling_number)
            _write_cached_point_cloud(gt_cache_path, gt_pcd)
            tqdm.write(f"[ScanNet] sample GT done  | {scene} | {time.perf_counter() - start_gt:.2f}s")
        else:
            tqdm.write(
                f"[ScanNet] sample GT cache hit | {scene} | "
                f"points={len(gt_pcd.points):,} | {time.perf_counter() - start_gt:.2f}s"
            )

        start_read = time.perf_counter()
        pred_pcd = o3d.io.read_point_cloud(fuse_path)
        tqdm.write(f"[ScanNet] read pred done | {scene} | points={len(pred_pcd.points):,} | {time.perf_counter() - start_read:.2f}s")
        pred_eval_pcd = pred_pcd

        start_crop = time.perf_counter()
        aabb = gt_pcd.get_axis_aligned_bounding_box()
        points = np.asarray(pred_pcd.points)
        if points.size > 0:
            inside_mask = (
                (points[:, 0] >= aabb.min_bound[0] - 0.1)
                & (points[:, 0] <= aabb.max_bound[0] + 0.1)
                & (points[:, 1] >= aabb.min_bound[1] - 0.1)
                & (points[:, 1] <= aabb.max_bound[1] + 0.1)
                & (points[:, 2] >= aabb.min_bound[2] - 0.1)
                & (points[:, 2] <= aabb.max_bound[2] + 0.1)
            )
            pred_eval_pcd = pred_pcd.select_by_index(np.nonzero(inside_mask)[0])
        tqdm.write(
            f"[ScanNet] AABB crop done | {scene} | "
            f"eval_points={len(pred_eval_pcd.points):,}/{len(pred_pcd.points):,} | {time.perf_counter() - start_crop:.2f}s"
        )

        start_geom = time.perf_counter()
        result = evaluate_3d_reconstruction_l2(
            pred_eval_pcd,
            gt_pcd,
            progress_desc=f"{scene} {Path(fuse_path).stem} L2 NN",
        )
        tqdm.write(f"[ScanNet] geometry metric done | {scene} | {result} | {time.perf_counter() - start_geom:.2f}s")
        del gt_pcd, pred_eval_pcd, points
        gc.collect()

        export_manifest = Path(fuse_path).parents[1] / "vipe_manifest.json"
        render_data = self._load_gt_meta(str(export_manifest)) or full_gt_data
        result.update(self._compute_render_metrics(scene, Path(fuse_path).stem, pred_pcd, render_data, fuse_path))
        tqdm.write(f"[ScanNet] eval3d done | {scene} | {Path(fuse_path).stem} | {time.perf_counter() - start_eval:.2f}s")
        return result

    def _compute_render_metrics(
        self,
        scene: str,
        method_name: str,
        pcd: o3d.geometry.PointCloud,
        scene_data: Dict,
        fuse_path: str,
    ) -> TDict[str, float]:
        start_total = time.perf_counter()
        label = f"{scene} {method_name}"
        tqdm.write(
            f"[ScanNet] render metric voxelize start | {label} | voxel={self.render_voxel_size}m"
        )
        cache_name = f"{Path(fuse_path).stem}_render_vox_{str(self.render_voxel_size).replace('.', 'p')}.npz"
        render_cache_path = Path(fuse_path).with_name(cache_name)
        points, colors = _load_cached_render_arrays(render_cache_path, fuse_path, self.render_voxel_size) or (None, None)
        if points is None:
            points, colors = _voxelized_render_arrays(pcd, self.render_voxel_size)
            _write_cached_render_arrays(render_cache_path, fuse_path, self.render_voxel_size, points, colors)
            tqdm.write(
                f"[ScanNet] render metric voxelize done  | {label} | "
                f"voxels={len(points):,} | {time.perf_counter() - start_total:.2f}s"
            )
        else:
            tqdm.write(
                f"[ScanNet] render metric voxelize cache hit | {label} | "
                f"voxels={len(points):,} | {time.perf_counter() - start_total:.2f}s"
            )

        frame_indices = np.arange(len(scene_data.image_files))
        if self.render_num_images != -1:
            sample_count = min(self.render_num_images, len(frame_indices))
            rng = np.random.default_rng(seed=42)
            frame_indices = np.sort(rng.choice(frame_indices, size=sample_count, replace=False))

        tasks = [
            (
                int(idx),
                str(scene_data.image_files[idx]),
                scene_data.extrinsics[idx],
                scene_data.intrinsics[idx],
            )
            for idx in frame_indices
        ]
        psnr_values = []
        ssim_values = []
        tqdm.write(
            f"[ScanNet] render metric project start | {label} | "
            f"frames={len(tasks):,}/{len(scene_data.image_files):,} device=cuda:0"
        )

        iterator = tqdm(total=len(tasks), desc=f"{label} render PSNR/SSIM", unit="frame", leave=False)
        device = torch.device("cuda:0")
        points_t = torch.as_tensor(np.ascontiguousarray(points), dtype=torch.float32, device=device)
        colors_t = torch.as_tensor(np.ascontiguousarray(colors), dtype=torch.float32, device=device)
        offset_shells = _make_offset_shells(self.render_radius_cap, device)
        for _, image_file, extrinsic, intrinsic in tasks:
            bgr = cv2.imread(image_file, cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(image_file)
            target = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            height, width = target.shape[:2]
            render = _render_voxel_cloud_cuda(
                points_t,
                colors_t,
                extrinsic,
                intrinsic,
                height,
                width,
                self.render_voxel_size,
                self.render_chunk_size,
                self.render_radius_cap,
                offset_shells,
            )
            psnr_values.append(_image_psnr(render, target))
            ssim_values.append(_image_ssim(render, target))
            iterator.update()
            iterator.set_postfix(psnr=f"{np.mean(psnr_values):.2f}", ssim=f"{np.mean(ssim_values):.4f}")
        del points_t, colors_t, offset_shells
        torch.cuda.empty_cache()
        iterator.close()

        tqdm.write(
            f"[ScanNet] render metric project done  | {label} | "
            f"{time.perf_counter() - start_total:.2f}s"
        )
        return {
            "psnr": float(np.mean(psnr_values)),
            "ssim": float(np.mean(ssim_values)),
        }

    def _load_gt_meta(self, result_path: str) -> Dict:
        gt_meta_path = os.path.join(os.path.dirname(result_path), "gt_meta.npz")
        if os.path.exists(gt_meta_path):
            data = np.load(gt_meta_path, allow_pickle=True)
            return Dict(
                {
                    "extrinsics": data["extrinsics"],
                    "intrinsics": data["intrinsics"],
                    "image_files": list(data["image_files"]),
                }
            )
        return None

    def result_path(self, export_dir: str) -> str:
        return os.path.join(export_dir, "exports", "vipe_manifest.json")

    def result_exists(self, result_path: str) -> bool:
        return os.path.exists(result_path)

    def _load_vipe_manifest(self, result_path: str) -> dict:
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_vipe_pose_map(self, manifest: dict) -> dict[int, np.ndarray]:
        pose_npz = np.load(manifest["pose_path"])
        return {int(idx): pose.astype(np.float32) for idx, pose in zip(pose_npz["inds"], pose_npz["data"])}

    def _load_vipe_intrinsics(self, manifest: dict) -> np.ndarray:
        with open(manifest["intrinsics_path"], "r", encoding="utf-8") as f:
            intr_data = json.load(f)
        fx, fy, cx, cy = np.asarray(intr_data["params"][:4], dtype=np.float32)
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    def load_pred_extrinsics(self, result_path: str) -> np.ndarray:
        manifest = self._load_vipe_manifest(result_path)
        pose_map = self._load_vipe_pose_map(manifest)
        extrinsics = [
            np.linalg.inv(pose_map[int(frame_idx)]).astype(np.float32)
            for frame_idx in manifest["frame_indices"]
        ]
        return np.stack(extrinsics)

    def _load_prediction_data(self, result_path: str, progress_desc: str) -> Dict:
        manifest = self._load_vipe_manifest(result_path)
        pose_map = self._load_vipe_pose_map(manifest)
        intrinsics = self._load_vipe_intrinsics(manifest)
        depths = []
        extrinsics = []
        intrinsics_out = []

        with zipfile.ZipFile(manifest["depth_path"], "r") as depth_zip:
            iterator = tqdm(manifest["frame_indices"], desc=progress_desc, leave=False)
            for frame_idx in iterator:
                frame_idx = int(frame_idx)
                raw = depth_zip.read(f"{frame_idx:06d}.npy")
                depths.append(np.load(BytesIO(raw), allow_pickle=False).astype(np.float16, copy=False))
                extrinsics.append(np.linalg.inv(pose_map[frame_idx]).astype(np.float32))
                intrinsics_out.append(intrinsics.copy())

        return Dict(
            {
                "depth": np.stack(depths).astype(np.float16, copy=False),
                "extrinsics": np.stack(extrinsics).astype(np.float32, copy=False),
                "intrinsics": np.stack(intrinsics_out).astype(np.float32, copy=False),
            }
        )

    def fuse3d(self, scene: str, result_path: str, fuse_path: str, mode: str) -> None:
        self.fuse3d_method(scene, result_path, fuse_path, mode, method="tsdf")

    def fuse3d_method(self, scene: str, result_path: str, fuse_path: str, mode: str, method: str) -> None:
        tqdm.write(f"[ScanNet] fuse start | {mode} | {method} | {scene}")
        full_gt_data = self.get_data(scene)

        gt_meta = self._load_gt_meta(result_path)
        if gt_meta is not None:
            gt_data = gt_meta
            full_image_index = {f: i for i, f in enumerate(full_gt_data.image_files)}
            image_indices = [full_image_index[f] for f in gt_data.image_files if f in full_image_index]
        else:
            gt_data = full_gt_data
            image_indices = list(range(len(full_gt_data.image_files)))

        pred_data = self._load_prediction_data(result_path, f"{scene} {mode} load ViPE depths")

        images = []
        orig_sizes = []
        for img_idx in tqdm(image_indices, desc=f"{scene} {mode} load RGB", leave=False):
            img_path = full_gt_data.image_files[img_idx]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            orig_sizes.append((img.shape[0], img.shape[1]))
        images = np.stack(images, axis=0)

        if mode == "recon_unposed":
            depths, intrinsics, extrinsics = self._prep_unposed(
                pred_data, gt_data, full_gt_data, image_indices, orig_sizes
            )
        elif mode == "recon_posed":
            depths, intrinsics, extrinsics = self._prep_posed(
                pred_data, gt_data, full_gt_data, image_indices, orig_sizes
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if method == "tsdf":
            volume = create_tsdf_volume(
                voxel_length=self.voxel_length,
                sdf_trunc=self.sdf_trunc,
            )
            mesh = fuse_depth_to_tsdf(
                volume,
                depths,
                images,
                intrinsics,
                extrinsics,
                max_depth=self.max_depth,
                progress_desc=f"{scene} {mode} tsdf frames",
            )
            pcd = sample_points_from_mesh(mesh, self.sampling_number)
        elif method == "backproject":
            pcd = self._backproject_point_cloud(
                depths,
                images,
                intrinsics,
                extrinsics,
                self.sampling_number,
                progress_desc=f"{scene} {mode} backproject frames",
            )
        else:
            raise ValueError(f"Invalid reconstruction method: {method}")

        os.makedirs(os.path.dirname(fuse_path), exist_ok=True)
        o3d.io.write_point_cloud(fuse_path, pcd)
        tqdm.write(f"[ScanNet] fuse done  | {mode} | {method} | {scene}")

    def _backproject_point_cloud(
        self,
        depths: np.ndarray,
        images: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        num_points: int,
        progress_desc: str,
    ) -> o3d.geometry.PointCloud:
        valid_counts = []
        for depth in depths:
            valid = np.isfinite(depth) & (depth > 0.0) & (depth <= self.max_depth)
            valid_counts.append(int(valid.sum()))

        total_valid = sum(valid_counts)
        sample_total = min(num_points, total_valid)
        if sample_total == 0:
            return o3d.geometry.PointCloud()

        raw_quotas = np.asarray(valid_counts, dtype=np.float64) * (sample_total / total_valid)
        quotas = np.floor(raw_quotas).astype(np.int64)
        remainder = sample_total - int(quotas.sum())
        if remainder > 0:
            fractions = raw_quotas - quotas
            for idx in np.argsort(-fractions)[:remainder]:
                quotas[idx] += 1

        rng = np.random.default_rng(seed=42)
        sampled_points = []
        sampled_colors = []
        iterator = tqdm(range(len(depths)), desc=progress_desc, leave=False)
        for i in iterator:
            quota = int(quotas[i])
            if quota == 0:
                continue

            depth = depths[i]
            valid = np.isfinite(depth) & (depth > 0.0) & (depth <= self.max_depth)
            valid_flat = np.flatnonzero(valid.ravel())
            if quota < len(valid_flat):
                valid_flat = rng.choice(valid_flat, size=quota, replace=False)

            height, width = depth.shape
            ys, xs = np.divmod(valid_flat, width)
            zs = depth.ravel()[valid_flat].astype(np.float32)
            ixt = intrinsics[i]
            fx, fy, cx, cy = ixt[0, 0], ixt[1, 1], ixt[0, 2], ixt[1, 2]

            points_cam = np.empty((len(valid_flat), 4), dtype=np.float32)
            points_cam[:, 0] = (xs.astype(np.float32) - cx) * zs / fx
            points_cam[:, 1] = (ys.astype(np.float32) - cy) * zs / fy
            points_cam[:, 2] = zs
            points_cam[:, 3] = 1.0

            c2w = np.linalg.inv(extrinsics[i]).astype(np.float32)
            sampled_points.append((c2w @ points_cam.T).T[:, :3])
            sampled_colors.append(images[i].reshape(-1, 3)[valid_flat].astype(np.float32) / 255.0)

        points = np.concatenate(sampled_points, axis=0)
        colors = np.concatenate(sampled_colors, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
        return pcd

    def _prep_unposed(
        self,
        pred_data: Dict,
        gt_data: Dict,
        full_gt_data: Dict,
        image_indices: list,
        orig_sizes: list,
    ) -> tuple:
        _, _, scale, extrinsics = align_poses_umeyama(
            gt_data.extrinsics.copy(),
            pred_data.extrinsics.copy(),
            return_aligned=True,
            ransac=True,
            random_state=42,
        )

        model_h, model_w = pred_data.depth.shape[1], pred_data.depth.shape[2]
        depths_out = []
        intrinsics_out = []
        for i in range(len(pred_data.depth)):
            orig_h, orig_w = orig_sizes[i]
            img_idx = image_indices[i]

            depth = cv2.resize(
                pred_data.depth[i],
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32)

            gt_zero_mask = self._load_gt_mask(full_gt_data.aux.gt_depth_files[img_idx], (orig_h, orig_w))
            depth = self._mask_invalid_depth(depth, gt_zero_mask)
            depth *= scale

            h_ratio = orig_h / model_h
            w_ratio = orig_w / model_w
            ixt = pred_data.intrinsics[i].copy()
            ixt[0, :] *= w_ratio
            ixt[1, :] *= h_ratio

            depths_out.append(depth)
            intrinsics_out.append(ixt)

        return np.stack(depths_out), np.stack(intrinsics_out), extrinsics

    def _prep_posed(
        self,
        pred_data: Dict,
        gt_data: Dict,
        full_gt_data: Dict,
        image_indices: list,
        orig_sizes: list,
    ) -> tuple:
        _, _, scale, _ = align_poses_umeyama(
            gt_data.extrinsics.copy(),
            pred_data.extrinsics.copy(),
            return_aligned=True,
            ransac=True,
            random_state=42,
        )

        depths_out = []
        intrinsics_out = []
        extrinsics_out = []
        for i in range(len(pred_data.depth)):
            orig_h, orig_w = orig_sizes[i]
            img_idx = image_indices[i]

            depth = cv2.resize(
                pred_data.depth[i],
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32)

            gt_zero_mask = self._load_gt_mask(full_gt_data.aux.gt_depth_files[img_idx], (orig_h, orig_w))
            depth = self._mask_invalid_depth(depth, gt_zero_mask)
            depth *= scale

            depths_out.append(depth)
            intrinsics_out.append(full_gt_data.intrinsics[img_idx].copy())
            extrinsics_out.append(full_gt_data.extrinsics[img_idx].copy())

        return (
            np.stack(depths_out),
            np.stack(intrinsics_out),
            np.stack(extrinsics_out),
        )

    def _load_gt_mask(self, gt_depth_path: str, target_hw=None) -> np.ndarray:
        if not os.path.exists(gt_depth_path):
            return None

        gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_UNCHANGED)
        if gt_depth is None:
            return None
        valid_mask = gt_depth > 0

        if target_hw is not None:
            target_h, target_w = target_hw
            valid_mask = cv2.resize(
                valid_mask.astype(np.uint8),
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        return valid_mask

    def _mask_invalid_depth(self, depth: np.ndarray, gt_zero_mask: np.ndarray = None) -> np.ndarray:
        depth = depth.copy()
        if gt_zero_mask is not None:
            pred_invalid = np.isnan(depth) | np.isinf(depth)
            combined_mask = np.logical_and(gt_zero_mask, np.logical_not(pred_invalid))
            depth = depth * combined_mask.astype(np.float32)
        else:
            invalid_mask = np.isnan(depth) | np.isinf(depth) | (depth <= 0)
            depth[invalid_mask] = 0.0
        return depth
