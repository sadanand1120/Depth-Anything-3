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

import os
from pathlib import Path
from typing import Dict as TDict

import cv2
import numpy as np
import open3d as o3d
from addict import Dict
from tqdm import tqdm

from depth_anything_3.bench.dataset import Dataset, _wait_for_file_ready
from depth_anything_3.bench.registries import MONO_REGISTRY, MV_REGISTRY
from depth_anything_3.bench.utils import (
    create_tsdf_volume,
    evaluate_3d_reconstruction,
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


@MV_REGISTRY.register(name="scannet")
@MONO_REGISTRY.register(name="scannet")
class ScanNetDataset(Dataset):
    """
    ScanNet benchmark wrapper for DA3 streaming evaluation.

    Metrics:
    - Pose: AUC on relative pose error, identical to the native evaluator
    - Recon: TSDF fusion + nearest-neighbor Acc/Comp/F-score
    """

    data_root = _scannet_input_root()
    raw_root = _scannet_raw_root()
    SCENES = _load_scene_list()

    max_depth = 5.0
    sampling_number = 1_000_000
    voxel_length = 0.02
    sdf_trunc = 0.15
    eval_threshold = 0.05
    down_sample = 0.02

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

        for img_path in image_files:
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
        gt_data = self.get_data(scene)
        gt_mesh = o3d.io.read_triangle_mesh(gt_data.aux.gt_mesh_path)
        gt_pcd = sample_points_from_mesh(gt_mesh, self.sampling_number)

        pred_pcd = o3d.io.read_point_cloud(fuse_path)

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
            pred_pcd = pred_pcd.select_by_index(np.nonzero(inside_mask)[0])

        return evaluate_3d_reconstruction(
            pred_pcd,
            gt_pcd,
            threshold=self.eval_threshold,
            down_sample=self.down_sample,
        )

    def _load_gt_meta(self, result_path: str) -> Dict:
        export_dir = os.path.dirname(result_path)
        gt_meta_path = os.path.join(os.path.dirname(result_path), "..", "gt_meta.npz")
        gt_meta_path = os.path.normpath(gt_meta_path)
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

    def fuse3d(self, scene: str, result_path: str, fuse_path: str, mode: str) -> None:
        tqdm.write(f"[ScanNet] fuse start | {mode} | {scene}")
        full_gt_data = self.get_data(scene)

        gt_meta = self._load_gt_meta(result_path)
        if gt_meta is not None:
            gt_data = gt_meta
            image_indices = [
                full_gt_data.image_files.index(f)
                for f in gt_data.image_files
                if f in full_gt_data.image_files
            ]
        else:
            gt_data = full_gt_data
            image_indices = list(range(len(full_gt_data.image_files)))

        _wait_for_file_ready(result_path)
        pred_data = Dict({k: v for k, v in np.load(result_path).items()})

        images = []
        orig_sizes = []
        for img_idx in image_indices:
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
            progress_desc=f"{scene} {mode} frames",
        )
        pcd = sample_points_from_mesh(mesh, self.sampling_number)

        os.makedirs(os.path.dirname(fuse_path), exist_ok=True)
        o3d.io.write_point_cloud(fuse_path, pcd)
        tqdm.write(f"[ScanNet] fuse done  | {mode} | {scene}")

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
