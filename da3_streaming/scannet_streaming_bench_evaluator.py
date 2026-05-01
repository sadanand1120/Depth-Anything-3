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
Standalone ScanNet streaming benchmark evaluator.

This wraps DA3-Streaming plus the native benchmark evaluation stack to produce:
- pose metrics (Auc3 / Auc30)
- recon_unposed metrics (F-score / Overall)
- recon_posed metrics (F-score / Overall)

Example:
  python da3_streaming/scannet_streaming_bench_evaluator.py \
      --streaming-config ./da3_streaming/configs/loopoff.yaml \
      --work-dir ./workspace/evaluation_scannet_streaming_loopoff \
      --scenes scene0000_00 scene0011_00 \
      --use-gt-pose
"""

import argparse
import importlib.util
import os
import subprocess
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _load_streaming_benchmark_evaluator():
    streaming_bench_module_path = os.path.join(CURRENT_DIR, "streaming_bench_evaluator.py")
    streaming_bench_spec = importlib.util.spec_from_file_location(
        "da3_streaming_scannet_bench_runtime",
        streaming_bench_module_path,
    )
    streaming_bench_module = importlib.util.module_from_spec(streaming_bench_spec)
    streaming_bench_spec.loader.exec_module(streaming_bench_module)
    return streaming_bench_module.StreamingBenchmarkEvaluator


DEFAULT_INPUT_ROOT = "/robodata/smodak/repos/ovo/data/input/ScanNet"
DEFAULT_RAW_ROOT = "/robodata/smodak/datasets/scannet_v2/scans"
WORKER_ENV = "_DA3_SCANNET_STREAMING_BENCH_WORKER"


def build_parser():
    parser = argparse.ArgumentParser(description="ScanNet DA3-Streaming benchmark evaluator")
    parser.add_argument(
        "--streaming-config",
        required=True,
        help="Path to DA3-Streaming YAML config (e.g. ./configs/loopoff.yaml)",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Benchmark workspace/output directory",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Explicit ScanNet scenes to evaluate (e.g. scene0000_00 scene0011_00)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["pose", "recon_unposed", "recon_posed"],
        choices=["pose", "recon_unposed", "recon_posed"],
        help="Evaluation modes to run",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional local checkpoint directory override",
    )
    parser.add_argument(
        "--input-root",
        default=DEFAULT_INPUT_ROOT,
        help="Processed ScanNet root containing scene folders",
    )
    parser.add_argument(
        "--raw-root",
        default=DEFAULT_RAW_ROOT,
        help="Raw ScanNet scans root containing GT meshes",
    )
    parser.add_argument(
        "--use-gt-pose",
        action="store_true",
        help="Use ScanNet GT poses/intrinsics inside DA3-Streaming",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Maximum frames per scene (-1 for all frames)",
    )
    parser.add_argument(
        "--num-fusion-workers",
        type=int,
        default=4,
        help="Number of workers for TSDF fusion",
    )
    parser.add_argument(
        "--ref-view-strategy",
        default=None,
        help="Optional override for Model.ref_view_strategy",
    )
    parser.add_argument(
        "--ref-view-strategy-loop",
        default=None,
        help="Optional override for Model.ref_view_strategy_loop",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip inference and evaluate existing outputs only",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print saved metrics only",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable benchmark debug mode",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--total-gpus", type=int, default=1, help=argparse.SUPPRESS)
    return parser


def _fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def _get_nested(d, *keys):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def print_scannet_summary(metrics):
    pose_mean = _get_nested(metrics, "scannet_pose", "mean") or {}
    recon_u_mean = _get_nested(metrics, "scannet_recon_unposed", "mean") or {}
    recon_p_mean = _get_nested(metrics, "scannet_recon_posed", "mean") or {}

    auc3 = None
    for key in ["Auc_3", "auc03", "auc_3", "auc3", "Auc3"]:
        if key in pose_mean:
            auc3 = pose_mean[key]
            break
    auc30 = None
    for key in ["Auc_30", "auc30", "auc_30", "Auc30"]:
        if key in pose_mean:
            auc30 = pose_mean[key]
            break

    col1 = 15
    col2 = 14

    print("\n" + "=" * 44)
    print("SCANNET STREAMING BENCHMARK SUMMARY")
    print("=" * 44)

    print("\nPOSE ESTIMATION")
    print("-" * (col1 + col2))
    print(f"{'Metric':<{col1}}{'ScanNet':<{col2}}")
    print("-" * (col1 + col2))
    print(f"{'Auc3':<{col1}}{_fmt(auc3):<{col2}}")
    print(f"{'Auc30':<{col1}}{_fmt(auc30):<{col2}}")

    print("\nRECON_UNPOSED (Pred Pose)")
    print("-" * (col1 + col2))
    print(f"{'Metric':<{col1}}{'ScanNet':<{col2}}")
    print("-" * (col1 + col2))
    print(f"{'F-score':<{col1}}{_fmt(recon_u_mean.get('fscore')):<{col2}}")
    print(f"{'Overall':<{col1}}{_fmt(recon_u_mean.get('overall')):<{col2}}")

    print("\nRECON_POSED (GT Pose)")
    print("-" * (col1 + col2))
    print(f"{'Metric':<{col1}}{'ScanNet':<{col2}}")
    print("-" * (col1 + col2))
    print(f"{'F-score':<{col1}}{_fmt(recon_p_mean.get('fscore')):<{col2}}")
    print(f"{'Overall':<{col1}}{_fmt(recon_p_mean.get('overall')):<{col2}}")


def build_evaluator(args):
    os.environ["DA3_SCANNET_INPUT_ROOT"] = os.path.abspath(args.input_root)
    os.environ["DA3_SCANNET_RAW_ROOT"] = os.path.abspath(args.raw_root)

    streaming_benchmark_evaluator = _load_streaming_benchmark_evaluator()

    return streaming_benchmark_evaluator(
        work_dir=args.work_dir,
        datas=["scannet"],
        modes=args.modes,
        scenes=args.scenes,
        debug=args.debug,
        num_fusion_workers=args.num_fusion_workers,
        max_frames=args.max_frames,
        ref_view_strategy="unused_by_streaming",
        streaming_config_path=args.streaming_config,
        streaming_ref_view_strategy=args.ref_view_strategy,
        streaming_ref_view_strategy_loop=args.ref_view_strategy_loop,
        streaming_use_gt_pose=args.use_gt_pose,
        streaming_model_path=args.model_path,
        gpu_id=args.gpu_id,
        total_gpus=args.total_gpus,
    )


def maybe_spawn_workers(args):
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_devices or os.environ.get(WORKER_ENV) == "1":
        return False

    gpu_list = [g.strip() for g in cuda_devices.split(",") if g.strip()]
    if len(gpu_list) <= 1 or args.eval_only or args.print_only:
        return False

    base_cmd = [sys.executable, os.path.abspath(__file__)]
    base_cmd += ["--streaming-config", args.streaming_config]
    base_cmd += ["--work-dir", args.work_dir]
    if args.scenes:
        base_cmd += ["--scenes", *args.scenes]
    if args.modes:
        base_cmd += ["--modes", *args.modes]
    if args.model_path:
        base_cmd += ["--model-path", args.model_path]
    base_cmd += ["--input-root", args.input_root]
    base_cmd += ["--raw-root", args.raw_root]
    if args.use_gt_pose:
        base_cmd += ["--use-gt-pose"]
    base_cmd += ["--max-frames", str(args.max_frames)]
    base_cmd += ["--num-fusion-workers", str(args.num_fusion_workers)]
    if args.ref_view_strategy is not None:
        base_cmd += ["--ref-view-strategy", args.ref_view_strategy]
    if args.ref_view_strategy_loop is not None:
        base_cmd += ["--ref-view-strategy-loop", args.ref_view_strategy_loop]
    if args.debug:
        base_cmd += ["--debug"]

    print(f"[INFO] Detected {len(gpu_list)} GPUs for ScanNet streaming benchmark: {gpu_list}")
    processes = []
    for idx, visible_gpu in enumerate(gpu_list):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible_gpu
        env[WORKER_ENV] = "1"
        cmd = base_cmd + ["--gpu-id", str(idx), "--total-gpus", str(len(gpu_list))]
        print(f"[INFO] Starting worker {idx} on GPU {visible_gpu}")
        processes.append(subprocess.Popen(cmd, env=env))

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise SystemExit(process.returncode)

    print("[INFO] All ScanNet streaming workers completed")
    return True


def main():
    parser = build_parser()
    args = parser.parse_args()
    is_worker = os.environ.get(WORKER_ENV) == "1"

    evaluator = build_evaluator(args)

    if args.print_only:
        metrics = evaluator._load_metrics()
        print_scannet_summary(metrics)
        return

    if args.eval_only:
        metrics = evaluator.eval()
        print_scannet_summary(metrics)
        return

    spawned = maybe_spawn_workers(args)
    if spawned:
        metrics = evaluator.eval()
        print_scannet_summary(metrics)
        return

    evaluator.infer()
    if is_worker:
        return
    metrics = evaluator.eval()
    print_scannet_summary(metrics)


if __name__ == "__main__":
    main()
