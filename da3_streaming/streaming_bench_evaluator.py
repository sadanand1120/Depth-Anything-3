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

import gc
import importlib.util
import os
import shutil
import sys

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

STREAMING_MODULE_PATH = os.path.join(CURRENT_DIR, "da3_streaming.py")
STREAMING_SPEC = importlib.util.spec_from_file_location("da3_streaming_runtime", STREAMING_MODULE_PATH)
da3_streaming_module = importlib.util.module_from_spec(STREAMING_SPEC)
STREAMING_SPEC.loader.exec_module(da3_streaming_module)

from loop_utils.config_utils import load_config as load_streaming_config
from depth_anything_3.bench.evaluator import Evaluator
from depth_anything_3.cfg import load_config as load_bench_config

DA3_Streaming = da3_streaming_module.DA3_Streaming
warmup_numba = da3_streaming_module.warmup_numba
merge_ply_files = da3_streaming_module.merge_ply_files


class StreamingBenchmarkEvaluator(Evaluator):
    def __init__(
        self,
        streaming_config_path,
        streaming_ref_view_strategy=None,
        streaming_ref_view_strategy_loop=None,
        streaming_use_gt_pose=False,
        streaming_model_path=None,
        gpu_id=0,
        total_gpus=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        unsupported = set(self.datas) - {"hiroom", "scannetpp", "scannet"}
        if unsupported:
            raise ValueError(
                "Streaming benchmark currently supports only ['hiroom', 'scannetpp', 'scannet'], "
                f"got {sorted(unsupported)}"
            )

        self.streaming_config_path = os.path.abspath(streaming_config_path)
        self.streaming_ref_view_strategy = streaming_ref_view_strategy
        self.streaming_ref_view_strategy_loop = streaming_ref_view_strategy_loop
        self.streaming_use_gt_pose = streaming_use_gt_pose
        self.streaming_model_path = streaming_model_path
        self.gpu_id = gpu_id
        self.total_gpus = total_gpus
        self._numba_warmed = False

    def _streaming_dir(self, data, scene):
        return os.path.join(self.work_dir, "model_results", data, scene, "streaming")

    def _build_streaming_config(self):
        config = load_streaming_config(self.streaming_config_path)
        config["__config_dir__"] = os.path.dirname(self.streaming_config_path)
        config["Model"]["save_depth_conf_result"] = True

        if self.streaming_ref_view_strategy is not None:
            config["Model"]["ref_view_strategy"] = self.streaming_ref_view_strategy
        if self.streaming_ref_view_strategy_loop is not None:
            config["Model"]["ref_view_strategy_loop"] = self.streaming_ref_view_strategy_loop

        if self.streaming_model_path:
            model_path = os.path.abspath(self.streaming_model_path)
            if os.path.isdir(model_path):
                config["Weights"]["DA3"] = os.path.join(model_path, "model.safetensors")
                config["Weights"]["DA3_CONFIG"] = os.path.join(model_path, "config.json")
            else:
                config["Weights"]["DA3"] = model_path
                config["Weights"]["DA3_CONFIG"] = os.path.join(
                    os.path.dirname(model_path), "config.json"
                )

        if config["Model"]["align_lib"] == "numba" and not self._numba_warmed:
            warmup_numba()
            self._numba_warmed = True

        return config

    def _run_streaming_scene(self, data, scene, scene_data):
        streaming_dir = self._streaming_dir(data, scene)
        if os.path.exists(streaming_dir):
            shutil.rmtree(streaming_dir)
        os.makedirs(streaming_dir, exist_ok=True)

        config = self._build_streaming_config()
        image_dir = os.path.dirname(scene_data.image_files[0])
        streamer = DA3_Streaming(
            image_dir=image_dir,
            save_dir=streaming_dir,
            config=config,
            image_files=scene_data.image_files,
            use_gt_pose=self.streaming_use_gt_pose,
        )

        try:
            streamer.run()
            pcd_dir = os.path.join(streaming_dir, "pcd")
            merge_ply_files(pcd_dir, os.path.join(pcd_dir, "combined_pcd.ply"))
            result_path = streamer.export_benchmark_results(streaming_dir)
        finally:
            streamer.close()
            del streamer
            torch.cuda.empty_cache()
            gc.collect()

        need_unposed = {"pose", "recon_unposed"} & self.modes
        need_posed = {"recon_posed", "view_syn"} & self.modes

        if need_unposed:
            unposed_dir = self._export_dir(data, scene, posed=False)
            os.makedirs(os.path.join(unposed_dir, "exports", "mini_npz"), exist_ok=True)
            shutil.copy2(result_path, os.path.join(unposed_dir, "exports", "mini_npz", "results.npz"))
            self._save_gt_meta(unposed_dir, scene_data)

        if need_posed:
            posed_dir = self._export_dir(data, scene, posed=True)
            os.makedirs(os.path.join(posed_dir, "exports", "mini_npz"), exist_ok=True)
            shutil.copy2(result_path, os.path.join(posed_dir, "exports", "mini_npz", "results.npz"))
            self._save_gt_meta(posed_dir, scene_data)

    def infer(self, api=None, model_path=None):
        all_tasks = []
        for data in self.datas:
            dataset = self.datasets[data]
            for scene in self._get_scenes(dataset):
                all_tasks.append((data, scene))

        if self.total_gpus > 1:
            tasks = [t for i, t in enumerate(all_tasks) if i % self.total_gpus == self.gpu_id]
            print(
                f"[INFO] Streaming worker {self.gpu_id}/{self.total_gpus}: "
                f"{len(tasks)}/{len(all_tasks)} tasks"
            )
        else:
            tasks = all_tasks
            print(f"[INFO] Total streaming inference tasks: {len(all_tasks)}")

        for data, scene in tasks:
            dataset = self.datasets[data]
            scene_data = dataset.get_data(scene)
            scene_data = self._sample_frames(scene_data, scene)
            self._run_streaming_scene(data, scene, scene_data)


if __name__ == "__main__":
    _default_config = os.path.join(
        os.path.dirname(__file__), "configs", "eval_bench_streaming.yaml"
    )

    argv = sys.argv[1:]
    config_path = _default_config
    if "--config" in argv:
        config_idx = argv.index("--config")
        if config_idx + 1 < len(argv):
            config_path = argv[config_idx + 1]
            argv = argv[:config_idx] + argv[config_idx + 2 :]

    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            """
DA3-Streaming Benchmark Evaluation

Usage:
  python da3_streaming/streaming_bench_evaluator.py [OPTIONS] [KEY=VALUE ...]

Configuration:
  --config PATH                         Config YAML file

Config Overrides:
  model.path=VALUE                      Optional local checkpoint directory override
  workspace.work_dir=VALUE              Working directory for outputs
  eval.datasets=[hiroom|scannetpp|scannet]      Datasets to evaluate
  eval.modes=[pose,recon_unposed,recon_posed]
  eval.scenes=[scene1,scene2]           Specific scenes to evaluate
  eval.max_frames=VALUE                 Max frames per scene (-1=no limit)
  eval.eval_only=true                   Only run evaluation
  eval.print_only=true                  Only print saved metrics
  streaming.config=VALUE                Streaming YAML config path
  streaming.ref_view_strategy=VALUE     Optional override for Model.ref_view_strategy
  streaming.ref_view_strategy_loop=VALUE
  streaming.use_gt_pose=true            Use ScanNet GT poses inside DA3-Streaming

Examples:
  python da3_streaming/streaming_bench_evaluator.py \\
      model.path=/path/to/da3nestedgiantlarge1.1 \\
      workspace.work_dir=./workspace/evaluation_streaming \\
      eval.datasets=[hiroom] \\
      eval.modes=[pose,recon_unposed,recon_posed] \\
      eval.max_frames=-1 \\
      streaming.config=./da3_streaming/configs/base_config.yaml
            """
        )
        sys.exit(0)

    config = load_bench_config(config_path, argv=argv)

    gpu_id = 0
    total_gpus = 1
    for arg in argv:
        if arg.startswith("gpu_id="):
            gpu_id = int(arg.split("=")[1])
        elif arg.startswith("total_gpus="):
            total_gpus = int(arg.split("=")[1])

    streaming_config_path = config.streaming.config
    if not os.path.isabs(streaming_config_path):
        streaming_config_path = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(config_path)), streaming_config_path)
        )

    evaluator = StreamingBenchmarkEvaluator(
        work_dir=config.workspace.work_dir,
        datas=config.eval.datasets,
        modes=config.eval.modes,
        scenes=config.eval.scenes,
        debug=config.inference.debug,
        num_fusion_workers=config.inference.num_fusion_workers,
        max_frames=config.eval.max_frames,
        ref_view_strategy="unused_by_streaming",
        streaming_config_path=streaming_config_path,
        streaming_ref_view_strategy=config.streaming.ref_view_strategy,
        streaming_ref_view_strategy_loop=config.streaming.ref_view_strategy_loop,
        streaming_use_gt_pose=config.streaming.use_gt_pose,
        streaming_model_path=config.model.path,
        gpu_id=gpu_id,
        total_gpus=total_gpus,
    )

    if config.eval.print_only:
        evaluator.print_metrics()
    elif config.eval.eval_only:
        metrics = evaluator.eval()
        evaluator.print_metrics(metrics)
    else:
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_devices is not None and cuda_devices.strip():
            gpu_list = [g.strip() for g in cuda_devices.split(",") if g.strip()]
        else:
            num_available = torch.cuda.device_count()
            gpu_list = [str(i) for i in range(num_available)] if num_available > 0 else ["0"]

        is_worker = os.environ.get("_DA3_STREAMING_BENCH_WORKER") == "1"

        if len(gpu_list) > 1 and not is_worker:
            import subprocess

            num_gpus = len(gpu_list)
            print(f"[INFO] Detected {num_gpus} GPUs for streaming benchmark: {gpu_list}")
            print(f"[INFO] Launching {num_gpus} streaming workers...")

            base_cmd = [sys.executable, os.path.abspath(__file__)]
            if config_path != _default_config:
                base_cmd += ["--config", config_path]
            if config.model.path is not None:
                base_cmd += [f"model.path={config.model.path}"]
            base_cmd += [f"workspace.work_dir={config.workspace.work_dir}"]
            base_cmd += [f"eval.datasets=[{','.join(config.eval.datasets)}]"]
            base_cmd += [f"eval.modes=[{','.join(config.eval.modes)}]"]
            if config.eval.scenes:
                base_cmd += [f"eval.scenes=[{','.join(config.eval.scenes)}]"]
            base_cmd += [f"eval.max_frames={config.eval.max_frames}"]
            base_cmd += [f"eval.eval_only={str(config.eval.eval_only).lower()}"]
            base_cmd += [f"eval.print_only={str(config.eval.print_only).lower()}"]
            base_cmd += [f"inference.debug={str(config.inference.debug).lower()}"]
            base_cmd += [f"inference.num_fusion_workers={config.inference.num_fusion_workers}"]
            base_cmd += [f"streaming.config={config.streaming.config}"]
            if config.streaming.ref_view_strategy is not None:
                base_cmd += [f"streaming.ref_view_strategy={config.streaming.ref_view_strategy}"]
            if config.streaming.ref_view_strategy_loop is not None:
                base_cmd += [
                    f"streaming.ref_view_strategy_loop={config.streaming.ref_view_strategy_loop}"
                ]
            base_cmd += [f"streaming.use_gt_pose={str(config.streaming.use_gt_pose).lower()}"]

            processes = []
            for idx, visible_gpu in enumerate(gpu_list):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = visible_gpu
                env["_DA3_STREAMING_BENCH_WORKER"] = "1"
                cmd = base_cmd + [f"gpu_id={idx}", f"total_gpus={num_gpus}"]
                print(f"[INFO] Starting streaming worker {idx} on GPU {visible_gpu}")
                processes.append(subprocess.Popen(cmd, env=env))

            for process in processes:
                process.wait()

            print(f"[INFO] All {num_gpus} streaming workers completed")
            metrics = evaluator.eval()
            evaluator.print_metrics(metrics)
        else:
            evaluator.infer()
            if not is_worker:
                metrics = evaluator.eval()
                evaluator.print_metrics(metrics)
