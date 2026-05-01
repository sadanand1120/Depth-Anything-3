# DA3-Streaming Architecture Notes

This document reflects the code paths in:

- `da3_streaming/da3_streaming.py`
- `src/depth_anything_3/model/da3.py`
- `src/depth_anything_3/model/dualdpt.py`
- `src/depth_anything_3/model/dpt.py`
- `src/depth_anything_3/model/dinov2/vision_transformer.py`
- `da3_streaming/loop_utils/*`

It is written specifically for the current streaming pipeline used by:

```bash
python3 da3_streaming.py --image_dir ... --config ./configs/base_config.yaml --output_dir ...
```

In every Mermaid diagram below, orange nodes mark the blocks used by the current code path for this streaming run:

- `base_config.yaml`
- `use_ray_pose=False`
- `loop_enable=True`
- `ref_view_strategy=saddle_balanced`

## 1. High-Level Streaming Pipeline

```mermaid
flowchart TD
    A[Image directory] --> B[Sort *.jpg/*.png into img_list]
    B --> C[Split sequence into overlapping chunks<br/>chunk_size=120, overlap=60]

    C --> D[For each chunk]
    D --> E[Preprocess chunk images<br/>resize, make dims divisible by 14, normalize]
    E --> F[Run DA3 nested model on whole chunk]
    F --> G[Chunk-local outputs<br/>depth, conf, extrinsics, intrinsics, processed_images]
    G --> H[Save raw chunk prediction to _tmp_results_unaligned/chunk_k.npy]

    H --> I{Previous chunk exists?}
    I -->|No| J[Continue]
    I -->|Yes| K[Take overlapping frames only<br/>prev tail 60, current head 60]
    K --> L[Convert overlap depth+pose to chunk-local point maps]
    L --> M[Estimate relative chunk Sim3<br/>current chunk -> previous chunk]
    M --> N[Append to sequential sim3_list]
    N --> J

    J --> O{All chunks processed?}
    O -->|No| D
    O -->|Yes| P{loop_enable == True?}

    P -->|No| Q[Skip loop closure]
    P -->|Yes| R[Run SALAD loop detector on entire image sequence]
    R --> S[Get frame-level loop pairs]
    S --> T[Convert each pair into two short frame windows]
    T --> U[Re-run DA3 on each loop window pair]
    U --> V[Estimate loop Sim3 constraints between chunks]
    V --> W[Sim3 pose-graph optimization over chunk chain]
    W --> Q

    Q --> X[Accumulate chunk transforms to chunk-0 world]
    X --> Y[Apply accumulated Sim3 to every chunk point map]
    Y --> Z[For each chunk:<br/>save chunk PLY and optionally save unique frame NPZs]
    Z --> AA[Save camera_poses.txt, intrinsic.txt, camera_poses.ply]
    AA --> AB[Delete temporary chunk files if enabled]
    AB --> AC[Merge all *_pcd.ply files into pcd/combined_pcd.ply]

    classDef active fill:#f59e0b,stroke:#c2410c,stroke-width:2px,color:#111827;
    class A,B,C,D,E,F,G,H,I,K,L,M,N,J,O,P,R,S,T,U,V,W,Q,X,Y,Z,AA,AB,AC active;
```

## 2. Exact Current Frame/Chunk Path

This diagram shows the concrete path for a frame processed inside a non-initial chunk `k > 0`. For chunk `0`, the overlap-stitch node is skipped.

```mermaid
flowchart TD
    A[Raw frame f_i<br/>1296 x 968 x 3 uint8] --> B[Frame is collected into a 120-frame chunk]
    B --> C[Preprocess every frame in chunk<br/>resize longest side to 504<br/>then make dims divisible by 14]
    C --> D[Processed per-frame tensor<br/>3 x 378 x 504]
    D --> E[Stack chunk before batch dim<br/>120 x 3 x 378 x 504]
    E --> F[Add batch dim on API side<br/>1 x 120 x 3 x 378 x 504]

    F --> G[Run NestedDepthAnything3Net]
    G --> H[Any-view vitg branch<br/>reference-view selection = saddle_balanced]
    G --> I[Metric vitl branch]

    H --> J[DualDPT outputs<br/>depth: 1 x 120 x 378 x 504<br/>conf: 1 x 120 x 378 x 504<br/>ray: 1 x 120 x 378 x 504 x 6<br/>ray_conf: 1 x 120 x 378 x 504]
    H --> K[CameraDec path<br/>use_ray_pose = False]
    K --> L[Pose decode outputs<br/>extrinsics: 1 x 120 x 3 x 4<br/>intrinsics: 1 x 120 x 3 x 3]
    I --> M[Metric DPT outputs<br/>metric depth: 1 x 120 x 378 x 504<br/>sky: 1 x 120 x 378 x 504]

    J --> N[Nested metric alignment<br/>scale any-view depth and translation<br/>set sky to far depth]
    L --> N
    M --> N

    N --> O[Convert to Prediction object<br/>depth/conf/extrinsics/intrinsics]
    O --> P[Add processed_images<br/>120 x 378 x 504 x 3]
    P --> Q[Streaming postprocess<br/>conf = conf - 1]
    Q --> R[Save _tmp_results_unaligned/chunk_k.npy]

    R --> S[Overlap stitch on 60-frame overlap<br/>point maps: 60 x 378 x 504 x 3<br/>estimate Sim3 s,R,t]
    S --> T[After all chunks: loop closure path is enabled<br/>SALAD + short DA3 reruns + pose-graph]
    T --> U[Accumulate chunk transforms]
    U --> V[Apply final transforms to per-pixel world points]
    V --> W[Write k_pcd.ply and camera pose files]
    W --> X[Merge *_pcd.ply -> pcd/combined_pcd.ply]

    classDef active fill:#f59e0b,stroke:#c2410c,stroke-width:2px,color:#111827;
    class A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X active;
```

## 3. What Happens Inside One Chunk Inference

```mermaid
flowchart TD
    A[Chunk image paths] --> B[InputProcessor]
    B --> C[Per-image load RGB]
    C --> D[Boundary resize to process_res]
    D --> E[Make H,W divisible by 14]
    E --> F[ImageNet normalization]
    F --> G[Batch tensor shape 1 x S x 3 x H x W]

    G --> H[DepthAnything3.inference]
    H --> I[No input extrinsics/intrinsics provided in streaming]
    I --> J[Call nested model forward]

    J --> K[Nested any-view branch]
    J --> L[Nested metric branch]

    K --> M[Any-view outputs<br/>depth, depth_conf, pose, intrinsics]
    L --> N[Metric outputs<br/>depth, sky]

    M --> O[Nested postprocessing]
    N --> O

    O --> P[Prediction object]
    P --> Q[processed_images added]
    Q --> R[Streaming postprocess:<br/>depth squeeze, conf = conf - 1]
    R --> S[Returned chunk prediction]

    classDef active fill:#f59e0b,stroke:#c2410c,stroke-width:2px,color:#111827;
    class A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S active;
```

## 4. Nested Model Architecture

```mermaid
flowchart TD
    A[Input chunk tensor<br/>1 x S x 3 x H x W] --> B[NestedDepthAnything3Net]

    B --> C[Any-view branch: DepthAnything3Net]
    B --> D[Metric branch: DepthAnything3Net]

    C --> E[vitg DinoV2 backbone]
    E --> F[DualDPT head]
    E --> G[CameraDec pose head]

    F --> H[depth]
    F --> I[depth_conf]
    F --> J[ray]
    F --> K[ray_conf]

    G --> L[pose encoding]
    L --> M[extrinsics w2c]
    L --> N[intrinsics]

    D --> O[vitl DinoV2 backbone]
    O --> P[DPT head]
    P --> Q[metric depth]
    P --> R[sky]

    H --> S[Nested alignment logic]
    I --> S
    M --> S
    N --> S
    Q --> S
    R --> S

    S --> T[metric-scaled depth]
    S --> U[scaled extrinsics translation]
    S --> V[sky handled depth/conf]

    classDef active fill:#f59e0b,stroke:#c2410c,stroke-width:2px,color:#111827;
    class A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V active;
```

## 5. Any-View Branch Detail

```mermaid
flowchart TD
    A[1 x S x 3 x H x W] --> B[DinoV2 vitg patch embedding]
    B --> C[Transformer blocks]

    C --> D{At alt_start - 1 and S >= 3?}
    D -->|Yes| E[Select reference view<br/>ref_view_strategy=saddle_balanced]
    E --> F[Reorder views so reference view is first]
    D -->|No| G[Keep original order]
    F --> H[Continue transformer]
    G --> H

    H --> I[Take 4 backbone stages]
    I --> J[DualDPT main head]
    I --> K[DualDPT auxiliary head]
    I --> L[CameraDec input token]

    J --> M[depth logits -> exp -> depth]
    J --> N[conf logits -> expp1 -> depth_conf_raw]

    K --> O[6-channel ray prediction]
    K --> P[1-channel ray_conf_raw]

    L --> Q[CameraDec MLP]
    Q --> R[t]
    Q --> S[qvec]
    Q --> T[fov_h, fov_w]
    R --> U[pose_encoding_to_extri_intri]
    S --> U
    T --> U
    U --> V[extrinsics w2c 3x4]
    U --> W[intrinsics 3x3]

    M --> X[Chunk-local relative depth]
    N --> Y[Chunk-local confidence]
    V --> Z[Chunk-local camera poses]
    W --> AA[Chunk-local intrinsics]

    classDef active fill:#f59e0b,stroke:#c2410c,stroke-width:2px,color:#111827;
    class A,B,C,D,E,F,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,AA active;
```

## 6. Pose Path: Camera Decoder vs Ray Pose

In the current streaming code, `use_ray_pose` is not passed to `model.inference`, so it uses the API default `False`. That means the active path is the camera decoder path, not the ray-pose path.

```mermaid
flowchart TD
    A[DualDPT produces ray and ray_conf] --> B{use_ray_pose?}

    B -->|False: current streaming run| C[Ignore ray/ray_conf for pose]
    C --> D[Use CameraDec on deepest backbone token]
    D --> E[Predict t, quaternion, fov_h, fov_w]
    E --> F[Convert to extrinsics/intrinsics]

    B -->|True| G[Use dense ray field for pose]
    G --> H[Interpret 6-channel ray output as camray]
    H --> I[Estimate rotation + focal + principal point from ray geometry]
    I --> J[Average translation votes from ray tail channels]
    J --> K[Build extrinsics/intrinsics from ray field]

    classDef active fill:#f59e0b,stroke:#c2410c,stroke-width:2px,color:#111827;
    class A,B,C,D,E,F active;
```

## 7. Metric Branch and Nested Postprocessing

```mermaid
flowchart TD
    A[Any-view outputs:<br/>depth, depth_conf, extrinsics, intrinsics] --> G[Nested postprocess]
    B[Metric outputs:<br/>depth, sky] --> G

    G --> H[Scale metric depth by focal / 300]
    H --> I[Compute non_sky_mask from metric sky < 0.3]
    A --> J[Take any-view depth_conf over non-sky]
    J --> K[Median confidence threshold]
    I --> L[Build alignment mask:<br/>non-sky, confident, positive depths]
    H --> L
    A --> L

    L --> M[Least-squares problem<br/>s* = argmin_s || D_m[align_mask] - s D_av[align_mask] ||^2]
    M --> N[Scale any-view depth by scale_factor]
    M --> O[Scale any-view extrinsics translation by scale_factor]
    B --> P[Use sky mask to set sky depth to far value]
    P --> Q[Set sky confidence to 1.0]

    N --> R[Final nested metric depth]
    O --> S[Final nested metric-scaled poses]
    Q --> T[Final nested sky-handled confidence]

    classDef active fill:#f59e0b,stroke:#c2410c,stroke-width:2px,color:#111827;
    class A,B,G,H,I,J,K,L,M,N,O,P,Q,R,S,T active;
```

## 8. Exact Metric Calibration Pipeline

This calibration happens once per nested forward call, so in the current streaming run it is one scalar per chunk, not one scalar per frame.

```mermaid
flowchart TD
    A["Any-view depth D_av<br>1 x S x H x W"] --> G["Calibration logic"]
    B["Any-view confidence C_av<br>1 x S x H x W"] --> G
    C["Any-view intrinsics K<br>1 x S x 3 x 3"] --> G
    D["Metric raw depth D_m_raw<br>1 x S x H x W"] --> G
    E["Metric sky Sky<br>1 x S x H x W"] --> G

    G --> H["Compute focal per frame<br>f = (fx + fy) / 2"]
    H --> I["Metric scaling<br>D_m = D_m_raw * f / 300"]
    E --> J["Non-sky mask<br>non_sky = Sky &lt; 0.3"]

    B --> K["Take C_av on non-sky pixels"]
    J --> K
    K --> L["Sample up to 100000 values"]
    L --> M["Median confidence threshold"]

    A --> N["Build align_mask"]
    B --> N
    I --> N
    J --> N
    M --> N
    N --> O["align_mask =<br>(C_av &gt;= median_conf)<br>&amp; non_sky<br>&amp; (D_m &gt; 1e-2)<br>&amp; (D_av &gt; 1e-3)"]

    A --> P["Flatten valid any-view depths<br>b = D_av[align_mask]"]
    I --> Q["Flatten valid metric depths<br>a = D_m[align_mask]"]
    O --> P
    O --> Q

    P --> R["Least-squares problem<br>s* = argmin_s || a - s b ||^2"]
    Q --> R
    R --> S["Closed-form solution<br>s* = (a^T b) / (b^T b)"]

    S --> T["Scale whole chunk depth<br>D_out = scale_factor * D_av"]
    S --> U["Scale whole chunk translation<br>t_out = scale_factor * t_av"]
    S --> V["Store output.scale_factor"]

    classDef active fill:#f59e0b,stroke:#c2410c,stroke-width:2px,color:#111827;
    class A,B,C,D,E,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V active;
```

## 9. Chunk-to-Chunk Alignment

```mermaid
flowchart TD
    A[Previous chunk prediction] --> C[Take overlap tail]
    B[Current chunk prediction] --> D[Take overlap head]

    C --> E[depth + intrinsics + extrinsics -> point_map_prev]
    D --> F[depth + intrinsics + extrinsics -> point_map_cur]

    E --> G[Keep pixels where both confidences are high]
    F --> G
    G --> H[Weight correspondences by sqrt(conf_prev * conf_cur)]
    H --> I[Robust weighted Sim3 alignment<br/>cur -> prev]
    I --> J[Output relative s, R, t]
    J --> K[Append to sim3_list]

    classDef active fill:#f59e0b,stroke:#c2410c,stroke-width:2px,color:#111827;
    class A,B,C,D,E,F,G,H,I,J,K active;
```

## 10. Loop Closure Detail

```mermaid
flowchart TD
    A[All original images] --> B[SALAD place-recognition model]
    B --> C[Descriptor per frame]
    C --> D[FAISS top-k similarity search]
    D --> E[Threshold by similarity and frame distance]
    E --> F[Optional NMS filtering]
    F --> G[Frame-level loop pairs]

    G --> H[Convert each pair into two short windows]
    H --> I[Re-run DA3 on concatenated loop windows]
    I --> J[Get loop-window point maps]

    J --> K[Align loop-window part A to original chunk A]
    J --> L[Align loop-window part B to original chunk B]
    K --> M[Compute chunk-level loop Sim3 A->B]
    L --> M

    M --> N[Sim3 pose-graph optimizer]
    N --> O[Optimized chunk transform chain]

    classDef active fill:#f59e0b,stroke:#c2410c,stroke-width:2px,color:#111827;
    class A,B,C,D,E,F,G,H,I,J,K,L,M,N,O active;
```

## 11. Final Point Cloud Export

```mermaid
flowchart TD
    A[Chunk-local predictions] --> B[Accumulate chunk transforms to chunk-0 frame]
    B --> C[Apply accumulated Sim3 to every point map]
    C --> D[Per-chunk confident filtering + random sampling]
    D --> E[Write k_pcd.ply for each chunk]
    E --> F[Binary PLY concatenation only]
    F --> G[pcd/combined_pcd.ply]

    A --> H[Trim overlaps for unique frame outputs]
    H --> I[Save results_output/frame_*.npz]
    I --> J[npz_output_process.py can rebuild a single PLY]

    classDef active fill:#f59e0b,stroke:#c2410c,stroke-width:2px,color:#111827;
    class A,B,C,D,E,F,G,H,I,J active;
```

## 12. Important Practical Notes

- The streaming pipeline is not doing TSDF fusion, voxel fusion, surfel fusion, ICP map fusion, or bundle adjustment over all frames.
- The final `combined_pcd.ply` is produced by merging per-chunk PLY files, not by re-optimizing or deduplicating all 3D points.
- Overlap regions are duplicated in the chunk PLY export path, because each chunk PLY is written from the full aligned chunk. The `results_output/frame_*.npz` path trims overlaps and keeps each frame once.
- Loop closure is optional and controlled by `Model.loop_enable`.
- In the default streaming path, pose comes from `CameraDec`, because `use_ray_pose=False`.
- The nested model already outputs metric depth per chunk, but the streaming stitcher still estimates chunk-to-chunk Sim3, including scale, to align chunk coordinate systems.
