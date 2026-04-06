# privmotion

Privacy-preserving motion analytics for videos, images, and frame folders.

`privmotion` turns identity-bearing visual input into anonymized motion outputs:
skeleton keypoints, silhouettes, low-resolution depth-like surrogates, feature
records, benchmark reports, and preview videos. The default policy is
`no-raw-rgb`: raw RGB frames are used only in memory for processing and are not
saved to the output directory.

This is a Python prototype, not an official OpenCV module.

## What It Does

- Processes MP4 files, image files, or folders of image frames.
- Exports anonymized motion data as JSON and mask/depth artifacts.
- Uses YOLO-Pose automatically when available, with prototype fallback.
- Validates that raw RGB frames were not retained.
- Benchmarks utility/privacy proxy metrics.
- Renders anonymized GIF or MP4 previews.
- Runs simple manifest-based dataset evaluations.

## Install

Basic install:

```bash
python -m pip install -e .
```

Install optional YOLO-Pose support:

```bash
python -m pip install -e ".[pose]"
```

YOLO may download model weights such as `yolo11n-pose.pt` on first use. Model
weights are ignored by git.

## Process An MP4

```bash
PYTHONPATH=src python -m privmotion.cli.process \
  --input examples/videoplayback.mp4 \
  --output out/my_video \
  --mode skeleton,silhouette,depth-surrogate,features
```

By default, `--pose-backend auto` tries YOLO-Pose when installed. If YOLO is not
available, it falls back to the prototype backend and records the reason in
`metadata.json`.

Force YOLO:

```bash
PYTHONPATH=src python -m privmotion.cli.process \
  --input examples/videoplayback.mp4 \
  --output out/my_video_yolo \
  --mode skeleton,silhouette,depth-surrogate,features \
  --pose-backend yolo \
  --pose-model yolo11n-pose.pt
```

Force the lightweight prototype backend:

```bash
PYTHONPATH=src python -m privmotion.cli.process \
  --input examples/videoplayback.mp4 \
  --output out/my_video_prototype \
  --mode skeleton,silhouette,depth-surrogate,features \
  --pose-backend prototype
```

## Output Files

A processed output directory contains files like:

```text
metadata.json
skeletons.json
features.json
retention_report.json
silhouettes/
depth_surrogates/
```

Important metadata fields:

- `backends.requested_pose_backend`: what the user asked for.
- `backends.pose`: the backend actually used.
- `backends.pose_fallback_reason`: why `auto` fell back, if it did.
- `retention.raw_rgb_written`: should be `false`.

## Validate Privacy Retention

```bash
PYTHONPATH=src python -m privmotion.cli.validate \
  --output out/my_video
```

Expected result:

```text
passed=true
violations=0
```

## Render An Anonymized Preview

MP4 preview:

```bash
PYTHONPATH=src python -m privmotion.cli.visualize \
  --output out/my_video \
  --visualization out/my_video/preview.mp4 \
  --fps 8 \
  --size 1280x720
```

GIF preview:

```bash
PYTHONPATH=src python -m privmotion.cli.visualize \
  --output out/my_video \
  --visualization out/my_video/preview.gif \
  --frames-dir out/my_video/preview_frames
```

The preview is rendered from anonymized outputs only. It does not reload or
display the original raw RGB video.

## Benchmark An Output

```bash
PYTHONPATH=src python -m privmotion.cli.benchmark \
  --output out/my_video \
  --report out/my_video/benchmark_report.json
```

The benchmark report includes deterministic proxy metrics for:

- processed frame count
- keypoint coverage
- average keypoint confidence
- silhouette coverage
- depth surrogate coverage
- raw RGB retention status
- output size and artifact availability

These are local proxy metrics, not full identity-leakage or pose-accuracy
benchmarks.

## Dataset Manifest Evaluation

Create a manifest:

```json
{
  "samples": [
    {
      "id": "walk_001",
      "input": "videos/walk_001.mp4",
      "label": "walk",
      "split": "test",
      "expected_frames": 120
    }
  ]
}
```

Run it:

```bash
PYTHONPATH=src python -m privmotion.cli.dataset_eval \
  --manifest datasets/demo_manifest.json \
  --output out/dataset_eval \
  --visualize \
  --visualization-ext .mp4
```

This writes per-sample outputs plus:

```text
out/dataset_eval/dataset_report.json
```

## Development

Run tests:

```bash
python -m pytest
```

Useful help commands:

```bash
PYTHONPATH=src python -m privmotion.cli.process --help
PYTHONPATH=src python -m privmotion.cli.visualize --help
PYTHONPATH=src python -m privmotion.cli.benchmark --help
PYTHONPATH=src python -m privmotion.cli.dataset_eval --help
```

## Limitations

- This is a prototype, not a production privacy guarantee.
- Skeletons, silhouettes, depth surrogates, and motion features can still leak
  identity through body shape or motion.
- YOLO-Pose improves skeleton placement but can still miss or misplace
  keypoints.
- The benchmark metrics are simple local proxies.
- Reversible access controls and encrypted recovery are not implemented.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
