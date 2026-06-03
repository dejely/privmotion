# privmotion

Privacy-preserving kinematic analytics for RGB, RGB-D, depth, and video
streams.

`privmotion` is a proposed OpenCV-style module for transforming
identity-bearing visual streams into anonymized motion representations while
measuring the tradeoff between analytic utility and identity leakage. It is not
an existing OpenCV module today; this repository is a proposal and reference
design for a future implementation that could mature toward an
`opencv_contrib`-style module.

The goal is to let applications analyze human motion without retaining raw
appearance data by default. Instead of exporting full RGB frames, the module
would emit skeletons, silhouettes, low-resolution depth surrogates, or encrypted
kinematic features together with reports that quantify what downstream analytic
value remains and what privacy risk is still present.

## Phase 2 Prototype

This repository now includes a lightweight Python prototype for Phase 2. It is
intended to exercise the data flow, export formats, retention validation, and
CLI shape before any production pose or segmentation model is selected.

Install locally:

```bash
python -m pip install -e .
```

Install optional real-pose support:

```bash
python -m pip install -e ".[pose]"
```

Run on an image, video, or directory of image frames:

```bash
privmotion-process \
  --input examples/session01.mp4 \
  --output out/session01 \
  --mode skeleton,silhouette,depth-surrogate,features \
  --retention no-raw-rgb

privmotion-validate \
  --output out/session01 \
  --policy no-raw-rgb
```

By default, `--pose-backend auto` attempts YOLO-Pose when the `pose` extra is
installed and model loading succeeds. If YOLO is unavailable, it falls back to
the deterministic prototype backend and records the fallback reason in
`metadata.json`. To force the prototype backend:

```bash
privmotion-process \
  --input examples/session01.mp4 \
  --output out/session01_prototype \
  --mode skeleton,silhouette,depth-surrogate,features \
  --pose-backend prototype
```

Use YOLO-Pose explicitly for model-backed 2D skeletons:

```bash
privmotion-process \
  --input examples/session01.mp4 \
  --output out/session01_yolo \
  --mode skeleton,silhouette,depth-surrogate,features \
  --retention no-raw-rgb \
  --pose-backend yolo \
  --pose-model yolo11n-pose.pt
```

The same commands can be run as modules during development:

```bash
python -m privmotion.cli.process --help
python -m privmotion.cli.validate --help
```

MP4 input is supported through OpenCV when available, with an `imageio`/ffmpeg
fallback for environments where `cv2` is not installed. If a video codec is not
available locally, extract the video into an image-frame directory and pass that
directory as `--input`.

Current outputs:

- `metadata.json` records input summary, config, backend names, retention
  policy, and prototype limitations.
- `skeletons.json` records timestamps, frame indices, track IDs, mask-derived
  placeholder keypoints, confidence values, and simple keypoint velocities.
- `silhouettes/` contains binary anonymized person-mask images.
- `depth_surrogates/` contains downsampled and quantized depth-like surrogate
  images.
- `features.json` contains machine-readable geometry features reserved for
  later encrypted-feature support.
- `retention_report.json` records whether the `no-raw-rgb` policy passed.

The current segmentation and skeleton backends are deterministic placeholders:
they use luminance and mask geometry to keep the prototype runnable without
model downloads. They are not production person segmentation or anatomical pose
estimators. YOLO-Pose is preferred automatically when the `pose` extra is
installed; it may download model weights on first use. When YOLO is active, the
silhouette mask is constrained to the detected person bounding box so broad
background foreground masks do not become full-scene silhouettes. Phase 2 also
does not implement reversible access controls or encryption; those remain Phase
5 topics.

## Phase 3 Benchmark Harness

This repository also includes a local Phase 3 benchmark harness. It reads an
existing `privmotion-process` output directory and produces deterministic
utility, privacy, and systems proxy metrics.

Run a benchmark:

```bash
privmotion-benchmark \
  --output out/session01 \
  --report out/session01/benchmark_report.json
```

Development form:

```bash
python -m privmotion.cli.benchmark --help
```

The benchmark report includes:

- **Utility metrics:** processed frame count, skeleton record count, keypoint
  coverage, average keypoint confidence, silhouette coverage, depth surrogate
  count, feature record count, and keypoint-velocity stability proxy.
- **Privacy metrics:** raw-RGB retention result, mask-only visual output check,
  surrogate resolution reduction, feature uniqueness proxy, and residual-risk
  notes.
- **Systems metrics:** output file count, output byte size, skipped frame count,
  available artifacts, and backend metadata.

These metrics are deterministic local proxies. They are not model-backed
face-recognition, person re-identification, gait-identification, or pose
accuracy scores. ORPose-Depth and Market-1501-style dataset adapters remain
future integrations.

## Phase 3.5 Visualization Demo

Phase 3.5 adds an anonymized visualization layer for demo and inspection. It
reads only existing `privmotion-process` outputs and does not access the raw RGB
input. The default output is a GIF because it is portable and testable without
OpenCV video codecs or ffmpeg.

Create an anonymized preview:

```bash
privmotion-visualize \
  --output out/session01 \
  --visualization out/session01/preview.gif \
  --frames-dir out/session01/preview_frames
```

Create an MP4 preview:

```bash
privmotion-visualize \
  --output out/session01 \
  --visualization out/session01/preview.mp4 \
  --fps 8 \
  --size 640x360
```

Development form:

```bash
python -m privmotion.cli.visualize --help
```

The visualizer renders:

- Silhouette and depth-surrogate panels when those artifacts are available.
- Skeleton overlays from `skeletons.json`.
- Frame index, track identifiers, available artifact labels, and retention
  status.
- Optional per-frame PNG previews with `--frames-dir`.

The visualizer is a demo aid, not a production renderer. It supports GIF and
MP4 output and intentionally avoids raw RGB comparison views to preserve the
privacy-by-default behavior.

## Dataset Manifest Evaluation

Use a local manifest to evaluate real videos/images without committing dataset
files to the repository:

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

Run the evaluation:

```bash
privmotion-dataset-eval \
  --manifest datasets/demo_manifest.json \
  --output out/dataset_eval \
  --visualize \
  --visualization-ext .mp4
```

Each sample gets its own processed output directory and benchmark report. The
dataset output also includes `dataset_report.json` with sample count, processed
frames, retention pass rate, and average keypoint coverage.

## Motivation

Motion analytics is useful in rehabilitation, ergonomics, safety monitoring,
sports analysis, retail flow analysis, and human-computer interaction. These
same workflows can also expose faces, clothing, body shape, gait, location
context, and other identifying signals.

OpenCV already provides adjacent building blocks for video processing, image
segmentation, tracking, calibration, RGB-D processing, and DNN inference. A
dedicated privacy-preserving motion module would connect those capabilities into
a pipeline that treats anonymization quality as a first-class output rather than
an afterthought.

`privmotion` should be designed around four principles:

- **Anonymize by default:** raw RGB identity-bearing output is not retained or
  exported unless explicitly enabled for a controlled validation workflow.
- **Measure the tradeoff:** every anonymization mode should be evaluated for
  both utility retention and identity leakage.
- **Keep outputs useful:** downstream consumers should receive structured,
  machine-readable kinematic data, not only blurred images.
- **Document residual risk:** pose, gait, silhouette, and depth cues can still
  leak identity, so the module should report limitations instead of claiming
  absolute privacy.

## Supported Inputs

The proposed module should support the following input classes:

- RGB frames or videos from ordinary cameras.
- RGB-D streams with synchronized color and depth.
- Depth-only streams from commodity depth sensors.
- Batched image sequences for benchmark and offline evaluation.
- Live video streams where only anonymized outputs are persisted.

Input adapters should preserve timestamps, camera metadata, calibration data,
and stream identity when available so exported kinematic sequences remain useful
for temporal analytics.

## Anonymized Outputs

`privmotion` should support several output modes because privacy requirements
and analytic tasks vary:

- **Skeletons:** 2D or 3D keypoints, confidence scores, track identifiers, and
  temporal metadata.
- **Silhouettes:** binary or soft person masks with face and appearance
  suppression.
- **Low-resolution depth surrogates:** downsampled, quantized, or otherwise
  privacy-filtered depth maps intended to retain coarse pose and occupancy.
- **Kinematic features:** velocities, joint angles, gait descriptors, pose
  embeddings, or action features stored without raw appearance.
- **Encrypted feature records:** optional sealed records for deployments that
  require controlled recovery or policy-gated audit workflows.

The default configuration should avoid storing raw RGB frames, unmasked faces,
or high-resolution appearance crops.

## Core Pipeline

A reference `privmotion` pipeline should include these stages:

1. **Frame ingest:** read RGB, RGB-D, depth-only, or video data while preserving
   timing and camera metadata.
2. **Person segmentation:** detect people and produce body masks or regions of
   interest.
3. **Face and appearance suppression:** remove, blur, mask, or avoid exporting
   identifying appearance cues.
4. **Pose and motion extraction:** estimate skeletal keypoints, tracks, depth
   pose, or temporal motion features.
5. **Anonymized export:** write skeletons, silhouettes, depth surrogates, or
   kinematic features in a machine-readable format.
6. **Utility evaluation:** compare anonymized outputs against task metrics such
   as pose accuracy, action recognition, or temporal stability.
7. **Privacy evaluation:** test identity leakage using face recognition,
   person re-identification, gait leakage, silhouette leakage, and
   reconstruction attempts.
8. **Storage policy enforcement:** apply retention, encryption, audit, and
   access-control settings before data is persisted.

## Proposed API Concepts

The first implementation should define APIs around behavior rather than a
single model architecture. The names below are proposal-level concepts, not a
committed ABI.

### `PrivMotionPipeline`

End-to-end stream processor that accepts frames or video streams and emits
anonymized motion outputs plus optional evaluation reports.

Expected responsibilities:

- Load and validate input stream metadata.
- Run segmentation, suppression, pose extraction, and export stages.
- Apply storage and retention policies.
- Expose hooks for benchmark evaluators.

### `AnonymizationConfig`

Configuration object for selecting privacy and output behavior.

Expected fields:

- Output mode: skeleton, silhouette, depth surrogate, kinematic features, or
  encrypted feature record.
- Spatial resolution and depth quantization settings.
- Face, body texture, clothing, and background suppression policy.
- Raw frame retention policy.
- Encryption and key-management options for optional recoverable workflows.
- Evaluator settings for utility and privacy reports.

### `KinematicFrame` and `KinematicSequence`

Structured representations for anonymized motion data.

Expected fields:

- Timestamp and frame index.
- Person or track identifier.
- 2D or 3D keypoints with confidence values.
- Optional joint angles, velocities, and derived motion descriptors.
- Links to anonymized masks or depth surrogates when exported.

### `UtilityReport`

Benchmark result describing how well anonymized outputs preserve analytic value.

Expected metrics:

- Pose accuracy against labeled keypoints.
- Action-recognition retention compared with raw or less-anonymized baselines.
- Temporal stability across frames.
- Segmentation quality.
- Depth surrogate usefulness for downstream pose or activity tasks.

### `PrivacyReport`

Benchmark result describing residual identity leakage.

Expected metrics:

- Face-recognition success after suppression.
- Person re-identification accuracy.
- Gait-identification leakage.
- Silhouette or body-shape leakage.
- Reconstruction risk from depth surrogates or kinematic embeddings.

## Evaluation Plan

Evaluation should report privacy and utility together. A mode that preserves
excellent action recognition but leaks identity through gait or silhouettes
should not be treated as successful without that caveat.

### Analytic Utility

- Pose accuracy and keypoint confidence.
- Action-recognition retention.
- Temporal stability and track consistency.
- Segmentation quality.
- Downstream usefulness of low-resolution depth surrogates.

### Identity Leakage

- Face-recognition success after anonymization.
- Market-1501-style person re-identification leakage.
- Gait and motion-signature leakage.
- Silhouette, body-shape, and clothing leakage.
- Reconstruction attacks against depth surrogates or learned kinematic
  embeddings.

### Systems Behavior

- Latency per frame and end-to-end throughput.
- Storage reduction relative to raw RGB or RGB-D streams.
- Robustness across lighting, occlusion, camera viewpoint, and depth noise.
- Behavior under missing depth, dropped frames, and partial segmentation.

## Recommended Data Sources

- **ORPose-Depth:** RGB-D and depth pose evaluation, especially for utility
  retention in pose-centric workflows.
- **Market-1501-style re-identification datasets:** leakage testing for
  person identity, clothing, and body-shape cues.
- **Consented custom video:** policy validation, storage-retention testing,
  deployment-specific camera geometry, and optional reversible-access
  workflows.

Any dataset use should be reviewed for license terms, subject consent,
benchmark suitability, and whether the data itself contains sensitive
identity-bearing content.

## Reversible Access Controls

The default privacy model should be irreversible anonymization. Optional
recoverable workflows should be treated as a future extension, not as the
baseline behavior.

If reversible access is implemented, it should require:

- Explicit encryption configuration.
- Clear key-management boundaries.
- Access policy checks before recovery.
- Audit logging for every recovery attempt.
- Consent-oriented validation with deployment-specific review.
- Separate evaluation of recovery risk and misuse scenarios.

This extension should never silently preserve raw RGB data under an
"anonymized" label.

## Future Prototype Acceptance Criteria

A first runnable prototype should meet these criteria:

- Produces no raw RGB identity-bearing output by default.
- Exports machine-readable anonymized motion data.
- Supports at least one skeletal output mode and one silhouette or depth
  surrogate output mode.
- Reports both utility retention and identity leakage.
- Includes clear documentation of limitations, residual risks, and misuse
  scenarios.
- Provides benchmark scripts that can run on public datasets and consented
  local video.

## Roadmap

### Phase 1: Documentation and Reference Design

- Define the module proposal, API concepts, output formats, and evaluation
  requirements.
- Align naming and structure with OpenCV and `opencv_contrib` conventions.
- Document privacy assumptions, residual risks, and excluded claims.

### Phase 2: Python Prototype

The Phase 2 prototype demonstrates an end-to-end anonymized motion export
workflow from RGB, video, or image-directory input. It remains a Python
prototype, not the final OpenCV-contrib module shape.

Implemented stack:

- OpenCV for video/image I/O, frame transforms, masking, resizing, depth-map
  handling, and visualization helpers when `cv2` is available.
- Imageio/ffmpeg fallback support for MP4 and other common video inputs when
  OpenCV is not available.
- NumPy for deterministic fallback processing and portable test fixtures.
- Pluggable segmentation, pose, and tracking backends.
- Ultralytics YOLO-Pose as the preferred `auto` backend when installed with
  `python -m pip install -e ".[pose]"`, with prototype fallback.
- Structured JSON exports for skeletons, metadata, and feature records.
- Portable grayscale mask images for silhouettes and depth surrogates.

Implemented pipeline stages:

1. Read frames from RGB, RGB-D, depth-only, or video input.
2. Detect or segment people and assign track identifiers where possible.
3. Suppress faces, body texture, clothing detail, and background appearance
   before any persisted visual output is written.
4. Extract pose keypoints, depth pose, or derived motion features.
5. Export anonymized skeletons, silhouettes, depth surrogates, or feature
   records.
6. Validate that raw RGB frames are not written by default.

Implemented outputs:

- **Skeleton export:** timestamps, frame indices, track IDs, 2D or 3D
  keypoints, confidence values, and optional derived motion features such as
  joint angles or velocities.
- **Silhouette export:** binary or soft person masks with no retained face or
  clothing texture.
- **Depth surrogate export:** downsampled or quantized depth maps intended to
  preserve coarse pose and occupancy while reducing reconstruction risk.
- **Feature record export:** machine-readable kinematic records designed for
  later encryption support, without implementing reversible access in Phase 2.

Implemented commands:

```bash
privmotion-process \
  --input examples/session01.ppm \
  --output out/session01 \
  --mode skeleton,silhouette \
  --retention no-raw-rgb \
  --segmentation-backend auto \
  --pose-backend auto

privmotion-process \
  --input examples/session01.mp4 \
  --output out/session01_yolo \
  --mode skeleton,silhouette,depth-surrogate,features \
  --retention no-raw-rgb \
  --pose-backend yolo \
  --pose-model yolo11n-pose.pt

privmotion-validate \
  --output out/session01 \
  --policy no-raw-rgb \
  --report out/session01/retention_report.json
```

Implemented configuration concepts:

- Input path and stream type.
- Output mode selection.
- Segmentation, pose, tracking, and depth backends.
- Pose model selection for opt-in YOLO-Pose.
- Requested pose backend, resolved pose backend, pose model, segmentation mode,
  and fallback reason metadata.
- Retention policy, with `no-raw-rgb` as the default.
- Export paths for skeleton JSON, silhouette masks, depth surrogates, and
  feature records.
- Metadata fields needed by future utility and privacy benchmarks.

Acceptance criteria:

- Raw RGB identity-bearing data is not written by default.
- Skeleton, silhouette, depth surrogate, and feature modes are available.
- Outputs are machine-readable and include timestamps or frame indices.
- Exported metadata is sufficient for later benchmark evaluation.
- Limitations and residual privacy risks are visible in metadata and docs.

### Phase 3: Benchmark Harness

The Phase 3 benchmark harness is implemented locally. It reads Phase 2 output
directories and writes a `benchmark_report.json` with utility, privacy, and
systems proxy metrics.

Implemented command:

```bash
privmotion-benchmark \
  --output out/session01 \
  --report out/session01/benchmark_report.json
```

Implemented behavior:

- Reads `metadata.json`, `skeletons.json`, `features.json`,
  `retention_report.json`, `silhouettes/`, and `depth_surrogates/` when
  available.
- Requires `metadata.json`; optional artifacts are reported as unavailable
  metrics rather than crashes.
- Reports deterministic proxy metrics for keypoint coverage, confidence,
  silhouette coverage, surrogate reduction, feature uniqueness, retention
  status, output file counts, and backend metadata.
- Clearly labels the report as proxy-only and not a substitute for
  face-recognition, re-identification, gait, or labeled-pose evaluation.

Future dataset integrations:

- Add ORPose-Depth-style annotation adapters for pose-utility evaluation.
- Add Market-1501-style adapters for person re-identification leakage testing.
- Add consented-video benchmark manifests for deployment-policy validation.

### Phase 3.5: Visualization Demo

The Phase 3.5 visualization demo is implemented locally. It creates anonymized
GIF or MP4 previews and optional PNG frames from Phase 2 output directories.

Implemented command:

```bash
privmotion-visualize \
  --output out/session01 \
  --visualization out/session01/preview.mp4 \
  --frames-dir out/session01/preview_frames \
  --fps 4 \
  --size 640x360
```

Implemented behavior:

- Requires `metadata.json`.
- Reads optional `skeletons.json`, `retention_report.json`, `silhouettes/`, and
  `depth_surrogates/`.
- Renders mask/depth panels, skeleton overlays, frame labels, track IDs,
  available artifact labels, and no-raw-RGB retention status.
- Writes GIF or MP4 output based on the visualization file extension and PNG
  preview frames when `--frames-dir` is provided.
- Uses anonymized outputs only; it does not load or render raw RGB input.

### Dataset Manifest Evaluation

Manifest-based dataset evaluation is implemented locally. It processes each
manifest sample, benchmarks it, optionally renders a preview, and writes an
aggregate `dataset_report.json`.

Implemented command:

```bash
privmotion-dataset-eval \
  --manifest datasets/demo_manifest.json \
  --output out/dataset_eval \
  --visualize \
  --visualization-ext .mp4
```

Implemented behavior:

- Supports JSON manifests with `id`, `input`, optional `label`, optional
  `split`, and optional `expected_frames`.
- Resolves relative sample inputs relative to the manifest file.
- Writes per-sample processed artifacts and `benchmark_report.json`.
- Aggregates sample count, processed frame count, retention pass rate, and
  average keypoint coverage.
- Does not download public datasets or commit dataset media.

### Phase 4: OpenCV-Contrib-Style Module Scaffold

- Introduce C++ and Python module structure.
- Add headers, bindings, samples, documentation, and tests.
- Keep model-specific dependencies optional where possible.

### Phase 5: Optional Recoverable Controls

- Add encrypted feature records and policy-gated recovery only after the
  irreversible anonymization path is stable.
- Include audit logging, access-control checks, and explicit deployment
  guidance.
- Evaluate recovery risk separately from ordinary anonymized outputs.

## Non-Goals for the Initial Proposal

- Claiming that `privmotion` is part of OpenCV today.
- Guaranteeing anonymity from skeleton, silhouette, gait, or depth outputs.
- Choosing a single pose-estimation architecture as the permanent design.
- Persisting raw RGB data as a default behavior.
- Treating reversible recovery as a required v1 feature.

## License

This repository is licensed under the Apache License 2.0. See
[`LICENSE`](LICENSE) for details.
