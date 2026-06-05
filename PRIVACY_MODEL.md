# Privacy Model

`privmotion` is designed to provide, what data it handles, what it stores, and what it does not guarantee.

`privmotion` is a research prototype. It reduces raw visual exposure by default,
but it does not prove anonymity and it does not replace consent, access control,
or legal review.

## Summary

By default, `privmotion` follows a `no-raw-rgb` retention model:

- Raw RGB video or image frames may enter the system for processing.
- Raw RGB frames are used in memory only.
- Raw RGB frames are not copied into the output directory by default.
- The tool stores anonymized or reduced motion artifacts instead.
- The generated artifacts can still be privacy-sensitive.

The intended privacy level is **raw visual suppression and retention
minimization**, not cryptographic anonymity or formal de-identification.

## What Data Enters the System?

The system can receive:

- RGB image files.
- MP4 or other OpenCV-readable video files.
- Directories of image frames.
- RGB-D or depth-like inputs when provided through supported readable formats.
- CLI configuration such as output modes, backend choices, retention policy,
  frame limits, and output paths.
- Optional model configuration such as a YOLO-Pose model path.

When YOLO-Pose is used, raw frames are passed to the model in memory for
inference. The model-backed path should still be treated as processing
identity-bearing data, even though the frames are not retained by default.

## What Data Is Stored?

Depending on the selected modes and commands, a processed output directory can
store:

- `metadata.json`: input summary, backend names, selected modes, retention
  policy, frame counts, skipped-frame counts, and fallback notes.
- `skeletons.json`: frame indices, timestamps, track IDs, keypoints,
  confidence values, bounding boxes, centroids, and simple motion fields.
- `features.json`: machine-readable kinematic feature records.
- `features.json` with `encrypted_records`: opt-in Fernet-encrypted kinematic
  feature records when Phase 5 feature encryption is enabled.
- `access_policy.json`: normalized access policy copied for encrypted-feature
  runs.
- `audit_log.jsonl`: append-only JSONL audit events for encrypted-feature runs
  and recovery-policy inspection.
- `retention_report.json`: validation results for the configured retention
  policy.
- `benchmark_report.json`: deterministic utility, privacy-proxy, and systems
  metrics.
- `dataset_report.json`: aggregate metrics for manifest-based dataset
  evaluation.
- `silhouettes/`: binary or soft mask images.
- `depth_surrogates/`: downsampled or quantized depth-like surrogate images.
- `preview.gif` or `preview.mp4`: rendered previews created from anonymized
  outputs.

These files are not raw RGB copies, but they should still be handled as
sensitive motion data.

## What Data Is Discarded?

Under the default policy, `privmotion` discards:

- Decoded raw RGB frames after each frame is processed.
- Intermediate full-resolution image arrays used for segmentation, pose
  inference, mask generation, and visualization-source construction.
- Unrequested output modes.
- Raw input video or image bytes; the input is read, not copied into the output
  directory.

The validation command checks output directories for likely retention-policy
violations. It is a practical guardrail, not a complete forensic proof that no
raw data exists anywhere else on the machine.

## What Artifacts Are Generated?

The main generated artifacts are:

| Artifact | Purpose | Privacy sensitivity |
| --- | --- | --- |
| `skeletons.json` | Pose and motion analysis | Can leak gait, body proportions, and movement style. |
| `features.json` | Compact kinematic records | Can leak rare motion patterns or track-level identity. |
| `silhouettes/` | Person mask inspection | Can leak body shape, posture, clothing outline, and carried objects. |
| `depth_surrogates/` | Low-detail geometry proxy | Can leak posture, height proxy, and body volume cues. |
| `preview.gif` / `preview.mp4` | Human inspection of anonymized outputs | Can combine skeleton, mask, and timing cues. |
| `metadata.json` | Reproducibility and audit context | Can leak file paths, labels, backend choices, and sample identifiers. |
| Reports | Validation and benchmark results | Can reveal dataset composition or sample-level behavior. |

## Can Skeletons, Silhouettes, or Motion Features Still Identify Someone?

Yes. Removing raw RGB and face appearance reduces direct visual identification,
but it does not eliminate all identity leakage.

Possible leakage paths include:

- Gait patterns across time.
- Body proportions inferred from keypoints.
- Height, posture, or mobility signatures.
- Silhouette shape and clothing outline.
- Unique gestures or action sequences.
- Track IDs that link the same person across frames.
- Metadata such as file names, labels, locations, timestamps, or sample IDs.
- Downstream models trained to re-identify people from motion or shape.

For that reason, `privmotion` outputs should be treated as privacy-sensitive
derived data, not public-safe anonymized data by default.

## What Privacy Level Does the Tool Actually Provide?

The current prototype provides:

- **Raw RGB non-retention by default**: source RGB frames are not written into
  the output directory.
- **Appearance suppression**: exported artifacts avoid direct face and texture
  appearance when default modes are used.
- **Motion-focused export**: outputs are structured around kinematics rather
  than full visual reconstruction.
- **Retention validation**: `privmotion-validate` reports likely raw RGB
  retention violations in an output directory.
- **Proxy privacy reporting**: `privmotion-benchmark` reports deterministic
  local proxy metrics such as raw-retention status and artifact availability.
- **Opt-in encrypted feature records**: Phase 5 can encrypt kinematic feature
  records with Fernet authenticated encryption, an access policy, and audit log.

This is best understood as **privacy risk reduction**, not guaranteed
anonymization.

## What Does the Tool Not Guarantee?

`privmotion` does not guarantee:

- That a person cannot be re-identified from skeletons, silhouettes, depth
  surrogates, or motion features.
- Formal k-anonymity, differential privacy, or cryptographic privacy.
- Removal of all biometric signals.
- Legal compliance with privacy, biometric, workplace, education, medical, or
  surveillance regulations.
- Consent management or subject-rights handling.
- Complete secure storage, access control, or key management. Phase 5 provides
  opt-in encrypted feature records and audit metadata only.
- That downstream tools will not reconstruct or infer identity.
- That model-backed pose output is accurate.
- That validation can detect raw data saved outside the selected output
  directory.
- That generated previews are safe to publish.
- Decryption or recovery export from the CLI.
- Raw RGB recovery.

Optional encrypted feature controls are available only when explicitly enabled.
They do not change the default `no-raw-rgb` privacy model.

## Recommended Operating Practices

- Use consented or synthetic input whenever possible.
- Keep raw input data outside the repository.
- Store generated motion artifacts with the same care as other sensitive
  derived data.
- Prefer aggregate benchmark reports over publishing per-person artifacts.
- Run `privmotion-validate` before sharing an output directory.
- Review metadata for paths, labels, timestamps, or IDs that could identify a
  person or collection site.
- Do not publish demo media unless it is generated from anonymized outputs and
  has been reviewed for residual identity leakage.

## Default Policy

The default policy is:

```text
retention = no-raw-rgb
```

Any future option that stores raw RGB, decrypts records, or enables broader
reversible recovery should be explicit, documented as unsafe or restricted,
covered by tests, and paired with stronger access policy, audit logging, and
consent-oriented validation.
