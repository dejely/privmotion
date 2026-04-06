# PrivMotion

`privmotion` is a proposed OpenCV-contrib-style module for privacy-preserving
kinematic analytics. It transforms visual input into anonymized motion
representations such as skeletons, silhouettes, depth surrogates, and kinematic
feature records.

This Phase 4 scaffold defines the intended C++ module surface. The existing
Python package remains the runnable implementation.

## Privacy Default

The scaffold does not persist raw RGB by default. Future implementations should
keep raw-frame retention explicit, audited, and disabled for ordinary
anonymized processing.

## API Concepts

- `cv::privmotion::AnonymizationConfig`
- `cv::privmotion::KinematicKeypoint`
- `cv::privmotion::KinematicFrame`
- `cv::privmotion::PrivacyReport`
- `cv::privmotion::UtilityReport`
- `cv::privmotion::PrivMotionPipeline`

## Non-Goals

- This scaffold is not a complete OpenCV module implementation.
- It does not load YOLO, Ultralytics, or any other model runtime.
- It does not implement generated Python bindings in this repository.
- It does not claim to guarantee anonymity.

## Relationship To The Python Prototype

The Python prototype currently provides processing, validation, benchmarking,
dataset manifest evaluation, and preview rendering. The C++ scaffold mirrors
the same concepts so a future OpenCV-contrib implementation can evolve without
changing the public model too sharply.

