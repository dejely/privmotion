# Contributing to privmotion

Thanks for helping improve `privmotion`. This project is a runnable Python
prototype plus an OpenCV-contrib-style scaffold for privacy-preserving motion
analytics. Contributions should keep that scope clear: the Python package is
the working implementation, while `modules/privmotion/` is a future integration
shape.

Please also follow the project [Code of Conduct](CODE_OF_CONDUCT.md).

## Project Principles

- Keep `no-raw-rgb` as the default retention model.
- Do not commit raw identity-bearing videos, private datasets, generated output
  directories, YOLO weights, or model downloads.
- Treat skeletons, silhouettes, depth surrogates, and motion features as
  potentially privacy-sensitive because they can still leak identity.
- Prefer deterministic, local tests over downloads or external services.
- Keep prototype behavior clearly labeled when it is not production-grade.

## Development Setup

Create and activate a virtual environment, then install the package in editable
mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

Install optional YOLO-Pose support only when you need the real-pose backend:

```bash
python -m pip install -e ".[pose,test]"
```

YOLO may download model weights such as `yolo11n-pose.pt` on first use. These
files should remain untracked.

## Run Checks

Run the test suite:

```bash
python -m pytest
```

Useful CLI smoke checks:

```bash
PYTHONPATH=src python -m privmotion.cli.process --help
PYTHONPATH=src python -m privmotion.cli.validate --help
PYTHONPATH=src python -m privmotion.cli.visualize --help
PYTHONPATH=src python -m privmotion.cli.benchmark --help
PYTHONPATH=src python -m privmotion.cli.dataset_eval --help
```

For a local end-to-end run:

```bash
PYTHONPATH=src python -m privmotion.cli.process \
  --input examples/videoplayback.mp4 \
  --output out/demo \
  --mode skeleton,silhouette,depth-surrogate,features

PYTHONPATH=src python -m privmotion.cli.validate \
  --output out/demo

PYTHONPATH=src python -m privmotion.cli.visualize \
  --output out/demo \
  --visualization out/demo/preview.mp4 \
  --fps 8 \
  --size 1280x720
```

## Contribution Areas

- Python pipeline and backend improvements in `src/privmotion/`.
- CLI improvements in `src/privmotion/cli/`.
- Benchmark and dataset-evaluation improvements.
- Anonymized visualization improvements for GIF/MP4 previews.
- Tests under `tests/`.
- OpenCV-style scaffold updates under `modules/privmotion/`.
- Documentation, examples, and anonymized demo media.

## Privacy and Data Rules

Do not add raw videos of identifiable people unless the data is explicitly
consented, legally shareable, and necessary for the change. Synthetic fixtures
are preferred for tests.

Generated outputs should stay under ignored directories such as `out/`,
`outputs/`, or `privmotion-output/`. Demo media intended for GitHub can live in
`assets/`, but it should be anonymized output such as a rendered preview GIF or
MP4, not the original source video.

If a contribution changes retention behavior, validation, exporters, or output
formats, include tests that prove raw RGB is not written by default.

## Pull Request Checklist

Before opening a pull request, check that:

- `python -m pytest` passes.
- New behavior has focused tests.
- README or docs are updated when public commands, outputs, or assumptions
  change.
- No raw RGB outputs, model weights, caches, virtual environments, or generated
  output directories are staged.
- Privacy limitations are documented when the change could affect identity
  leakage, retention, or reversibility.

## Reporting Issues

When reporting a bug, include:

- The command you ran.
- The input type, for example image, MP4, or frame directory.
- The selected modes and backend flags.
- Relevant error output.
- Whether YOLO-Pose was installed.

Avoid attaching raw identity-bearing media to public issues. Use synthetic or
anonymized reproductions whenever possible.
