from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "modules" / "privmotion"


def test_opencv_scaffold_files_exist() -> None:
    expected = [
        MODULE / "CMakeLists.txt",
        MODULE / "include" / "opencv2" / "privmotion.hpp",
        MODULE / "include" / "opencv2" / "privmotion" / "privmotion.hpp",
        MODULE / "src" / "privmotion.cpp",
        MODULE / "samples" / "privmotion_demo.cpp",
        MODULE / "test" / "test_privmotion.cpp",
        MODULE / "doc" / "privmotion.markdown",
    ]

    missing = [path for path in expected if not path.exists()]

    assert not missing


def test_public_header_declares_expected_api() -> None:
    header = (MODULE / "include" / "opencv2" / "privmotion" / "privmotion.hpp").read_text(
        encoding="utf-8"
    )

    for name in [
        "namespace privmotion",
        "AnonymizationConfig",
        "KinematicKeypoint",
        "KinematicFrame",
        "PrivacyReport",
        "UtilityReport",
        "PrivMotionPipeline",
        "process(InputArray frame)",
    ]:
        assert name in header


def test_module_docs_state_raw_rgb_default() -> None:
    doc = (MODULE / "doc" / "privmotion.markdown").read_text(encoding="utf-8").lower()

    assert "does not persist raw rgb by default" in doc
    assert "python package remains the runnable implementation" in doc


def test_cmake_declares_opencv_module() -> None:
    cmake = (MODULE / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "ocv_define_module(privmotion" in cmake
    assert "WRAP python" in cmake
