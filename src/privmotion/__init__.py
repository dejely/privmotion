"""Phase 2 Python prototype for privacy-preserving kinematic analytics."""

from privmotion.config import ProcessConfig, parse_output_modes
from privmotion.pipeline import ProcessResult, PrivMotionPipeline
from privmotion.validation import RetentionValidationResult, validate_output_dir
from privmotion.benchmark import BenchmarkReport, benchmark_output_dir
from privmotion.visualization import VisualizationResult, visualize_output_dir
from privmotion.dataset_eval import DatasetEvaluationReport, evaluate_dataset_manifest

__all__ = [
    "BenchmarkReport",
    "DatasetEvaluationReport",
    "PrivMotionPipeline",
    "ProcessConfig",
    "ProcessResult",
    "RetentionValidationResult",
    "VisualizationResult",
    "benchmark_output_dir",
    "evaluate_dataset_manifest",
    "parse_output_modes",
    "validate_output_dir",
    "visualize_output_dir",
]

__version__ = "0.1.0"
