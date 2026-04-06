#include "opencv2/privmotion.hpp"

namespace cv {
namespace privmotion {

AnonymizationConfig::AnonymizationConfig()
    : outputModes(OUTPUT_SKELETON | OUTPUT_SILHOUETTE),
      surrogateMaxSize(64),
      retainRawRgb(false),
      poseBackend("prototype"),
      retentionPolicy("no-raw-rgb")
{
}

KinematicKeypoint::KinematicKeypoint()
    : name(), point(0.f, 0.f), confidence(0.f)
{
}

KinematicKeypoint::KinematicKeypoint(const String& name_, Point2f point_, float confidence_)
    : name(name_), point(point_), confidence(confidence_)
{
}

PrivMotionPipeline::PrivMotionPipeline(const AnonymizationConfig& config)
    : config_(config), frameCounter_(0)
{
    // The scaffold never persists raw RGB frames. Future implementations should
    // keep raw-frame retention opt-in and auditable.
    config_.retainRawRgb = false;
    config_.retentionPolicy = "no-raw-rgb";

    privacyReport_.rawRgbWritten = false;
    privacyReport_.retentionPassed = true;
    privacyReport_.residualRiskNotes.push_back(
        "Skeletons, silhouettes, and depth surrogates can still leak identity.");

    utilityReport_.processedFrameCount = 0;
    utilityReport_.skeletonRecordCount = 0;
    utilityReport_.keypointFrameCoverage = 0.0;
    utilityReport_.averageKeypointConfidence = 0.0;
}

KinematicFrame PrivMotionPipeline::process(InputArray frame)
{
    Mat input = frame.getMat();

    KinematicFrame output;
    output.frameIndex = frameCounter_++;
    output.timestampMs = static_cast<double>(output.frameIndex);
    output.trackId = input.empty() ? 0 : 1;

    if (!input.empty())
    {
        output.bbox = Rect(0, 0, input.cols, input.rows);
        output.centroid = Point2f(input.cols * 0.5f, input.rows * 0.5f);
        output.keypoints.push_back(KinematicKeypoint("center", output.centroid, 0.25f));
    }

    utilityReport_.processedFrameCount += 1;
    utilityReport_.skeletonRecordCount += output.keypoints.empty() ? 0 : 1;
    utilityReport_.keypointFrameCoverage =
        utilityReport_.processedFrameCount == 0
            ? 0.0
            : static_cast<double>(utilityReport_.skeletonRecordCount) /
                  static_cast<double>(utilityReport_.processedFrameCount);
    utilityReport_.averageKeypointConfidence = output.keypoints.empty() ? 0.0 : 0.25;

    return output;
}

PrivacyReport PrivMotionPipeline::privacyReport() const
{
    return privacyReport_;
}

UtilityReport PrivMotionPipeline::utilityReport() const
{
    return utilityReport_;
}

AnonymizationConfig PrivMotionPipeline::config() const
{
    return config_;
}

} // namespace privmotion
} // namespace cv

