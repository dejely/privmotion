#ifndef OPENCV_PRIVMOTION_PRIVMOTION_HPP
#define OPENCV_PRIVMOTION_PRIVMOTION_HPP

#include "opencv2/core.hpp"

#include <string>
#include <vector>

#ifndef CV_EXPORTS_W
#define CV_EXPORTS_W
#endif

namespace cv {
namespace privmotion {

//! Output modes supported by the scaffolded privacy pipeline.
enum OutputMode
{
    OUTPUT_SKELETON = 1,
    OUTPUT_SILHOUETTE = 2,
    OUTPUT_DEPTH_SURROGATE = 4,
    OUTPUT_FEATURES = 8
};

//! Configuration for anonymized kinematic export.
struct CV_EXPORTS_W AnonymizationConfig
{
    int outputModes;
    int surrogateMaxSize;
    bool retainRawRgb;
    String poseBackend;
    String retentionPolicy;

    AnonymizationConfig();
};

//! One 2D kinematic keypoint.
struct CV_EXPORTS_W KinematicKeypoint
{
    String name;
    Point2f point;
    float confidence;

    KinematicKeypoint();
    KinematicKeypoint(const String& name_, Point2f point_, float confidence_);
};

//! Per-frame anonymized motion record.
struct CV_EXPORTS_W KinematicFrame
{
    int frameIndex;
    double timestampMs;
    int trackId;
    Rect bbox;
    Point2f centroid;
    std::vector<KinematicKeypoint> keypoints;
};

//! Privacy report placeholder matching the Python prototype concepts.
struct CV_EXPORTS_W PrivacyReport
{
    bool rawRgbWritten;
    bool retentionPassed;
    std::vector<String> residualRiskNotes;
};

//! Utility report placeholder matching the Python prototype concepts.
struct CV_EXPORTS_W UtilityReport
{
    int processedFrameCount;
    int skeletonRecordCount;
    double keypointFrameCoverage;
    double averageKeypointConfidence;
};

//! OpenCV-contrib-style scaffold for privacy-preserving motion analytics.
//!
//! The Phase 4 C++ scaffold intentionally avoids model loading and raw RGB
//! persistence. The Python package remains the runnable implementation.
//! Intended Python binding surface:
//! - cv.privmotion.AnonymizationConfig
//! - cv.privmotion.PrivMotionPipeline
//! - cv.privmotion.KinematicFrame
class CV_EXPORTS_W PrivMotionPipeline
{
public:
    CV_WRAP explicit PrivMotionPipeline(const AnonymizationConfig& config = AnonymizationConfig());

    CV_WRAP KinematicFrame process(InputArray frame);

    CV_WRAP PrivacyReport privacyReport() const;
    CV_WRAP UtilityReport utilityReport() const;
    CV_WRAP AnonymizationConfig config() const;

private:
    AnonymizationConfig config_;
    PrivacyReport privacyReport_;
    UtilityReport utilityReport_;
    int frameCounter_;
};

} // namespace privmotion
} // namespace cv

#endif // OPENCV_PRIVMOTION_PRIVMOTION_HPP

