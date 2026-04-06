#include "opencv2/privmotion.hpp"
#include "opencv2/ts.hpp"

namespace opencv_test {
namespace {

TEST(PrivMotion_Scaffold, DoesNotPersistRawRgbByDefault)
{
    cv::privmotion::PrivMotionPipeline pipeline;
    cv::Mat frame(32, 32, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::privmotion::KinematicFrame output = pipeline.process(frame);
    cv::privmotion::PrivacyReport privacy = pipeline.privacyReport();

    EXPECT_EQ(0, output.frameIndex);
    EXPECT_FALSE(privacy.rawRgbWritten);
    EXPECT_TRUE(privacy.retentionPassed);
}

}} // namespace

