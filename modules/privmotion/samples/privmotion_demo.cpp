#include "opencv2/privmotion.hpp"

#include <iostream>

int main()
{
    cv::privmotion::AnonymizationConfig config;
    cv::privmotion::PrivMotionPipeline pipeline(config);

    cv::Mat frame(240, 320, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::privmotion::KinematicFrame result = pipeline.process(frame);
    cv::privmotion::PrivacyReport privacy = pipeline.privacyReport();

    std::cout << "frame_index=" << result.frameIndex << std::endl;
    std::cout << "keypoints=" << result.keypoints.size() << std::endl;
    std::cout << "raw_rgb_written=" << (privacy.rawRgbWritten ? "true" : "false") << std::endl;
    return privacy.rawRgbWritten ? 1 : 0;
}

