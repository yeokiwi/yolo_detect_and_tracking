#pragma once

#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include "detection.h"
#include <vector>
#include <string>

class YOLOv9Detector {
public:
    struct Config {
        std::string modelPath;
        float confThreshold = 0.25f;
        float nmsThreshold  = 0.45f;
        int inputWidth      = 640;
        int inputHeight     = 640;
        bool useCUDA        = false;
    };

    explicit YOLOv9Detector(const Config& cfg);

    // Run detection on a BGR frame. Returns detections in original-frame coords.
    std::vector<Detection> detect(const cv::Mat& frame);

private:
    // Letterbox resize to input dimensions, returns blob
    cv::Mat preprocess(const cv::Mat& frame, float& scaleX, float& scaleY,
                       int& padLeft, int& padTop);

    // Decode network output, apply NMS, map coords to original frame
    std::vector<Detection> postprocess(const cv::Mat& output,
                                       float scaleX, float scaleY,
                                       int padLeft, int padTop,
                                       int origWidth, int origHeight);

    cv::dnn::Net net_;
    Config cfg_;
    std::vector<std::string> classNames_;

    void loadClassNames();
};
