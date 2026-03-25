#pragma once

#include <string>

struct AppConfig {
    // Paths
    std::string videoPath  = "input.mp4";
    std::string modelPath  = "models/yolov9.onnx";

    // Detection
    float confThreshold    = 0.25f;
    float nmsThreshold     = 0.45f;
    int   inputSize        = 640;
    bool  useCUDA          = false;

    // Tracking
    float searchPadding    = 2.0f;
    float nccMinConfidence = 0.5f;
    int   maxTemplateSize  = 128;

    // Association
    float iouThreshold     = 0.3f;
    int   maxFramesLost    = 15;

    // Pipeline
    int   detectInterval   = 30;
};
