#pragma once

#include <opencv2/core.hpp>
#include <string>

struct Detection {
    cv::Rect2f bbox;        // Bounding box (x, y, width, height) in original frame coords
    int classId = -1;
    float confidence = 0.0f;
    std::string label;
};

struct Track {
    int id = -1;                // Unique track ID
    Detection lastDetection;    // Most recent associated detection
    cv::Rect2f predictedBox;    // Current position (from tracker or detection)
    float trackConfidence = 0.0f; // NCC match confidence from last tracking step
    int framesLost = 0;         // Consecutive frames with low confidence or no match
    bool active = true;
};
