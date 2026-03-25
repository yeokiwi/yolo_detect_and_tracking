#pragma once

#include <opencv2/core.hpp>
#include "detection.h"
#include <vector>

// Compute Intersection-over-Union between two rectangles
float computeIoU(const cv::Rect2f& a, const cv::Rect2f& b);

// Draw bounding boxes and labels on the frame
void drawTracks(cv::Mat& frame, const std::vector<Track>& tracks);

// Generate a stable color from a track ID
cv::Scalar colorFromId(int id);
