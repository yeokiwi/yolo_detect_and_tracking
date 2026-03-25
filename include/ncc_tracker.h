#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

class NCCTracker {
public:
    struct Config {
        float searchRegionPadding = 2.0f;   // Search area = bbox * this factor
        float minConfidence       = 0.5f;   // Below this, track is considered lost
        float templateUpdateThresh = 0.7f;  // Only update template above this confidence
        int maxTemplateSize       = 128;    // Cap template dimension to avoid slow matching
    };

    explicit NCCTracker(const Config& cfg = {});

    // Initialize tracker with a frame and bounding box (from detection)
    void init(const cv::Mat& frame, const cv::Rect2f& bbox);

    // Update tracker on a new frame. Returns true if confidence >= threshold.
    bool update(const cv::Mat& frame, cv::Rect2f& updatedBox, float& confidence);

    // Re-initialize template from a fresh detection
    void resetTemplate(const cv::Mat& frame, const cv::Rect2f& bbox);

private:
    cv::Mat template_;           // Stored ROI patch (grayscale)
    cv::Rect2f lastBox_;         // Last known bounding box
    Config cfg_;
    bool initialized_ = false;

    // Extract grayscale template from frame at given bbox
    cv::Mat extractTemplate(const cv::Mat& frame, const cv::Rect2f& bbox);

    // Compute expanded search region clamped to frame bounds
    cv::Rect computeSearchRegion(const cv::Size& frameSize);

    // Clamp a Rect2f to frame boundaries and convert to integer Rect
    static cv::Rect clampRect(const cv::Rect2f& rect, const cv::Size& frameSize);
};
