#pragma once

#include "detection.h"
#include "ncc_tracker.h"
#include <vector>
#include <utility>

class MultiTracker {
public:
    struct Config {
        float iouThreshold    = 0.3f;   // Minimum IoU to associate detection with track
        int maxFramesLost     = 15;     // Remove track after this many lost frames
        NCCTracker::Config trackerCfg;
    };

    explicit MultiTracker(const Config& cfg = {});

    // Called on detection frames: associate detections with tracks, manage lifecycle
    void onDetection(const cv::Mat& frame, const std::vector<Detection>& detections);

    // Called on tracking-only frames: update all active tracks via NCC
    void onTrackingFrame(const cv::Mat& frame);

    // Get current active tracks for visualization
    const std::vector<Track>& getTracks() const;

private:
    std::vector<Track> tracks_;
    std::vector<NCCTracker> trackers_; // Parallel to tracks_
    Config cfg_;
    int nextTrackId_ = 0;

    // Greedy IoU matching: returns (detectionIdx, trackIdx) pairs
    std::vector<std::pair<int, int>> associateDetections(
        const std::vector<Detection>& detections);

    // Remove inactive tracks
    void pruneInactiveTracks();
};
