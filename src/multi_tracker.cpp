#include "multi_tracker.h"
#include "utils.h"
#include <algorithm>
#include <set>

MultiTracker::MultiTracker(const Config& cfg) : cfg_(cfg) {}

const std::vector<Track>& MultiTracker::getTracks() const {
    return tracks_;
}

std::vector<std::pair<int, int>> MultiTracker::associateDetections(
    const std::vector<Detection>& detections) {

    std::vector<std::pair<int, int>> matches;
    if (detections.empty() || tracks_.empty()) return matches;

    int numDets = static_cast<int>(detections.size());
    int numTrks = static_cast<int>(tracks_.size());

    // Build IoU matrix
    std::vector<std::vector<float>> iouMatrix(numDets, std::vector<float>(numTrks, 0.0f));
    for (int d = 0; d < numDets; ++d) {
        for (int t = 0; t < numTrks; ++t) {
            iouMatrix[d][t] = computeIoU(detections[d].bbox, tracks_[t].predictedBox);
        }
    }

    // Greedy matching: pick the highest IoU pair repeatedly
    std::set<int> matchedDets, matchedTrks;

    while (true) {
        float bestIoU = cfg_.iouThreshold;
        int bestDet = -1, bestTrk = -1;

        for (int d = 0; d < numDets; ++d) {
            if (matchedDets.count(d)) continue;
            for (int t = 0; t < numTrks; ++t) {
                if (matchedTrks.count(t)) continue;
                if (iouMatrix[d][t] > bestIoU) {
                    bestIoU = iouMatrix[d][t];
                    bestDet = d;
                    bestTrk = t;
                }
            }
        }

        if (bestDet < 0) break;

        matches.emplace_back(bestDet, bestTrk);
        matchedDets.insert(bestDet);
        matchedTrks.insert(bestTrk);
    }

    return matches;
}

void MultiTracker::onDetection(const cv::Mat& frame,
                                const std::vector<Detection>& detections) {
    auto matches = associateDetections(detections);

    // Track which detections and tracks are matched
    std::set<int> matchedDetIdx, matchedTrkIdx;
    for (auto& [dIdx, tIdx] : matches) {
        matchedDetIdx.insert(dIdx);
        matchedTrkIdx.insert(tIdx);
    }

    // Update matched tracks with fresh detections
    for (auto& [dIdx, tIdx] : matches) {
        tracks_[tIdx].lastDetection = detections[dIdx];
        tracks_[tIdx].predictedBox = detections[dIdx].bbox;
        tracks_[tIdx].trackConfidence = detections[dIdx].confidence;
        tracks_[tIdx].framesLost = 0;
        tracks_[tIdx].active = true;
        trackers_[tIdx].resetTemplate(frame, detections[dIdx].bbox);
    }

    // Create new tracks for unmatched detections
    for (int d = 0; d < static_cast<int>(detections.size()); ++d) {
        if (matchedDetIdx.count(d)) continue;

        Track newTrack;
        newTrack.id = nextTrackId_++;
        newTrack.lastDetection = detections[d];
        newTrack.predictedBox = detections[d].bbox;
        newTrack.trackConfidence = detections[d].confidence;
        newTrack.framesLost = 0;
        newTrack.active = true;

        NCCTracker tracker(cfg_.trackerCfg);
        tracker.init(frame, detections[d].bbox);

        tracks_.push_back(newTrack);
        trackers_.push_back(std::move(tracker));
    }

    // Increment framesLost for unmatched tracks
    for (int t = 0; t < static_cast<int>(tracks_.size()); ++t) {
        if (matchedTrkIdx.count(t)) continue;
        tracks_[t].framesLost++;
        if (tracks_[t].framesLost > cfg_.maxFramesLost) {
            tracks_[t].active = false;
        }
    }

    pruneInactiveTracks();
}

void MultiTracker::onTrackingFrame(const cv::Mat& frame) {
    for (size_t i = 0; i < tracks_.size(); ++i) {
        if (!tracks_[i].active) continue;

        cv::Rect2f newBox;
        float conf;
        bool success = trackers_[i].update(frame, newBox, conf);

        tracks_[i].predictedBox = newBox;
        tracks_[i].trackConfidence = conf;

        if (success) {
            tracks_[i].framesLost = 0;
        } else {
            tracks_[i].framesLost++;
            if (tracks_[i].framesLost > cfg_.maxFramesLost) {
                tracks_[i].active = false;
            }
        }
    }

    pruneInactiveTracks();
}

void MultiTracker::pruneInactiveTracks() {
    size_t writeIdx = 0;
    for (size_t i = 0; i < tracks_.size(); ++i) {
        if (tracks_[i].active) {
            if (writeIdx != i) {
                tracks_[writeIdx] = std::move(tracks_[i]);
                trackers_[writeIdx] = std::move(trackers_[i]);
            }
            writeIdx++;
        }
    }
    tracks_.resize(writeIdx);
    trackers_.resize(writeIdx);
}
