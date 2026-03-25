#include "ncc_tracker.h"
#include <algorithm>

NCCTracker::NCCTracker(const Config& cfg) : cfg_(cfg) {}

void NCCTracker::init(const cv::Mat& frame, const cv::Rect2f& bbox) {
    template_ = extractTemplate(frame, bbox);
    lastBox_ = bbox;
    initialized_ = !template_.empty();
}

void NCCTracker::resetTemplate(const cv::Mat& frame, const cv::Rect2f& bbox) {
    init(frame, bbox);
}

cv::Mat NCCTracker::extractTemplate(const cv::Mat& frame, const cv::Rect2f& bbox) {
    cv::Rect roi = clampRect(bbox, frame.size());
    if (roi.width <= 0 || roi.height <= 0) return {};

    cv::Mat patch = frame(roi);

    // Convert to grayscale
    cv::Mat gray;
    if (patch.channels() == 3) {
        cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = patch.clone();
    }

    // Resize if too large (preserving aspect ratio)
    int maxDim = std::max(gray.cols, gray.rows);
    if (maxDim > cfg_.maxTemplateSize) {
        float scale = static_cast<float>(cfg_.maxTemplateSize) / maxDim;
        cv::resize(gray, gray, cv::Size(), scale, scale);
    }

    return gray;
}

cv::Rect NCCTracker::computeSearchRegion(const cv::Size& frameSize) {
    float centerX = lastBox_.x + lastBox_.width / 2.0f;
    float centerY = lastBox_.y + lastBox_.height / 2.0f;

    float searchW = lastBox_.width * cfg_.searchRegionPadding;
    float searchH = lastBox_.height * cfg_.searchRegionPadding;

    cv::Rect2f searchRect(centerX - searchW / 2.0f,
                          centerY - searchH / 2.0f,
                          searchW, searchH);

    return clampRect(searchRect, frameSize);
}

bool NCCTracker::update(const cv::Mat& frame, cv::Rect2f& updatedBox, float& confidence) {
    if (!initialized_ || template_.empty()) {
        confidence = 0.0f;
        return false;
    }

    // Compute search region
    cv::Rect searchRegion = computeSearchRegion(frame.size());
    if (searchRegion.width <= 0 || searchRegion.height <= 0) {
        confidence = 0.0f;
        return false;
    }

    // Crop and convert to grayscale
    cv::Mat searchPatch = frame(searchRegion);
    cv::Mat searchGray;
    if (searchPatch.channels() == 3) {
        cv::cvtColor(searchPatch, searchGray, cv::COLOR_BGR2GRAY);
    } else {
        searchGray = searchPatch.clone();
    }

    // Resize search region proportionally if template was resized
    // The search region must be at least as large as the template
    if (searchGray.cols < template_.cols || searchGray.rows < template_.rows) {
        confidence = 0.0f;
        return false;
    }

    // Perform NCC template matching
    cv::Mat result;
    cv::matchTemplate(searchGray, template_, result, cv::TM_CCORR_NORMED);

    // Find best match
    double maxVal;
    cv::Point maxLoc;
    cv::minMaxLoc(result, nullptr, &maxVal, nullptr, &maxLoc);

    confidence = static_cast<float>(maxVal);

    // Compute matched box position in frame coordinates
    // The match location is relative to the search region
    // Template size may differ from lastBox_ size due to resizing, so we
    // keep the original box dimensions but update the position
    float matchScale = 1.0f;
    int maxDim = std::max(static_cast<int>(lastBox_.width), static_cast<int>(lastBox_.height));
    if (maxDim > cfg_.maxTemplateSize) {
        matchScale = static_cast<float>(maxDim) / cfg_.maxTemplateSize;
    }

    float matchX = searchRegion.x + maxLoc.x * matchScale;
    float matchY = searchRegion.y + maxLoc.y * matchScale;

    updatedBox = cv::Rect2f(matchX, matchY, lastBox_.width, lastBox_.height);

    // Clamp to frame
    updatedBox.x = std::max(0.0f, std::min(updatedBox.x, static_cast<float>(frame.cols) - updatedBox.width));
    updatedBox.y = std::max(0.0f, std::min(updatedBox.y, static_cast<float>(frame.rows) - updatedBox.height));

    lastBox_ = updatedBox;

    // Update template only when confidence is high (prevents drift)
    if (confidence >= cfg_.templateUpdateThresh) {
        template_ = extractTemplate(frame, updatedBox);
    }

    return confidence >= cfg_.minConfidence;
}

cv::Rect NCCTracker::clampRect(const cv::Rect2f& rect, const cv::Size& frameSize) {
    int x = std::max(0, static_cast<int>(rect.x));
    int y = std::max(0, static_cast<int>(rect.y));
    int w = static_cast<int>(rect.width);
    int h = static_cast<int>(rect.height);

    // Clamp to frame boundaries
    if (x + w > frameSize.width)  w = frameSize.width - x;
    if (y + h > frameSize.height) h = frameSize.height - y;

    if (w <= 0 || h <= 0) return cv::Rect(0, 0, 0, 0);
    return cv::Rect(x, y, w, h);
}
