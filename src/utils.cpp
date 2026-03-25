#include "utils.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <sstream>
#include <iomanip>

float computeIoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);

    float interW = std::max(0.0f, x2 - x1);
    float interH = std::max(0.0f, y2 - y1);
    float interArea = interW * interH;

    float areaA = a.width * a.height;
    float areaB = b.width * b.height;
    float unionArea = areaA + areaB - interArea;

    if (unionArea <= 0.0f) return 0.0f;
    return interArea / unionArea;
}

cv::Scalar colorFromId(int id) {
    // Generate distinct colors by distributing hues evenly
    int hue = (id * 47) % 180; // 47 is coprime to 180 for good distribution
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 220, 220));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    auto pixel = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(pixel[0], pixel[1], pixel[2]);
}

void drawTracks(cv::Mat& frame, const std::vector<Track>& tracks) {
    for (const auto& track : tracks) {
        if (!track.active) continue;

        cv::Scalar color = colorFromId(track.id);
        cv::Rect box(static_cast<int>(track.predictedBox.x),
                     static_cast<int>(track.predictedBox.y),
                     static_cast<int>(track.predictedBox.width),
                     static_cast<int>(track.predictedBox.height));

        cv::rectangle(frame, box, color, 2);

        // Build label text: "label #id conf"
        std::ostringstream oss;
        oss << track.lastDetection.label
            << " #" << track.id
            << " " << std::fixed << std::setprecision(2) << track.trackConfidence;
        std::string label = oss.str();

        // Draw label background
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                             0.5, 1, &baseline);
        int labelY = std::max(box.y, textSize.height + 4);
        cv::rectangle(frame,
                      cv::Point(box.x, labelY - textSize.height - 4),
                      cv::Point(box.x + textSize.width + 4, labelY),
                      color, cv::FILLED);

        // Draw label text
        cv::putText(frame, label,
                    cv::Point(box.x + 2, labelY - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);
    }
}
