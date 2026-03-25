#include "yolov9_detector.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <iostream>

YOLOv9Detector::YOLOv9Detector(const Config& cfg) : cfg_(cfg) {
    net_ = cv::dnn::readNetFromONNX(cfg_.modelPath);
    if (net_.empty()) {
        throw std::runtime_error("Failed to load ONNX model: " + cfg_.modelPath);
    }

    if (cfg_.useCUDA) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        std::cout << "[YOLOv9] Using CUDA backend\n";
    } else {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "[YOLOv9] Using CPU backend\n";
    }

    loadClassNames();
    std::cout << "[YOLOv9] Model loaded: " << cfg_.modelPath << "\n";
}

cv::Mat YOLOv9Detector::preprocess(const cv::Mat& frame, float& scaleX, float& scaleY,
                                    int& padLeft, int& padTop) {
    int origW = frame.cols;
    int origH = frame.rows;

    // Compute letterbox scale
    float scale = std::min(static_cast<float>(cfg_.inputWidth) / origW,
                           static_cast<float>(cfg_.inputHeight) / origH);

    int newW = static_cast<int>(origW * scale);
    int newH = static_cast<int>(origH * scale);

    // Padding to center the resized image
    padLeft = (cfg_.inputWidth - newW) / 2;
    padTop  = (cfg_.inputHeight - newH) / 2;

    // Resize
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(newW, newH));

    // Create letterbox canvas (gray fill)
    cv::Mat letterbox(cfg_.inputHeight, cfg_.inputWidth, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(letterbox(cv::Rect(padLeft, padTop, newW, newH)));

    // Scale factors for mapping back to original coordinates
    scaleX = scale;
    scaleY = scale;

    // Create blob: swap R/B, normalize to [0,1]
    cv::Mat blob = cv::dnn::blobFromImage(letterbox, 1.0 / 255.0,
                                           cv::Size(cfg_.inputWidth, cfg_.inputHeight),
                                           cv::Scalar(), true, false);
    return blob;
}

std::vector<Detection> YOLOv9Detector::detect(const cv::Mat& frame) {
    float scaleX, scaleY;
    int padLeft, padTop;

    cv::Mat blob = preprocess(frame, scaleX, scaleY, padLeft, padTop);
    net_.setInput(blob);

    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    // Use the first output
    cv::Mat output = outputs[0];

    return postprocess(output, scaleX, scaleY, padLeft, padTop, frame.cols, frame.rows);
}

std::vector<Detection> YOLOv9Detector::postprocess(const cv::Mat& output,
                                                     float scaleX, float scaleY,
                                                     int padLeft, int padTop,
                                                     int origWidth, int origHeight) {
    // Output shape can be [1, N, 84+] or [1, 84+, N] (transposed)
    // For standard YOLOv9: [1, N, 84] where 84 = 4 (box) + 80 (classes)
    // Some exports include objectness: [1, N, 85] = 4 + 1 + 80

    int dims = output.dims;
    int rows, cols;
    cv::Mat data;

    if (dims == 3) {
        int d1 = output.size[1];
        int d2 = output.size[2];

        // Reshape to 2D for easier processing
        cv::Mat mat2d = output.reshape(1, d1); // [d1 x d2]

        // Determine orientation: if d1 is small (84/85) and d2 is large, it's transposed
        if (d1 <= 85 && d2 > 85) {
            cv::transpose(mat2d, data);
            rows = d2;
            cols = d1;
        } else {
            data = mat2d;
            rows = d1;
            cols = d2;
        }
    } else {
        std::cerr << "[YOLOv9] Unexpected output dimensions: " << dims << "\n";
        return {};
    }

    bool hasObjectness = (cols == 85); // 4 + 1 + 80
    int numClasses = hasObjectness ? (cols - 5) : (cols - 4);

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;

    for (int i = 0; i < rows; ++i) {
        const float* row = data.ptr<float>(i);

        float cx = row[0];
        float cy = row[1];
        float w  = row[2];
        float h  = row[3];

        const float* classScores;
        float objConf = 1.0f;

        if (hasObjectness) {
            objConf = row[4];
            if (objConf < cfg_.confThreshold) continue;
            classScores = row + 5;
        } else {
            classScores = row + 4;
        }

        // Find best class
        int bestClassId = 0;
        float bestClassScore = classScores[0];
        for (int c = 1; c < numClasses; ++c) {
            if (classScores[c] > bestClassScore) {
                bestClassScore = classScores[c];
                bestClassId = c;
            }
        }

        float finalConf = objConf * bestClassScore;
        if (finalConf < cfg_.confThreshold) continue;

        // Convert from letterbox coords to original frame coords
        float x1 = (cx - w / 2.0f - padLeft) / scaleX;
        float y1 = (cy - h / 2.0f - padTop) / scaleY;
        float bw = w / scaleX;
        float bh = h / scaleY;

        // Clamp to frame boundaries
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(origWidth - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(origHeight - 1)));
        bw = std::min(bw, static_cast<float>(origWidth) - x1);
        bh = std::min(bh, static_cast<float>(origHeight) - y1);

        boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1),
                           static_cast<int>(bw), static_cast<int>(bh));
        confidences.push_back(finalConf);
        classIds.push_back(bestClassId);
    }

    // Apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, cfg_.confThreshold, cfg_.nmsThreshold, indices);

    std::vector<Detection> detections;
    detections.reserve(indices.size());

    for (int idx : indices) {
        Detection det;
        det.bbox = cv::Rect2f(static_cast<float>(boxes[idx].x),
                               static_cast<float>(boxes[idx].y),
                               static_cast<float>(boxes[idx].width),
                               static_cast<float>(boxes[idx].height));
        det.classId = classIds[idx];
        det.confidence = confidences[idx];
        det.label = (classIds[idx] >= 0 && classIds[idx] < static_cast<int>(classNames_.size()))
                    ? classNames_[classIds[idx]] : "unknown";
        detections.push_back(det);
    }

    return detections;
}

void YOLOv9Detector::loadClassNames() {
    classNames_ = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    };
}
