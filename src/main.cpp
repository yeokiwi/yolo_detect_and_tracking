#include "yolov9_detector.h"
#include "multi_tracker.h"
#include "utils.h"
#include "app_config.h"
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n"
              << "Options:\n"
              << "  --video <path>       Input video file (default: input.mp4)\n"
              << "  --model <path>       YOLOv9 ONNX model (default: models/yolov9.onnx)\n"
              << "  --interval <N>       Run detection every N frames (default: 30)\n"
              << "  --conf <threshold>   Confidence threshold (default: 0.25)\n"
              << "  --nms <threshold>    NMS threshold (default: 0.45)\n"
              << "  --cuda               Enable CUDA backend\n"
              << "  --help               Show this message\n";
}

AppConfig parseArgs(int argc, char** argv) {
    AppConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--video" && i + 1 < argc) {
            cfg.videoPath = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            cfg.modelPath = argv[++i];
        } else if (arg == "--interval" && i + 1 < argc) {
            cfg.detectInterval = std::stoi(argv[++i]);
        } else if (arg == "--conf" && i + 1 < argc) {
            cfg.confThreshold = std::stof(argv[++i]);
        } else if (arg == "--nms" && i + 1 < argc) {
            cfg.nmsThreshold = std::stof(argv[++i]);
        } else if (arg == "--cuda") {
            cfg.useCUDA = true;
        } else if (arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage(argv[0]);
            std::exit(1);
        }
    }

    return cfg;
}

int main(int argc, char** argv) {
    AppConfig appCfg = parseArgs(argc, argv);

    // Configure detector
    YOLOv9Detector::Config detCfg;
    detCfg.modelPath = appCfg.modelPath;
    detCfg.confThreshold = appCfg.confThreshold;
    detCfg.nmsThreshold = appCfg.nmsThreshold;
    detCfg.inputWidth = appCfg.inputSize;
    detCfg.inputHeight = appCfg.inputSize;
    detCfg.useCUDA = appCfg.useCUDA;

    std::cout << "Loading YOLOv9 model...\n";
    YOLOv9Detector detector(detCfg);

    // Configure multi-tracker
    MultiTracker::Config trkCfg;
    trkCfg.iouThreshold = appCfg.iouThreshold;
    trkCfg.maxFramesLost = appCfg.maxFramesLost;
    trkCfg.trackerCfg.searchRegionPadding = appCfg.searchPadding;
    trkCfg.trackerCfg.minConfidence = appCfg.nccMinConfidence;
    trkCfg.trackerCfg.maxTemplateSize = appCfg.maxTemplateSize;
    MultiTracker tracker(trkCfg);

    // Open video
    cv::VideoCapture cap(appCfg.videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file: " << appCfg.videoPath << "\n";
        return 1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "Video: " << appCfg.videoPath
              << " (" << totalFrames << " frames, " << fps << " FPS)\n";
    std::cout << "Detection interval: every " << appCfg.detectInterval << " frames\n";
    std::cout << "Press ESC to quit, SPACE to pause/resume\n\n";

    cv::Mat frame;
    int frameCount = 0;
    bool paused = false;

    while (true) {
        if (!paused) {
            if (!cap.read(frame)) break;

            bool isDetectionFrame = (frameCount % appCfg.detectInterval == 0);

            if (isDetectionFrame) {
                auto detections = detector.detect(frame);
                tracker.onDetection(frame, detections);
                std::cout << "\rFrame " << frameCount
                          << ": Detected " << detections.size() << " objects, "
                          << "Tracking " << tracker.getTracks().size() << " tracks"
                          << std::flush;
            } else {
                tracker.onTrackingFrame(frame);
            }

            // Draw visualization
            cv::Mat display = frame.clone();
            drawTracks(display, tracker.getTracks());

            // Show frame counter
            std::string frameText = "Frame: " + std::to_string(frameCount) + "/" + std::to_string(totalFrames);
            cv::putText(display, frameText, cv::Point(10, 25),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

            cv::imshow("YOLOv9 + NCC Tracking", display);
            frameCount++;
        }

        int key = cv::waitKey(paused ? 30 : 1);
        if (key == 27) break;         // ESC
        if (key == 32) paused = !paused; // SPACE
    }

    std::cout << "\nProcessed " << frameCount << " frames.\n";
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
