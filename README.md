# YOLOv9 Detection + NCC Tracking

A C++ Windows application that performs real-time object detection using YOLOv9 and object tracking using Normalized Cross-Correlation (NCC) template matching. The application reads video files via the OpenCV API.

## How It Works

The application uses a **detect-and-track** pipeline:

1. **Detection frames** (every N frames, default 30): YOLOv9 runs inference on the frame via OpenCV's DNN module with an ONNX model. Detected objects are associated with existing tracks using IoU (Intersection over Union) matching. New tracks are created for unmatched detections, and stale tracks are removed.

2. **Tracking frames** (all other frames): Each tracked object is updated using NCC template matching (`cv::matchTemplate` with `TM_CCORR_NORMED`). The tracker searches an expanded region around the last known position, finds the best match, and updates the bounding box. Templates are only updated when match confidence is high to prevent drift.

This approach balances accuracy (YOLOv9 detection) with speed (lightweight NCC tracking between detections).

### Architecture

```
src/main.cpp              → Pipeline orchestration, CLI argument parsing
src/yolov9_detector.cpp   → YOLOv9 ONNX inference (letterbox, NMS, coord mapping)
src/ncc_tracker.cpp        → Single-object NCC template matching tracker
src/multi_tracker.cpp      → Multi-object management with IoU association
src/utils.cpp              → IoU computation, bounding box drawing, color generation
```

## Requirements

- **CMake** 3.15 or later
- **OpenCV** 4.8+ with the DNN module (core, imgproc, highgui, videoio, dnn)
- **C++17** compatible compiler (MSVC 2019+, MinGW-w64, or GCC 7+)
- **YOLOv9 ONNX model** (exported for 640x640 input with COCO 80-class output)

### Optional

- OpenCV built with CUDA support for GPU-accelerated inference

## Installation

### 1. Install OpenCV

**Option A — Pre-built binaries (Windows):**

Download OpenCV from [opencv.org/releases](https://opencv.org/releases/) and extract it. Set the `OpenCV_DIR` environment variable to the build directory:

```cmd
set OpenCV_DIR=C:\opencv\build
```

**Option B — Build from source:**

```bash
git clone https://github.com/opencv/opencv.git
cd opencv && mkdir build && cd build
cmake .. -DBUILD_opencv_dnn=ON -DWITH_CUDA=OFF
cmake --build . --config Release --target install
```

### 2. Get a YOLOv9 ONNX Model

Export a YOLOv9 model to ONNX format (640x640 input) and place it in the `models/` directory:

```
models/yolov9.onnx
```

You can export from the official YOLOv9 repository using:

```bash
python export.py --weights yolov9-c.pt --include onnx --imgsz 640
```

### 3. Build the Application

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

On Windows with Visual Studio:

```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

The executable `YoloDetectAndTracking` (or `YoloDetectAndTracking.exe` on Windows) will be generated in the build output directory. On Windows, OpenCV DLLs are automatically copied next to the executable.

## Usage

```
YoloDetectAndTracking [options]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--video <path>` | Input video file | `input.mp4` |
| `--model <path>` | YOLOv9 ONNX model file | `models/yolov9.onnx` |
| `--interval <N>` | Run detection every N frames | `30` |
| `--conf <float>` | Detection confidence threshold | `0.25` |
| `--nms <float>` | NMS IoU threshold | `0.45` |
| `--cuda` | Enable CUDA backend for inference | disabled |
| `--help` | Show help message | |

### Examples

Basic usage:

```bash
YoloDetectAndTracking --video traffic.mp4 --model models/yolov9.onnx
```

With custom thresholds and faster detection interval:

```bash
YoloDetectAndTracking --video traffic.mp4 --model models/yolov9.onnx --interval 15 --conf 0.4
```

With CUDA acceleration:

```bash
YoloDetectAndTracking --video traffic.mp4 --model models/yolov9.onnx --cuda
```

### Controls

- **ESC** — Quit the application
- **SPACE** — Pause / Resume playback

## Configuration Defaults

These defaults are defined in `config/app_config.h` and can be overridden via command-line arguments:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Confidence threshold | 0.25 | Minimum detection confidence |
| NMS threshold | 0.45 | Non-maximum suppression IoU threshold |
| Input size | 640x640 | YOLOv9 model input resolution |
| Detection interval | 30 | Frames between full detections |
| Search padding | 2.0x | NCC search region multiplier around last known position |
| NCC min confidence | 0.5 | Below this, a track is considered lost |
| Template update threshold | 0.7 | Only update template when confidence exceeds this |
| IoU threshold | 0.3 | Minimum IoU to associate a detection with an existing track |
| Max frames lost | 15 | Remove a track after this many consecutive lost frames |

## License

This project is provided as-is for educational and research purposes.
