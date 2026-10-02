# Real-Time Fire Detection and Telegram Alert System using YOLOv8

An individual Embedded Software Design course project that detects fire from a live webcam stream using a custom-trained YOLOv8s model and sends Telegram alerts when fire is confirmed.

## Overview

The system processes frames from a laptop webcam, runs real-time fire detection with a custom YOLOv8s model, confirms detections across consecutive frames to reduce one-frame false alarms, and sends an alert image through Telegram.

Current prototype pipeline:

`Webcam -> YOLOv8s inference -> multi-frame fire confirmation -> bounding-box display -> Telegram alert`

## Project Highlights

- Custom YOLOv8s fire-detection model
- Real-time webcam inference with OpenCV
- Multi-frame confirmation to reduce transient false positives
- Telegram alert with the detected frame
- Configurable confidence threshold and camera settings
- Asynchronous Telegram sending so alerts do not block the detection loop

## Dataset and Training

The initial fire-detection dataset was sourced from Roboflow and was extended with additional real-world samples collected by the author.

Additional data included:
- self-collected fire images
- red-light images used as hard-negative samples to reduce false positives from visually similar light sources

The model was trained using Google Colab.

> Note: The full training dataset is not included in this repository. Add the original Roboflow dataset URL here if you want to make the data source directly traceable.

## Current Deployment

The current prototype has been tested on a **laptop webcam** with inference running on the computer.

This repository does **not** claim an Edge AI deployment yet. A future direction is to optimize and deploy the model on resource-constrained edge hardware.

## Project Structure

```text
real-time-fire-detection-yolov8/
├── main.py
├── fire_detector.py
├── telegram_alert.py
├── requirements.txt
├── .gitignore
└── models/
    └── fire.pt
```

## Installation

Python 3.13 is recommended for the current tested environment.

```bash
python -m pip install -r requirements.txt
```

## Run Without Telegram

```bash
py -3.13 main.py --no-telegram
```

or:

```bash
python main.py --no-telegram
```

## Telegram Setup

Create a Telegram bot using BotFather, then set the following environment variables before running the application.

### PowerShell

```powershell
$env:TG_TOKEN="YOUR_BOT_TOKEN"
$env:TG_CHAT_ID="YOUR_CHAT_ID"
```

Run:

```powershell
py -3.13 main.py
```

Do **not** hard-code or commit your Telegram token.

## Useful Options

Lower the confidence threshold:

```bash
py -3.13 main.py --no-telegram --confidence 0.4
```

Use a different camera:

```bash
py -3.13 main.py --camera 1
```

Use a CUDA device when available:

```bash
py -3.13 main.py --device 0
```

## Future Work

- Evaluate precision, recall, mAP, false positives, and inference latency systematically
- Improve robustness under difficult lighting and visually similar red/orange objects
- Compare smaller YOLO variants for faster inference
- Apply quantization or model compression
- Deploy the model on an edge platform
- Benchmark latency, memory usage, and power consumption after edge deployment

## Disclaimer

This is an academic prototype for learning and research. It is **not a certified fire-safety system** and should not be used as a substitute for approved fire-detection equipment.
