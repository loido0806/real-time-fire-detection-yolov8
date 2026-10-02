"""Run real-time fire detection from a webcam and send Telegram alerts."""

from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2

from fire_detector import FireDetector
from telegram_alert import TelegramAlert


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Real-time fire detection using a custom YOLOv8 model "
            "with optional Telegram alerts."
        )
    )

    parser.add_argument(
        "--model",
        default="models/fire.pt",
        help="Path to the trained YOLO model. Default: models/fire.pt",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index. Default: 0",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.50,
        help="YOLO confidence threshold. Default: 0.50",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Inference device, for example "cpu", "0", or "cuda:0". Default: cpu',
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Requested camera width. Default: 640",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Requested camera height. Default: 480",
    )
    parser.add_argument(
        "--required-frames",
        type=int,
        default=3,
        help=(
            "Number of consecutive positive frames required to confirm fire. "
            "Default: 3"
        ),
    )
    parser.add_argument(
        "--clear-frames",
        type=int,
        default=3,
        help=(
            "Number of consecutive negative frames required to clear an alarm. "
            "Default: 3"
        ),
    )
    parser.add_argument(
        "--alert-cooldown",
        type=float,
        default=30.0,
        help="Minimum seconds between Telegram alerts. Default: 30",
    )
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Disable Telegram alerts.",
    )

    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera index {index}.")

    return camera


def main() -> int:
    args = parse_args()
    configure_logging()

    if args.required_frames < 1:
        raise ValueError("--required-frames must be at least 1.")

    if args.clear_frames < 1:
        raise ValueError("--clear-frames must be at least 1.")

    detector = FireDetector(
        model_path=args.model,
        confidence=args.confidence,
        device=args.device,
    )

    camera = open_camera(args.camera, args.width, args.height)

    telegram = TelegramAlert(
        cooldown_seconds=args.alert_cooldown,
        enabled=not args.no_telegram,
    )

    positive_frames = 0
    negative_frames = 0
    alarm_active = False

    previous_time = time.perf_counter()
    fps = 0.0

    LOGGER.info("Fire-detection system started. Press Q to quit.")

    try:
        while True:
            ok, frame = camera.read()

            if not ok or frame is None:
                LOGGER.warning("Could not read a frame from the camera.")
                time.sleep(0.05)
                continue

            detected, annotated_frame, detections = detector.detect(frame)

            if detected:
                positive_frames += 1
                negative_frames = 0
            else:
                positive_frames = 0
                negative_frames += 1

            # Confirm fire only after several consecutive positive frames.
            if not alarm_active and positive_frames >= args.required_frames:
                alarm_active = True

                best_confidence = max(
                    (d.confidence for d in detections),
                    default=0.0,
                )

                message = (
                    "🔥 FIRE ALERT\n"
                    f"Fire detected by YOLOv8 | confidence={best_confidence:.2f}"
                )

                telegram.send_fire_alert(
                    frame=annotated_frame,
                    message=message,
                )

                LOGGER.warning(
                    "Fire confirmed | best confidence=%.2f",
                    best_confidence,
                )

            # Clear the alarm only after several consecutive negative frames.
            if alarm_active and negative_frames >= args.clear_frames:
                alarm_active = False
                LOGGER.info("Fire alarm cleared.")

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time

            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0.0 else (0.9 * fps + 0.1 * instant_fps)

            state_text = "FIRE" if alarm_active else "NORMAL"

            cv2.putText(
                annotated_frame,
                f"State: {state_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255) if alarm_active else (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("YOLOv8 Fire Detection", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # Q or ESC
                break

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")

    finally:
        LOGGER.info("Shutting down...")
        camera.release()
        telegram.shutdown()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        logging.exception("Application terminated because of an error: %s", exc)
        raise
