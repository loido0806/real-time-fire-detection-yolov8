"""Fire detection module based on a custom YOLOv8 model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import logging

import numpy as np
from ultralytics import YOLO


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """One object detection returned by the YOLO model."""

    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


class FireDetector:
    """Load a trained YOLO model and run fire detection on image frames."""

    def __init__(
        self,
        model_path: str | Path = "models/fire.pt",
        confidence: float = 0.50,
        device: str = "cpu",
    ) -> None:
        if not 0.0 < confidence <= 1.0:
            raise ValueError("confidence must be in the range (0, 1].")

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path.resolve()}"
            )

        self.confidence = confidence
        self.device = device

        try:
            self.model = YOLO(str(self.model_path))
        except Exception as exc:
            raise RuntimeError(
                f"Could not load YOLO model from {self.model_path}"
            ) from exc

        LOGGER.info(
            "Loaded fire-detection model: %s | device=%s | confidence=%.2f",
            self.model_path,
            self.device,
            self.confidence,
        )

    @staticmethod
    def _class_name(names: object, class_id: int) -> str:
        """Resolve a YOLO class id to a readable class name."""
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))

        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])

        return str(class_id)

    def detect(
        self, frame: np.ndarray
    ) -> tuple[bool, np.ndarray, List[Detection]]:
        """
        Detect fire in one BGR frame.

        Returns:
            detected:
                True when at least one object is detected.
            annotated_frame:
                Copy of the frame with YOLO bounding boxes and labels.
            detections:
                Structured information for every detection.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty.")

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            LOGGER.exception("YOLO inference failed.")
            raise RuntimeError("YOLO inference failed.") from exc

        if not results:
            return False, frame.copy(), []

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return False, frame.copy(), []

        detections: List[Detection] = []

        xyxy_values = boxes.xyxy.cpu().tolist()
        confidence_values = boxes.conf.cpu().tolist()
        class_values = boxes.cls.cpu().tolist()

        for xyxy, confidence, class_id_raw in zip(
            xyxy_values, confidence_values, class_values
        ):
            x1, y1, x2, y2 = (int(value) for value in xyxy[:4])
            class_id = int(class_id_raw)

            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(confidence),
                    class_id=class_id,
                    class_name=self._class_name(result.names, class_id),
                )
            )

        annotated_frame = result.plot()
        return True, annotated_frame, detections
