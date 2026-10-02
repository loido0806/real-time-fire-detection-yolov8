"""Asynchronous Telegram alerts for the fire-detection application."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import requests


LOGGER = logging.getLogger(__name__)


@dataclass
class Alert:
    """Data placed in the background Telegram worker queue."""

    message: str
    frame: Optional[np.ndarray]


class TelegramAlert:
    """
    Send fire alerts to Telegram without blocking the camera/inference loop.

    Credentials are read from:
        TG_TOKEN
        TG_CHAT_ID

    They can also be passed directly to the constructor, but environment
    variables are safer for a public GitHub repository.
    """

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        cooldown_seconds: float = 30.0,
        max_retries: int = 3,
        request_timeout: float = 10.0,
        enabled: bool = True,
    ) -> None:
        self.token = token or os.getenv("TG_TOKEN")
        self.chat_id = chat_id or os.getenv("TG_CHAT_ID")

        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.max_retries = max(1, int(max_retries))
        self.request_timeout = max(1.0, float(request_timeout))

        self.enabled = bool(enabled and self.token and self.chat_id)

        self._queue: queue.Queue[Alert | None] = queue.Queue(maxsize=10)
        self._stop_event = threading.Event()
        self._last_sent_at = 0.0
        self._worker: threading.Thread | None = None

        if enabled and not self.enabled:
            LOGGER.warning(
                "Telegram alerts are disabled because TG_TOKEN or TG_CHAT_ID "
                "is missing."
            )

        if self.enabled:
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="telegram-alert-worker",
                daemon=True,
            )
            self._worker.start()
            LOGGER.info("Telegram alert worker started.")

    def send_fire_alert(
        self,
        frame: np.ndarray | None,
        message: str = "🔥 FIRE ALERT: Fire detected!",
    ) -> bool:
        """
        Queue a Telegram alert.

        Returns False when Telegram is disabled or the queue is full.
        """
        if not self.enabled:
            return False

        alert = Alert(
            message=message,
            frame=frame.copy() if frame is not None else None,
        )

        try:
            self._queue.put_nowait(alert)
            return True
        except queue.Full:
            LOGGER.warning("Telegram alert queue is full; alert was dropped.")
            return False

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                alert = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                if alert is None:
                    return

                now = time.monotonic()
                if now - self._last_sent_at < self.cooldown_seconds:
                    LOGGER.info("Telegram alert skipped due to cooldown.")
                    continue

                if self._send(alert):
                    self._last_sent_at = time.monotonic()
            finally:
                self._queue.task_done()

    def _send(self, alert: Alert) -> bool:
        send_message_url = (
            f"https://api.telegram.org/bot{self.token}/sendMessage"
        )
        send_photo_url = f"https://api.telegram.org/bot{self.token}/sendPhoto"

        for attempt in range(1, self.max_retries + 1):
            try:
                if alert.frame is not None:
                    ok, encoded_image = cv2.imencode(".jpg", alert.frame)
                    if not ok:
                        raise RuntimeError("Could not encode alert frame as JPEG.")

                    response = requests.post(
                        send_photo_url,
                        data={
                            "chat_id": self.chat_id,
                            "caption": alert.message,
                        },
                        files={
                            "photo": (
                                "fire_alert.jpg",
                                encoded_image.tobytes(),
                                "image/jpeg",
                            )
                        },
                        timeout=self.request_timeout,
                    )
                else:
                    response = requests.post(
                        send_message_url,
                        data={
                            "chat_id": self.chat_id,
                            "text": alert.message,
                        },
                        timeout=self.request_timeout,
                    )

                response.raise_for_status()
                LOGGER.info("Telegram alert sent successfully.")
                return True

            except (requests.RequestException, RuntimeError) as exc:
                LOGGER.warning(
                    "Telegram alert attempt %d/%d failed: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 4))

        LOGGER.error("Telegram alert failed after all retries.")
        return False

    def shutdown(self) -> None:
        """Stop the background worker cleanly."""
        if self._worker is None:
            return

        self._stop_event.set()

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        self._worker.join(timeout=2.0)
        LOGGER.info("Telegram alert worker stopped.")

    def __enter__(self) -> "TelegramAlert":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.shutdown()
