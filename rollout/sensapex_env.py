"""Thin ROS2 client used by the MicroACT rollout script.

MicroACT is trained on the 8-D Sensapex state/action vector:
    [x1, y1, z1, d1, x2, y2, z2, d2]

This is intentionally different from the OpenPI rollout this was adapted from,
which included a ninth ODrive motor dimension. This environment subscribes only
to the camera and two Sensapex live topics, and publishes only two Sensapex
absolute target commands.
"""

from __future__ import annotations

import io
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32MultiArray


@dataclass
class SensapexObs:
    image_rgb: np.ndarray
    state: np.ndarray


def _decode_compressed_jpeg_to_rgb(msg: CompressedImage) -> np.ndarray:
    pil = PILImage.open(io.BytesIO(bytes(msg.data))).convert("RGB")
    return np.array(pil, dtype=np.uint8)


class _SensapexROSNode(Node):
    """Minimal subscriber/publisher node owned by `SensapexEnv`."""

    def __init__(
        self,
        *,
        save_preview: bool = True,
        preview_path: str = "microact_live.png",
        preview_every_n_frames: int = 5,
    ):
        super().__init__("microact_sensapex_bridge")

        self.sub_img = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self._on_img, 10
        )
        self.sub_ump1_live = self.create_subscription(
            Int32MultiArray, "/ump/live", self._on_ump1_live, 10
        )
        self.sub_ump2_live = self.create_subscription(
            Int32MultiArray, "/ump2/live", self._on_ump2_live, 10
        )

        self.pub_ump1_target = self.create_publisher(Int32MultiArray, "/ump/target", 10)
        self.pub_ump2_target = self.create_publisher(Int32MultiArray, "/ump2/target", 10)

        self._lock = threading.Lock()
        self._latest_image_rgb = None
        self._latest_ump1 = None
        self._latest_ump2 = None

        self._save_preview = bool(save_preview)
        self._preview_path = str(preview_path)
        self._preview_every_n_frames = max(1, int(preview_every_n_frames))
        self._frame_counter = 0

    def _on_img(self, msg: CompressedImage) -> None:
        try:
            rgb = _decode_compressed_jpeg_to_rgb(msg)
        except Exception as e:
            self.get_logger().warn(f"Image decode failed: {e}")
            return

        with self._lock:
            self._latest_image_rgb = rgb

        if self._save_preview:
            self._frame_counter += 1
            if self._frame_counter % self._preview_every_n_frames == 0:
                try:
                    # Write to a temp file then atomically replace, so a reader
                    # (or a killed process) never sees a half-written PNG. Keep
                    # the real extension so PIL can infer the format.
                    root, ext = os.path.splitext(self._preview_path)
                    tmp_path = f"{root}.tmp{ext}"
                    PILImage.fromarray(rgb).save(tmp_path)
                    os.replace(tmp_path, self._preview_path)
                except Exception as e:
                    self.get_logger().warn(f"Preview save failed: {e}")

    def _on_ump1_live(self, msg: Int32MultiArray) -> None:
        if len(msg.data) < 4:
            return
        with self._lock:
            self._latest_ump1 = [int(v) for v in msg.data[:4]]

    def _on_ump2_live(self, msg: Int32MultiArray) -> None:
        if len(msg.data) < 4:
            return
        with self._lock:
            self._latest_ump2 = [int(v) for v in msg.data[:4]]

    def get_latest(self):
        with self._lock:
            img = None if self._latest_image_rgb is None else self._latest_image_rgb.copy()
            ump1 = None if self._latest_ump1 is None else list(self._latest_ump1)
            ump2 = None if self._latest_ump2 is None else list(self._latest_ump2)
        return img, ump1, ump2

    def send_action_absolute(
        self,
        x1,
        y1,
        z1,
        d1,
        x2,
        y2,
        z2,
        d2,
        speed=100,
    ) -> None:
        ump1_msg = Int32MultiArray()
        ump1_msg.data = [int(x1), int(y1), int(z1), int(d1), int(speed)]
        self.pub_ump1_target.publish(ump1_msg)

        ump2_msg = Int32MultiArray()
        ump2_msg.data = [int(x2), int(y2), int(z2), int(d2), int(speed)]
        self.pub_ump2_target.publish(ump2_msg)


class SensapexEnv:
    """Synchronous wrapper around `_SensapexROSNode`."""

    def __init__(
        self,
        *,
        save_preview: bool = True,
        preview_path: str = "microact_live.png",
        preview_every_n_frames: int = 5,
        default_speed: int = 100,
        wait_timeout_s: float = 10.0,
    ):
        self.default_speed = int(default_speed)

        rclpy.init(args=None)
        self.node = _SensapexROSNode(
            save_preview=save_preview,
            preview_path=preview_path,
            preview_every_n_frames=preview_every_n_frames,
        )
        self._executor_thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), daemon=True
        )
        self._executor_thread.start()

        self._wait_for_first_messages(timeout_s=wait_timeout_s)

    def _wait_for_first_messages(self, timeout_s: float = 10.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            img, ump1, ump2 = self.node.get_latest()
            if img is not None and ump1 is not None and ump2 is not None:
                return
            time.sleep(0.05)
        raise RuntimeError(
            "Timed out waiting for /camera/image/compressed, /ump/live, /ump2/live"
        )

    def get_observation(self) -> SensapexObs:
        img, ump1, ump2 = self.node.get_latest()
        if img is None or ump1 is None or ump2 is None:
            raise RuntimeError("Missing observation components (image/ump1/ump2).")

        x1, y1, z1, d1 = ump1
        x2, y2, z2, d2 = ump2
        state = np.array([x1, y1, z1, d1, x2, y2, z2, d2], dtype=np.float32)
        return SensapexObs(image_rgb=img, state=state)

    def step_absolute(self, action_8d: np.ndarray) -> None:
        """Send an absolute target [x1,y1,z1,d1,x2,y2,z2,d2]."""
        action_8d = np.asarray(action_8d).reshape(-1)
        if action_8d.shape != (8,):
            raise ValueError(f"Expected action shape (8,), got {action_8d.shape}")

        x1, y1, z1, d1, x2, y2, z2, d2 = action_8d
        self.node.send_action_absolute(
            x1, y1, z1, d1, x2, y2, z2, d2, speed=self.default_speed
        )

    def close(self) -> None:
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

