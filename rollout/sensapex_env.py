"""Thin ROS2 client used by the MicroVLA rollout script.

MicroVLA's state/action vector is one manipulator per 4 axes (x, y, z, d):

    1 manipulator -> [x, y, z, d]                       (/ump only)
    2 manipulators -> [x1, y1, z1, d1, x2, y2, z2, d2]  (/ump + /ump2)

``num_manipulators`` (from ``config.NUM_MANIPULATORS``) decides how many Sensapex
live topics this node subscribes to and how many absolute target commands it
publishes. The camera topic is always subscribed.
"""

from __future__ import annotations

import io
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, Float64, Int32MultiArray

from config import vla_config as C

_AXES = C.AXES_PER_MANIPULATOR

# Live pipette-resistance topic message types (selectable at rollout). The value
# is read as a scalar: msg.data for Float32/Float64, msg.data[0] for Int32MultiArray.
_RESISTANCE_MSG_TYPES = {"float32": Float32, "float64": Float64, "int_array": Int32MultiArray}


@dataclass
class SensapexObs:
    image_rgb: np.ndarray
    state: np.ndarray
    resistance: Optional[float] = None


def _decode_compressed_jpeg_to_rgb(msg: CompressedImage) -> np.ndarray:
    pil = PILImage.open(io.BytesIO(bytes(msg.data))).convert("RGB")
    return np.array(pil, dtype=np.uint8)


class _SensapexROSNode(Node):
    """Minimal subscriber/publisher node owned by `SensapexEnv`."""

    def __init__(
        self,
        *,
        num_manipulators: int = 1,
        save_preview: bool = True,
        preview_path: str = "microvla_live.png",
        preview_every_n_frames: int = 5,
        resistance_topic: Optional[str] = None,
        resistance_type: str = "float32",
    ):
        super().__init__("microvla_sensapex_bridge")
        self.num_manipulators = int(num_manipulators)
        self.dual = self.num_manipulators >= 2

        self.sub_img = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self._on_img, 10
        )
        self.sub_ump1_live = self.create_subscription(
            Int32MultiArray, "/ump/live", self._on_ump1_live, 10
        )
        self.pub_ump1_target = self.create_publisher(Int32MultiArray, "/ump/target", 10)
        self.sub_ump2_live = None
        self.pub_ump2_target = None
        if self.dual:
            self.sub_ump2_live = self.create_subscription(
                Int32MultiArray, "/ump2/live", self._on_ump2_live, 10
            )
            self.pub_ump2_target = self.create_publisher(Int32MultiArray, "/ump2/target", 10)

        # Optional live pipette resistance (patch clamp). Off unless a topic is given.
        self.sub_resistance = None
        self._resistance_is_array = (resistance_type == "int_array")
        if resistance_topic:
            msg_cls = _RESISTANCE_MSG_TYPES.get(resistance_type, Float32)
            self.sub_resistance = self.create_subscription(
                msg_cls, resistance_topic, self._on_resistance, 10
            )
            self.get_logger().info(f"subscribed to resistance topic {resistance_topic} ({resistance_type})")

        self._lock = threading.Lock()
        self._latest_image_rgb = None
        self._latest_ump1 = None
        self._latest_ump2 = None
        self._latest_resistance = None

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
                    # (or a killed process) never sees a half-written PNG.
                    root, ext = os.path.splitext(self._preview_path)
                    tmp_path = f"{root}.tmp{ext}"
                    PILImage.fromarray(rgb).save(tmp_path)
                    os.replace(tmp_path, self._preview_path)
                except Exception as e:
                    self.get_logger().warn(f"Preview save failed: {e}")

    def _on_ump1_live(self, msg: Int32MultiArray) -> None:
        if len(msg.data) < _AXES:
            return
        with self._lock:
            self._latest_ump1 = [int(v) for v in msg.data[:_AXES]]

    def _on_ump2_live(self, msg: Int32MultiArray) -> None:
        if len(msg.data) < _AXES:
            return
        with self._lock:
            self._latest_ump2 = [int(v) for v in msg.data[:_AXES]]

    def _on_resistance(self, msg) -> None:
        try:
            val = float(msg.data[0]) if self._resistance_is_array else float(msg.data)
        except (TypeError, IndexError, ValueError):
            return
        with self._lock:
            self._latest_resistance = val

    def get_latest(self):
        with self._lock:
            img = None if self._latest_image_rgb is None else self._latest_image_rgb.copy()
            ump1 = None if self._latest_ump1 is None else list(self._latest_ump1)
            ump2 = None if self._latest_ump2 is None else list(self._latest_ump2)
        return img, ump1, ump2

    def latest_resistance(self):
        with self._lock:
            return self._latest_resistance

    def send_action_absolute(self, values, speed: int = 100) -> None:
        """Publish absolute targets. ``values`` is a flat [x,y,z,d(,x2,..,d2)]."""
        values = [int(v) for v in np.asarray(values).reshape(-1)]
        ump1_msg = Int32MultiArray()
        ump1_msg.data = values[:_AXES] + [int(speed)]
        self.pub_ump1_target.publish(ump1_msg)
        if self.dual and self.pub_ump2_target is not None:
            ump2_msg = Int32MultiArray()
            ump2_msg.data = values[_AXES:2 * _AXES] + [int(speed)]
            self.pub_ump2_target.publish(ump2_msg)


class SensapexEnv:
    """Synchronous wrapper around `_SensapexROSNode`."""

    def __init__(
        self,
        *,
        num_manipulators: int = 1,
        save_preview: bool = True,
        preview_path: str = "microvla_live.png",
        preview_every_n_frames: int = 5,
        default_speed: int = 100,
        wait_timeout_s: float = 10.0,
        resistance_topic: Optional[str] = None,
        resistance_type: str = "float32",
    ):
        self.num_manipulators = int(num_manipulators)
        self.dual = self.num_manipulators >= 2
        self.default_speed = int(default_speed)

        rclpy.init(args=None)
        self.node = _SensapexROSNode(
            num_manipulators=self.num_manipulators,
            save_preview=save_preview,
            preview_path=preview_path,
            preview_every_n_frames=preview_every_n_frames,
            resistance_topic=resistance_topic,
            resistance_type=resistance_type,
        )
        self._executor_thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), daemon=True
        )
        self._executor_thread.start()

        self._wait_for_first_messages(timeout_s=wait_timeout_s)

    def _have_all(self):
        img, ump1, ump2 = self.node.get_latest()
        ready = img is not None and ump1 is not None and (ump2 is not None or not self.dual)
        return ready, (img, ump1, ump2)

    def _wait_for_first_messages(self, timeout_s: float = 10.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            ready, _ = self._have_all()
            if ready:
                return
            time.sleep(0.05)
        topics = "/camera/image/compressed, /ump/live" + (", /ump2/live" if self.dual else "")
        raise RuntimeError(f"Timed out waiting for {topics}")

    def get_observation(self) -> SensapexObs:
        ready, (img, ump1, ump2) = self._have_all()
        if not ready:
            raise RuntimeError("Missing observation components (image/ump live).")
        state = list(ump1) + (list(ump2) if self.dual else [])
        return SensapexObs(image_rgb=img, state=np.array(state, dtype=np.float32),
                           resistance=self.node.latest_resistance())

    def step_absolute(self, action: np.ndarray) -> None:
        """Send an absolute target [x,y,z,d(,x2,y2,z2,d2)]."""
        action = np.asarray(action).reshape(-1)
        expected = self.num_manipulators * _AXES
        if action.shape != (expected,):
            raise ValueError(f"Expected action shape ({expected},), got {action.shape}")
        self.node.send_action_absolute(action, speed=self.default_speed)

    def close(self) -> None:
        try:
            self.node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
