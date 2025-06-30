# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Provides the OpenCVCamera class for capturing frames from cameras using OpenCV.
"""

import logging
import math
import platform
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Dict, List

import cv2
import numpy as np
import pyzed.sl as sl

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from .configuration_zed import ZedCameraConfig

logger = logging.getLogger(__name__)


def zed_status_to_str(status: sl.ERROR_CODE) -> str:
    return f"<{status.name} ({status.value})>: {status}"


class ZedCamera(Camera):
    def __init__(self, config: ZedCameraConfig):
        """
        Initializes the OpenCVCamera instance.

        Args:
            config: The configuration settings for the camera.
        """
        super().__init__(config)

        self.config = config
        self.camera_serial = config.camera_serial

        self.fps = config.fps if config.fps else 15

        self.camera = sl.Camera()
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.new_frame_event: Event = Event()

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height

        self.init_params = sl.InitParameters()
        self.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080
        self.init_params.camera_fps = self.fps
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.camera_disable_self_calib = True

        self.runtime_parameters = sl.RuntimeParameters()
        self.runtime_parameters.enable_depth = False

        self.color_image = sl.Mat()

        self._is_connected = False

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.camera_serial})"

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, warmup: bool = True) -> None:
        open_status = self.camera.open(self.init_params)
        if open_status != sl.ERROR_CODE.SUCCESS:
            raise DeviceNotConnectedError(f"Failed to open ZED camera: {zed_status_to_str(open_status)}")

    @staticmethod
    def find_cameras() -> List[Dict[str, Any]]:
        cameras = sl.Camera.get_device_list()
        infos = []
        for camera in cameras:
            info = {
                "type": "ZED",
                "id": camera.serial_number,
                "camera_model": camera.camera_model.name,
                "camera_state": camera.camera_state.name,
                "camera_index": camera.id,
                "input_type": camera.input_type.name,
                "path": camera.path,
            }
            infos.append(info)
        return infos

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        status = self.camera.grab(self.runtime_parameters)
        if status != sl.ERROR_CODE.SUCCESS:
            raise DeviceNotConnectedError(
                f"Failed to grab frame from ZED camera: {zed_status_to_str(status)}"
            )
        self.camera.retrieve_image(self.color_image, sl.VIEW.LEFT)
        color_image_data = self.color_image.get_data()[..., 0:3]
        if color_mode is None or color_mode == ColorMode.RGB:
            return cv2.cvtColor(color_image_data, cv2.COLOR_BGR2RGB)
        elif color_mode == ColorMode.BGR:
            return color_image_data
        else:
            raise ValueError(f"Unsupported color mode: {color_mode}")

    def async_read(self, timeout_ms: float = 0.1) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join(timeout=0.1)
            self.thread = None
        self.camera.close()

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _read_loop(self):
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration:
        1. Reads a color frame with 500ms timeout
        2. Stores result in latest_frame (thread-safe)
        3. Sets new_frame_event to notify listeners

        Stops on DeviceNotConnectedError, logs other errors and continues.
        """
        while not self.stop_event.is_set():
            try:
                color_image = self.read()

                with self.frame_lock:
                    self.latest_frame = color_image
                self.new_frame_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")
