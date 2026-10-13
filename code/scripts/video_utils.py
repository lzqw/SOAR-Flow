"""Video helpers shared by the standalone JAX Safe-SAC recorder."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

GREEN = (70, 220, 100)
RED = (60, 70, 245)
CYAN = (245, 205, 45)
ORANGE = (50, 165, 255)
WHITE = (245, 245, 245)
DARK = (24, 27, 33)


class _OpenCVEncoderStdin:
    def __init__(self, writer, width: int, height: int):
        self.writer = writer
        self.width = int(width)
        self.height = int(height)
        self.closed = False

    def write(self, data: bytes) -> int:
        if self.closed:
            raise RuntimeError("OpenCV video encoder is closed")
        array = np.frombuffer(data, dtype=np.uint8)
        expected = self.width * self.height * 3
        if array.size != expected:
            raise ValueError(f"video frame has {array.size} bytes, expected {expected}")
        self.writer.write(array.reshape(self.height, self.width, 3))
        return len(data)

    def close(self) -> None:
        if not self.closed:
            self.writer.release()
            self.closed = True


class _OpenCVEncoder:
    """Minimal subprocess-compatible wrapper used when ffmpeg CLI is absent."""

    def __init__(self, output: Path, width: int, height: int, fps: int):
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (int(width), int(height)),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not initialize the MP4 encoder")
        self.stdin = _OpenCVEncoderStdin(writer, width, height)
        self.stderr = None

    def wait(self) -> int:
        self.stdin.close()
        return 0


def _start_encoder(
    output: Path,
    *,
    width: int,
    height: int,
    fps: int,
    crf: int,
):
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        return _OpenCVEncoder(output, width, height, fps)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vf",
        "setfield=prog",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _alpha_box(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int] = DARK,
    alpha: float = 0.82,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, thickness=-1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def _text(
    frame: np.ndarray,
    value: str,
    position: tuple[int, int],
    *,
    color: tuple[int, int, int] = WHITE,
    scale: float = 0.62,
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame,
        value,
        position,
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _format_number(value, precision: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:+.{precision}f}" if np.isfinite(number) else "n/a"


def _format_action(value) -> str:
    action = np.asarray(value, dtype=np.float32).reshape(-1)
    if action.size < 2 or not np.all(np.isfinite(action[:2])):
        return "[n/a, n/a]"
    return f"[{action[0]:+.3f}, {action[1]:+.3f}]"


def _screen_position(renderer, world_position, ego_position, shape):
    height, width = shape[:2]
    world_pixel = np.asarray(
        renderer._frame_canvas.pos2pix(*np.asarray(world_position)[:2]),
        dtype=np.float32,
    )
    ego_pixel = np.asarray(
        renderer._frame_canvas.pos2pix(*np.asarray(ego_position)[:2]),
        dtype=np.float32,
    )
    point = world_pixel - ego_pixel + np.array([width / 2.0, height / 2.0])
    return round(point[0]), round(point[1])


def _draw_vehicle_labels(frame: np.ndarray, base_env) -> None:
    renderer = base_env.engine.top_down_renderer
    if renderer is None:
        return
    ego_point = _screen_position(
        renderer,
        base_env.vehicle.position,
        base_env.vehicle.position,
        frame.shape,
    )
    cv2.circle(frame, ego_point, 36, CYAN, 2, cv2.LINE_AA)
    _text(
        frame,
        "EGO",
        (ego_point[0] - 24, ego_point[1] - 44),
        color=CYAN,
        scale=0.58,
        thickness=2,
    )
    for obstacle in list(getattr(base_env, "_parked_objs", [])):
        point = _screen_position(
            renderer,
            obstacle.position,
            base_env.vehicle.position,
            frame.shape,
        )
        if (
            -50 <= point[0] <= frame.shape[1] + 50
            and -50 <= point[1] <= frame.shape[0] + 50
        ):
            cv2.circle(frame, point, 36, RED, 2, cv2.LINE_AA)
            _text(
                frame,
                "STOPPED CAR",
                (point[0] - 66, point[1] - 44),
                color=RED,
                scale=0.52,
                thickness=2,
            )


def _write_frame(encoder, frame: np.ndarray) -> None:
    if encoder.stdin is None:
        raise RuntimeError("ffmpeg stdin is unavailable")
    encoder.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
