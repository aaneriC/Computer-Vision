"""
Intel RealSense D435i — RGB Video Stream
Requires:
    pip install pyrealsense2 opencv-python numpy
"""

import pyrealsense2 as rs
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

def create_pipeline() -> tuple[rs.pipeline, rs.config]:
    """Configure and return a RealSense pipeline set up for RGB streaming."""
    pipeline = rs.pipeline()
    config = rs.config()

    # ── Device discovery ────────────────────────────────────────────────────
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("[ERROR] No RealSense device detected. Check USB connection.")
        sys.exit(1)

    device = devices[0]
    print(f"[INFO] Connected to: {device.get_info(rs.camera_info.name)}")
    print(f"[INFO] Serial number : {device.get_info(rs.camera_info.serial_number)}")
    print(f"[INFO] Firmware      : {device.get_info(rs.camera_info.firmware_version)}")

    # ── Stream configuration ─────────────────────────────────────────────────
    # D435i colour sensor supports up to 1920×1080 @ 30 fps
    config.enable_stream(
        rs.stream.color,
        1280, 720,   # resolution  (try 1920×1080 if you want full HD)
        rs.format.bgr8,
        30           # fps
    )

    return pipeline, config


def run_stream() -> None:
    """Start the pipeline and display the RGB feed until 'q' is pressed."""
    pipeline, config = create_pipeline()

    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        print(f"[ERROR] Could not start pipeline: {exc}")
        sys.exit(1)

    # ── Optional: auto-exposure warmup ──────────────────────────────────────
    print("[INFO] Warming up camera (30 frames) …")
    for _ in range(30):
        pipeline.wait_for_frames()

    print("[INFO] Streaming — press  q  to quit,  s  to save a snapshot.")

    frame_count = 0
    snapshot_count = 0

    try:
        while True:
            # Block until a coherent frame set arrives (timeout = 5 s)
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()

            if not color_frame:
                print("[WARN] Dropped frame, retrying …")
                continue

            # Convert to NumPy array — already BGR thanks to rs.format.bgr8
            color_image = np.asanyarray(color_frame.get_data())
            frame_count += 1

            # ── HUD overlay ─────────────────────────────────────────────────
            h, w = color_image.shape[:2]
            overlay = color_image.copy()

            cv2.putText(
                overlay,
                f"Intel RealSense D435i  |  {w}x{h} @ 30fps",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"Frame: {frame_count:06d}   [q] quit  [s] snapshot",
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow("RealSense D435i — RGB Stream", overlay)

            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quit requested.")
                break
            elif key == ord("s"):
                fname = f"snapshot_{snapshot_count:04d}.png"
                cv2.imwrite(fname, color_image)
                snapshot_count += 1
                print(f"[INFO] Snapshot saved → {fname}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"[INFO] Stream ended after {frame_count} frames.")


if __name__ == "__main__":
    run_stream()