import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

#developed from claude's code, but modularized and improved for error handling, status display, and performance.

# ----------------------------------------------------------------------------
# Program settings
# ----------------------------------------------------------------------------
MODEL_PATH = "yolo26_best.pt"
MIN_CONFIDENCE = 0.5
IMAGE_SIZE = 1024
STREAM_WIDTH = 1920
STREAM_HEIGHT = 1080
STREAM_FPS = 15

RETINA_MASKS = True            # Full-resolution masks; the costliest option. Set False for more FPS.
DISPLAY_SCALE = 1.0            # e.g. 0.5 shrinks the output window and saves drawing time.
FRAME_TIMEOUT_MS = 5000        # Max wait for a camera frame.
MAX_FRAME_FAILURES = 30        # Consecutive bad frames before giving up.
WARMUP_RUNS = 2                # Dummy inferences so the first real frame isn't slow.
STATUS_CLOSE_DELAY_MS = 1200   # How long the "all OK" status stays visible.

STATUS_WINDOW = "Device Status"
OUTPUT_WINDOW = "YOLO11 Output"
LABEL_WINDOW = "Detected Labels"


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------
class CameraError(RuntimeError):
    """Raised when the RealSense camera or its stream fails."""


class ModelError(RuntimeError):
    """Raised when the YOLO model fails to load or run."""


# ----------------------------------------------------------------------------
# Startup status window
# ----------------------------------------------------------------------------
class StatusWindow:
    """A small window listing startup checks; closes itself once all pass."""

    WIDTH, HEIGHT = 640, 300
    BG = (30, 30, 30)
    STATES = {
        "pending": ("...", (160, 160, 160)),
        "ok": ("OK", (80, 200, 80)),
        "fail": ("FAIL", (60, 60, 230)),
    }

    def __init__(self, titles):
        self.items = [[title, "pending", ""] for title in titles]
        self._base = np.full((self.HEIGHT, self.WIDTH, 3), self.BG, dtype=np.uint8)
        cv2.putText(self._base, "Device status check", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.namedWindow(STATUS_WINDOW, cv2.WINDOW_AUTOSIZE)
        self.draw()

    def set(self, index, state, detail=""):
        self.items[index][1] = state
        self.items[index][2] = detail
        self.draw()

    def draw(self):
        canvas = self._base.copy()
        y = 90
        for title, state, detail in self.items:
            tag, color = self.STATES[state]
            cv2.putText(canvas, f"[{tag}]", (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2, cv2.LINE_AA)
            cv2.putText(canvas, title, (130, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 1, cv2.LINE_AA)
            if detail:
                cv2.putText(canvas, detail[:70], (130, y + 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (190, 190, 190), 1, cv2.LINE_AA)
            y += 55
        cv2.imshow(STATUS_WINDOW, canvas)
        cv2.waitKey(1)

    def hold_until_key(self):
        """Keep the window open on failure so the error can be read."""
        cv2.putText(self._base, "Press any key to exit", (20, self.HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        self.draw()
        cv2.waitKey(0)

    def close(self):
        cv2.waitKey(STATUS_CLOSE_DELAY_MS)
        cv2.destroyWindow(STATUS_WINDOW)
        cv2.waitKey(1)  # Let the GUI process the close event.


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def load_model():
    """Load the YOLO model, move it to the best device, and warm it up.

    Returns (model, predict_kwargs) where predict_kwargs are fixed inference
    options built once instead of on every frame.
    """
    if not Path(MODEL_PATH).is_file():
        raise ModelError(f"Model file not found: {MODEL_PATH}")

    try:
        model = YOLO(MODEL_PATH)
    except Exception as exc:
        raise ModelError(f"Could not load model: {exc}") from exc

    use_cuda = torch.cuda.is_available()
    predict_kwargs = {
        "imgsz": IMAGE_SIZE,
        "conf": MIN_CONFIDENCE,
        "retina_masks": RETINA_MASKS,
        "device": 0 if use_cuda else "cpu",
        "half": use_cuda,          # FP16 is much faster on GPU; unsupported on CPU.
        "verbose": False,
    }

    try:
        dummy = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
        for _ in range(WARMUP_RUNS):
            model.predict(source=dummy, **predict_kwargs)
    except Exception as exc:
        raise ModelError(f"Model warm-up failed: {exc}") from exc

    return model, predict_kwargs


def send_to_model(model, predict_kwargs, color_image):
    """Run YOLO inference on one color image and return the first result."""
    try:
        return model.predict(source=color_image, **predict_kwargs)[0]
    except Exception as exc:
        raise ModelError(f"Inference failed: {exc}") from exc


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------
def detect_camera():
    """Find the first connected RealSense camera and return its device."""
    try:
        devices = rs.context().query_devices()
    except RuntimeError as exc:
        raise CameraError(f"Could not query RealSense devices: {exc}") from exc

    if len(devices) == 0:
        raise CameraError("No RealSense device detected. Check the USB connection.")

    return devices[0]


def describe_camera(device):
    """Return a short description string (name / serial / USB type) for display."""
    def info(key):
        try:
            return device.get_info(key) if device.supports(key) else "?"
        except RuntimeError:
            return "?"

    return (f"{info(rs.camera_info.name)} | S/N {info(rs.camera_info.serial_number)} "
            f"| USB {info(rs.camera_info.usb_type_descriptor)}")


def configure_stream(device):
    """Create the pipeline/config for the color stream and verify it is supported."""
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(device.get_info(rs.camera_info.serial_number))
        config.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT,
                             rs.format.bgr8, STREAM_FPS)

        if not config.can_resolve(rs.pipeline_wrapper(pipeline)):
            raise CameraError(
                f"Camera does not support {STREAM_WIDTH}x{STREAM_HEIGHT} "
                f"@ {STREAM_FPS} FPS (BGR8)."
            )
    except CameraError:
        raise
    except RuntimeError as exc:
        raise CameraError(f"Could not configure stream: {exc}") from exc

    return pipeline, config


def start_stream(pipeline, config):
    """Start the pipeline."""
    try:
        pipeline.start(config)
    except RuntimeError as exc:
        raise CameraError(f"Could not start camera stream: {exc}") from exc


def get_color_frame(pipeline):
    """Wait for a frame and return it as a NumPy image, or None if it is unusable.

    The array wraps the frame buffer (no copy), which is safe because it is
    consumed by the model before the next frame is requested.
    """
    try:
        frames = pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
    except RuntimeError as exc:
        raise CameraError(f"Timed out or lost the camera while waiting for a frame: {exc}") from exc

    color_frame = frames.get_color_frame()
    if not color_frame:
        return None

    return np.asanyarray(color_frame.get_data())


# ----------------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------------
class LabelWindow:
    """Draws the label panel, re-rendering only when its text changes."""

    WIDTH, HEIGHT = 800, 500

    def __init__(self):
        self._empty = np.full((self.HEIGHT, self.WIDTH, 3), 30, dtype=np.uint8)
        self._last_lines = None
        self._last_image = self._empty

    def render(self, result):
        names = result.names
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            classes = boxes.cls.tolist()
            confs = boxes.conf.tolist()
            lines = tuple(f"{names[int(c)]}: {p:.2f}" for c, p in zip(classes, confs))
        else:
            lines = ("No objects detected",)

        if lines == self._last_lines:
            return self._last_image

        image = self._empty.copy()
        y = 45
        for line in lines:
            if y > self.HEIGHT - 10:   # Don't draw off the bottom of the panel.
                break
            cv2.putText(image, line, (25, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2, cv2.LINE_AA)
            y += 40

        self._last_lines = lines
        self._last_image = image
        return image


def display_results(result, label_window):
    """Show annotated detections and labels; return True if Q was pressed."""
    annotated = result.plot()
    if DISPLAY_SCALE != 1.0:
        annotated = cv2.resize(annotated, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                               interpolation=cv2.INTER_AREA)

    cv2.imshow(OUTPUT_WINDOW, annotated)
    cv2.imshow(LABEL_WINDOW, label_window.render(result))

    return (cv2.waitKey(1) & 0xFF) == ord("q")


# ----------------------------------------------------------------------------
# Startup checks and main loop
# ----------------------------------------------------------------------------
def run_startup_checks():
    """Run every startup check with a live status window.

    The window closes automatically once everything is operational. On a
    failure it stays open until a key is pressed, then the error is re-raised.
    Returns (model, predict_kwargs, pipeline).
    """
    status = StatusWindow([
        "RealSense camera detected",
        "YOLO model loaded & warmed up",
        "Camera stream started",
        "Receiving frames",
    ])
    pipeline = None
    step = 0

    try:
        device = detect_camera()
        status.set(0, "ok", describe_camera(device))

        step = 1
        status.set(1, "pending", "Loading model...")
        model, predict_kwargs = load_model()
        status.set(1, "ok", f"{MODEL_PATH} on {'GPU (FP16)' if predict_kwargs['half'] else 'CPU'}")

        step = 2
        pipeline, config = configure_stream(device)
        start_stream(pipeline, config)
        status.set(2, "ok", f"{STREAM_WIDTH}x{STREAM_HEIGHT} @ {STREAM_FPS} FPS")

        step = 3
        # The first frames after start-up can be empty; allow a few attempts.
        for _ in range(MAX_FRAME_FAILURES):
            if get_color_frame(pipeline) is not None:
                break
        else:
            raise CameraError("Camera started but no color frames were received.")
        status.set(3, "ok", "Frames are flowing")

        status.close()
        return model, predict_kwargs, pipeline

    except Exception as exc:
        status.set(step, "fail", str(exc))
        print(f"Startup failed: {exc}", file=sys.stderr)
        if pipeline is not None:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
        status.hold_until_key()
        cv2.destroyAllWindows()
        raise


def main():
    """Verify the device, run detection, and display results until Q is pressed."""
    model, predict_kwargs, pipeline = run_startup_checks()
    label_window = LabelWindow()
    failures = 0

    try:
        while True:
            color_image = get_color_frame(pipeline)

            if color_image is None:
                failures += 1
                if failures >= MAX_FRAME_FAILURES:
                    raise CameraError("Too many consecutive empty frames from the camera.")
                continue
            failures = 0

            result = send_to_model(model, predict_kwargs, color_image)

            if display_results(result, label_window):
                break
    finally:
        try:
            pipeline.stop()
        except RuntimeError:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except (CameraError, ModelError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(2)
