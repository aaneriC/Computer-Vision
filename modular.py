import sys

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

#developed after cam_w_model.py, but modularized and improved for error handling, status display, and performance.

# Program settings
MODEL_PATH = "best2.pt"
MIN_CONFIDENCE = 0.5
IMAGE_SIZE = 1024
STREAM_WIDTH = 1920
STREAM_HEIGHT = 1080
STREAM_FPS = 15

def load_model():
    """Load and return the trained YOLO model."""
    return YOLO(MODEL_PATH)

def detect_camera():
    """Find the first connected RealSense camera and return its device."""
    context = rs.context()
    devices = context.query_devices()

    if len(devices) == 0:
        raise RuntimeError("No RealSense device detected. Check the USB connection.")

    return devices[0]


def configure_stream():
    """Create and configure the RealSense color stream."""
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        STREAM_WIDTH,
        STREAM_HEIGHT,
        rs.format.bgr8,
        STREAM_FPS
    )

    return pipeline, config


def get_color_frame(pipeline):
    """Wait for a frame and convert the color frame to a NumPy image."""
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if not color_frame:
        return None

    return np.asanyarray(color_frame.get_data())


def send_to_model(model, color_image):
    """Run YOLO inference on one color image and return the results."""
    return model.predict(
        source=color_image,
        imgsz=IMAGE_SIZE,
        conf=MIN_CONFIDENCE,
        retina_masks=True,
        verbose=False
    )


def create_label_window(results):
    """Create an image containing detected labels and confidence scores."""
    label_window = np.zeros((500, 800, 3), dtype=np.uint8)
    label_window[:] = (30, 30, 30)
    label_y = 45
    detected_label = False

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for class_id, confidence in zip(class_ids, confidences):
            label_name = result.names[int(class_id)]
            label_text = f"{label_name}: {confidence:.2f}"

            cv2.putText(
                label_window,
                label_text,
                (25, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            label_y += 40
            detected_label = True

    if not detected_label:
        cv2.putText(
            label_window,
            "No objects detected",
            (25, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return label_window


def display_results(results, original_image):
    """Display annotated detections and labels, then return whether Q was pressed."""
    if results:
        annotated_frame = results[0].plot()
    else:
        annotated_frame = original_image

    label_window = create_label_window(results)

    cv2.imshow("YOLO11 Output", annotated_frame)
    cv2.imshow("Detected Labels", label_window)

    return (cv2.waitKey(1) & 0xFF) == ord("q")


def main():
    """Start the camera, run detection, and display results until Q is pressed."""
    model = load_model()
    detect_camera()
    pipeline, config = configure_stream()

    pipeline.start(config)

    try:
        while True:
            color_image = get_color_frame(pipeline)

            if color_image is None:
                continue

            results = send_to_model(model, color_image)

            if display_results(results, color_image):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()







if __name__ == "__main__":
    try:
        main()
    except RuntimeError:
        # Keep runtime messages out of the terminal; close any OpenCV windows.
        cv2.destroyAllWindows()
        sys.exit(1)
