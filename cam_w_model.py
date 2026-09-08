import os
import sys
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

#define the .pt model
model = "best.pt"
min_thresh =  0.6 # minimum confidence level model can output


#detect camera
ctx = rs.context()
devices = ctx.query_devices()
if len(devices) == 0:
    print("[ERROR] No RealSense device detected. Check USB connection.")
    sys.exit(1)

#print camera info if detected
device = devices[0]
print(f"[INFO] Connected to: {device.get_info(rs.camera_info.name)}")
print(f"[INFO] Serial number : {device.get_info(rs.camera_info.serial_number)}")
print(f"[INFO] Firmware      : {device.get_info(rs.camera_info.firmware_version)}")


#configure D435i camera pipeline for streaming
pipeline = rs.pipeline()
config = rs.config()

#enable RGB stream; define resolution & FPS
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30) 
pipeline.start(config)


#if variables are working
try:

    while True:
        #wait for frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        #convert obtained image into OpenCV image
        color_image = np.asanyarray(color_frame.get_data())

        #pass image to YOLO11 model
        results = model(color_image, stream = True)
    
        for result in results:
            #segmentation images
            annotated_frame = result.plot()

        #Output from YOLO
        cv2.imshow("YOLO11 Output", annotated_frame)

        #wait to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#always run this section of code
finally:
    print("stopping")
    pipeline.stop
    cv2.destroyAllWindows()
