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

