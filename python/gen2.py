import cv2
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------
# Python script for annotating images with quadrilateral regions-of-interest
# Note: Plot the points in a rotating fashion to avoid crossing the lines of the polygon
# -----------------------------------------------

image_path = r'image.jpg'
img = cv2.imread(image_path)

if img is None:
    print("ERROR: Image not found, check your path!")
    exit()

clicks = []
quads = []

# --- Set this to fit your screen comfortably ---
DISPLAY_WIDTH  = 1920
DISPLAY_HEIGHT = 1200

# Calculate scale factor
orig_h, orig_w = img.shape[:2]
scale_x = orig_w / DISPLAY_WIDTH
scale_y = orig_h / DISPLAY_HEIGHT

img_display = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
img_display_copy = img_display.copy()

# -----------------------------------------------
# PHASE 1 — Click to collect coordinates
# -----------------------------------------------

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        real_x = int(x * scale_x)
        real_y = int(y * scale_y)
        clicks.append((real_x, real_y))
        count = len(clicks)

        cv2.circle(img_display_copy, (x, y), 6, (0, 255, 0), -1)
        
        if count % 4 == 0:
            points = np.array([[item[0] / scale_x, item[1] / scale_y] for item in clicks], np.int32)
            quads.append(points)
            cv2.polylines(img_display_copy, [points], True, (0, 255, 0), 2)
            clicks.clear()
            
        cv2.imshow("Image", img_display_copy)

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", DISPLAY_WIDTH, DISPLAY_HEIGHT)
cv2.imshow("Image", img_display_copy)
cv2.setMouseCallback("Image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------------------------------
# PHASE 2 — Build ROI list
# -----------------------------------------------

# ROIs already built in quads

# -----------------------------------------------
# PHASE 3 — Draw Outlines ONLY
# -----------------------------------------------

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_with_rois = img_rgb.copy()

# Using a single color (Green) for a clean look, or keep the list for variety
for quad in quads:
    # Drawing only the rectangle, no text labels
    cv2.polylines(img_display_copy, [quad], True, (0, 255, 0), 2)

# -----------------------------------------------
# PHASE 4 — Plot Result # Uncalled function
# -----------------------------------------------
def plotImageWithAnnotations():
    plt.figure(figsize=(12, 8))
    plt.imshow(img_with_rois)
    plt.title("Image with ROI Outlines", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# -----------------------------------------------
# PHASE 5 — Summary
# -----------------------------------------------

print("\nFinal Coordinates:")
for idx, quad in enumerate(quads):
    print(f"ROI {idx}: ({quad[0][0]}, {quad[0][1]}, {quad[1][0]}, {quad[1][1]})")