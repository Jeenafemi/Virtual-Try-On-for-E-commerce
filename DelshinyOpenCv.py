import cv2
import numpy as np

def detect_fire_smoke(frame):
    # Convert frame to HSV and Grayscale
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Fire color range in HSV
    fire_lower = np.array([18, 50, 50])
    fire_upper = np.array([35, 255, 255])
    fire_mask = cv2.inRange(hsv, fire_lower, fire_upper)

    # Smoke detection using mid-gray range
    smoke_mask = cv2.inRange(gray, 100, 200)

    # Combine masks
    combined_mask = cv2.bitwise_or(fire_mask, smoke_mask)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Detection logic
    fire_detected = cv2.countNonZero(fire_mask) > 500
    smoke_detected = cv2.countNonZero(smoke_mask) > 500

    if fire_detected and smoke_detected:
        classification = "Fire and Smoke Detected"
    elif fire_detected:
        classification = "Fire Detected"
    elif smoke_detected:
        classification = "Smoke Detected"
    else:
        classification = "No Threat"

    return combined_mask, classification

# === Load image (change path as needed) ===
image_path = "fire and smoke.jpg"  # ← Replace with your file name
frame = cv2.imread(image_path)

if frame is None:
    print("Image not found or invalid path.")
else:
    mask, classification = detect_fire_smoke(frame)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Add classification text
    cv2.putText(frame, f'Status: {classification}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show results
    cv2.imshow("Image Frame", frame)
    cv2.imshow("Detection Output", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
