import cv2
import numpy as np

def detect_animals(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_brown = np.array([5, 50, 50])
    upper_brown = np.array([25, 255, 255])
    animal_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(animal_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detected += 1

    # Show status on the image
    status_text = "Animal Detected" if detected > 0 else "No Animal Detected"
    cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if detected == 0 else (0, 255, 0), 2)

    return image, detected

# === Load input image ===
image_path = "image.jpeg"  # Replace with your image
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found.")
else:
    result, animal_count = detect_animals(image)
    print(f"Animals Detected: {animal_count}")

    # Display result
    cv2.imshow("Wildlife Monitoring Output", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()