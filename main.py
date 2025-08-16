import cv2
import numpy as np
import time
import os

# -------------------------
# Load Face Detector
# -------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------
# Filter paths and names
# -------------------------
filter_paths = [
    r"C:/Users/LENOVO/Desktop/AI/C4/sunglasses.png",
    r"C:/Users/LENOVO/Desktop/AI/C4/hat.png",
    r"C:/Users/LENOVO/Desktop/AI/C4/moustache.png",
    r"C:/Users/LENOVO/Desktop/AI/C4/crown.png",
]
filter_names = ["Sunglasses", "Hat", "Moustache", "Crown"]

# -------------------------
# Load Filters
# -------------------------
filters = []
for path in filter_paths:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: Could not load {path}")
        exit()
    filters.append(img)

# -------------------------
# Helper to overlay filter
# -------------------------
def add_filter(frame, overlay_img, x, y, w, h):
    overlay_img = cv2.resize(overlay_img, (w, h))
    b, g, r, a = cv2.split(overlay_img)
    mask = a / 255.0
    for c in range(3):
        frame[y:y+h, x:x+w, c] = (1 - mask) * frame[y:y+h, x:x+w, c] + mask * overlay_img[:, :, c]
    return frame

# -------------------------
# Setup camera
# -------------------------
cap = cv2.VideoCapture(0)

current_filter = 0
is_recording = False
out = None

# Create output folder
os.makedirs("captures", exist_ok=True)

print("Controls:")
print("N - Next filter")
print("P - Previous filter")
print("V - Start/Stop recording")
print("S - Save snapshot")
print("Q - Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 7)

    show_adjust_msg = False  # Flag for warning

    for (x, y, w, h) in faces:
        name = filter_names[current_filter]

        if name == "Sunglasses":
            fw, fh = w, int(h * 0.3)
            fx, fy = x, y + int(h * 0.3)
        elif name in ["Hat", "Crown"]:
            fw, fh = int(w * 1.2), int(h * 0.6)
            fx, fy = x - int(w * 0.1), y - int(h * 0.6)
        elif name == "Moustache":
            fw, fh = int(w * 0.5), int(h * 0.18)
            fx, fy = x + int(w * 0.25), y + int(h * 0.65)
        else:
            fw, fh = w, int(h * 0.3)
            fx, fy = x, y + int(h * 0.2)

        # Check if filter goes out of frame
        if fx < 0 or fy < 0 or fx + fw > frame.shape[1] or fy + fh > frame.shape[0]:
            show_adjust_msg = True
        else:
            frame = add_filter(frame, filters[current_filter], fx, fy, fw, fh)

        break  # Only process first detected face

    # Show filter name
    cv2.putText(frame, filter_names[current_filter], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show warning if needed
    if show_adjust_msg:
        cv2.putText(frame, "Adjust face position", (50, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Video recording
    if is_recording:
        cv2.putText(frame, "Recording...", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out.write(frame)

    cv2.imshow("Fun Filters", frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('n'):
        current_filter = (current_filter + 1) % len(filters)
    elif key == ord('p'):
        current_filter = (current_filter - 1) % len(filters)
    elif key == ord('v'):
        if not is_recording:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            filename = f"captures/recording_{time.strftime('%Y%m%d_%H%M%S')}.avi"
            out = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            is_recording = True
            print(f"Started recording: {filename}")
        else:
            is_recording = False
            out.release()
            print("Stopped recording.")
    elif key == ord('s'):
        img_filename = f"captures/snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(img_filename, frame)
        print(f"Snapshot saved: {img_filename}")

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
