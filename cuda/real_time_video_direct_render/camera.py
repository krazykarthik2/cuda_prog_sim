import cv2
import subprocess
import sys
import time
import numpy as np

EXE_PATH = "raylib_cuda.exe"

# Target Resolution (Must match C++ VID_W/VID_H)
WIDTH = 1280
HEIGHT = 720
FPS_TARGET = 30

current_filter = 0  # 0: original, 1: grayscale, 2: inversion, 3: custom, 4: gaussian, 5: sharpen, 6: high-pass, 7: low-pass, 8: edge
current_radius = 1

def mouse_callback(event, x, y, flags, param):
    global current_filter
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x <= 110 and 10 <= y <= 40:
            current_filter = 1  # Grayscale
        elif 120 <= x <= 220 and 10 <= y <= 40:
            current_filter = 2  # Inversion
        elif 230 <= x <= 330 and 10 <= y <= 40:
            current_filter = 3  # Custom
        elif 10 <= x <= 110 and 50 <= y <= 80:
            current_filter = 4  # Gaussian
        elif 120 <= x <= 220 and 50 <= y <= 80:
            current_filter = 5  # Sharpen
        elif 230 <= x <= 330 and 50 <= y <= 80:
            current_filter = 6  # High-pass
        elif 10 <= x <= 110 and 90 <= y <= 120:
            current_filter = 7  # Low-pass
        elif 120 <= x <= 220 and 90 <= y <= 120:
            current_filter = 8  # Edge
        elif 230 <= x <= 330 and 90 <= y <= 120:
            current_filter = 0  # Normal

def radius_callback(val):
    global current_radius
    current_radius = val

def main():
    # 1. Open Video
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # 2. Launch C++ Process
    try:
        proc = subprocess.Popen(
            [EXE_PATH], 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=sys.stderr
        )
    except FileNotFoundError:
        print(f"Error: Could not find {EXE_PATH}. Did you build it?")
        return

    cv2.namedWindow("Video")
    cv2.setMouseCallback("Video", mouse_callback)
    cv2.createTrackbar("Radius", "Video", 1, 20, radius_callback)

    frame_time = 1.0 / FPS_TARGET

    while True:
        start_t = time.time()

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # 3. Process Frame
        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        # Draw buttons
        cv2.rectangle(frame, (10, 10), (110, 40), (255, 255, 255), -1)
        cv2.putText(frame, "Grayscale", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(frame, (120, 10), (220, 40), (255, 255, 255), -1)
        cv2.putText(frame, "Inversion", (125, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(frame, (230, 10), (330, 40), (255, 255, 255), -1)
        cv2.putText(frame, "Custom", (235, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(frame, (10, 50), (110, 80), (255, 255, 255), -1)
        cv2.putText(frame, "Gaussian", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(frame, (120, 50), (220, 80), (255, 255, 255), -1)
        cv2.putText(frame, "Sharpen", (125, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(frame, (230, 50), (330, 80), (255, 255, 255), -1)
        cv2.putText(frame, "High-pass", (235, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(frame, (10, 90), (110, 120), (255, 255, 255), -1)
        cv2.putText(frame, "Low-pass", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(frame, (120, 90), (220, 120), (255, 255, 255), -1)
        cv2.putText(frame, "Edge", (125, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(frame, (230, 90), (330, 120), (255, 255, 255), -1)
        cv2.putText(frame, "Normal", (235, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Convert BGR to RGBA
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        # Send filter command
        filter_cmd = f"FILTER:{current_filter}:{current_radius}\n".encode()
        proc.stdin.write(filter_cmd)
        proc.stdin.flush()

        # Send frame
        proc.stdin.write(frame_rgba.tobytes())
        proc.stdin.flush()

        # Receive processed frame
        processed_bytes = proc.stdout.read(WIDTH * HEIGHT * 4)
        if len(processed_bytes) != WIDTH * HEIGHT * 4:
            break
        processed_frame = np.frombuffer(processed_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, 4))
        processed_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGBA2BGR)

        # Display
        cv2.imshow("Video", processed_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Maintain FPS
        elapsed = time.time() - start_t
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

    cap.release()
    cv2.destroyAllWindows()
    proc.terminate()

if __name__ == "__main__":
    main()
