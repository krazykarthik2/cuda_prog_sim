import cv2
import subprocess
import sys
import time

EXE_PATH = "raylib_cuda.exe"

# Target Resolution (Must match C++ VID_W/VID_H)
WIDTH = 1280
HEIGHT = 720
FPS_TARGET = 120

def main():
    # 1. Open Video
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"Error: Could not open camera")
        return

    # 2. Launch C++ Process
    # We pipe stdout=None so it goes to the real terminal (to see ASCII)
    # We pipe stdin=PIPE to send frames
    try:
        proc = subprocess.Popen(
            [EXE_PATH], 
            stdin=subprocess.PIPE, 
            stdout=None, 
            stderr=sys.stderr
        )
    except FileNotFoundError:
        print(f"Error: Could not find {EXE_PATH}. Did you build it?")
        return

    print("Playing video... Press 'q' in this terminal to stop (ctrl+c might work too)")

    frame_time = 1.0 / FPS_TARGET

    while True:
        start_t = time.time()

        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # 3. Process Frame
        # Resize if needed
        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

        # Convert BGR to RGBA (for uchar4 in CUDA)
        # We add an alpha channel = 255
        frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        # 4. Send to C++
        try:
            proc.stdin.write(frame_rgba.tobytes())
            proc.stdin.flush()
        except BrokenPipeError:
            print("Process exited.")
            break

        # 5. Maintain FPS
        elapsed = time.time() - start_t
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

    cap.release()
    proc.terminate()

if __name__ == "__main__":
    main()
