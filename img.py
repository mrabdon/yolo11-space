import cv2
import time

cpt = 0
maxFrames = 1  # Number of frames to save

count = 0

# Replace with your RTSP URL
rtsp_url = "rtsp://admin:%40ict2025%40@192.168.1.3:554/stream?rtsp_transport=tcp"

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Unable to open RTSP stream.")
    exit()

while cpt < maxFrames:
    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame.")
        break

    count += 1

    # Process every 3rd frame
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1080, 600))

    cv2.imshow("RTSP Stream", frame)

    cv2.imwrite(
        f"D:\Dev\space/img_{cpt}.jpg",
        frame
    )

    time.sleep(0.01)
    cpt += 1

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()