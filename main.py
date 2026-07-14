import cv2
import threading
import time
from parking import ParkingManagement

class RTSPStreamReader:
    """
    Reads RTSP streams on a background thread so the main processing loop 
    always gets the freshest frame without network lag.
    """
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        # Force FFMPEG backend and optimize buffer size for low latency
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Limit internal buffer to 1 frame
        
        self.ret = False
        self.frame = None
        self.is_running = True
        
        # Start background frame grabber
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.is_running:
            if not self.cap.isOpened():
                # Attempt reconnection if stream drops
                time.sleep(1)
                self.cap.open(self.rtsp_url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            ret, frame = self.cap.read()
            if ret:
                self.ret = True
                self.frame = frame
            else:
                self.ret = False
                time.sleep(0.01)  # Prevent CPU spiking when stream stalls

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.is_running = False
        if self.thread.is_alive():
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()


# ==========================================
# MAIN EXECUTION
# ==========================================

# Replace with your actual RTSP credentials/addresses
RTSP_URL_1 = "rtsp://username:password@ip_address_1:554/stream"
RTSP_URL_2 = "rtsp://username:password@ip_address_2:554/stream"

# 1. Initialize threaded streams
stream1 = RTSPStreamReader(RTSP_URL_1)
stream2 = RTSPStreamReader(RTSP_URL_2)

# 2. Initialize separate parking managers (one for each unique camera view)
parking_manager_s1 = ParkingManagement(
    model="yolo11s.pt",
    classes=[2],
    json_file="bounding_boxes_s1.json",  # Specific map for Camera 1
)

parking_manager_s2 = ParkingManagement(
    model="yolo11s.pt",
    classes=[2],
    json_file="bounding_boxes_s2.json",  # Specific map for Camera 2
)

print("Starting streams... Press ESC on video windows to close.")

try:
    while True:
        # Retrieve the most recent frames from memory buffer (no network wait)
        ret1, frame1 = stream1.read()
        ret2, frame2 = stream2.read()

        # Process stream 1 if frame is ready
        if ret1 and frame1 is not None:
            im01 = cv2.resize(frame1, (1080, 600))
            processed_s1 = parking_manager_s1.process_data(im01)
            cv2.imshow("RTSP Camera 1", processed_s1)

        # Process stream 2 if frame is ready
        if ret2 and frame2 is not None:
            im02 = cv2.resize(frame2, (1080, 600))
            processed_s2 = parking_manager_s2.process_data(im02)
            cv2.imshow("RTSP Camera 2", processed_s2)

        # Break loop if ESC is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Safely clean up threads and cameras
    print("Closing streams...")
    stream1.release()
    stream2.release()
    cv2.destroyAllWindows()