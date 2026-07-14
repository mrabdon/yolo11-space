import cv2
import numpy as np
import threading
import time
from flask import Flask, render_template, jsonify, Response
from parking import ParkingManagement

# ==========================================
# 1. FLASK WEB SERVER SETUP
# ==========================================
app = Flask(__name__)

# Global storage for live occupancy metrics
parking_stats = {
    "camera_1": {"occupied": 0, "available": 0, "total": 0},
    "camera_2": {"occupied": 0, "available": 0, "total": 0},
    "total_lot": {"occupied": 0, "available": 0, "total": 0}
}

# Separate global thread-safe buffers to hold individual video frames
output_frames = {
    "camera_1": None,
    "camera_2": None
}
frame_lock = threading.Lock()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/parking")
def api_parking():
    return jsonify(parking_stats)

def generate_mjpeg_stream(camera_id):
    """Generates JPEG frame byte-streams for browser rendering based on camera_id."""
    global output_frames, frame_lock
    while True:
        with frame_lock:
            frame = output_frames.get(camera_id)
            if frame is None:
                time.sleep(0.03)
                continue
            # Encode frame to JPEG
            ret, encoded_image = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = encoded_image.tobytes()
        
        # Standard MJPEG boundary formatting
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)  # Throttle stream roughly to ~30 FPS

@app.route("/video_feed/<camera_id>")
def video_feed(camera_id):
    """Streaming route tailored for individual camera_ids."""
    if camera_id not in ["camera_1", "camera_2"]:
        return "Camera not found", 404
    return Response(generate_mjpeg_stream(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)


# ==========================================
# 2. RTSP STREAM READER
# ==========================================
class RTSPStreamReader:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.ret = False
        self.frame = None
        self.is_running = True
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.is_running:
            if not self.cap.isOpened():
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
                time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.is_running = False
        if self.thread.is_alive():
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()


# ==========================================
# 3. MAIN CV PROCESSING LOOP
# ==========================================
def main_cv_loop():
    global output_frames, frame_lock

    RTSP_URL_1 = "rtsp://admin:%40ict2025%40@192.168.1.3:554/stream?rtsp_transport=tcp"
    RTSP_URL_2 = "rtsp://admin:%40ict2025%40@192.168.1.11:554/stream?rtsp_transport=tcp"

    stream1 = RTSPStreamReader(RTSP_URL_1)
    stream2 = RTSPStreamReader(RTSP_URL_2)

    parking_manager_s1 = ParkingManagement(
        model="yolo11s.pt", classes=[2], json_file="bounding_boxes_s1.json"
    )
    parking_manager_s2 = ParkingManagement(
        model="yolo11s.pt", classes=[2], json_file="bounding_boxes_s2.json"
    )

    FRAME_WIDTH, FRAME_HEIGHT = 1080, 600

    try:
        while True:
            ret1, frame1 = stream1.read()
            ret2, frame2 = stream2.read()

            # Process Stream 1
            if ret1 and frame1 is not None:
                im01 = cv2.resize(frame1, (FRAME_WIDTH, FRAME_HEIGHT))
                processed_s1 = parking_manager_s1.process_data(im01)
                occ1 = parking_manager_s1.pr_info.get("Occupancy", 0)
                empty1 = parking_manager_s1.pr_info.get("Available", 0)
                parking_stats["camera_1"] = {"occupied": occ1, "available": empty1, "total": occ1 + empty1}
            else:
                processed_s1 = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(processed_s1, "Camera 1 Offline", (100, FRAME_HEIGHT // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                parking_stats["camera_1"] = {"occupied": 0, "available": 0, "total": 0}

            # Process Stream 2
            if ret2 and frame2 is not None:
                im02 = cv2.resize(frame2, (FRAME_WIDTH, FRAME_HEIGHT))
                processed_s2 = parking_manager_s2.process_data(im02)
                occ2 = parking_manager_s2.pr_info.get("Occupancy", 0)
                empty2 = parking_manager_s2.pr_info.get("Available", 0)
                parking_stats["camera_2"] = {"occupied": occ2, "available": empty2, "total": occ2 + empty2}
            else:
                processed_s2 = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(processed_s2, "Camera 2 Offline", (100, FRAME_HEIGHT // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                parking_stats["camera_2"] = {"occupied": 0, "available": 0, "total": 0}

            # Aggregating Lot Statistics (remains identical)
            t_occupied = parking_stats["camera_1"]["occupied"] + parking_stats["camera_2"]["occupied"]
            t_available = parking_stats["camera_1"]["available"] + parking_stats["camera_2"]["available"]
            parking_stats["total_lot"] = {"occupied": t_occupied, "available": t_available, "total": t_occupied + t_available}

            # Write to thread-safe separate camera pointers
            with frame_lock:
                output_frames["camera_1"] = processed_s1.copy()
                output_frames["camera_2"] = processed_s2.copy()

            time.sleep(0.01) # Yield core control minorly
    finally:
        stream1.release()
        stream2.release()

if __name__ == "__main__":
    print("Launching AI backend and background web services...")
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Run the main loop directly on the main thread
    main_cv_loop()