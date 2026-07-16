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

# Combined global storage for simplified API retrieval
parking_data = {
    "lot": {
        "camera_1": {"occupied": 0, "available": 0, "total": 0},
        "camera_2": {"occupied": 0, "available": 0, "total": 0},
        "total": {"occupied": 0, "available": 0, "total": 0}
    },
    "aeb": {
        "camera_aeb1": {"occupied": 0, "available": 0, "total": 0},
        "camera_aeb2": {"occupied": 0, "available": 0, "total": 0},
        "total": {"occupied": 0, "available": 0, "total": 0}
    },
      "gym": {
        "camera_gym1": {"occupied": 0, "available": 0, "total": 0},
        "camera_gym2": {"occupied": 0, "available": 0, "total": 0},
        "total": {"occupied": 0, "available": 0, "total": 0}
    }
}

# Separate global thread-safe buffers to hold individual video frames
output_frames = {
    "camera_1": None,
    "camera_2": None,
    "camera_aeb1": None,
    "camera_aeb2": None,
    "camera_gym1": None,
    "camera_gym2": None
}
frame_lock = threading.Lock()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/parking")
def api_parking():
    return jsonify(parking_data)

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
    if camera_id not in ["camera_1", "camera_2", "camera_aeb1", "camera_aeb2", "camera_gym1",  "camera_gym2"]:
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
    RTSP_AEB_1 = "rtsp://admin:%40ict2025%40@192.168.1.6:554/stream?rtsp_transport=tcp"
    RTSP_AEB_2 = "rtsp://admin:%40ict2025%40@192.168.1.12:554/stream?rtsp_transport=tcp"
    RTSP_GYM_1 = "rtsp://admin:%40ict2025%40@192.168.1.4:554/stream?rtsp_transport=tcp"
    RTSP_GYM_2 = "rtsp://admin:%40ict2025%40@192.168.1.5:554/stream?rtsp_transport=tcp"

    stream1 = RTSPStreamReader(RTSP_URL_1)
    stream2 = RTSPStreamReader(RTSP_URL_2)
    stream3 = RTSPStreamReader(RTSP_AEB_1)
    stream4 = RTSPStreamReader(RTSP_AEB_2)
    stream5 = RTSPStreamReader(RTSP_GYM_1)
    stream6 = RTSPStreamReader(RTSP_GYM_2)

    parking_manager_s1 = ParkingManagement(
        model="yolo11s.pt", classes=[2], json_file="bounding_boxes_s1.json"
    )
    parking_manager_s2 = ParkingManagement(
        model="yolo11s.pt", classes=[2], json_file="bounding_boxes_s2.json"
    )
    parking_manager_aeb1 = ParkingManagement(
        model="yolo11s.pt", classes=[2], json_file="bounding_boxes_aeb1.json"
    )
    parking_manager_aeb2 = ParkingManagement(
        model="yolo11s.pt", classes=[2], json_file="bounding_boxes_aeb2.json"
    )
    parking_manager_gym1 = ParkingManagement(
        model="yolo11s.pt", classes=[2], json_file="bounding_boxes_gym1.json"
    )
    parking_manager_gym2 = ParkingManagement(
        model="yolo11s.pt", classes=[2], json_file="bounding_boxes_gym2.json"
    )

    FRAME_WIDTH, FRAME_HEIGHT = 1080, 600

    try:
        while True:
            ret1, frame1 = stream1.read()
            ret2, frame2 = stream2.read()
            ret3, frame3 = stream3.read()
            ret4, frame4 = stream4.read()
            ret5, frame5 = stream5.read()
            ret6, frame6 = stream6.read()

            # Process Stream 1
            if ret1 and frame1 is not None:
                im01 = cv2.resize(frame1, (FRAME_WIDTH, FRAME_HEIGHT))
                processed_s1 = parking_manager_s1.process_data(im01)
                occ1 = parking_manager_s1.pr_info.get("Occupancy", 0)
                empty1 = parking_manager_s1.pr_info.get("Available", 0)
                parking_data["lot"]["camera_1"] = {"occupied": occ1, "available": empty1, "total": occ1 + empty1}
            else:
                processed_s1 = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(processed_s1, "Camera 1 Offline", (100, FRAME_HEIGHT // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                parking_data["lot"]["camera_1"] = {"occupied": 0, "available": 0, "total": 0}

            # Process Stream 2
            if ret2 and frame2 is not None:
                im02 = cv2.resize(frame2, (FRAME_WIDTH, FRAME_HEIGHT))
                processed_s2 = parking_manager_s2.process_data(im02)
                occ2 = parking_manager_s2.pr_info.get("Occupancy", 0)
                empty2 = parking_manager_s2.pr_info.get("Available", 0)
                parking_data["lot"]["camera_2"] = {"occupied": occ2, "available": empty2, "total": occ2 + empty2}
            else:
                processed_s2 = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(processed_s2, "Camera 2 Offline", (100, FRAME_HEIGHT // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                parking_data["lot"]["camera_2"] = {"occupied": 0, "available": 0, "total": 0}

            # Process Stream 3 (AEB 1)
            if ret3 and frame3 is not None:
                im03 = cv2.resize(frame3, (FRAME_WIDTH, FRAME_HEIGHT))
                processed_s3 = parking_manager_aeb1.process_data(im03)
                occ3 = parking_manager_aeb1.pr_info.get("Occupancy", 0)
                empty3 = parking_manager_aeb1.pr_info.get("Available", 0)
                parking_data["aeb"]["camera_aeb1"] = {"occupied": occ3, "available": empty3, "total": occ3 + empty3}
            else:
                processed_s3 = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(processed_s3, "Camera AEB 1 Offline", (100, FRAME_HEIGHT // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                parking_data["aeb"]["camera_aeb1"] = {"occupied": 0, "available": 0, "total": 0}

            # Process Stream 4 (AEB 2)
            if ret4 and frame4 is not None:
                im04 = cv2.resize(frame4, (FRAME_WIDTH, FRAME_HEIGHT))
                processed_s4 = parking_manager_aeb2.process_data(im04)
                occ4 = parking_manager_aeb2.pr_info.get("Occupancy", 0)
                empty4 = parking_manager_aeb2.pr_info.get("Available", 0)
                parking_data["aeb"]["camera_aeb2"] = {"occupied": occ4, "available": empty4, "total": occ4 + empty4}
            else:
                processed_s4 = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(processed_s4, "Camera AEB 2 Offline", (100, FRAME_HEIGHT // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                parking_data["aeb"]["camera_aeb2"] = {"occupied": 0, "available": 0, "total": 0}

              # Process Stream 5 (GYM 1)
            if ret5 and frame5 is not None:
                im05 = cv2.resize(frame5, (FRAME_WIDTH, FRAME_HEIGHT))
                processed_s5 = parking_manager_gym1.process_data(im05)
                occ5 = parking_manager_gym1.pr_info.get("Occupancy", 0)
                empty5 = parking_manager_gym1.pr_info.get("Available", 0)
                parking_data["gym"]["camera_gym1"] = {"occupied": occ5, "available": empty5, "total": occ5 + empty5}
            else:
                processed_s5 = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(processed_s5, "Camera GYM 1 Offline", (100, FRAME_HEIGHT // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                parking_data["gym"]["camera_gym1"] = {"occupied": 0, "available": 0, "total": 0}

            # Process Stream 6 (GYM 2)
            if ret6 and frame6 is not None:
                im06 = cv2.resize(frame6, (FRAME_WIDTH, FRAME_HEIGHT))
                processed_s6 = parking_manager_gym2.process_data(im06)
                occ6 = parking_manager_gym2.pr_info.get("Occupancy", 0)
                empty6 = parking_manager_gym2.pr_info.get("Available", 0)
                parking_data["gym"]["camera_gym2"] = {"occupied": occ6, "available": empty6, "total": occ6 + empty6}
            else:
                processed_s6 = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(processed_s6, "Camera GYM 2 Offline", (100, FRAME_HEIGHT // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                parking_data["gym"]["camera_gym2"] = {"occupied": 0, "available": 0, "total": 0}    

            # Aggregate Lot Statistics
            t_occupied = parking_data["lot"]["camera_1"]["occupied"] + parking_data["lot"]["camera_2"]["occupied"]
            t_available = parking_data["lot"]["camera_1"]["available"] + parking_data["lot"]["camera_2"]["available"]
            parking_data["lot"]["total"] = {"occupied": t_occupied, "available": t_available, "total": t_occupied + t_available}

            # Aggregate AEB Statistics
            t_occupied_aeb = parking_data["aeb"]["camera_aeb1"]["occupied"] + parking_data["aeb"]["camera_aeb2"]["occupied"]
            t_available_aeb = parking_data["aeb"]["camera_aeb1"]["available"] + parking_data["aeb"]["camera_aeb2"]["available"]
            parking_data["aeb"]["total"] = {"occupied": t_occupied_aeb, "available": t_available_aeb, "total": t_occupied_aeb + t_available_aeb}

            # Aggregate GYM Statistics
            t_occupied_gym = parking_data["gym"]["camera_gym1"]["occupied"] + parking_data["gym"]["camera_gym2"]["occupied"]
            t_available_gym = parking_data["gym"]["camera_gym1"]["available"] + parking_data["gym"]["camera_gym2"]["available"]
            parking_data["gym"]["total"] = {"occupied": t_occupied_gym, "available": t_available_gym, "total": t_occupied_gym + t_available_gym}

            # Thread-safe copy of processed frames to global pointers
            with frame_lock:
                output_frames["camera_1"] = processed_s1.copy()
                output_frames["camera_2"] = processed_s2.copy()
                output_frames["camera_aeb1"] = processed_s3.copy()
                output_frames["camera_aeb2"] = processed_s4.copy()
                output_frames["camera_gym1"] = processed_s5.copy()
                output_frames["camera_gym2"] = processed_s6.copy()

            time.sleep(0.01)
    finally:
        stream1.release()
        stream2.release()
        stream3.release()
        stream4.release()
        stream5.release()
        stream6.release()

if __name__ == "__main__":
    print("Launching AI backend and background web services...")
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    main_cv_loop()