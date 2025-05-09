# main.py

import cv2
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from infer import YOLODetector
#from class_selector import get_classes_from_prompt

import uvicorn
import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import psutil
import logging
import queue
import multiprocessing
import requests
from datetime import datetime
from collections import defaultdict

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CPU Threads
TOTAL_CORES = multiprocessing.cpu_count()
NUM_THREADS = max(1, int(TOTAL_CORES * 0.75))
logger.info(f"Using {NUM_THREADS} threads out of {TOTAL_CORES} total cores")
thread_pool = ThreadPoolExecutor(max_workers=NUM_THREADS)

# FPS Control
MIN_FPS = 15
MAX_FPS = 60
CURRENT_FPS = 30
FRAME_INTERVAL = 1.0 / CURRENT_FPS
last_frame_time = time.time()
frame_count = 0
current_fps = 0

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

class FPSControl(BaseModel):
    target_fps: int

@app.post("/set-fps")
async def set_fps(control: FPSControl):
    global CURRENT_FPS, FRAME_INTERVAL
    if not MIN_FPS <= control.target_fps <= MAX_FPS:
        raise HTTPException(status_code=400, detail=f"FPS must be between {MIN_FPS} and {MAX_FPS}")
    CURRENT_FPS = control.target_fps
    FRAME_INTERVAL = 1.0 / CURRENT_FPS
    logger.info(f"Set FPS to {CURRENT_FPS}")
    return {
        "status": "success",
        "message": f"FPS set to {CURRENT_FPS}",
        "current_fps": current_fps,
        "cpu_usage": psutil.cpu_percent()
    }

@app.get("/fps-status")
async def get_fps_status():
    return {
        "current_fps": current_fps,
        "target_fps": CURRENT_FPS,
        "min_fps": MIN_FPS,
        "max_fps": MAX_FPS,
        "cpu_usage": psutil.cpu_percent()
    }

def update_fps():
    global last_frame_time, frame_count, current_fps
    frame_count += 1
    now = time.time()
    if now - last_frame_time >= 1.0:
        current_fps = frame_count
        frame_count = 0
        last_frame_time = now
        print(f"\rFPS: {current_fps} | CPU: {psutil.cpu_percent():.1f}% | Threads: {NUM_THREADS}/{TOTAL_CORES}", end="", flush=True)

@app.get("/")
async def root():
    return {"message": "Server is running!", "endpoints": ["/video_feed", "/health", "/latest-detections"]}

# Globals
yolo_detector = YOLODetector()
latest_detections = {"success": True, "detections": []}
latest_lock = threading.Lock()
video_frame_lock = threading.Lock()
video_frame = None
overlay_frame = None
overlay_lock = threading.Lock()

detection_history = defaultdict(float)
DETECTION_COOLDOWN = 5
WEBHOOK_URL = "http://majsssss.com/clip.mp4"

class ImageRequest(BaseModel):
    image_data: str


def process_frame_for_detection(frame):
    try:
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = yolo_detector.model(frame_rgb, conf=0.25, verbose=False)

        detections = []
        overlay = frame.copy()

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = yolo_detector.class_names[class_id]

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class": class_name,
                    "class_id": class_id
                })

                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(overlay, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                detection_key = class_name.lower()
                now = time.time()
                last_sent = detection_history[detection_key]

                if (now - last_sent > DETECTION_COOLDOWN):
                    #print(yolo_detector.target_classes)

                    if yolo_detector.target_classes != None:

                        if (class_name in yolo_detector.target_classes):
                            payload = {
                                "event": f"{class_name} detected",
                                "timestamp": datetime.now().isoformat(),
                                "feedId": "1",
                                "videoUrl": WEBHOOK_URL
                            }
                            try:
                                requests.post("http://172.160.225.181:7000/detection_output", json=payload, timeout=2)
                                detection_history[detection_key] = now
                                logger.info(f"Webhook sent for {detection_key}")
                            except Exception as e:
                                logger.error(f"Webhook error for {detection_key}: {e}")
                else:
                    logger.debug(f"Cooldown active for {detection_key}, skipping webhook")
        with latest_lock:
            global latest_detections
            latest_detections = {"success": True, "detections": detections}

        with overlay_lock:
            global overlay_frame
            overlay_frame = overlay

        return detections
    except Exception as e:
        logger.error(f"process_frame error: {e}")
        return None


def mjpeg_streamer():
    while True:
        try:
            with video_frame_lock:
                if video_frame is None:
                    continue
                frame = video_frame.copy()
            
            with overlay_lock:
                if overlay_frame is not None:
                    # Just use the overlay frame directly instead of blending
                    frame = overlay_frame

            # Increase JPEG quality and use better compression settings
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, 85,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ]
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            if not ret:
                continue
                
            # Add a small delay to maintain consistent frame rate
            time.sleep(FRAME_INTERVAL)
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            logger.error(f"Stream error: {e}")

def camera_thread():
    stream = "http://localhost:6969/video_feed"
    cap = cv2.VideoCapture(stream)
    if cap.isOpened():
        logger.info(f"Camera opened on {stream}")

    cap.release()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 120)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    detection_queue = queue.Queue(maxsize=5)

    def detection_worker():
        while True:
            try:
                frame = detection_queue.get_nowait()
                if frame is None:
                    break
                process_frame_for_detection(frame)
            except queue.Empty:
                time.sleep(0.001)
            except Exception as e:
                logger.error(f"detection_worker error: {e}")

    threading.Thread(target=detection_worker, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Camera read failed, retrying...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture("http://localhost:6969/video_feed")
            continue

        with video_frame_lock:
            global video_frame
            video_frame = frame

        try:
            detection_queue.put_nowait(frame)
        except queue.Full:
            continue

        update_fps()

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        mjpeg_streamer(),
        media_type='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Connection': 'keep-alive'
        }
    )

@app.on_event("startup")
def start_camera_thread():
    threading.Thread(target=camera_thread, daemon=True).start()
    logger.info("Camera thread started")

"""Lägger till detta för payloaden från frontend för prompt 
och detection mode. Detection mode används inte fullt just nu."""
class Payload(BaseModel):
    feed_id: str
    detection_mode: str
    prompt: str
    
@app.post("/recieve")
async def receive_json(payload: Payload):
    feed_id = payload.feed_id
    detection_mode = payload.detection_mode
    prompt = payload.prompt

    print(f"Feed ID: {feed_id}")
    print(f"Detection Mode: {detection_mode}")
    print(f"Prompt: {prompt}")
    yolo_detector.set_target_classes(str(prompt))

    return {
        "status": "ok",
        "target_classes": "ajajajajja",
        "detection_mode": detection_mode
    }

if __name__ == "__main__":
    print(f"\nStarting FastAPI server with {NUM_THREADS} threads ({NUM_THREADS/TOTAL_CORES*100:.1f}%)")
    uvicorn.run(app, host="0.0.0.0", port=4200)