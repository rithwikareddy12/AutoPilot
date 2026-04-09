#!/usr/bin/env python3
import cv2, json, base64, time, threading, serial, os
import numpy as np
import paho.mqtt.client as mqtt

# =======================
# Config
# =======================
BROKER = "BROKER_IP"
PORT = 1883
TOPIC_VIDEO = "video/stream"
TOPIC_CMD = "rover/cmd"
TOPIC_STATUS = "rover/status"   # new status topic

DDSM_PORT = "/dev/ttyACM1"
LEFT_ID, RIGHT_ID = 1, 2
CAM_WIDTH, CAM_HEIGHT = 640, 480
FRAME_INTERVAL = 1.0

USE_YOLO = False
try:
    from ultralytics import YOLO
    USE_YOLO = True
except ImportError:
    pass

YOLO_PATH = "yolov8n.pt"
if USE_YOLO and os.path.exists(YOLO_PATH):
    model = YOLO(YOLO_PATH)
    print(f"[YOLO] Loaded {YOLO_PATH}")
else:
    print("[YOLO] Not found, using MOG2 fallback")
    model = cv2.createBackgroundSubtractorMOG2(300, 16, False)

# =======================
# Globals
# =======================
speed_left = 0
speed_right = 0
target_distance = 0
executing = False
obstacle_detected = False
last_cmd_time = time.time()
last_move_time = 0

# =======================
# Helper Functions
# =======================
def build_motor_command(mid, val):
    return {"T": 10010, "id": mid, "cmd": int(val), "act": 3}

def send_motor(ser, cmd):
    ser.write((json.dumps(cmd) + "\n").encode())

def stop_motors(ser):
    send_motor(ser, build_motor_command(LEFT_ID, 0))
    send_motor(ser, build_motor_command(RIGHT_ID, 0))
    print("🛑 Motors stopped")

def publish_status(client, status):
    payload = json.dumps({"timestamp": int(time.time()*1000), "status": status})
    client.publish(TOPIC_STATUS, payload, qos=1)

# =======================
# MQTT Callbacks
# =======================
def on_connect(c, u, f, rc, properties=None):
    # Accept optional `properties` argument for newer paho-mqtt versions
    print("[MQTT] Connected", rc)
    c.subscribe(TOPIC_CMD)

def on_message(c, u, msg):
    global speed_left, speed_right, target_distance, executing, last_cmd_time, obstacle_detected
    if obstacle_detected:
        print("🚫 Command ignored — obstacle present.")
        return
    try:
        data = json.loads(msg.payload.decode())
        speed_left = float(data.get("speed_left", 0))
        speed_right = float(data.get("speed_right", 0))
        target_distance = float(data.get("distance_m", 0))
        executing = (speed_left != 0 or speed_right != 0)
        last_cmd_time = time.time()
        print(f"[CMD] L={speed_left:.1f} R={speed_right:.1f} dist={target_distance:.1f}")
    except Exception as e:
        print("[CMD Parse Error]", e)

# =======================
# Obstacle Detection (Camera + LiDAR placeholder)
# =======================
def detect_obstacle(frame, lidar_distance=None):
    """
    Returns True if obstacle detected either by camera (YOLO/MOG2) or LiDAR distance.
    """
    # --- LiDAR check ---
    if lidar_distance is not None and lidar_distance < 0.5:  # meters
        return True

    # --- Camera check ---
    if USE_YOLO and os.path.exists(YOLO_PATH):
        results = model(frame, verbose=False)
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > 0.5:
                _, y1, _, y2 = map(int, box.xyxy[0])
                if y2 > frame.shape[0] * 0.6:  # close in view
                    return True
        return False
    else:
        mask = model.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        area = h * w
        # Use a lower area threshold but require the contour to appear in the lower part
        # of the image (likely the rover's path). This reduces false positives from
        # distant or side objects.
        large = []
        for c in contours:
            a = cv2.contourArea(c)
            if a <= area * 0.02:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            # require the bottom of the bbox to be in the lower 40% of the frame
            if (y + hh) >= int(h * 0.6):
                large.append(c)
        return len(large) > 0

# =======================
# Main Loop
# =======================
def main():
    global executing, obstacle_detected, last_move_time

    mqttc = mqtt.Client()
    mqttc.on_connect = on_connect
    mqttc.on_message = on_message
    mqttc.connect(BROKER, PORT, 60)
    mqttc.loop_start()

    print("[DDSM] Connecting...")
    ser = serial.Serial(DDSM_PORT, 115200, timeout=1)
    print("[DDSM] Connected")

    cap = cv2.VideoCapture(
        "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=%d,height=%d,framerate=30/1 ! "
        "nvvidconv flip-method=2 ! video/x-raw,format=BGRx ! videoconvert ! appsink"
        % (CAM_WIDTH, CAM_HEIGHT),
        cv2.CAP_GSTREAMER,
    )
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera failed.")
        return

    print("[Camera] Ready")
    last_send = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # --- Send video feed to Gemini ---
            if time.time() - last_send >= FRAME_INTERVAL:
                _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                mqttc.publish(TOPIC_VIDEO, base64.b64encode(buf))
                last_send = time.time()

            # --- Check obstacle (LiDAR optional integration placeholder) ---
            lidar_distance = None  # plug in your LiDAR read function here
            new_obstacle = detect_obstacle(frame, lidar_distance)

            # --- Obstacle reaction ---
            if new_obstacle:
                if not obstacle_detected:
                    print("🚨 Obstacle Detected — Stopping!")
                    stop_motors(ser)
                    publish_status(mqttc, "STOPPED_OBSTACLE")
                obstacle_detected = True
                executing = False
                time.sleep(0.2)
                continue
            else:
                if obstacle_detected:
                    print("✅ Path Cleared — Resuming Gemini commands.")
                    publish_status(mqttc, "PATH_CLEARED")
                obstacle_detected = False

            # --- Execute movement only if safe ---
            if executing and not obstacle_detected:
                send_motor(ser, build_motor_command(LEFT_ID, int(speed_left)))
                send_motor(ser, build_motor_command(RIGHT_ID, -int(speed_right)))
                print(f"[MOVE] L={speed_left:.1f}% R={speed_right:.1f}%")
                last_move_time = time.time()

            # --- Timeout safety ---
            # Only trigger the timeout stop if we were executing a command. This
            # prevents repeatedly calling stop_motors when no command is active.
            if executing and (time.time() - last_cmd_time > 10):
                stop_motors(ser)
                executing = False

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Manual Stop")
        stop_motors(ser)

    finally:
        cap.release()
        mqttc.loop_stop()
        ser.close()

if __name__ == "__main__":
    main()
