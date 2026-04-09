import paho.mqtt.client as mqtt
import cv2
import numpy as np
import base64, io, os, re, json, time
from PIL import Image
import google.generativeai as genai

# =========================
# Config
# =========================
BROKER = "BROKER_IP"
PORT = 1883
TOPIC_VIDEO = "video/stream"
TOPIC_CMD = "rover/cmd"
TOPIC_OBS = "rover/obstacle"
SAVE_DIR = "motion_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# NOTE: API key is embedded directly in the code per user request.
# Replace the string below with your actual API key.
API_KEY = "API_KEY"
GEMINI_AVAILABLE = False
model = None
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
        print("✅ Gemini model ready")
        GEMINI_AVAILABLE = True
    except Exception as e:
        print("[Gemini] Initialization failed:", e)
        GEMINI_AVAILABLE = False
else:
    print("[Gemini] No API key set in code. Falling back to local planner.")

prev_frame = None
last_sent = 0.0
frame_count = 0
rover_has_obstacle = False
last_cmd_was_stop = False
last_local_override = 0.0

# throttle: minimum seconds between model invocations
MIN_INTERVAL = 0.2   # seconds (reduced so commands publish more frequently and avoid long gaps)

# max duration we allow a command to ask the rover to run (seconds)
MAX_CMD_DURATION = 1.0
# Local detection tuning
# Detections with estimated distance greater than this are ignored as "far" objects
PATH_MAX_CONSIDER_DISTANCE = 3.0  # meters (tune: lower -> more conservative)
# Minimum area ratio (bbox area / frame area) for on-path detections to be considered
MIN_ON_PATH_AREA_RATIO = 0.002

# =========================
# Helper Functions
# =========================
def decode_image(base64_bytes):
    img_bytes = base64.b64decode(base64_bytes)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def detect_motion(frame, prev_frame, threshold=6000):
    # If we don't have a previous frame, treat as motion so the pipeline records the first frame
    if prev_frame is None:
        return True

    # Guard against frames that differ in size or channel count (this caused the arithm_op error)
    try:
        if prev_frame.shape != frame.shape:
            # Different sizes or channels -> can't do pixel-wise diff safely. Treat as motion
            # and allow the caller to reset/replace the previous frame.
            print("[MotionDetect] frame size/channel mismatch; resetting previous frame")
            return True

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        return np.count_nonzero(mask) > threshold
    except Exception as e:
        # Catch any unexpected OpenCV errors and treat as motion (safer than crashing)
        print("[MotionDetect] error during diff:", e)
        return True

def get_prediction_from_gemini(frame):
    _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    img_bytes = buf.tobytes()
    prompt = """
You are an autonomous driving AI assistant. 
Respond ONLY in JSON (no extra explanation):
{
  "scene_description": "brief summary",
  "speed_m_s": <float>,
  "steering_deg": <float>,
  "distance_m": <float>
}
Rules:
- If path clear: speed 1–3 m/s, distance 0.5–3 m
- Turn ± steering_deg if needed (-35 to 35)
- If obstacle ahead: speed 0, distance 0
"""
    try:
        # If Gemini isn't available, use a local planner instead of calling API
        if not GEMINI_AVAILABLE or model is None:
            return local_fallback_planner(frame)

        # send prompt + image
        result = model.generate_content([prompt, {"mime_type":"image/jpeg","data":img_bytes}])
        return result.text or ""
    except Exception as e:
        # Log the error (API expired/invalid keys commonly show up here)
        print("[Gemini] Error:", e)
        # Fall back to local planner so rover remains operational
        try:
            return local_fallback_planner(frame)
        except Exception:
            return ""


    

def local_fallback_planner(frame):
    """Produce a small JSON decision when Gemini is unavailable.

    Heuristic: detect large foreground blobs (likely obstacle) -> stop.
    Otherwise produce a modest forward speed.
    """
    try:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, th = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = h * w
        large = [c for c in contours if cv2.contourArea(c) > area * 0.08]
        if len(large) > 0:
            data = {"scene_description": "local_obstacle_detected", "speed_m_s": 0.0, "steering_deg": 0.0, "distance_m": 0.0}
            return json.dumps(data)

        # Increase fallback speed so rover moves noticeably faster when Gemini is unavailable.
        data = {"scene_description": "local_clear", "speed_m_s": 2.0, "steering_deg": 0.0, "distance_m": 1.0}
        return json.dumps(data)
    except Exception as e:
        print("[LocalPlanner] error:", e)
        return json.dumps({"scene_description": "error", "speed_m_s": 0.0, "steering_deg": 0.0, "distance_m": 0.0})


# -------------------------
# Local detection utilities (top-level)
# -------------------------
# Background subtractor and HOG person detector to reduce false positives
_BG_SUB = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=True)
_HOG = cv2.HOGDescriptor()
try:
    _HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
except Exception:
    _HOG = None


def detect_objects_on_path(frame, roi_w=0.6, roi_h=0.5, min_area_ratio=0.001):
    """Detect moving objects and classify roughly as person/vehicle/object.

    Returns a list of dicts: {type, bbox(x,y,w,h), center_on_path(bool), est_distance_m(float)}
    """
    h, w = frame.shape[:2]
    frame_area = h * w

    # compute ROI (center bottom region)
    roi_x1 = int(w * (0.5 - roi_w / 2.0))
    roi_x2 = int(w * (0.5 + roi_w / 2.0))
    roi_y1 = int(h * (1.0 - roi_h))
    roi_y2 = h

    detections = []
    try:
        fg = _BG_SUB.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < frame_area * min_area_ratio:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            cx = x + ww // 2
            cy = y + hh // 2

            # classify roughly: height>width -> person-like, width>height -> vehicle-like
            if hh > ww * 1.1:
                typ = "person"
            elif ww > hh * 1.2:
                typ = "vehicle"
            else:
                typ = "object"

            # estimate distance from bounding-box height heuristically
            # mapping: est_m = (frame_height / bbox_height) * scale
            # choose scale so that bbox_height ~= frame_height/4 -> ~2.0m
            scale = 0.5
            est_distance = (h / float(hh)) * scale if hh > 0 else 99.0

            # determine if center lies inside the path ROI
            on_path = (cx >= roi_x1 and cx <= roi_x2 and cy >= roi_y1 and cy <= roi_y2)

            detections.append({
                "type": typ,
                "bbox": (int(x), int(y), int(ww), int(hh)),
                "center": (int(cx), int(cy)),
                "on_path": on_path,
                "est_distance_m": float(est_distance),
                "area": float(area),
            })

        # Also run HOG people detector to improve person detection confidence
        if _HOG is not None:
            rects, weights = _HOG.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
            for (x, y, ww, hh), wgt in zip(rects, weights):
                cx = x + ww // 2
                cy = y + hh // 2
                on_path = (cx >= roi_x1 and cx <= roi_x2 and cy >= roi_y1 and cy <= roi_y2)
                est_distance = (h / float(hh)) * 0.5 if hh > 0 else 99.0
                detections.append({
                    "type": "person",
                    "bbox": (int(x), int(y), int(ww), int(hh)),
                    "center": (int(cx), int(cy)),
                    "on_path": on_path,
                    "est_distance_m": float(est_distance),
                    "area": float(ww * hh),
                    "hog_score": float(wgt)
                })

    except Exception as e:
        print("[LocalDetect] error:", e)

    return detections

def parse_json_output(text):
    cleaned = re.sub(r"```json|```", "", text).strip()
    try:
        data = json.loads(cleaned)
    except:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return 0.0, 0.0, 0.0, text
        try:
            data = json.loads(m.group())
        except Exception:
            return 0.0, 0.0, 0.0, text
    return float(data.get("speed_m_s", 0.0)), float(data.get("steering_deg", 0.0)), float(data.get("distance_m", 0.0)), text

def to_pwm(speed, angle):
    # Map model speed (m/s) to PWM %.
    # The Gemini prompt expects speeds in roughly 1-3 m/s; map 0..MAX_MODEL_SPEED -> 0..100%
    MAX_MODEL_SPEED = 3.0  # m/s - tune this if your rover's top speed differs
    s = float(speed)
    # Convert to percentage of max speed
    base = (s / MAX_MODEL_SPEED) * 100.0
    # Steering angle contribution (keeps previous scaling)
    diff = (angle / 35.0) * 50.0
    left = np.clip(base + diff, -100, 100)
    right = np.clip(base - diff, -100, 100)
    return float(left), float(right)

def publish_cmd(client, l, r, d, duration_s):
    payload = json.dumps({
        "timestamp": int(time.time()*1000),
        "speed_left": l,
        "speed_right": r,
        "distance_m": d,
        "duration_s": duration_s
    })
    client.publish(TOPIC_CMD, payload, qos=1)
    print(f"📡 CMD → L={l:.1f}% R={r:.1f}% Dist={d:.2f}m Dur={duration_s:.2f}s")

# =========================
# MQTT
# =========================
def on_connect(client, userdata, flags, rc, properties=None):
    # Accept optional `properties` for newer paho-mqtt callback API (silences deprecation warnings)
    print("[MQTT] Connected rc=", rc)
    client.subscribe(TOPIC_VIDEO)
    client.subscribe(TOPIC_OBS)

def on_message(client, userdata, msg):
    global prev_frame, frame_count, last_sent, rover_has_obstacle
    global last_cmd_was_stop
    global last_local_override
    try:
        if msg.topic == TOPIC_OBS:
            try:
                d = json.loads(msg.payload.decode())
                rover_has_obstacle = bool(d.get("obstacle", False))
            except Exception:
                # tolerate simple boolean payloads
                try:
                    rover_has_obstacle = msg.payload.decode().strip().lower() in ("1", "true", "yes")
                except Exception:
                    rover_has_obstacle = False
            if rover_has_obstacle:
                print("⛔ Rover reports obstacle — will NOT query Gemini")
            return

        # video frame
        frame = decode_image(msg.payload)

        # If previous frame exists but has different shape, reset it so we don't try an invalid absdiff
        if prev_frame is not None and prev_frame.shape != frame.shape:
            prev_frame = None
        cv2.imshow("Rover Feed", frame)

        # If rover already reports obstacle, skip everything
        if rover_has_obstacle:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                client.disconnect(); cv2.destroyAllWindows()
            return

        # throttle model invocations
        if time.time() - last_sent < MIN_INTERVAL:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                client.disconnect(); cv2.destroyAllWindows()
            return

        if detect_motion(frame, prev_frame):
            frame_count += 1
            prev_frame = frame.copy()
            path = os.path.join(SAVE_DIR, f"frame_{frame_count}.jpg")
            cv2.imwrite(path, frame)
            print(f"\n⚡ Motion detected → {path}")

            # Run local object detection focused on the path
            detections = detect_objects_on_path(frame)
            # Filter detections: only keep ones on the path and not too far away
            on_path_dets = [d for d in detections if d.get("on_path")]
            # further filter by estimated distance and area to ignore far/small false positives
            frame_area = frame.shape[0] * frame.shape[1]
            filtered_on_path = []
            for d in on_path_dets:
                if d.get("est_distance_m", 999.0) > PATH_MAX_CONSIDER_DISTANCE:
                    continue
                if d.get("area", 0.0) < frame_area * MIN_ON_PATH_AREA_RATIO:
                    continue
                filtered_on_path.append(d)

            if len(filtered_on_path) > 0:
                print(f"[LocalDetect] {len(filtered_on_path)} filtered on-path detections:")
                for d in filtered_on_path:
                    print(f"  - {d['type']} @ {d['est_distance_m']:.2f}m bbox={d['bbox']}")

            # If we have an on-path object, override model commands with deterministic behavior
            now = time.time()
            if len(filtered_on_path) > 0 and now - last_local_override > 0.5:
                # pick nearest on-path object from filtered list
                nearest = min(filtered_on_path, key=lambda x: x.get("est_distance_m", 999))
                d_est = nearest.get("est_distance_m", 999.0)
                # If object is >= 2.0m away, instruct rover to move exactly 2.0m (straight)
                if d_est >= 2.0:
                    s = 1.5
                    a = 0.0
                    distance = 2.0
                    print(f"[Override] Object at {d_est:.2f}m on path: moving {distance}m then stop")
                    l, r = to_pwm(s, a)
                    publish_cmd(client, l, r, distance, float(min(MAX_CMD_DURATION, max(0.15, distance / max(s,0.05)))))
                    last_local_override = now
                    last_sent = now
                    last_cmd_was_stop = False
                    # skip calling Gemini for this frame
                    return
                else:
                    # object is closer than 2m -> stop immediately
                    print(f"[Override] Object at {d_est:.2f}m (closer than 2m): stopping")
                    publish_cmd(client, 0.0, 0.0, 0.0, 0.0)
                    last_local_override = now
                    last_sent = now
                    last_cmd_was_stop = True
                    return

            # if no local override, fall back to Gemini model (if available)
            start = time.time()
            raw = get_prediction_from_gemini(frame)
            s,a,distance, raw_text = parse_json_output(raw)
            latency = time.time() - start
            print(f"--- Gemini --- s={s:.2f}, a={a:.1f}, d={distance:.2f}, latency={latency:.2f}s\n{raw_text}\n")

            # compute duration: distance / speed (if speed > 0) but cap to MAX_CMD_DURATION
            # Ensure a small minimum duration to avoid immediate stop/gap between cmds
            duration_s = 0.0
            if s > 0.05 and distance > 0.01:
                # speed s is m/s from the model; protect division by zero and unrealistic numbers
                safe_speed = max(s, 0.05)
                duration_s = distance / safe_speed
                # clip duration to a short value so rover can be responsive
                duration_s = float(min(duration_s, MAX_CMD_DURATION))
                # enforce a small minimum duration to avoid short stop gaps
                duration_s = max(duration_s, 0.15)
            else:
                duration_s = 0.0

            # If model wants to stop (s small) then publish stop
            if s <= 0.05:
                publish_cmd(client, 0.0, 0.0, 0.0, 0.0)
                last_sent = time.time()
                last_cmd_was_stop = True
            else:
                # If the previous command was a stop (likely due to an obstacle),
                # force the first resumed command to have zero steering so the rover
                # moves straight instead of veering.
                if last_cmd_was_stop:
                    orig_a = a
                    a = 0.0
                    print(f"[CMD] Clearing steering after stop (was {orig_a}) to ensure straight resume")

                # convert to PWM % and send duration
                l, r = to_pwm(s, a)
                publish_cmd(client, l, r, distance, duration_s)
                last_sent = time.time()
                last_cmd_was_stop = False

        if cv2.waitKey(1) & 0xFF == ord('q'):
            client.disconnect(); cv2.destroyAllWindows()

    except Exception as e:
        print("❌ Frame error:", e)

# =========================
# Main
# =========================
def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    client.loop_forever()

if __name__ == "__main__":
    main()
