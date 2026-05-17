from ultralytics import YOLO
import cv2
import time
import threading
import serial

# ── Serial configuration (ESP8266) ───────────────────────────────────────────
SERIAL_PORT = "COM3"
SERIAL_BAUDRATE = 9600
SERIAL_TIMEOUT = 1

# ── PIR / camera timing ───────────────────────────────────────────────────────
# Camera stays ON for this many seconds after the last detected motion.
ACTIVE_DURATION = 15  # seconds

# ── Global state ──────────────────────────────────────────────────────────────
arduino_serial = None

# ── Load YOLOv8 model ─────────────────────────────────────────────────────────
model = YOLO("yolov8n.pt")

# ── Alert categories ──────────────────────────────────────────────────────────
RED_ALERT_ANIMALS = [
    "bear",
    "tiger",
    "lion",
    "leopard",
    "wolf",
    "snake",
    "elephant",
    "gorilla",
]

YELLOW_ALERT_ANIMALS = [
    "dog",
    "cat",
    "horse",
    "cow",
    "sheep",
    "bird",
    "monkey",
    "deer",
    "rabbit",
    "squirrel",
    "raccoon",
    "fox",
]

# ── Emergency contact ─────────────────────────────────────────────────────────
EMERGENCY_PHONE = "9663201915"

call_made = False
last_call_time = 0


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────


def make_emergency_call(animal_name):
    """Trigger an emergency call (runs in a background thread)."""
    global call_made, last_call_time

    current_time = time.time()

    # Only call once every 2 minutes to avoid spam
    if current_time - last_call_time < 120:
        print(
            f"⚠️  Call cooldown active. "
            f"Next call in {int(120 - (current_time - last_call_time))}s"
        )
        return

    print("🚨🚨🚨 EMERGENCY CALL INITIATED 🚨🚨🚨")
    print(f"📞 CALLING: {EMERGENCY_PHONE}")
    print(f"🐯 DANGEROUS ANIMAL DETECTED: {animal_name}")
    print(f"🕐 Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔊 Phone is ringing...")

    call_made = True
    last_call_time = current_time


def setup_esp8266():
    """Open serial connection to the ESP8266."""
    global arduino_serial
    try:
        arduino_serial = serial.Serial(
            SERIAL_PORT, SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT
        )
        time.sleep(2)  # Allow ESP8266 to reset after DTR toggle
        print(f"🔌 Connected to ESP8266 on {SERIAL_PORT}")
    except Exception as e:
        arduino_serial = None
        print(f"⚠️  ESP8266 serial unavailable: {e}")
        print("    Running in camera-only mode (no PIR / buzzer).")


def send_buzzer_command(command):
    """Send a single-character buzzer command to the ESP8266."""
    if arduino_serial is None:
        return
    try:
        arduino_serial.write(command.encode("utf-8"))
        arduino_serial.flush()
    except Exception as e:
        print(f"⚠️  Failed to send buzzer command: {e}")


def check_motion():
    """
    Return True if the ESP8266 has sent at least one 'M' (motion) byte
    since the last check.  Drains the entire input buffer each call so
    bytes don't pile up.
    """
    if arduino_serial is None:
        return False
    try:
        waiting = arduino_serial.in_waiting
        if waiting > 0:
            data = arduino_serial.read(waiting)
            return b"M" in data
    except Exception as e:
        print(f"⚠️  Serial read error: {e}")
    return False


def improve_night_vision(frame):
    """Brighten and denoise a frame for better low-light detection."""
    enhanced = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)
    enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return enhanced


def process_frame(cap):
    """
    Capture one frame, run YOLO, draw annotations, send buzzer command.
    Returns the annotated frame, or None if the camera read failed.
    """
    ret, frame = cap.read()
    if not ret:
        return None

    enhanced_frame = improve_night_vision(frame)
    results = model(enhanced_frame)
    frame_alert = None

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            if class_name.lower() in [a.lower() for a in RED_ALERT_ANIMALS]:
                color = (0, 0, 255)
                alert_text = "DANGEROUS!"
                frame_alert = "RED"
                if confidence > 0.7 and not call_made:
                    t = threading.Thread(target=make_emergency_call, args=(class_name,))
                    t.daemon = True
                    t.start()

            elif class_name.lower() in [a.lower() for a in YELLOW_ALERT_ANIMALS]:
                color = (0, 255, 255)
                alert_text = "CAUTION"
                if frame_alert != "RED":
                    frame_alert = "YELLOW"

            else:
                color = (0, 255, 0)
                alert_text = "SAFE"

            # Bounding box
            cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), color, 3)

            # Label
            label = f"{alert_text}: {class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(
                enhanced_frame, (x1, y1 - 30), (x1 + label_size[0], y1), color, -1
            )
            cv2.putText(
                enhanced_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    # Send buzzer command
    if frame_alert == "RED":
        send_buzzer_command("R")
    elif frame_alert == "YELLOW":
        send_buzzer_command("Y")
    else:
        send_buzzer_command("S")

    # Overlay: status line
    cv2.putText(
        enhanced_frame,
        f"Night Vision: ON | Emergency: {EMERGENCY_PHONE}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Overlay: call status
    if call_made:
        cv2.putText(
            enhanced_frame,
            f"CALL MADE TO {EMERGENCY_PHONE}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return enhanced_frame


# ─────────────────────────────────────────────────────────────────────────────
# Main — PIR-triggered state machine
# ─────────────────────────────────────────────────────────────────────────────

print("🟢 Starting Advanced Detection System (PIR-triggered mode)...")
print(f"📞 Emergency number  : {EMERGENCY_PHONE}")
print(f"⏱️  Camera auto-off  : {ACTIVE_DURATION}s after last motion")
print("🔴 Red: Dangerous  |  🟡 Yellow: Caution  |  🟢 Green: Safe")
print("Press 'q' to quit\n")

setup_esp8266()

# States: 'IDLE' → camera off, waiting for PIR
#         'ACTIVE' → camera on, running detection
STATE = "IDLE"
cap = None
last_motion_time = 0.0

print("🟡 IDLE — Camera OFF. Waiting for motion from PIR sensor...")

try:
    while True:
        # ── IDLE STATE ────────────────────────────────────────────────────────
        if STATE == "IDLE":
            if check_motion():
                print("🟢 Motion detected! Activating camera...")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("❌ Failed to open camera. Remaining in IDLE.")
                    cap = None
                else:
                    last_motion_time = time.time()
                    STATE = "ACTIVE"
                    print("📷 Camera ON — Detection running.")
            else:
                time.sleep(0.1)  # Avoid busy-wait; ~10 checks per second

        # ── ACTIVE STATE ──────────────────────────────────────────────────────
        elif STATE == "ACTIVE":
            # Refresh timer on continued/new motion
            if check_motion():
                last_motion_time = time.time()

            # Auto-off check
            elapsed = time.time() - last_motion_time
            time_left = max(0, int(ACTIVE_DURATION - elapsed))

            if elapsed > ACTIVE_DURATION:
                print(f"🟡 No motion for {ACTIVE_DURATION}s — Shutting camera down.")
                send_buzzer_command("S")
                cap.release()
                cv2.destroyAllWindows()
                cap = None
                STATE = "IDLE"
                print("🟡 IDLE — Camera OFF. Waiting for motion from PIR sensor...")
                continue

            # Run detection on this frame
            annotated = process_frame(cap)
            if annotated is None:
                print("⚠️  Camera read failed. Returning to IDLE.")
                cap.release()
                cv2.destroyAllWindows()
                cap = None
                STATE = "IDLE"
                print("🟡 IDLE — Camera OFF. Waiting for motion from PIR sensor...")
                continue

            # Countdown overlay
            cv2.putText(
                annotated,
                f"Auto-off in: {time_left}s",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2,
            )

            cv2.imshow("🚨 Advanced Animal Detection System", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("🔴 Quit requested by user.")
                break

finally:
    # ── Cleanup ───────────────────────────────────────────────────────────────
    send_buzzer_command("S")
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    if arduino_serial is not None:
        arduino_serial.close()
    print("✅ System shut down cleanly.")
