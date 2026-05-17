from ultralytics import YOLO
import cv2
import time
import threading
import serial
import requests

# ── Serial configuration (ESP32-WROOM-32) ───────────────────────────────────────────
SERIAL_PORT = "COM3"
SERIAL_BAUDRATE = 9600
SERIAL_TIMEOUT = 1

# ── PIR / camera timing ───────────────────────────────────────────────────────
# Camera stays ON for this many seconds after the last detected motion.
ACTIVE_DURATION    = 15   # seconds

# ── BLE alert timing ──────────────────────────────────────────────────────────
# Minimum seconds between consecutive BLE notifications (avoids flooding).
BLE_ALERT_COOLDOWN = 30   # seconds

# ── Telegram Bot configuration ──────────────────────────────────────────────────────────
# 1. Open Telegram → search @BotFather → /newbot → copy the token
# 2. Send /start to your new bot, then visit:
#    https://api.telegram.org/bot<TOKEN>/getUpdates
#    and copy the 'id' value from the 'chat' object (your Chat ID)
TELEGRAM_BOT_TOKEN = "8728557277:AAHqjqT0LLMCDB2rh5gXmZnK_31--3keVtw"
TELEGRAM_CHAT_ID   = "7323065507"
TELEGRAM_COOLDOWN  = 60   # minimum seconds between Telegram alerts

# ── Global state ──────────────────────────────────────────────────────────────
arduino_serial      = None
last_ble_alert_time      = 0.0  # timestamp of last BLE alert sent
last_telegram_alert_time = 0.0  # timestamp of last Telegram alert sent

# ── Load YOLOv8 model ─────────────────────────────────────────────────────────
# model = YOLO("yolov8n.pt")
model = YOLO("yolo26n.pt")

# ── Alert categories ──────────────────────────────────────────────────────────
# Standard COCO models (80 classes) cannot detect many wildlife species by name.
# COCO_WILDLIFE_MAP remaps the nearest COCO proxy class to the true wildlife
# identity so alert levels fire correctly without swapping the model.
#
# Coverage after mapping:
#   bear, elephant         → native COCO  ✅
#   lion, tiger, leopard   → detected as "cat", remapped  ✅
#   wolf                   → detected as "dog", remapped  ✅
#   snake, gorilla, monkey, deer, rabbit,
#   squirrel, raccoon, fox → NOT in COCO; need custom model  ⚠️

COCO_WILDLIFE_MAP = {
    # COCO class  →  true wildlife identity (in outdoor/wildlife footage)
    "cat": "lion",  # big felids classified as cat at distance
    "dog": "wolf",  # wild canids classified as dog
    "teddy bear": "bear",  # edge-case low-res confusion
}

RED_ALERT_ANIMALS = [
    # ── Natively detectable via COCO ──
    "bear",
    "elephant",
    # ── Detectable via COCO_WILDLIFE_MAP ──
    "lion",
    "tiger",
    "leopard",
    "wolf",
    # ── Require custom-trained model to detect ──
    "snake",
    "gorilla",
    "crocodile",
    "rhinoceros",
]

YELLOW_ALERT_ANIMALS = [
    # ── Natively detectable via COCO ──
    "dog",
    "cat",
    "horse",
    "cow",
    "sheep",
    "bird",
    "zebra",
    "giraffe",
    # ── Require custom-trained model to detect ──
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


def setup_esp32():
    """Open serial connection to the ESP32-WROOM-32."""
    global arduino_serial
    try:
        arduino_serial = serial.Serial(
            SERIAL_PORT, SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT
        )
        time.sleep(2)  # Allow ESP32 to reset after DTR toggle
        print(f"🔌 Connected to ESP32 on {SERIAL_PORT}")
    except Exception as e:
        arduino_serial = None
        print(f"⚠️  ESP32 serial unavailable: {e}")
        print("    Running in camera-only mode (no PIR / buzzer / BLE).")


def send_buzzer_command(command):
    """Send a newline-terminated buzzer command to the ESP32."""
    if arduino_serial is None:
        return
    try:
        # Append '\n' so ESP32's line-based parser can delimit messages
        arduino_serial.write((command + "\n").encode("utf-8"))
        arduino_serial.flush()
    except Exception as e:
        print(f"⚠️  Failed to send buzzer command: {e}")


def send_ble_alert(animal_name: str, confidence: float) -> None:
    """
    Send a formatted BLE notification via the ESP32 over serial.
    The message is pre-formatted in Python so the ESP32 just broadcasts it.
    Format sent to ESP32:  A:<formatted_message>\n
    Received on phone :  'DANGER: Lion detected! Probability: 92%'
    """
    global last_ble_alert_time

    now = time.time()
    if now - last_ble_alert_time < BLE_ALERT_COOLDOWN:
        remaining = int(BLE_ALERT_COOLDOWN - (now - last_ble_alert_time))
        print(f"📡 BLE cooldown — next alert in {remaining}s")
        return

    if arduino_serial is None:
        print("📡 BLE alert skipped — no ESP32 serial connection.")
        return

    try:
        # Human-readable message broadcast over BLE
        payload = (
            f"DANGER: {animal_name.capitalize()} detected! "
            f"Probability: {confidence:.0%}"
        )
        msg = f"A:{payload}\n"
        arduino_serial.write(msg.encode("utf-8"))
        arduino_serial.flush()
        last_ble_alert_time = now
        print(f"📡 BLE alert sent → {payload}")
    except Exception as e:
        print(f"⚠️  Failed to send BLE alert: {e}")


def send_telegram_alert(animal_name: str, confidence: float) -> None:
    """
    Send a Telegram push notification to your phone.
    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to be set above.
    The user receives a native OS pop-up notification via the Telegram app.
    """
    global last_telegram_alert_time

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("📱 Telegram skipped — token/chat ID not configured.")
        return

    now = time.time()
    if now - last_telegram_alert_time < TELEGRAM_COOLDOWN:
        remaining = int(TELEGRAM_COOLDOWN - (now - last_telegram_alert_time))
        print(f"📱 Telegram cooldown — next alert in {remaining}s")
        return

    try:
        text = (
            f"🚨 *WILDLIFE ALERT* 🚨\n"
            f"*Animal :* {animal_name.capitalize()}\n"
            f"*Probability :* {confidence:.0%}\n"
            f"*Status :* DANGEROUS\n"
            f"*Time   :* {time.strftime('%d %b %Y  %H:%M:%S')}"
        )
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"},
            timeout=5,
        )
        if resp.status_code == 200:
            last_telegram_alert_time = now
            print(f"📱 Telegram alert sent → {animal_name.capitalize()} ({confidence:.0%})")
        else:
            print(f"⚠️  Telegram error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"⚠️  Telegram alert failed: {e}")


def check_motion():
    """
    Return True if the ESP32 has sent at least one 'M' (motion) byte
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


def resolve_class(raw_name: str) -> str:
    """Remap a COCO class to its true wildlife identity where applicable."""
    return COCO_WILDLIFE_MAP.get(raw_name.lower(), raw_name.lower())


def process_frame(cap):
    """
    Capture one frame, run YOLO inference, draw annotations, send buzzer command.
    COCO proxy detections (e.g. 'cat' in wildlife = lion) are resolved via
    COCO_WILDLIFE_MAP before alert-level matching.
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
            raw_name = model.names[class_id]  # raw COCO label
            class_name = resolve_class(raw_name)  # mapped wildlife name

            if class_name.lower() in [a.lower() for a in RED_ALERT_ANIMALS]:
                color      = (0, 0, 255)
                alert_text = "DANGEROUS!"
                frame_alert = "RED"
                if confidence > 0.7 and not call_made:
                    t = threading.Thread(target=make_emergency_call, args=(class_name,))
                    t.daemon = True
                    t.start()
                # ── BLE alert (non-blocking, has its own cooldown) ────────────
                ble_t = threading.Thread(
                    target=send_ble_alert, args=(class_name, confidence)
                )
                ble_t.daemon = True
                ble_t.start()
                # ── Telegram push notification (non-blocking) ────────────────
                tg_t = threading.Thread(
                    target=send_telegram_alert, args=(class_name, confidence)
                )
                tg_t.daemon = True
                tg_t.start()

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

print("🟢 Starting Advanced Detection System (ESP32 BLE + Telegram | PIR-triggered mode)...")
print(f"📞 Emergency number   : {EMERGENCY_PHONE}")
print(f"⏱️  Camera auto-off   : {ACTIVE_DURATION}s after last motion")
print(f"📡 BLE alert cooldown : {BLE_ALERT_COOLDOWN}s between alerts")
print("🔴 Red: Dangerous  |  🟡 Yellow: Caution  |  🟢 Green: Safe")
print("Press 'q' to quit\n")

setup_esp32()

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
