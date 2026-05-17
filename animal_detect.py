from ultralytics import YOLO
import cv2
import time
import threading
import serial

# Arduino serial configuration
SERIAL_PORT = 'COM3'  
SERIAL_BAUDRATE = 9600
SERIAL_TIMEOUT = 1

arduino_serial = None

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Open webcam
cap = cv2.VideoCapture(0)

# Dangerous animals (RED alert)
RED_ALERT_ANIMALS = [
    'bear', 'tiger', 'lion', 'leopard', 'wolf', 'crocodile', 
    'alligator', 'snake', 'shark', 'rhinoceros', 'hippopotamus',
    'elephant', 'gorilla', 'scorpion', 'spider'
]

# Less dangerous animals (YELLOW alert) 
YELLOW_ALERT_ANIMALS = [
    'dog', 'cat', 'horse', 'cow', 'sheep', 'bird', 'monkey',
    'deer', 'rabbit', 'squirrel', 'raccoon', 'fox'
]

# Emergency phone number
EMERGENCY_PHONE = "9663201915"

# Track if call was made
call_made = False
last_call_time = 0

def make_emergency_call(animal_name):
    """Function to call emergency number"""
    global call_made, last_call_time
    
    current_time = time.time()
    
    # Only call once every 2 minutes to avoid spam
    if current_time - last_call_time < 120:
        print(f"⚠️ Call cooldown active. Next call available in {int(120 - (current_time - last_call_time))} seconds")
        return
    
    print("🚨🚨🚨 EMERGENCY CALL INITIATED 🚨🚨🚨")
    print(f"📞 CALLING: {EMERGENCY_PHONE}")
    print(f"🐯 DANGEROUS ANIMAL DETECTED: {animal_name}")
    print(f"🕐 Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔊 Phone is ringing...")
    
    # In a real application, you would integrate with a calling API here
    # For demonstration, we'll just print the call action
    
    call_made = True
    last_call_time = current_time

def setup_arduino():
    """Initialize serial connection to Arduino."""
    global arduino_serial
    try:
        arduino_serial = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT)
        time.sleep(2)  # Allow Arduino to reset
        print(f"🔌 Connected to Arduino on {SERIAL_PORT}")
    except Exception as e:
        arduino_serial = None
        print(f"⚠️ Arduino serial unavailable: {e}")


def send_buzzer_command(command):
    """Send buzzer command to Arduino."""
    if arduino_serial is None:
        return
    try:
        arduino_serial.write(command.encode('utf-8'))
        arduino_serial.flush()
    except Exception as e:
        print(f"⚠️ Failed to send buzzer command: {e}")


def improve_night_vision(frame):
    """Enhance frame for night vision"""
    # Increase brightness and contrast
    alpha = 1.5  # Contrast control
    beta = 30    # Brightness control
    
    enhanced = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # Apply Gaussian blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    return enhanced

print("🟢 Starting Advanced Detection System...")
print(f"📞 Emergency number: {EMERGENCY_PHONE}")
print("🔴 Red: Dangerous animals")
print("🟡 Yellow: Less dangerous animals") 
print("🟢 Green: Other objects")
print("Press 'q' to quit")

setup_arduino()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply night vision enhancement
    enhanced_frame = improve_night_vision(frame)
    
    # Run YOLO detection
    results = model(enhanced_frame)
    
    # Track highest alert level in this frame
    frame_alert = None

    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get class and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            
            # Determine alert level and color
            if class_name.lower() in [animal.lower() for animal in RED_ALERT_ANIMALS]:
                color = (0, 0, 255)  # RED - Dangerous
                alert_text = "DANGEROUS!"
                frame_alert = 'RED'
                
                # Make emergency call for dangerous animals (high confidence)
                if confidence > 0.7 and not call_made:
                    call_thread = threading.Thread(target=make_emergency_call, args=(class_name,))
                    call_thread.daemon = True
                    call_thread.start()
                    
            elif class_name.lower() in [animal.lower() for animal in YELLOW_ALERT_ANIMALS]:
                color = (0, 255, 255)  # YELLOW - Less dangerous
                alert_text = "CAUTION"
                if frame_alert != 'RED':
                    frame_alert = 'YELLOW'
            else:
                color = (0, 255, 0)  # GREEN - Other objects
                alert_text = "SAFE"
            
            # Draw bounding box
            cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), color, 3)
            
            # Create label with confidence
            label = f"{alert_text}: {class_name} {confidence:.2f}"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(enhanced_frame, (x1, y1 - 30), (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(enhanced_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Send buzzer command for this frame
    if frame_alert == 'RED':
        send_buzzer_command('R')
    elif frame_alert == 'YELLOW':
        send_buzzer_command('Y')
    else:
        send_buzzer_command('S')

    # Add status information to frame
    status_text = f"Night Vision: ON | Emergency: {EMERGENCY_PHONE}"
    cv2.putText(enhanced_frame, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show call status
    if call_made:
        call_status = f"🚨 CALL MADE TO {EMERGENCY_PHONE}"
        cv2.putText(enhanced_frame, call_status, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow('🚨 Advanced Animal Detection System', enhanced_frame)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
send_buzzer_command('S')
if arduino_serial is not None:
    arduino_serial.close()

cap.release()
cv2.destroyAllWindows()