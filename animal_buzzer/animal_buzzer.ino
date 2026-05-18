// ── Wildlife Alert System — ESP32-WROOM-32 ────────────────────────────────────
//
// Responsibilities:
//   • PIR motion sensor (GPIO 13) → sends 'M' over serial to Python
//   • Passive buzzer     (GPIO 15) ← receives R / Y / S  commands from Python
//   • BLE server (GATT)            ← receives A:<animal>:<conf> from Python
//     → notifies every connected BLE client with the alert string
//
// Serial protocol (all messages newline-terminated '\n'):
//   Python → ESP32 :  "R\n"               red buzzer
//                     "Y\n"               yellow buzzer
//                     "S\n"               stop buzzer
//                     "A:lion:0.92\n"     BLE danger alert
//   ESP32  → Python:  'M'                 motion detected (single byte, no '\n')
//
// BLE client setup (nRF Connect app):
//   1. Scan → tap "WildlifeAlert" → Connect
//   2. Expand the service  →  find characteristic "WildlifeAlert"
//   3. Tap the  ↓  (down-arrow / Subscribe) icon  → notifications enabled
//   4. When Python detects danger you receive e.g.  "DANGER! lion:0.92"
//
// ─────────────────────────────────────────────────────────────────────────────

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <BLEDescriptor.h>

// ── Hardware pins ─────────────────────────────────────────────────────────────
static const int BUZZER_PIN = 2;
static const int PIR_PIN    = 4;
static const int LED_PIN    = 13;

// ── PIR timing ────────────────────────────────────────────────────────────────
static unsigned long lastMotionSend  = 0;
static const unsigned long MOTION_MS = 500;

// ── BLE UUIDs ─────────────────────────────────────────────────────────────────
// 128-bit custom UUIDs — unique to this project
#define BLE_DEVICE_NAME     "WildlifeAlert"
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

static BLECharacteristic* pCharacteristic = nullptr;
static bool               clientConnected = false;

// ── BLE connection callbacks ──────────────────────────────────────────────────
class BLEEvents : public BLEServerCallbacks {
    void onConnect(BLEServer* /*pSrv*/) override {
        clientConnected = true;
        Serial.println("BLE: client connected");
    }

    void onDisconnect(BLEServer* pSrv) override {
        clientConnected = false;
        Serial.println("BLE: client disconnected — restarting advertising");
        // Restart advertising so a new client can connect
        BLEDevice::startAdvertising();
    }
};

// ── Serial input buffer ───────────────────────────────────────────────────────
static String rxBuffer = "";

// ─────────────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(9600);
    pinMode(BUZZER_PIN, OUTPUT);
    pinMode(PIR_PIN,    INPUT);
    pinMode(LED_PIN,    OUTPUT);
    noTone(BUZZER_PIN);
    digitalWrite(LED_PIN, LOW);

    // ── BLE initialisation ─────────────────────────────────────────────────────
    BLEDevice::init(BLE_DEVICE_NAME);

    BLEServer*  pServer  = BLEDevice::createServer();
    pServer->setCallbacks(new BLEEvents());

    BLEService* pService = pServer->createService(SERVICE_UUID);

    // ── Characteristic with NOTIFY + INDICATE ─────────────────────────────────
    // Using both NOTIFY and INDICATE ensures nRF Connect shows the ↓ Subscribe
    // button regardless of which property the client prefers.
    pCharacteristic = pService->createCharacteristic(
        CHARACTERISTIC_UUID,
        BLECharacteristic::PROPERTY_READ    |
        BLECharacteristic::PROPERTY_NOTIFY  |
        BLECharacteristic::PROPERTY_INDICATE
    );

    // ── CCCD descriptor (0x2902) ──────────────────────────────────────────────
    // Required for notifications to work. Pre-enable both notify and indicate
    // so the client sees Subscribe immediately without needing to write to CCCD.
    BLE2902* pCCCD = new BLE2902();
    pCCCD->setNotifications(true);
    pCCCD->setIndications(true);
    pCharacteristic->addDescriptor(pCCCD);

    // ── User Description descriptor (0x2901) ──────────────────────────────────
    // Labels the characteristic in nRF Connect — makes it easy to identify.
    BLEDescriptor* pUserDesc = new BLEDescriptor(BLEUUID((uint16_t)0x2901));
    pUserDesc->setValue("WildlifeAlert");
    pCharacteristic->addDescriptor(pUserDesc);

    // Set initial readable value before first alert
    pCharacteristic->setValue("SYSTEM_READY");

    pService->start();

    // ── Advertising ───────────────────────────────────────────────────────────
    BLEAdvertising* pAdv = BLEDevice::getAdvertising();
    pAdv->addServiceUUID(SERVICE_UUID);
    pAdv->setScanResponse(true);
    pAdv->setMinPreferred(0x06);   // helps with iPhone connection stability
    pAdv->setMaxPreferred(0x12);
    BLEDevice::startAdvertising();

    Serial.println("BLE: WildlifeAlert advertising...");
    Serial.println("     Open nRF Connect → Scan → WildlifeAlert → Connect");
    Serial.println("     Tap the characteristic → press ↓ to Subscribe");
}

// ─────────────────────────────────────────────────────────────────────────────
// Handle one complete newline-terminated command received from Python
// ─────────────────────────────────────────────────────────────────────────────
void handleCommand(const String& cmd) {

    if (cmd.startsWith("A:")) {
        // ── BLE danger alert ─────────────────────────────────────────────────
        // Format: "A:lion:0.92"  →  broadcast "DANGER! lion:0.92" via BLE
        String payload = "DANGER! " + cmd.substring(2);
        pCharacteristic->setValue(payload.c_str());
        pCharacteristic->notify();      // pushes to NOTIFY subscribers
        pCharacteristic->indicate();    // pushes to INDICATE subscribers
        Serial.print("BLE alert: ");
        Serial.println(payload);

    } else if (cmd == "R") {
        digitalWrite(LED_PIN, HIGH);
        tone(BUZZER_PIN, 1000);
        delay(250);
        noTone(BUZZER_PIN);

    } else if (cmd == "Y") {
        digitalWrite(LED_PIN, HIGH);
        tone(BUZZER_PIN, 800);
        delay(120);
        noTone(BUZZER_PIN);

    } else if (cmd == "S") {
        noTone(BUZZER_PIN);
        digitalWrite(LED_PIN, LOW);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void loop() {

    // ── PIR: relay motion to Python ───────────────────────────────────────────
    if (digitalRead(PIR_PIN) == HIGH) {
        unsigned long now = millis();
        if (now - lastMotionSend >= MOTION_MS) {
            Serial.write('M');
            lastMotionSend = now;
        }
    }

    // ── Serial: accumulate chars and dispatch on newline ──────────────────────
    while (Serial.available() > 0) {
        char c = Serial.read();
        if (c == '\n') {
            rxBuffer.trim();
            if (rxBuffer.length() > 0) {
                handleCommand(rxBuffer);
            }
            rxBuffer = "";
        } else if (c != '\r') {
            rxBuffer += c;
        }
    }
}
