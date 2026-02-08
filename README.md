# Autonomous & IoT Sentry Turret

A dual-mode sentry turret powered by a Raspberry Pi. This system features an autonomous face-tracking mode using ONNX computer vision and a manual IoT override mode controllable via MQTT protocol.

## Features

* **Autonomous Tracking:** Real-time face detection using an ONNX model.
* **Motion Control:**
    * **Yaw:** Stepper motor driven by Adafruit Motor Hat.
    * **Pitch:** Dual angular servos for elevation control.
* **Targeting Logic:** Proportional control loop with coordinate smoothing (Exponential Moving Average) to center targets.
* **Firing Mechanism:** Relay-controlled automated firing when the target is locked.
* **IoT Manual Override:** Remote control capability over Wi-Fi using MQTT.

## Hardware Requirements

* **Raspberry Pi 4B** (Running Raspberry Pi OS Bullseye/Bookworm)
* **Adafruit Motor Hat** (for Stepper Motor)
* **Stepper Motor** (NEMA 17 or similar)
* **Servo Motors (x2)** (Standard 180 servos)
* **USB Webcam**
* **5V Relay Module** (for firing mechanism)
* **Gel Blaster** (or similar projectile device)

### Wiring Configuration

| Component | Pin / Port | Notes |
| :--- | :--- | :--- |
| **Servo 1** | GPIO 12 | Pitch Control (Left) |
| **Servo 2** | GPIO 18 | Pitch Control (Right) |
| **Stepper** | Motor Hat (I2C) | Controls Yaw |
| **Relay** | GPIO 19 | Trigger (Autonomous Mode) |
| **Relay** | GPIO 26 | Trigger (Manual/IoT Mode) |

Note: The current code uses GPIO 19 for the relay in Autonomous mode and GPIO 26 in Manual mode. Ensure your wiring matches the mode you are running, or update the relay_ch variable in the scripts to match your physical setup.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/sentry-turret.git](https://github.com/Winter-Contingency2552/turret_gun]
    ```

2.  **Install System Dependencies:**
    You need the pigpio daemon for smooth servo jitter reduction.
    ```bash
    sudo apt-get update
    sudo apt-get install pigpio python3-pigpio
    sudo systemctl enable pigpiod
    sudo systemctl start pigpiod
    ```

3.  **Install Python Libraries:**
    Create a virtual environment (recommended) and install the requirements.
    ```bash
    pip3 install opencv-python numpy onnxruntime gpiozero RPi.GPIO adafruit-circuitpython-motorkit adafruit-circuitpython-motor paho-mqtt
    ```

4.  **Model Weights:**
    Ensure you have the face detection model (version-RFB-320-int8.onnx) downloaded and update the path in autonomous_main.py:
    ```python
    self.face_detector = ort.InferenceSession("/path/to/your/model.onnx")
    ```

## Usage

The system operates in two distinct modes.

### 1. Autonomous Mode
Run this script on the Raspberry Pi. It initializes the camera, tracks faces, and fires automatically when the target is centered.

```bash
python3 autonomous_main.py
```
Controls: Press 'q' to quit the video feed window.

2. Manual IoT Mode
This requires two scripts running simultaneously: one on the robot (Subscriber) and one on your computer (Publisher).

Step A: Start the Turret Receiver (On Raspberry Pi) This script listens for MQTT commands to move the servos and fire the relay.

```bash
python3 turret_receiver.py
```
Step B: Start the Remote Controller (On Laptop/PC) This script captures keyboard input and sends it to the turret.

```bash
python3 remote_controller.py
```
## Controls

- **`w` / `s`**: Pitch Up / Down
- **`a` / `d`**: Turn Left / Right 
- **`f`**: Fire Relay  
- **`exit`**: Close connection  

---

## Code Structure

- **`autonomous_main.py`** *(FaceTrackingSystem)*  
  Contains the main class for CV logic, coordinate smoothing, and autonomous motor driving.

- **`remote_controller.py`** *(MQTT Publisher)*  
  The client-side script that takes user input and publishes it to the broker.

- **`turret_receiver.py`** *(MQTT Subscriber)*  
  The robot-side script that processes incoming MQTT messages and actuates the hardware manually.

---

## Tuning

You may need to adjust the following variables in `autonomous_main.py` to fit your specific build:

- **`self.stepper_speed`**: Adjust for faster or slower rotation.


## Liscense
This model (version‑RFB‑320‑int8.onnx) is redistributed under the MIT License (copyright from the original authors).
Source: [https://huggingface.co/onnxmodelzoo/version‑RFB‑320](https://huggingface.co/onnxmodelzoo/version-RFB-320)
