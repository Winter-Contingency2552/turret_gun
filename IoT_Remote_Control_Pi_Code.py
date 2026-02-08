import paho.mqtt.client as mqtt
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep
import time
import RPi.GPIO as GPIO
import struct
relay_ch = 26

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

GPIO.setup(relay_ch, GPIO.OUT)
# Create a PiGPIOFactory instance
factory = PiGPIOFactory()

# Use the factory when creating your servo
servo1 = AngularServo(18, min_angle=0, max_angle=180, min_pulse_width=0.0005, max_pulse_width=0.0025, pin_factory=factory)
servo2 = AngularServo(12, min_angle=0, max_angle=180, min_pulse_width=0.0005, max_pulse_width=0.0025, pin_factory=factory)
t=0
# Your servo control code here
def dual_servo_angle(angle):
    servo1.angle=90-angle
    servo2.angle=90+angle

# Global variable to store the current angle
current_angle = 0
dual_servo_angle(0)

def on_connect(client, userdata, flags, rc, properties=None):
    print("Connected successfully")
    client.subscribe("output_topic")

def on_message(client, userdata, msg):
    global current_angle
    message = msg.payload
    unpacked_data=struct.unpack('iii',message)
    print(f"Received message: {unpacked_data} on topic {msg.topic}")
    
    if message == 's':
        current_angle = max(-60, current_angle - 5)  # Decrease angle by 5 degrees, but not below 0
    elif message == 'w':
        current_angle = min(60, current_angle + 5)  # Increase angle by 5 degrees, but not above 180
    if message == 'f':
        GPIO.output(relay_ch, GPIO.HIGH)
        time.sleep(.5)
        GPIO.output(relay_ch, GPIO.LOW)
    else:
        GPIO.output(relay_ch, GPIO.LOW)
    
        
    dual_servo_angle(current_angle)
    print(f"Servo angle set to: {current_angle}")

client = mqtt.Client(protocol=mqtt.MQTTv5, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.username_pw_set("<user_name>", "<password>")
client.connect("IP adress>", <tcp port>, 60)

# Start the MQTT client loop in a non-blocking way
client.loop_start()

try:
    while True:
        sleep(0.1)  # Small delay to prevent CPU overuse
except KeyboardInterrupt:
    print("Exiting...")
    client.loop_stop()
    servo.close()
