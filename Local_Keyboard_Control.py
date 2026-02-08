import cv2
import numpy as np
import threading
import time
import RPi.GPIO as GPIO
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep
from adafruit_motor import stepper
from adafruit_motorkit import MotorKit
import board
class TurretGun:
    def __init__(self):
        # GPIO and Servo Setup
        self.relay_ch = 19
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.relay_ch, GPIO.OUT)
        self.angle = 0
        # Create PiGPIOFactory instance
        factory = PiGPIOFactory()
        self.kit = MotorKit(i2c=board.I2C())
        # Servo Setup
        self.servo1 = AngularServo(18, min_angle=0, max_angle=180, 
                                   min_pulse_width=0.0005, max_pulse_width=0.0025, 
                                   pin_factory=factory)
        self.servo2 = AngularServo(12, min_angle=0, max_angle=180, 
                                   min_pulse_width=0.0005, max_pulse_width=0.0025, 
                                   pin_factory=factory)

        # State Management
        self.running = True
        self.relay_on = False

        # Camera Setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            self.running = False

        # Get camera properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {self.width}x{self.height}")

    def add_crosshair_and_elements(self, frame, color=(0, 0, 0), thickness=2):
        """Add crosshair, center circle, and elevation marks to frame"""
        height, width = frame.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Extended crosshair lines
        cv2.line(frame, (0, center_y), (width, center_y), color, thickness)
        cv2.line(frame, (center_x, 0), (center_x, height), color, thickness)
        
        # Circle
        radius = min(width, height) // 4
        cv2.circle(frame, (center_x, center_y), radius, color, thickness)
        
        # Elevation marks
        mark_length = 20
        num_marks = 8
        for i in range(num_marks):
            angle = i * (360 / num_marks)
            radian = np.radians(angle)
            start_x = int(center_x + (radius + 5) * np.cos(radian))
            start_y = int(center_y + (radius + 5) * np.sin(radian))
            end_x = int(center_x + (radius + mark_length) * np.cos(radian))
            end_y = int(center_y + (radius + mark_length) * np.sin(radian))
            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)
        
        return frame

    def dual_servo_angle(self, angle):
        """Set both servos to a synchronized angle"""
        self.servo1.angle = 90 - angle
        self.servo2.angle = 90 + angle

    def servo_control(self):
        """Thread for managing servo and relay control"""
        while self.running:
            # Default servo position
            self.dual_servo_angle(self.angle)
            
            # Manage relay state
            GPIO.output(self.relay_ch, GPIO.HIGH if self.relay_on else GPIO.LOW)
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.1)

    def display_webcam(self):
        """Handle webcam display and user input"""
        cv2.namedWindow('Turret Control', cv2.WINDOW_NORMAL)
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Add crosshair and elements
                frame_with_elements = self.add_crosshair_and_elements(frame)
                
                # Add relay status text
                relay_status = "Relay: ON" if self.relay_on else "Relay: OFF"
                cv2.putText(frame_with_elements, relay_status, 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Turret Control', frame_with_elements)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('f'):
                    self.relay_on = not self.relay_on
                elif key == ord('s'):
                    self.angle += 2
                elif key == ord('w'):
                    self.angle -= 2
                elif key == ord('a'):
                    self.kit.stepper2.onestep(style=stepper.DOUBLE)
                    time.sleep(0.05)            
                elif key == ord('d'):
                    self.kit.stepper2.onestep(direction=stepper.BACKWARD, style=stepper.DOUBLE)
                    time.sleep(0.05)
        
        except Exception as e:
            print(f"Error in display thread: {e}")
        
        finally:
            # Cleanup resources
            self.cap.release()
            cv2.destroyAllWindows()

    def run(self):
        """Start threads and manage overall execution"""
        if not self.running:
            print("Cannot run - initialization failed")
            return

        try:
            # Create threads
            servo_thread = threading.Thread(target=self.servo_control)
            display_thread = threading.Thread(target=self.display_webcam)

            # Start threads
            servo_thread.start()
            display_thread.start()

            # Wait for threads to complete
            display_thread.join()
            servo_thread.join()

        except Exception as e:
            print(f"Run error: {e}")
        
        finally:
            # Ensure cleanup
            GPIO.cleanup()

# Main execution
if __name__ == "__main__":
    turret = TurretGun()
    turret.run()
