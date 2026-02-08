from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep
import RPi.GPIO as GPIO
import cv2
import numpy as np
import onnxruntime as ort
from collections import deque
import board
from adafruit_motor import stepper
from adafruit_motorkit import MotorKit
import threading
import time
from queue import Queue, Empty

class FaceTrackingSystem:
    def __init__(self):
        # Initialize GPIO
        self.relay_ch = 19
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.relay_ch, GPIO.OUT)

        # Initialize servos
        factory = PiGPIOFactory()
        self.servo1 = AngularServo(12, min_angle=0, max_angle=180, 
                                 min_pulse_width=0.0005, max_pulse_width=0.0025, 
                                 pin_factory=factory)
        self.servo2 = AngularServo(18, min_angle=0, max_angle=180, 
                                 min_pulse_width=0.0005, max_pulse_width=0.0025, 
                                 pin_factory=factory)
        
        # Initialize stepper motor and control queue
        self.kit = MotorKit(i2c=board.I2C())
        self.stepper_queue = Queue(maxsize= 5)
        self.running = True
        self.stepper_speed=0.05
        self.stepper_in_zone = False         
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")

        # Initialize face detector
        self.face_detector = ort.InferenceSession("turret_gun/version-RFB-320-int8.onnx")
        
        # Target zone for relay control
        self.target_zone_size = 0.10
        self.relay_state = False

        # Center servos at startup
        self.servo1.angle = 90
        self.servo2.angle = 90
        self.last_angle=90
        self.max_angle=5
        # Initialize coordinate averaging
        self.buffer_size_y = 5
        self.buffer_size_x = 5
    # Initialize buffers with center coordinates
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        center_x = self.width // 2
        center_y = self.height // 2
        self.x_coords = deque([center_x] * self.buffer_size_x, maxlen=self.buffer_size_x)
        self.y_coords = deque([center_y] * self.buffer_size_y, maxlen=self.buffer_size_y)
        self.weights_x = np.array([0.1 * (i + 1) for i in range(self.buffer_size_x)])
        self.weights_x = self.weights_x / np.sum(self.weights_x)
        self.weights_y = np.array([0.1 * (i + 1) for i in range(self.buffer_size_y)])
        self.weights_y = self.weights_y / np.sum(self.weights_y)        
        # Start stepper control thread
        self.stepper_thread = threading.Thread(target=self._stepper_worker, daemon=True)
        self.stepper_thread.start()

    def _stepper_worker(self):
        while self.running:
            try:
                target_angle_x = self.stepper_queue.get(timeout=1.0)
                
                steps = abs(int(target_angle_x / 1.8))
                step_delay = 0.01  # Consistent, short delay
                
                for i in range(steps):
                    if target_angle_x < 0:
                        self.kit.stepper2.onestep(style=stepper.INTERLEAVE)
                    else:
                        self.kit.stepper2.onestep(direction=stepper.BACKWARD, style=stepper.INTERLEAVE)
                    sleep(step_delay)
                
                self.kit.stepper2.release()
            
            except Exception as e:
                print(f"Stepper error: {e}")

    def get_smoothed_coordinates(self, center_x, center_y):
        self.x_coords.append(center_x)
        self.y_coords.append(center_y)
        
        if len(self.x_coords) >= self.buffer_size_x:
            x_array = np.array(self.x_coords)
            smoothed_x = np.sum(x_array * self.weights_x)
        else:
            smoothed_x = center_x
        
        if len(self.y_coords) >= self.buffer_size_y:
            y_array = np.array(self.y_coords)
            smoothed_y = np.sum(y_array * self.weights_y)
        else:
            smoothed_y = center_y
        
        # Apply exponential moving average
        alpha = 0.5  # Smoothing factor (0 < alpha < 1)
        if hasattr(self, 'last_smoothed_x'):
            smoothed_x = alpha * smoothed_x + (1 - alpha) * self.last_smoothed_x
            smoothed_y = alpha * smoothed_y + (1 - alpha) * self.last_smoothed_y
        
        self.last_smoothed_x = smoothed_x
        self.last_smoothed_y = smoothed_y
        
        return smoothed_x, smoothed_y


    def detect_faces(self, frame, threshold=0.7):
        """Detect faces in the frame using ONNX model"""
        if frame is None:
            return [], [], []
            
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)

        input_name = self.face_detector.get_inputs()[0].name
        confidences, boxes = self.face_detector.run(None, {input_name: image})
        boxes, labels, probs = self.predict(frame.shape[1], frame.shape[0], 
                                          confidences, boxes, threshold)
        return boxes, labels, probs

    def move_servos(self, target_angle):
        max_angle_change = 5  # Maximum angle change per frame
        current_angle = self.last_angle
        angle_diff = target_angle - current_angle
        
        if abs(angle_diff) > max_angle_change:
            target_angle = current_angle + max_angle_change * (1 if angle_diff > 0 else -1)
        
        angle = np.clip(target_angle, 45, 125)
        
        try:
            self.servo1.angle = angle
            self.servo2.angle = 180 - angle
        except Exception as e:
            print(f"Servo control error: {e}")
        
        self.last_angle = angle


    def update_relay(self, norm_x, norm_y, face_detected):
        """Update relay based on target position"""
        if not face_detected:
            if self.relay_state:
                self.relay_state = False
                GPIO.output(self.relay_ch, GPIO.LOW)
            return

        in_zone = (abs(norm_x) < self.target_zone_size and 
                  abs(norm_y) < self.target_zone_size)
                  
        if in_zone and not self.relay_state:
            self.relay_state = True
            GPIO.output(self.relay_ch, GPIO.HIGH)
        elif not in_zone and self.relay_state:
            self.relay_state = False
            GPIO.output(self.relay_ch, GPIO.LOW)


    def run(self):
        """Main control loop with improved error handling"""
        cv2.namedWindow('Face Tracking System')
        camera_error_count = 0
        max_camera_errors = 5
        
        while self.running:
            try:
                # Capture and process frame with timeout
                ret = False
                start_time = time.time()
                while not ret and time.time() - start_time < 2.0:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to read frame, retrying...")
                        time.sleep(0.1)
                
                if not ret:
                    camera_error_count += 1
                    print(f"Persistent camera read failure (attempt {camera_error_count})")
                    
                    if camera_error_count >= max_camera_errors:
                        print("Too many camera errors. Reinitializing...")
                        self.init_camera()
                        camera_error_count = 0
                    
                    continue
                
                # Reset error count on successful frame
                camera_error_count = 0
                
                height, width = frame.shape[:2]
                processed_frame = frame.copy()
                boxes, labels, probs = self.detect_faces(frame)
                
                face_detected = len(boxes) > 0
                
                if face_detected:
                    box = boxes[0]
                    center_x = int((box[0] + box[2]) / 2)
                    center_y = int((box[1] + box[3]) / 2)
                    
                    smoothed_x, smoothed_y = self.get_smoothed_coordinates(center_x, center_y)
                    target_angle = 45 + (smoothed_y / height) * 90
                    angle_x = (smoothed_x / width) * 66
                    print(angle_x,target_angle)
                    target_angle_stepper = angle_x - 33
                    
                    norm_x = smoothed_x / width - 0.5
                    norm_y = smoothed_y / height - 0.5
                    
                    # Update stepper in zone status
                    self.stepper_in_zone = abs(norm_x) < self.target_zone_size
                    
                    # Draw visualizations
                    cv2.rectangle(processed_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.circle(processed_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.circle(processed_frame, (int(smoothed_x), int(smoothed_y)), 5, (255, 0, 0), -1)
                    
                    # Draw target zone
                    zone_width = int(width * self.target_zone_size * 2)
                    zone_height = int(height * self.target_zone_size * 2)
                    zone_x = int(width/2 - zone_width/2)
                    zone_y = int(height/2 - zone_height/2)
                    
                    zone_color = (0, 255, 0) if self.relay_state else (0, 0, 255)
                    cv2.rectangle(processed_frame, (zone_x, zone_y),
                                (zone_x + zone_width, zone_y + zone_height),
                                zone_color, 2)
                    
                    cv2.putText(processed_frame, f"Angle: {target_angle:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Update controls
                    self.move_servos(target_angle)
                    if not self.stepper_in_zone and not self.stepper_queue.full():
                        self.stepper_queue.put(target_angle_stepper)
                    self.update_relay(norm_x, norm_y, face_detected)
                else:
                    # No face detected, center servos and reset stepper status
                    self.move_servos(90)
                    self.stepper_in_zone = False
                    self.update_relay(0, 0, False)
                
                cv2.imshow('Face Tracking System', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                sleep(0.02)  # ~50Hz update rate
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                try:
                    # Attempt to reinitialize camera on unexpected errors
                    self.init_camera()
                except Exception as init_error:
                    print(f"Failed to reinitialize camera: {init_error}")
                
                time.sleep(1)

        self.cleanup()

    def cleanup(self):
        """Enhanced cleanup method"""
        self.running = False
        time.sleep(0.5)  # Allow stepper thread to finish
        
        try:
            GPIO.cleanup()
        except Exception as e:
            print(f"GPIO cleanup error: {e}")
        
        try:
            self.kit.stepper2.release()
        except Exception as e:
            print(f"Stepper release error: {e}")
        
        if self.cap is not None:
            try:
                self.cap.release()
                print("Camera released successfully")
            except Exception as e:
                print(f"Camera release error: {e}")
        
        cv2.destroyAllWindows()

    def predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
        """Predict face locations"""
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = self.hard_nms(box_probs, iou_threshold=iou_threshold, top_k=top_k)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

    def hard_nms(self, box_scores, iou_threshold, top_k=-1, candidate_size=200):
        """Perform Non-Maximum Suppression"""
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        indexes = np.argsort(scores)
        indexes = indexes[-candidate_size:]
        while len(indexes) > 0:
            current = indexes[-1]
            picked.append(current)
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = self.iou_of(rest_boxes, np.expand_dims(current_box, axis=0))
            indexes = indexes[iou <= iou_threshold]
        return box_scores[picked, :]

    @staticmethod
    def iou_of(boxes0, boxes1, eps=1e-5):
        """Calculate intersection over union"""
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])
        
        hw = np.clip(overlap_right_bottom - overlap_left_top, 0.0, None)
        overlap_area = hw[..., 0] * hw[..., 1]
        
        area0 = FaceTrackingSystem.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = FaceTrackingSystem.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    @staticmethod
    def area_of(left_top, right_bottom):
        """Calculate area of a rectangle"""
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]

if __name__ == "__main__":
    try:
        tracker = FaceTrackingSystem()
        tracker.run()
    except Exception as e:
        print(f"Error: {e}")
