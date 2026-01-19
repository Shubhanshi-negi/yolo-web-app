import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import queue
from collections import defaultdict, Counter
from pathlib import Path
import sys

# Optional text-to-speech
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print(" pyttsx3 not installed - audio feedback disabled")
    print("    Install with: pip install pyttsx3")


class AudioFeedback:
    """Text-to-speech system for accessibility"""
    
    def __init__(self, enabled=True):
        self.enabled = enabled and TTS_AVAILABLE
        self.engine = None
        self.audio_queue = queue.Queue()
        self.thread = None
        self.running = False
        self.speaking = False
        
        if self.enabled:
            try:
                self.engine = pyttsx3.init()
                # Configure voice
                self.engine.setProperty('rate', 175)  # Speed
                self.engine.setProperty('volume', 0.9)
                
                # Use female voice if available
                voices = self.engine.getProperty('voices')
                if len(voices) > 1:
                    self.engine.setProperty('voice', voices[1].id)
                
                self._start_worker()
                print("✓ Audio feedback enabled")
            except Exception as e:
                print(f"⚠️  Audio initialization failed: {e}")
                self.enabled = False
    
    def _start_worker(self):
        """Start background audio thread"""
        self.running = True
        self.thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.thread.start()
    
    def _audio_worker(self):
        """Background thread for non-blocking audio"""
        while self.running:
            try:
                text = self.audio_queue.get(timeout=0.1)
                if text and self.engine:
                    self.speaking = True
                    self.engine.say(text)
                    self.engine.runAndWait()
                    self.speaking = False
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio error: {e}")
                self.speaking = False
    
    def announce(self, text):
        """Queue text for announcement"""
        if self.enabled and text and not self.speaking:
            # Clear old messages if queue is full
            if self.audio_queue.qsize() > 3:
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
            self.audio_queue.put(text)
    
    def toggle(self):
        """Toggle audio on/off"""
        self.enabled = not self.enabled
        return self.enabled
    
    def stop(self):
        """Stop audio system"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


class DetectionVisualizer:
    """Draw bounding boxes, labels, and UI elements"""
    
    # Indoor-focused object classes (from COCO dataset)
    INDOOR_CLASSES = {
        'person', 'chair', 'couch', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush', 'bottle', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'potted plant', 'backpack',
        'handbag', 'tie', 'suitcase', 'umbrella', 'shoe', 'door', 'window'
    }
    
    # Critical objects for priority alerts
    CRITICAL_OBJECTS = {'person', 'chair', 'stairs', 'door'}
    
    def __init__(self, window_name="YOLO Indoor Object Detection"):
        self.window_name = window_name
        self.colors = self._generate_colors()
        
    def _generate_colors(self):
        """Generate consistent colors for each class"""
        np.random.seed(42)
        colors = {}
        for i in range(80):  # COCO has 80 classes
            colors[i] = tuple(map(int, np.random.randint(100, 255, 3)))
        return colors
    
    def is_indoor_object(self, class_name):
        """Check if object is relevant for indoor detection"""
        return class_name.lower() in self.INDOOR_CLASSES
    
    def is_critical(self, class_name):
        """Check if object needs priority alert"""
        return class_name.lower() in self.CRITICAL_OBJECTS
    
    def draw_box_with_label(self, frame, x1, y1, x2, y2, label, color, conf):
        """Draw a single detection box with label"""
        # Draw box
        thickness = 3 if conf > 0.7 else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        label_text = f"{label} {conf:.2f}"
        
        # Calculate label size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, font_thickness
        )
        
        # Draw label background
        label_y1 = max(y1 - text_height - baseline - 10, 0)
        label_y2 = y1
        cv2.rectangle(
            frame,
            (x1, label_y1),
            (x1 + text_width + 10, label_y2),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label_text,
            (x1 + 5, y1 - baseline - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    def draw_detections(self, frame, results, class_names, indoor_only=True):
        """Draw all detections on frame"""
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Extract box info
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = class_names.get(cls, f"class_{cls}")
                
                # Filter indoor objects if enabled
                if indoor_only and not self.is_indoor_object(class_name):
                    continue
                
                # Store detection data
                detection_data = {
                    'class': class_name,
                    'conf': conf,
                    'bbox': (x1, y1, x2, y2),
                    'center_x': (x1 + x2) / 2 / w,  # Normalized 0-1
                    'center_y': (y1 + y2) / 2 / h,
                    'height': (y2 - y1) / h,
                    'width': (x2 - x1) / w,
                    'is_critical': self.is_critical(class_name)
                }
                detections.append(detection_data)
                
                # Draw visualization
                color = self.colors.get(cls, (255, 255, 255))
                self.draw_box_with_label(
                    annotated, x1, y1, x2, y2, class_name, color, conf
                )
        
        return annotated, detections
    
    def draw_info_panel(self, frame, fps, detection_counts, total_objects,
                       conf_threshold, audio_enabled):
        """Draw information panel overlay"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent background
        overlay = frame.copy()
        panel_width = 380
        panel_height = min(300, h - 100)
        cv2.rectangle(
            overlay,
            (10, 10),
            (panel_width, panel_height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_pos = 45
        line_height = 30
        
        # Title
        cv2.putText(
            frame,
            "YOLO Detection System",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        y_pos += line_height + 10
        
        # FPS
        fps_color = (0, 255, 0) if fps > 20 else (0, 165, 255) if fps > 10 else (0, 0, 255)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            fps_color,
            2
        )
        y_pos += line_height
        
        # Total objects
        cv2.putText(
            frame,
            f"Objects Detected: {total_objects}",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        y_pos += line_height
        
        # Settings
        audio_status = "ON" if audio_enabled else "OFF"
        audio_color = (0, 255, 0) if audio_enabled else (128, 128, 128)
        cv2.putText(
            frame,
            f"Audio: {audio_status}",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            audio_color,
            1
        )
        y_pos += line_height - 5
        
        cv2.putText(
            frame,
            f"Confidence: {conf_threshold:.2f}",
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        y_pos += line_height + 5
        
        # Detected objects list
        if detection_counts:
            cv2.putText(
                frame,
                "Detected:",
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
            y_pos += line_height - 5
            
            for class_name, count in sorted(detection_counts.items())[:5]:
                cv2.putText(
                    frame,
                    f"  {class_name}: {count}",
                    (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1
                )
                y_pos += 25
        
        # Controls at bottom
        controls_y = h - 80
        controls = [
            "Q: Quit",
            "A: Audio",
            "S: Screenshot",
            "+/-: Confidence"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(
                frame,
                control,
                (20, controls_y + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 180, 180),
                1
            )
        
        return frame


class SpatialAwareness:
    """Generate spatial descriptions for audio feedback"""
    
    def __init__(self):
        self.announcement_history = {}
        self.cooldown_time = 3.0  # seconds between same object announcements
    
    def get_direction(self, center_x):
        """Get horizontal direction"""
        if center_x < 0.33:
            return "on your left"
        elif center_x > 0.67:
            return "on your right"
        else:
            return "directly ahead"
    
    def get_distance(self, height):
        """Estimate distance from object height in frame"""
        if height > 0.6:
            return "very close"
        elif height > 0.35:
            return "approaching"
        elif height > 0.15:
            return "nearby"
        else:
            return "in the distance"
    
    def should_announce(self, class_name):
        """Check if enough time has passed since last announcement"""
        current_time = time.time()
        last_time = self.announcement_history.get(class_name, 0)
        
        if current_time - last_time >= self.cooldown_time:
            self.announcement_history[class_name] = current_time
            return True
        return False
    
    def generate_announcement(self, detections):
        """Generate natural language announcement"""
        if not detections:
            return None
        
        # Prioritize critical objects
        critical = [d for d in detections if d['is_critical']]
        normal = [d for d in detections if not d['is_critical']]
        
        # Choose what to announce
        to_announce = critical[:2] if critical else normal[:1]
        
        announcements = []
        for det in to_announce:
            class_name = det['class']
            
            if not self.should_announce(class_name):
                continue
            
            direction = self.get_direction(det['center_x'])
            distance = self.get_distance(det['height'])
            
            # Special handling for critical objects
            if class_name.lower() == 'person':
                msg = f"Person {direction}"
            elif class_name.lower() == 'chair':
                msg = f"Chair {direction}, {distance}"
            else:
                msg = f"{class_name} {direction}"
            
            announcements.append(msg)
        
        return ". ".join(announcements) if announcements else None
    
    def clear_history(self):
        """Clear announcement cooldown history"""
        self.announcement_history.clear()



class IndoorObjectDetector:
    """Main real-time detection system"""
    
    def __init__(self, camera_id=0, conf_threshold=0.45):
        print("\n" + "="*70)
        print(" YOLO INDOOR OBJECT DETECTION SYSTEM")
        print("="*70)
        print("\nInitializing...")
        
        # Load pretrained YOLO model (auto-downloads if needed)
        print("\n[1/4] Loading YOLOv8n model...")
        try:
            self.model = YOLO('yolov8n.pt')  # Pretrained on COCO dataset
            self.class_names = self.model.names
            print(f"✓ Model loaded ({len(self.class_names)} classes)")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            sys.exit(1)
        
        # Initialize camera
        print(f"\n[2/4] Opening camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print("✗ Failed to open camera")
            print("Try different camera_id: python realtime_detection_gui.py --camera 1")
            sys.exit(1)
        
        # Configure camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✓ Camera opened ({actual_width}x{actual_height})")
        
        # Initialize components
        print("\n[3/4] Initializing components...")
        self.audio = AudioFeedback(enabled=TTS_AVAILABLE)
        self.visualizer = DetectionVisualizer()
        self.spatial = SpatialAwareness()
        print("✓ Components ready")
        
        # Settings
        self.conf_threshold = conf_threshold
        self.iou_threshold = 0.45
        self.indoor_only = True
        
        # State
        self.running = False
        self.fps_history = []
        self.frame_count = 0
        self.screenshot_count = 0
        
        print("\n[4/4] System ready!")
        print("="*70)
    
    def adjust_confidence(self, delta):
        """Adjust confidence threshold"""
        self.conf_threshold = np.clip(self.conf_threshold + delta, 0.1, 0.9)
        msg = f"Confidence threshold: {self.conf_threshold:.2f}"
        print(msg)
        self.audio.announce(msg)
    
    def save_screenshot(self, frame):
        """Save screenshot with timestamp"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}.jpg"
        
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)
        filepath = screenshots_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        self.screenshot_count += 1
        
        msg = f"Screenshot {self.screenshot_count} saved"
        print(f"📸 {msg}: {filepath}")
        self.audio.announce(msg)
    
    def process_frame(self, frame):
        """Process single frame - detect and visualize"""
        # Run YOLO detection
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Visualize detections
        annotated, detections = self.visualizer.draw_detections(
            frame, results, self.class_names, self.indoor_only
        )
        
        # Count objects by class
        detection_counts = Counter(d['class'] for d in detections)
        
        return annotated, detections, detection_counts
    
    def run(self):
        """Main detection loop"""
        print("\n🎥 Starting detection...\n")
        print("Controls:")
        print("  Q - Quit")
        print("  A - Toggle audio")
        print("  S - Screenshot")
        print("  C - Clear detection history")
        print("  + - Increase confidence")
        print("  - - Decrease confidence")
        print("  I - Toggle indoor-only mode")
        print("\n" + "="*70 + "\n")
        
        self.running = True
        self.frame_count = 0
        
        try:
            while self.running:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Measure processing time
                start_time = time.time()
                
                # Process frame
                annotated, detections, detection_counts = self.process_frame(frame)
                
                # Calculate FPS
                process_time = time.time() - start_time
                fps = 1.0 / process_time if process_time > 0 else 0
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                avg_fps = np.mean(self.fps_history)
                
                # Audio feedback (every 30 frames ≈ 1 second)
                if self.frame_count % 30 == 0 and self.audio.enabled:
                    announcement = self.spatial.generate_announcement(detections)
                    if announcement:
                        self.audio.announce(announcement)
                
                # Draw info panel
                annotated = self.visualizer.draw_info_panel(
                    annotated,
                    avg_fps,
                    detection_counts,
                    len(detections),
                    self.conf_threshold,
                    self.audio.enabled
                )
                
                # Display frame
                cv2.imshow(self.visualizer.window_name, annotated)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    print("\n👋 Quitting...")
                    break
                    
                elif key == ord('a'):
                    enabled = self.audio.toggle()
                    status = "enabled" if enabled else "disabled"
                    print(f"🔊 Audio {status}")
                    if enabled:
                        self.audio.announce(f"Audio {status}")
                        
                elif key == ord('s'):
                    self.save_screenshot(annotated)
                    
                elif key == ord('c'):
                    self.spatial.clear_history()
                    print("🔄 Detection history cleared")
                    self.audio.announce("History cleared")
                    
                elif key == ord('+') or key == ord('='):
                    self.adjust_confidence(0.05)
                    
                elif key == ord('-') or key == ord('_'):
                    self.adjust_confidence(-0.05)
                    
                elif key == ord('i'):
                    self.indoor_only = not self.indoor_only
                    mode = "indoor-only" if self.indoor_only else "all objects"
                    print(f"🏠 Mode: {mode}")
                    self.audio.announce(f"Mode: {mode}")
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        print("\n🧹 Cleaning up...")
        self.running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        self.audio.stop()
        
        if self.fps_history:
            avg_fps = np.mean(self.fps_history)
            print(f"\n📊 Session Statistics:")
            print(f"   Frames processed: {self.frame_count}")
            print(f"   Average FPS: {avg_fps:.1f}")
            print(f"   Screenshots: {self.screenshot_count}")
        
        print("\n✓ Cleanup complete")
        print("="*70)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Application entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="YOLO Real-Time Indoor Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python realtime_detection_gui.py
  python realtime_detection_gui.py --camera 1
  python realtime_detection_gui.py --conf 0.5 --no-audio
  
For more information, visit: https://docs.ultralytics.com/
        """
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.45,
        help='Confidence threshold (default: 0.45)'
    )
    
    parser.add_argument(
        '--no-audio',
        action='store_true',
        help='Disable audio feedback'
    )
    
    args = parser.parse_args()
    
    try:
        # Create detector
        detector = IndoorObjectDetector(
            camera_id=args.camera,
            conf_threshold=args.conf
        )
        
        # Disable audio if requested
        if args.no_audio:
            detector.audio.enabled = False
            print("🔇 Audio feedback disabled")
        
        # Run detection
        detector.run()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()