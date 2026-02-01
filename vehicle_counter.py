"""
Vehicle Counter with Object Tracking and Line Crossing Detection
Tracks vehicles crossing a virtual line to count IN/OUT movements
"""

import cv2
import numpy as np
from collections import deque, defaultdict
from datetime import datetime
import math

class VehicleTracker:
    """Track individual vehicles across frames"""
    
    def __init__(self, track_id, bbox, category, confidence):
        self.track_id = track_id
        self.category = category
        self.confidence = confidence
        self.positions = deque(maxlen=30)  # Store last 30 positions
        self.bbox = bbox
        self.counted = False
        self.direction = None  # 'IN' or 'OUT'
        self.crossed_line = False
        self.last_seen = datetime.now()
        
        # Store initial position
        center = self.get_center(bbox)
        self.positions.append(center)
    
    def get_center(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def update(self, bbox):
        """Update tracker with new detection"""
        self.bbox = bbox
        center = self.get_center(bbox)
        self.positions.append(center)
        self.last_seen = datetime.now()
    
    def get_trajectory_direction(self):
        """Determine movement direction from trajectory"""
        if len(self.positions) < 5:
            return None
        
        # Compare first and last positions
        start_x = np.mean([p[0] for p in list(self.positions)[:5]])
        end_x = np.mean([p[0] for p in list(self.positions)[-5:]])
        
        displacement = end_x - start_x
        
        # Threshold for significant movement
        if abs(displacement) > 30:
            return 'RIGHT' if displacement > 0 else 'LEFT'
        
        return None
    
    def is_stale(self, max_age_seconds=2):
        """Check if tracker is too old"""
        age = (datetime.now() - self.last_seen).total_seconds()
        return age > max_age_seconds

class VehicleCounter:
    """Count vehicles crossing a virtual line"""
    
    def __init__(self, line_position=0.5, direction_mapping={'LEFT': 'OUT', 'RIGHT': 'IN'}):
        """
        Initialize vehicle counter
        
        Args:
            line_position: Position of counting line (0-1, fraction of frame width)
            direction_mapping: Map movement direction to count type
        """
        self.line_position = line_position
        self.direction_mapping = direction_mapping
        self.trackers = {}
        self.next_track_id = 1
        
        # Counting statistics
        self.counts = {
            'IN': defaultdict(int),
            'OUT': defaultdict(int)
        }
        
        self.total_counts = {
            'IN': 0,
            'OUT': 0
        }
        
        # Tracking parameters
        self.max_distance = 100  # Maximum distance for matching detections
        self.max_age = 2  # Maximum age in seconds before removing tracker
    
    def reset_counts(self):
        """Reset all counts"""
        self.counts = {
            'IN': defaultdict(int),
            'OUT': defaultdict(int)
        }
        self.total_counts = {'IN': 0, 'OUT': 0}
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def match_detections_to_trackers(self, detections):
        """Match new detections to existing trackers"""
        if len(self.trackers) == 0:
            return {}, list(range(len(detections)))
        
        if len(detections) == 0:
            return {}, []
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(self.trackers)))
        
        for d, det in enumerate(detections):
            det_bbox = det['bbox']
            for t, (track_id, tracker) in enumerate(self.trackers.items()):
                iou_matrix[d, t] = self.calculate_iou(det_bbox, tracker.bbox)
        
        # Match using Hungarian algorithm (simplified greedy approach)
        matched = {}
        unmatched_detections = []
        matched_trackers = set()
        
        # Greedy matching
        for d in range(len(detections)):
            best_iou = 0.3  # Minimum IoU threshold
            best_tracker_idx = -1
            
            for t in range(len(self.trackers)):
                if t in matched_trackers:
                    continue
                if iou_matrix[d, t] > best_iou:
                    best_iou = iou_matrix[d, t]
                    best_tracker_idx = t
            
            if best_tracker_idx >= 0:
                track_id = list(self.trackers.keys())[best_tracker_idx]
                matched[d] = track_id
                matched_trackers.add(best_tracker_idx)
            else:
                unmatched_detections.append(d)
        
        return matched, unmatched_detections
    
    def check_line_crossing(self, tracker, line_x):
        """Check if tracker has crossed the counting line"""
        if len(tracker.positions) < 2:
            return False
        
        # Check last two positions
        prev_pos = tracker.positions[-2]
        curr_pos = tracker.positions[-1]
        
        # Check if line is crossed
        crossed = (prev_pos[0] < line_x <= curr_pos[0]) or (prev_pos[0] > line_x >= curr_pos[0])
        
        return crossed
    
    def update(self, detections, frame_shape):
        """Update tracker with new detections"""
        height, width = frame_shape[:2]
        line_x = int(width * self.line_position)
        
        # Match detections to trackers
        matched, unmatched_detections = self.match_detections_to_trackers(detections)
        
        # Update matched trackers
        for det_idx, track_id in matched.items():
            detection = detections[det_idx]
            self.trackers[track_id].update(detection['bbox'])
            self.trackers[track_id].confidence = detection['confidence']
        
        # Create new trackers for unmatched detections
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            track_id = self.next_track_id
            self.next_track_id += 1
            
            tracker = VehicleTracker(
                track_id,
                detection['bbox'],
                detection['display_category'],
                detection['confidence']
            )
            self.trackers[track_id] = tracker
        
        # Check for line crossings and count
        trackers_to_remove = []
        
        for track_id, tracker in list(self.trackers.items()):
            # Remove stale trackers
            if tracker.is_stale(self.max_age):
                trackers_to_remove.append(track_id)
                continue
            
            # Check if crossed line and not yet counted
            if not tracker.counted and self.check_line_crossing(tracker, line_x):
                direction = tracker.get_trajectory_direction()
                
                if direction in self.direction_mapping:
                    count_type = self.direction_mapping[direction]
                    tracker.direction = count_type
                    tracker.counted = True
                    tracker.crossed_line = True
                    
                    # Increment count
                    self.counts[count_type][tracker.category] += 1
                    self.total_counts[count_type] += 1
                    
                    print(f"✅ {tracker.category} crossed line {direction} → {count_type}")
        
        # Remove stale trackers
        for track_id in trackers_to_remove:
            del self.trackers[track_id]
    
    def draw_on_frame(self, frame):
        """Draw tracking information on frame"""
        height, width = frame.shape[:2]
        line_x = int(width * self.line_position)
        
        # Draw counting line
        cv2.line(frame, (line_x, 0), (line_x, height), (0, 255, 255), 3)
        
        # Draw line labels
        cv2.putText(frame, "OUT ←", (line_x - 150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "→ IN", (line_x + 50, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw trackers
        for track_id, tracker in self.trackers.items():
            bbox = tracker.bbox
            x1, y1, x2, y2 = map(int, bbox)
            
            # Color based on counting status
            if tracker.counted:
                color = (0, 255, 0) if tracker.direction == 'IN' else (0, 0, 255)
            else:
                color = (255, 255, 0)  # Yellow for unco unted
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and category
            label = f"ID:{track_id} {tracker.category}"
            if tracker.counted:
                label += f" [{tracker.direction}]"
            
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw trajectory
            if len(tracker.positions) > 1:
                points = np.array(tracker.positions, dtype=np.int32)
                cv2.polylines(frame, [points], False, color, 2)
        
        # Draw count display
        self.draw_count_display(frame)
        
        return frame
    
    def draw_count_display(self, frame):
        """Draw count statistics on frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # IN counts (top-right)
        in_x, in_y = width - 250, 80
        cv2.rectangle(overlay, (in_x, in_y), (in_x + 230, in_y + 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "IN →", (in_x + 10, in_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        y_offset = in_y + 60
        for category in ['Car', '2-Wheeler', 'Bus', 'Truck']:
            count = self.counts['IN'][category]
            cv2.putText(frame, f"{category}: {count}", (in_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        # OUT counts (top-left)
        out_x, out_y = 20, 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (out_x, out_y), (out_x + 230, out_y + 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "← OUT", (out_x + 10, out_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        y_offset = out_y + 60
        for category in ['Car', '2-Wheeler', 'Bus', 'Truck']:
            count = self.counts['OUT'][category]
            cv2.putText(frame, f"{category}: {count}", (out_x + 10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
    
    def get_counts(self):
        """Get current counts"""
        return {
            'in_counts': dict(self.counts['IN']),
            'out_counts': dict(self.counts['OUT']),
            'total_in': self.total_counts['IN'],
            'total_out': self.total_counts['OUT'],
            'net_count': self.total_counts['IN'] - self.total_counts['OUT']
        }
    
    def get_parking_availability(self, total_capacity):
        """Calculate parking availability based on car counts"""
        cars_in = self.counts['IN']['Car']
        cars_out = self.counts['OUT']['Car']
        net_cars = cars_in - cars_out
        
        available = total_capacity - net_cars
        
        return {
            'total_capacity': total_capacity,
            'cars_in': cars_in,
            'cars_out': cars_out,
            'currently_parked': net_cars,
            'available': max(0, available),
            'occupancy_percent': (net_cars / total_capacity * 100) if total_capacity > 0 else 0
        }