"""
Video processing utilities for feature extraction
"""

import cv2
import numpy as np
import config


class VideoProcessor:
    """Handle video processing and feature extraction"""
    
    def __init__(self):
        self.prev_gray = None
        self.feature_history = []
        
    def extract_optical_flow(self, frame):
        """
        Calculate optical flow for motion analysis
        
        Args:
            frame: Current video frame
            
        Returns:
            Optical flow magnitude
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros_like(gray, dtype=np.float32)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray,
            None,
            **config.OPTICAL_FLOW_PARAMS
        )
        
        # Calculate magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        self.prev_gray = gray
        return magnitude
    
    def extract_spatial_features(self, detections, frame_shape):
        """
        Extract spatial features from detections
        
        Args:
            detections: YOLOv8 detection results
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            Dictionary of spatial features
        """
        if detections is None or len(detections) == 0:
            return {
                'person_count': 0,
                'density': 0.0,
                'avg_bbox_area': 0.0,
                'crowd_center': (0, 0)
            }
        
        height, width = frame_shape[:2]
        frame_area = height * width
        
        # Count persons
        person_count = len(detections)
        
        # Calculate density
        total_bbox_area = 0
        centers_x = []
        centers_y = []
        
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            bbox_area = (x2 - x1) * (y2 - y1)
            total_bbox_area += bbox_area
            
            # Calculate center
            centers_x.append((x1 + x2) / 2)
            centers_y.append((y1 + y2) / 2)
        
        density = total_bbox_area / frame_area if frame_area > 0 else 0
        avg_bbox_area = total_bbox_area / person_count if person_count > 0 else 0
        
        # Calculate crowd center
        crowd_center = (
            np.mean(centers_x) if centers_x else width // 2,
            np.mean(centers_y) if centers_y else height // 2
        )
        
        return {
            'person_count': person_count,
            'density': density,
            'avg_bbox_area': avg_bbox_area,
            'crowd_center': crowd_center
        }
    
    def extract_temporal_features(self, spatial_features, optical_flow):
        """
        Extract temporal features for LSTM
        
        Args:
            spatial_features: Dictionary of spatial features
            optical_flow: Optical flow magnitude array
            
        Returns:
            Feature vector for LSTM
        """
        # Calculate motion statistics
        flow_mean = np.mean(optical_flow)
        flow_std = np.std(optical_flow)
        flow_max = np.max(optical_flow)
        
        # Create feature vector
        features = np.array([
            spatial_features['person_count'],
            spatial_features['density'],
            spatial_features['avg_bbox_area'],
            flow_mean,
            flow_std,
            flow_max
        ])
        
        # Add to history
        self.feature_history.append(features)
        
        # Keep only recent features for sequence
        if len(self.feature_history) > config.SEQUENCE_LENGTH:
            self.feature_history = self.feature_history[-config.SEQUENCE_LENGTH:]
        
        return np.array(self.feature_history)
    
    def reset(self):
        """Reset processor state"""
        self.prev_gray = None
        self.feature_history = []
    
    @staticmethod
    def resize_frame(frame, max_width=None, max_height=None):
        """
        Resize frame while maintaining aspect ratio
        
        Args:
            frame: Input frame
            max_width: Maximum width
            max_height: Maximum height
            
        Returns:
            Resized frame
        """
        if max_width is None:
            max_width = config.MAX_DISPLAY_WIDTH
        if max_height is None:
            max_height = config.MAX_DISPLAY_HEIGHT
        
        height, width = frame.shape[:2]
        
        # Calculate scaling factor
        scale = min(max_width / width, max_height / height, 1.0)
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        return frame
