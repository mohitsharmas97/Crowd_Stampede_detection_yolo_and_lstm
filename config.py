"""
Configuration file for Crowd Stampede Detection System
"""

# Model Paths
YOLO_MODEL_PATH = "best.pt"
LSTM_MODEL_PATH = "lstm_risk_model.pt"

# Risk Thresholds
RISK_THRESHOLD_LOW = 30  # Green zone: 0-30
RISK_THRESHOLD_HIGH = 70  # Yellow zone: 30-70, Red zone: 70-100

# Alert Colors
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)

# YOLOv8 Detection Settings
CONFIDENCE_THRESHOLD = 0.25  # Lowered for better crowd detection (was 0.5)
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 1000

# Video Processing Settings
DEFAULT_FPS = 25
FRAME_SKIP = 1  # Process every nth frame
SEQUENCE_LENGTH = 10  # Number of frames for LSTM temporal analysis

# Density Map Settings
HEATMAP_ALPHA = 0.6
HEATMAP_COLORMAP = 'jet'

# Feature Extraction
OPTICAL_FLOW_PARAMS = {
    'pyr_scale': 0.5,
    'levels': 3,
    'winsize': 15,
    'iterations': 3,
    'poly_n': 5,
    'poly_sigma': 1.2,
    'flags': 0
}

# Display Settings
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720
