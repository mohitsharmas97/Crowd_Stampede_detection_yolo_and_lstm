"""
Visualization utilities for displaying detection results and risk scores
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import config


def get_risk_color(risk_score):
    """
    Get color based on risk score
    
    Args:
        risk_score: Risk score (0-100)
        
    Returns:
        Color tuple (B, G, R) for OpenCV
    """
    if risk_score < config.RISK_THRESHOLD_LOW:
        return config.COLOR_GREEN
    elif risk_score < config.RISK_THRESHOLD_HIGH:
        return config.COLOR_YELLOW
    else:
        return config.COLOR_RED


def get_risk_label(risk_score):
    """
    Get risk level label
    
    Args:
        risk_score: Risk score (0-100)
        
    Returns:
        Risk level text
    """
    if risk_score < config.RISK_THRESHOLD_LOW:
        return "LOW RISK"
    elif risk_score < config.RISK_THRESHOLD_HIGH:
        return "MODERATE RISK"
    else:
        return "HIGH RISK"


def draw_detections(frame, detections, show_boxes=True):
    """
    Draw bounding boxes on frame
    
    Args:
        frame: Input frame
        detections: YOLOv8 detection results
        show_boxes: Whether to show bounding boxes
        
    Returns:
        Frame with drawn detections
    """
    frame_copy = frame.copy()
    
    if detections is None or len(detections) == 0:
        return frame_copy
    
    if show_boxes:
        for det in detections:
            x1, y1, x2, y2, conf = det[:5]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            label = f'{conf:.2f}'
            cv2.putText(frame_copy, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return frame_copy


def create_heatmap(frame, detections):
    """
    Create density heatmap
    
    Args:
        frame: Input frame
        detections: YOLOv8 detection results
        
    Returns:
        Frame with heatmap overlay
    """
    height, width = frame.shape[:2]
    
    # Create empty heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    if detections is not None and len(detections) > 0:
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure coordinates are within bounds
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))
            
            # Add to heatmap
            if x2 > x1 and y2 > y1:
                heatmap[y1:y2, x1:x2] += 1
    
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Apply Gaussian blur
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    
    # Convert to color
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    
    # Overlay on frame
    frame_copy = frame.copy()
    frame_copy = cv2.addWeighted(
        frame_copy,
        1 - config.HEATMAP_ALPHA,
        heatmap_colored,
        config.HEATMAP_ALPHA,
        0
    )
    
    return frame_copy


def display_risk_score(frame, risk_score, person_count):
    """
    Display risk score and metrics on frame
    
    Args:
        frame: Input frame
        risk_score: Current risk score
        person_count: Number of detected persons
        
    Returns:
        Frame with risk information
    """
    frame_copy = frame.copy()
    height, width = frame.shape[:2]
    
    # Get risk color and label
    risk_color = get_risk_color(risk_score)
    risk_label = get_risk_label(risk_score)
    
    # Create semi-transparent overlay for text background
    overlay = frame_copy.copy()
    
    # Draw background rectangle
    cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame_copy, 0.3, 0, frame_copy)
    
    # Display risk score
    cv2.putText(frame_copy, f'Risk Score: {risk_score:.1f}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, risk_color, 3)
    
    # Display risk level
    cv2.putText(frame_copy, risk_label, (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, risk_color, 2)
    
    # Display person count
    cv2.putText(frame_copy, f'People: {person_count}', (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame_copy


def create_risk_gauge(risk_score):
    """
    Create a gauge visualization for risk score
    
    Args:
        risk_score: Risk score (0-100)
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create gauge
    categories = ['Low\n(0-30)', 'Moderate\n(30-70)', 'High\n(70-100)']
    colors = ['green', 'yellow', 'red']
    
    # Draw gauge bars
    for i, (cat, color) in enumerate(zip(categories, colors)):
        start = i * 33.33
        end = (i + 1) * 33.33
        ax.barh(0, end - start, left=start, height=0.5, color=color, alpha=0.3)
    
    # Draw current risk indicator
    ax.barh(0, risk_score, height=0.5, color='black', alpha=0.7)
    
    # Add text
    ax.text(risk_score, 0, f'{risk_score:.1f}', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([0, 30, 70, 100])
    ax.set_yticks([])
    ax.set_xlabel('Risk Score', fontsize=12)
    ax.set_title('Stampede Risk Level', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    return fig
