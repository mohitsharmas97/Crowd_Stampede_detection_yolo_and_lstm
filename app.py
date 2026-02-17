"""
AI-Based Early Prediction and Prevention of Crowd Stampede
Using Spatio-Temporal Crowd Behavior Analysis

Main Streamlit Application
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

# Import custom modules
from utils.model_loader import load_yolo_model, load_lstm_model, predict_risk_score
from utils.video_processor import VideoProcessor
from utils.visualizer import (
    draw_detections, create_heatmap, display_risk_score,
    create_risk_gauge, get_risk_label
)
import config

# Page configuration
st.set_page_config(
    page_title="Crowd Stampede Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        text-align: center;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
    }
    .risk-moderate {
        background-color: #fff3cd;
        color: #856404;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    """Main application"""
    
    # Title
    st.markdown('<div class="main-header">🚨 Crowd Stampede Detection System</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Based Early Prediction and Prevention Using Spatio-Temporal Analysis</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model loading
        st.subheader("📦 Models")
        with st.spinner("Loading models..."):
            yolo_model = load_yolo_model()
            lstm_model = load_lstm_model()
        
        if yolo_model is None or lstm_model is None:
            st.error("❌ Failed to load models. Please check model files.")
            st.stop()
        
        st.divider()
        
        # Detection settings
        st.subheader("🎯 Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,  # Changed from 0.5 to 0.25 for better crowd detection
            step=0.05,
            help="Lower values detect more people but may include false positives"
        )
        
        show_boxes = st.checkbox("Show Bounding Boxes", value=True)
        show_heatmap = st.checkbox("Show Density Heatmap", value=True)
        
        st.divider()
        
        # Risk thresholds
        st.subheader("⚠️ Alert Thresholds")
        st.info(f"""
        - 🟢 Low Risk: 0-{config.RISK_THRESHOLD_LOW}
        - 🟡 Moderate Risk: {config.RISK_THRESHOLD_LOW}-{config.RISK_THRESHOLD_HIGH}
        - 🔴 High Risk: {config.RISK_THRESHOLD_HIGH}-100
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📹 Video Upload", "ℹ️ About", "📊 How It Works"])
    
    with tab1:
        process_video_tab(yolo_model, lstm_model, confidence_threshold, 
                         show_boxes, show_heatmap)
    
    with tab2:
        show_about_tab()
    
    with tab3:
        show_workflow_tab()


def process_video_tab(yolo_model, lstm_model, confidence_threshold, 
                      show_boxes, show_heatmap):
    """Video upload and processing tab"""
    
    st.header("📹 Upload Video for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a video file (MP4/AVI)",
        type=['mp4', 'avi', 'mov', 'mkv']
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # Display video info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.video(video_path)
        
        with col2:
            st.markdown("### 📝 Video Info")
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            st.metric("Duration", f"{duration:.1f}s")
            st.metric("FPS", fps)
            st.metric("Resolution", f"{width}x{height}")
            st.metric("Frames", frame_count)
        
        st.divider()
        
        # Process button
        if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
            process_video(video_path, yolo_model, lstm_model, 
                         confidence_threshold, show_boxes, show_heatmap)
        
        # Cleanup
        try:
            os.unlink(video_path)
        except:
            pass
    else:
        st.info("👆 Please upload a video file to begin analysis")


def process_video(video_path, yolo_model, lstm_model, confidence_threshold,
                  show_boxes, show_heatmap):
    """Process video and display results"""
    
    st.subheader("🔄 Processing Video...")
    
    # Initialize video processor
    video_processor = VideoProcessor()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create placeholders
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Display columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎥 Original Frame")
        frame_placeholder = st.empty()
    
    with col2:
        st.markdown("### 🔥 Density Heatmap")
        heatmap_placeholder = st.empty()
    
    with col3:
        st.markdown("### 📊 Risk Analysis")
        risk_placeholder = st.empty()
    
    # Metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    risk_metric = metric_col1.empty()
    count_metric = metric_col2.empty()
    status_metric = metric_col3.empty()
    frame_metric = metric_col4.empty()
    
    # Risk gauge
    gauge_placeholder = st.empty()
    
    # Process frames
    frame_idx = 0
    risk_scores = []
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for faster processing
            if frame_idx % config.FRAME_SKIP != 0:
                frame_idx += 1
                continue
            
            # Resize frame for display
            display_frame = video_processor.resize_frame(frame)
            
            # YOLOv8 detection
            results = yolo_model(display_frame, conf=confidence_threshold, verbose=False)
            
            # Extract detections - IMPROVED LOGIC
            detections = None
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if len(boxes) > 0:
                    # Get class IDs
                    cls_ids = boxes.cls.cpu().numpy()
                    
                    # Check if model has 'names' attribute (class mapping)
                    has_person_class = False
                    person_mask = None
                    
                    if hasattr(yolo_model, 'names'):
                        # Check if model was trained on COCO (class 0 = person)
                        if 0 in yolo_model.names and 'person' in str(yolo_model.names[0]).lower():
                            # Standard COCO model - filter for person class
                            person_mask = cls_ids == 0
                            has_person_class = person_mask.any()
                        else:
                            # Custom trained model - check for person-like classes
                            for cls_id, cls_name in yolo_model.names.items():
                                if 'person' in str(cls_name).lower() or 'people' in str(cls_name).lower() \
                                   or 'human' in str(cls_name).lower() or 'crowd' in str(cls_name).lower():
                                    person_mask = cls_ids == cls_id
                                    has_person_class = person_mask.any()
                                    if has_person_class:
                                        break
                    
                    # If no person class found, use ALL detections (custom crowd model)
                    if not has_person_class or person_mask is None:
                        # Fallback: Use all detections
                        person_mask = np.ones(len(cls_ids), dtype=bool)
                    
                    if person_mask.any():
                        # Extract person detections
                        detections = boxes.xyxy.cpu().numpy()[person_mask]
                        confs = boxes.conf.cpu().numpy()[person_mask]
                        detections = np.concatenate([detections, confs.reshape(-1, 1)], axis=1)


            
            # Extract features
            spatial_features = video_processor.extract_spatial_features(
                detections, display_frame.shape
            )
            
            optical_flow = video_processor.extract_optical_flow(display_frame)
            
            temporal_features = video_processor.extract_temporal_features(
                spatial_features, optical_flow
            )
            
            # Predict risk score
            risk_score = 0.0
            if len(video_processor.feature_history) >= 5:  # Minimum sequence length
                risk_score = predict_risk_score(lstm_model, temporal_features)
                risk_scores.append(risk_score)
            
            # Visualizations
            # Original with detections
            if show_boxes and detections is not None:
                vis_frame = draw_detections(display_frame, detections)
            else:
                vis_frame = display_frame
            
            # Add risk score overlay
            vis_frame = display_risk_score(
                vis_frame, risk_score, spatial_features['person_count']
            )
            
            # Heatmap
            if show_heatmap:
                heatmap_frame = create_heatmap(display_frame, detections)
            else:
                heatmap_frame = display_frame
            
            # Update display
            frame_placeholder.image(
                cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
            )
            
            heatmap_placeholder.image(
                cv2.cvtColor(heatmap_frame, cv2.COLOR_BGR2RGB)
            )
            
            # Update metrics
            risk_level = get_risk_label(risk_score)
            risk_metric.metric("Risk Score", f"{risk_score:.1f}")
            count_metric.metric("People Count", spatial_features['person_count'])
            status_metric.metric("Status", risk_level)
            frame_metric.metric("Frame", f"{frame_idx}/{total_frames}")
            
            # Update risk gauge
            if risk_score > 0:
                fig = create_risk_gauge(risk_score)
                gauge_placeholder.pyplot(fig)
            
            # Update progress
            progress = min((frame_idx + 1) / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_idx + 1}/{total_frames}")
            
            frame_idx += 1
    
    finally:
        cap.release()
        video_processor.reset()
    
    # Final statistics
    st.success("✅ Video processing complete!")
    
    if risk_scores:
        st.divider()
        st.subheader("📈 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Risk", f"{np.mean(risk_scores):.1f}")
        col2.metric("Max Risk", f"{np.max(risk_scores):.1f}")
        col3.metric("Min Risk", f"{np.min(risk_scores):.1f}")
        col4.metric("Frames Analyzed", len(risk_scores))
        
        # Risk score chart
        st.line_chart(risk_scores)


def show_about_tab():
    """About tab content"""
    
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### Project Overview
    This application uses advanced AI techniques to detect and predict crowd stampede risks 
    in real-time from video footage.
    
    ### Technology Stack
    - **YOLOv8**: State-of-the-art person detection
    - **LSTM**: Temporal pattern analysis for risk prediction
    - **Optical Flow**: Motion analysis for crowd dynamics
    - **Streamlit**: Interactive web interface
    
    ### Features
    - ✅ Real-time person detection and tracking
    - ✅ Spatial feature extraction (density, distribution)
    - ✅ Temporal pattern analysis
    - ✅ Risk score prediction (0-100 scale)
    - ✅ Color-coded alerts (Green/Yellow/Red)
    - ✅ Density heatmap visualization
    - ✅ Automated warnings
    
    ### Datasets
    - ShanghaiTech Crowd Counting Dataset
    - UCSD Pedestrian Database
    - Custom Crowd Counting Dataset
    """)


def show_workflow_tab():
    """Workflow tab content"""
    
    st.header("📊 How It Works")
    
    st.markdown("""
    ### Processing Pipeline
    
    1. **📹 Video Input**
       - Upload video file or use camera feed
       - Support for MP4, AVI, and other formats
    
    2. **👥 Person Detection (YOLOv8)**
       - Detect and track individuals frame-by-frame
       - Extract bounding box coordinates
    
    3. **📐 Spatial Feature Extraction**
       - Calculate crowd density
       - Analyze distribution patterns
       - Determine crowd center
    
    4. **🌊 Temporal Feature Extraction**
       - Optical flow for motion analysis
       - Movement pattern recognition
       - Sequential data preparation
    
    5. **🧠 LSTM Risk Prediction**
       - Analyze temporal sequences
       - Predict stampede risk patterns
       - Generate risk score (0-100)
    
    6. **🎨 Visualization**
       - Display detection results
       - Generate density heatmaps
       - Show risk scores and alerts
    
    7. **⚠️ Alert System**
       - 🟢 **Green (0-30)**: Low risk, normal crowd behavior
       - 🟡 **Yellow (30-70)**: Moderate risk, monitor closely
       - 🔴 **Red (70-100)**: High risk, immediate action required
    
    ### Alert Actions
    When risk exceeds threshold:
    - Visual and audio warnings
    - Automated notifications
    - Emergency protocol activation
    """)


if __name__ == "__main__":
    main()
