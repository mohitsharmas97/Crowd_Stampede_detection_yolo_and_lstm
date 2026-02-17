# AI-Based Early Prediction and Prevention of Crowd Stampede

## 🚨 Overview
This project implements an AI-based system for early detection and prediction of crowd stampede risks using spatio-temporal crowd behavior analysis. The system combines YOLOv8 for person detection and LSTM networks for temporal pattern recognition to provide real-time risk assessment.

## 🎯 Features
- ✅ **Real-time Person Detection**: YOLOv8-based detection and tracking
- ✅ **Spatial Analysis**: Crowd density and distribution patterns
- ✅ **Temporal Analysis**: Motion patterns using optical flow
- ✅ **Risk Prediction**: LSTM-based stampede risk scoring (0-100)
- ✅ **Visual Alerts**: Color-coded warnings (Green/Yellow/Red)
- ✅ **Density Heatmaps**: Visual representation of crowd concentration
- ✅ **Interactive Dashboard**: Streamlit-based web interface

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- 8GB RAM minimum (16GB recommended)

### Dependencies
See `requirements.txt` for complete list:
```bash
pip install -r requirements.txt
```

## 🚀 Installation

### 1. Clone or Download Project
```bash
cd Crowd_stampede_detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Model Files
Ensure the following trained models are present in the project directory:
- `best.pt` - YOLOv8 person detection model
- `lstm_risk_model.pt` - LSTM risk prediction model

## 💻 Usage

### Running the Application
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Interface

1. **Upload Video**
   - Click "Browse files" to upload a video (MP4/AVI)
   - Supported formats: MP4, AVI, MOV, MKV
   - Recommended: 25-30 FPS, clear crowd footage

2. **Configure Settings**
   - Adjust confidence threshold for detection
   - Enable/disable bounding boxes
   - Toggle density heatmap visualization

3. **Analyze Video**
   - Click "Analyze Video" button
   - Watch real-time processing
   - View risk scores and alerts

4. **Monitor Results**
   - **Risk Score**: 0-100 scale indicating stampede risk
   - **Alert Level**: 
     - 🟢 Green (0-30): Low risk
     - 🟡 Yellow (30-70): Moderate risk
     - 🔴 Red (70-100): High risk
   - **People Count**: Number of detected persons
   - **Density Heatmap**: Visual crowd concentration

## 🏗️ Project Structure
```
Crowd_stampede_detection/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration settings
├── best.pt                     # YOLOv8 trained model
├── lstm_risk_model.pt          # LSTM trained model
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── stampede-detection.ipynb    # Training notebook
└── utils/
    ├── __init__.py
    ├── model_loader.py         # Model loading utilities
    ├── video_processor.py      # Video processing functions
    └── visualizer.py           # Visualization utilities
```

## 🧠 How It Works

### Workflow
1. **Video Input**: Upload video or use camera feed
2. **Detection**: YOLOv8 detects persons frame-by-frame
3. **Spatial Features**: Extract density, distribution patterns
4. **Temporal Features**: Calculate optical flow and motion patterns
5. **Risk Prediction**: LSTM analyzes temporal sequences
6. **Visualization**: Display results with heatmaps and alerts

### Algorithms
- **YOLOv8**: State-of-the-art object detection
- **Optical Flow**: Farneback method for motion analysis
- **LSTM**: Long Short-Term Memory for temporal prediction
- **Risk Scoring**: Custom algorithm combining spatial and temporal features

## 📊 Datasets Used
- [ShanghaiTech Crowd Counting Dataset](https://www.kaggle.com/datasets/tthien/shanghaitech)
- [UCSD Pedestrian Database](https://www.kaggle.com/datasets/aryashah2k/ucsd-pedestrian-database)
- [Crowd Counting Dataset](https://www.kaggle.com/datasets/trainingdatapro/crowd-counting-dataset)

## ⚙️ Configuration

Edit `config.py` to customize:
- Model paths
- Risk thresholds
- Detection parameters
- Visualization settings
- Feature extraction parameters

## 🔧 Troubleshooting

### Common Issues

**Models not found:**
```
Ensure best.pt and lstm_risk_model.pt are in the project root directory
```

**CUDA out of memory:**
```
Reduce video resolution or process fewer frames (adjust FRAME_SKIP in config.py)
```

**Slow processing:**
```
- Use GPU if available
- Increase FRAME_SKIP value
- Reduce video resolution
```

## 📝 Training Custom Models

Refer to `stampede-detection.ipynb` for:
- Data preparation
- Model training
- Hyperparameter tuning
- Evaluation metrics

## 🎓 Credits

**Technology Stack:**
- Python
- Streamlit
- PyTorch
- OpenCV
- Ultralytics YOLOv8

**Datasets:**
- ShanghaiTech University
- UCSD Computer Vision Group

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or support, please open an issue in the project repository.

---

**Note:** This system is designed for early warning and should be used in conjunction with proper crowd management protocols and human oversight.
