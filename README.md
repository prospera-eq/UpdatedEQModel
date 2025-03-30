# 🧠 EQ Analysis System

An intelligent system that analyzes emotional intelligence and trust in communications using advanced AI techniques.

## 🌟 Key Features

### 📹 Video Analysis

- Analyzes facial expressions and body language
- Tracks attention and engagement levels
- Provides frame-by-frame emotional scoring
- Generates actionable feedback for improvement

### 🗣️ Voice Analysis

- Evaluates tone and emotional content in speech
- Detects confidence and authenticity markers
- Measures speaking pace and clarity

### 💬 Text Analysis

- Sentiment and emotion detection in written communication
- Trust indicators assessment
- Communication style evaluation

## 🚀 Quick Start

1. **Setup Environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Using the Agents**

   a. **Video Analysis**

   ```bash
   # Analyze a video file
   curl -X POST -F "file=@your_video.mp4" http://localhost:3000/analyze/video
   ```

   b. **Voice Analysis**

   ```bash
   # Analyze an audio file
   curl -X POST -F "file=@your_audio.wav" http://localhost:3000/analyze/voice
   ```

   c. **Text Analysis**

   ```bash
   # Analyze text content
   curl -X POST -H "Content-Type: application/json" \
        -d '{"text": "Your text here"}' \
        http://localhost:3000/analyze/text
   ```

   d. **Comprehensive Analysis**

   ```bash
   # Analyze all media types together
   curl -X POST \
        -F "video_file=@video.mp4" \
        -F "voice_file=@audio.wav" \
        -F "text=Your text here" \
        http://localhost:3000/analyze/comprehensive
   ```

## 📊 Output

### Video Analysis Results

- Frame-by-frame emotional scores (happy, neutral, surprised, focused)
- Body language and gesture analysis
- Attention tracking metrics
- CSV report with detailed frame analysis

### Voice Analysis Results

- Emotional tone mapping
- Speech clarity and confidence scores
- Key moments and emphasis points
- Transcription with sentiment markers

### Text Analysis Results

- Emotional intelligence indicators
- Trust and authenticity metrics
- Communication style assessment
- Suggested improvements

### Comprehensive Reports

- Combined EQ scoring across all mediums
- Cross-referenced emotional patterns
- Unified recommendations
- Exportable CSV/JSON data

## 🔧 Configuration

### Video Agent (`agents/video_agent.py`)

- Frame sampling rate (default: 1 frame/3s)
- Face detection confidence threshold
- Gesture recognition sensitivity

### Voice Agent (`agents/voice_agent.py`)

- Audio sampling rate
- Voice emotion detection threshold
- Transcription language settings

### Text Agent (`agents/text_agent.py`)

- Language model selection
- Sentiment analysis thresholds
- Response detail level


## 📈 Performance Notes

- Video analysis processes 1 frame every 3 seconds for optimal performance
- Supports common video formats (MP4, AVI, MOV)
- Typical processing time: ~2-3 minutes per minute of video
