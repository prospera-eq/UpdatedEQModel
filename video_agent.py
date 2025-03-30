from typing import Dict, Any, List
import cv2
import numpy as np
import mediapipe as mp
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
import pandas as pd
import os

class VideoFrame(BaseModel):
    frame_number: int
    emotions: Dict[str, float]
    gestures: List[str]
    attention_score: float

class VideoAnalysisAgent:
    def __init__(self):
        self.console = Console()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_holistic = mp.solutions.holistic
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _analyze_face(self, frame: np.ndarray) -> Dict[str, float]:
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return {}
        
        landmarks = results.multi_face_landmarks[0]
        emotions = {
            'neutral': 0.5,
            'happy': self._calculate_happiness(landmarks),
            'surprised': self._calculate_surprise(landmarks),
            'focused': self._calculate_focus(landmarks)
        }
        return emotions

    def _analyze_gestures(self, frame: np.ndarray) -> List[str]:
        results = self.holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return ["no_pose_detected"]
        
        # Basic gesture detection based on pose landmarks
        gestures = []
        # Add your gesture detection logic here
        gestures.append("neutral_pose")
        return gestures

    async def analyze_video(self, video_path: str) -> Dict[str, Any]:
        self.console.print("[bold green]Starting video analysis...[/bold green]")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = frame_count / fps
        
        # Analyze 1 frame every 3 seconds
        sample_interval = 3 * fps
        total_samples = frame_count // sample_interval + 1
        
        self.console.print(f"Video duration: {duration:.1f} seconds")
        self.console.print(f"Analyzing {total_samples} frames...")
        
        frames_analysis: List[VideoFrame] = []
        frame_number = 0
        samples_processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % sample_interval == 0:
                # Show progress
                samples_processed += 1
                self.console.print(f"[cyan]Progress: {samples_processed}/{total_samples} frames ({(samples_processed/total_samples*100):.1f}%)[/cyan]")
                
                # Resize frame for faster processing
                frame = cv2.resize(frame, (640, 360))
                
                # Analyze frame
                emotions = self._analyze_face(frame)
                gestures = self._analyze_gestures(frame)
                
                frames_analysis.append(VideoFrame(
                    frame_number=frame_number,
                    emotions=emotions,
                    gestures=gestures,
                    attention_score=emotions.get('focused', 0.5)
                ))

            frame_number += 1
            if frame_number >= frame_count:
                break

        cap.release()

        # Aggregate results
        overall_emotions = {}
        overall_attention = 0.0
        total_analyzed_frames = len(frames_analysis)

        for frame in frames_analysis:
            for emotion, score in frame.emotions.items():
                overall_emotions[emotion] = overall_emotions.get(emotion, 0) + score
            overall_attention += frame.attention_score

        if total_analyzed_frames > 0:
            for emotion in overall_emotions:
                overall_emotions[emotion] /= total_analyzed_frames
            overall_attention /= total_analyzed_frames

        self.console.print("[bold green]Video analysis complete![/bold green]")

        return {
            "overall_emotions": overall_emotions,
            "average_attention_score": float(overall_attention),
            "frames_analyzed": total_analyzed_frames,
            "video_length_seconds": frame_count / fps,
            "recommendations": self._generate_recommendations(frames_analysis)
        }

    def _calculate_happiness(self, landmarks) -> float:
        """Calculate happiness score from facial landmarks"""
        # In a real implementation, you'd calculate this based on mouth corners and other features
        # For now, we'll use a placeholder that varies based on the landmarks
        return np.random.uniform(0.4, 0.8)

    def _calculate_surprise(self, landmarks) -> float:
        """Calculate surprise score from facial landmarks"""
        # In a real implementation, you'd calculate this based on eyebrow position and eye openness
        return np.random.uniform(0.2, 0.6)

    def _calculate_focus(self, landmarks) -> float:
        """Calculate focus/attention score from facial landmarks"""
        # In a real implementation, you'd calculate this based on eye gaze and head pose
        return np.random.uniform(0.3, 0.9)

    def _generate_recommendations(self, frames_analysis: List[VideoFrame]) -> List[str]:
        """Generate recommendations based on frame analyses"""
        recommendations = []
        
        # Analyze emotions
        avg_emotions = {}
        for frame in frames_analysis:
            for emotion, score in frame.emotions.items():
                avg_emotions[emotion] = avg_emotions.get(emotion, 0) + score
        
        if frames_analysis:
            for emotion in avg_emotions:
                avg_emotions[emotion] /= len(frames_analysis)
            
            if avg_emotions.get('focused', 0) < 0.5:
                recommendations.append("Try to maintain better focus and attention")
            if avg_emotions.get('happy', 0) < 0.4:
                recommendations.append("Consider showing more positive expressions when appropriate")
        
        # Analyze gestures
        neutral_poses = sum(1 for frame in frames_analysis if "neutral_pose" in frame.gestures)
        if neutral_poses / len(frames_analysis) > 0.8:
            recommendations.append("Try to be more dynamic in your body language and gestures")
        
        return recommendations
