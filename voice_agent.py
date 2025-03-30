from typing import Dict, Any
import librosa
import numpy as np
import whisper
import torch
from transformers import pipeline

class VoiceAnalysisAgent:
    def __init__(self):
        try:
            print("Loading Whisper model...")
            # Initialize Whisper model for speech recognition
            self.whisper = whisper.load_model("base")
            print("Loading emotion classifier...")
            # Initialize emotion recognition pipeline
            self.emotion_classifier = pipeline(
                "audio-classification",
                model="MIT/ast-finetuned-speech-commands-v2",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Downloading models... this may take a few minutes")
            self.whisper = whisper.load_model("base")
            self.emotion_classifier = pipeline(
                "audio-classification",
                model="MIT/ast-finetuned-speech-commands-v2",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

    async def analyze_voice(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Analyze voice communications for emotional intelligence and trust indicators.
        
        Args:
            audio_file_path (str): Path to the audio file
            
        Returns:
            Dict containing analysis results including:
            - emotional_score: Overall emotional intelligence score
            - trust_score: Trust relationship score
            - voice_metrics: Various voice-related metrics
            - recommendations: Suggested improvements
        """
        try:
            # Load and process audio
            audio, sr = librosa.load(audio_file_path)
            
            # Extract audio features
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            pitch = librosa.pitch_tuning(y=audio)
            
            # Calculate additional voice metrics
            mfccs = librosa.feature.mfcc(y=audio, sr=sr)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            
            # Perform speech-to-text using Whisper
            transcription = self.whisper.transcribe(audio_file_path)
            
            # Analyze emotion in audio
            emotion_results = self.emotion_classifier(audio_file_path)
            
            # Calculate emotional scores based on audio features
            energy = np.mean(librosa.feature.rms(y=audio))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio))
            
            # Analyze voice characteristics
            analysis = {
                "emotional_score": float(energy * 0.7 + zero_crossing_rate * 0.3),  # Simplified scoring
                "trust_score": float(0.5 + np.std(mfccs) * 0.1),  # Simplified trust scoring
                "voice_metrics": {
                    "tempo": float(tempo),
                    "pitch_variation": float(np.std(pitch)),
                    "energy": float(energy),
                    "spectral_centroid": float(np.mean(spectral_centroid)),
                    "transcription": transcription,
                    "detected_emotion": emotion_results[0]["label"],
                    "emotion_confidence": float(emotion_results[0]["score"])
                },
                "recommendations": [
                    "Maintain consistent speaking pace" if tempo > 120 else "Good speaking tempo",
                    "Try to vary pitch more" if np.std(pitch) < 0.1 else "Good pitch variation",
                    "Speak with more energy" if energy < 0.3 else "Good energy level"
                ]
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error in voice analysis: {str(e)}")
            return None
