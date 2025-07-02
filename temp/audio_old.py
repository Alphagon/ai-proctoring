import moviepy.editor as mp
import librosa
import speech_recognition as speech_recog
import numpy as np

def analyze_video_audio(video_path):
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile("temp_audio.wav")

    # Loading audio file
    y, sr = librosa.load("temp_audio.wav")

    # Perform voice activity detection
    intervals = librosa.effects.split(y, top_db=20)

    # Check if there's any voice activity
    if len(intervals) > 0:
        print("Voice detected in the video")
        # Use speech recognition to detect multiple speakers
        r = speech_recog.Recognizer()
        with speech_recog.AudioFile("temp_audio.wav") as source:
            audio_data = r.record(source)
            try:
                # Attempt to recognize speech
                text = r.recognize_google(audio_data)
                print("Detected speech", text)

                # Simple heuristic to check for multiple speakers
                if len(text.split()) > 20:  # Adjust this threshold as needed
                    print("Multiple speakers likely present")
                else:
                    print("Likely single speaker")
            except speech_recog.UnknownValueError:
                print("Speech recognition could not understand the audio")
            except speech_recog.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
    else:
        print("No voice detected in the video")

    # Clean up temporary file
    import os
    os.remove("temp_audio.wav")

# Usage
analyze_video_audio("/home/yravi/test_2.mp4")