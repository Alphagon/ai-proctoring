import moviepy.editor as mp
import numpy as np
from pydub import AudioSegment
import tempfile
import os

def extract_audio(video_path):
    """Extract audio from video and save to a temporary file."""
    video = mp.VideoFileClip(video_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        temp_audio_path = temp_audio_file.name
    video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
    return temp_audio_path

def load_audio(audio_path):
    """Load audio file and convert to numpy array."""
    audio = AudioSegment.from_wav(audio_path)
    sample_rate = audio.frame_rate
    samples = np.array(audio.get_array_of_samples())

    # Checking if audio is stereo
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
        samples = np.mean(samples, axis=1)  # Converting to mono by averaging channels

    samples = samples / np.max(np.abs(samples))
    return samples, sample_rate


# This threshold is used to determine if a part of the audio is considered "voice" or just background noise. 
# The audio samples are normalized to a range of -1 to 1, so 0.02 represents 2% of the maximum possible amplitude. 
# Any sample with an absolute value above this threshold is considered part of a voice segment. 
# This value (0.02) is a starting point and may need adjustment depending on the audio characteristics of your videos. 
# If it's too low, it might pick up background noise; if it's too high, it might miss quieter speech.
def detect_voice_segments(samples, sample_rate, threshold=0.06):
    """Detect voice segments in the audio."""
    voice_segments = np.where(np.abs(samples) > threshold)[0]
    segments = np.split(voice_segments, np.where(np.diff(voice_segments) > sample_rate)[0] + 1)
    return segments

def analyze_segments(segments, sample_rate):
    """Analyze the detected segments and print information."""
    print("Speech segments:")
    for i, segment in enumerate(segments, 1):
        start_time = segment[0] / sample_rate
        end_time = segment[-1] / sample_rate
        duration = end_time - start_time
        q1, r1 = divmod(start_time, 60)
        q2, r2 = divmod(end_time, 60)
        print(f"Segment {i}: {int(q1)}:{r1:.2f}s - {int(q2)}:{r2:.2f}s (duration: {duration:.2f}s)")
    
    if len(segments) > 5:  # This threshold can be adjusted
        print("Multiple speakers likely present")
    else:
        print("Likely single speaker")

def analyze_video_audio(video_path):
    """Main function to analyze audio in a video."""
    temp_audio_path = extract_audio(video_path)
    
    samples, sample_rate = load_audio(temp_audio_path)
    
    segments = detect_voice_segments(samples, sample_rate)
    
    if len(segments) > 0:
        print("Voice detected in the video")
        analyze_segments(segments, sample_rate)
    else:
        print("No voice detected in the video")
    
    # Cleaning up temporary file
    os.remove(temp_audio_path)

if __name__ == "__main__":
    video_path = "/home/yravi/test_2.mp4"
    analyze_video_audio(video_path)