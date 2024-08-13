import cv2
from queue import Queue
from threading import Thread

def alert(condition, no_of_frames):
    if(condition):
        no_of_frames = no_of_frames + 1
        # if (no_of_frames > ALERT_THRESHOLD):
        #     log_alert(f"ALERT: {condition} condition met", frame_count, fps)
    else:
        no_of_frames=0
    return no_of_frames

def parse_video_path(value):
    try:
        # Try to convert the value to an integer
        return bool(value)
    except ValueError:
        # If conversion fails, return the value as a string
        return value