import logging
from math import ceil
from datetime import datetime

def alert(condition, no_of_frames, frame_count, fps, ALERT_THRESHOLD):
    if(condition):
        no_of_frames = no_of_frames + 1
        if (no_of_frames > ALERT_THRESHOLD):
            log_alert(f"ALERT: {condition} condition met", frame_count, fps)
    else:
        no_of_frames=0
    return no_of_frames

# Setup logging
logging.basicConfig(filename='proctoring_alerts.log', level=logging.INFO, format='%(asctime)s %(message)s')

def log_alert(message, frame_count, fps):
    current_time = ceil(frame_count / fps)
    logging.info(f"Second {current_time:.2f}: {message}")