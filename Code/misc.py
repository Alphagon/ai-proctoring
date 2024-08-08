import logging
from datetime import datetime
from online_proctoring_system import fps, crame_count

def alert(condition, no_of_frames):
    if(condition):
        no_of_frames = no_of_frames + 1
        log_alert(f"ALERT: {condition} condition met")
    else:
        no_of_frames=0
    return no_of_frames

# Setup logging
logging.basicConfig(filename='proctoring_alerts.log', level=logging.INFO, format='%(asctime)s %(message)s')

def log_alert(message):
    current_time = frame_count / fps
    logging.info(f"Second {current_time:.2f}: {message}")