def alert(condition, no_of_frames):
    if(condition):
        no_of_frames = no_of_frames + 1
        # if (no_of_frames > ALERT_THRESHOLD):
        #     log_alert(f"ALERT: {condition} condition met", frame_count, fps)
    else:
        no_of_frames=0
    return no_of_frames