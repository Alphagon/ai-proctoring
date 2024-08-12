from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import subprocess
import os

app = FastAPI()

# Endpoint to process the video
@app.post("/process-video/")
async def process_video(video_path: str, debug: str):
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    # Run the face detection and logging script
    log_file_path = "proctoring_alerts.log"
    
    # Ensuring the log file is cleared before running the script
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    # Run the processing script
    try:
        command = ["python", "offline_proctoring_system.py", "--video_path", video_path, "--debug", debug]
        subprocess.run(command, check=True)
        pass
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Read the log file after processing
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as log_file:
            log_content = log_file.read()
    else:
        log_content = "Log file not found."

    return HTMLResponse(content=f"{log_content}", status_code=200)

# Run the application with: uvicorn your_fastapi_file:app --reload
