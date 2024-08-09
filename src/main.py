from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import subprocess
import os

app = FastAPI()

# Endpoint to process the video
@app.post("/process-video/")
async def process_video(video_path: str):
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    # Run the face detection and logging script
    log_file_path = "/home/yravi/Documents/ai-proctoring/src/sproctoring_alerts.log"
    
    # Ensure the log file is cleared before running the script
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    # Run the processing script (you can run it as a subprocess)
    try:
        # Change the following command to point to your Python environment and script
        command = ["python", "online_proctoring_system.py", "--video_path", video_path]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Read the log file after processing
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as log_file:
            log_content = log_file.read()
    else:
        log_content = "Log file not found."

    return HTMLResponse(content=f"<pre>{log_content}</pre>", status_code=200)

# Run the application with: uvicorn your_fastapi_file:app --reload