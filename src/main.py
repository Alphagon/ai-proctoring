from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import subprocess
import os

app = FastAPI(title="AI-Proctoring - Automatic Run",
              version="0.1")

scheduler = BackgroundScheduler()
scheduler.start()

def process_video_task(debug: bool):
    command = ["python", "offline_proctoring_system.py", "--debug", str(debug)]
    try:
        print("command running")
        subprocess.run(command, check=True)
        print("ended")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def process_video(debug: bool = False, background_tasks: BackgroundTasks = None):
    scheduler.add_job(
        process_video_task,
        trigger=IntervalTrigger(minutes=5),  #"hours=1" for hourly runs
        args=[debug],
        id="process_video_job",
        replace_existing=True
    )
    return HTMLResponse(content="Processing started and scheduled.", status_code=200)

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()

# uvicorn src.main:app --reload