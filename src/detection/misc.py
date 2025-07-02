import cv2
import glob
import wget
from queue import Queue
from threading import Thread

def alert(condition, no_of_frames):
    if(condition):
        no_of_frames = no_of_frames + 10
    else:
        no_of_frames=0
    return no_of_frames

def parse_video_path(value):
    try:
        # Trying to convert the value to an integer
        return bool(value)
    except ValueError:
        # If conversion fails, return the value as a string
        return value

def download_files(candidate):
    video_url = candidate.videoUrl
    image_url = candidate.candidatePicture
    video_filename = wget.download(video_url, out="src/attendee_db/")
    print("#"*100)
    image_filename = wget.download(image_url, out="src/attendee_db/")
    return video_filename, image_filename

def clean_up_directory(directory="src/attendee_db/"):
    try:
        files = glob.glob(f'{directory}/*')
        for file in files:
            os.remove(file)
    except OSError as e:
        print(f"Error deleting files: {e}")

# Install the package first. [Not present in requirements because of size concerns]
# !pip install ultralytics
# 
# from ultralytics import YOLO
# def convert_to_onnx(model_path):
#     model = YOLO(model_path)
#     # Export the model to ONNX format
#     try:
#         # Exported to same path as model path
#         model.export(format="onnx")
#         return("Model exported ONNX format successfully")
#     except Exception as error:
#         return(f"Mondel conversion unsuccessful, facing following error {error}")