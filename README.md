# AI Proctoring

This is a batch processing proctoring tool. It is run once the video has been recorded and pushed to S3.

## Functionalities

So far, we've worked on the video part of the proctoring. The next step would be to proctor the audio part. The functionalities in the video proctoring are:
- **Object Detection**: Trained a custom object detection model using YOLOv8n with the Ultralytics library. Once trained, the `.pth` model is converted to an `onnx` model. The code can be found in `misc.py`. The following objects are detected: `["person", "laptop", "remote", "cell phone", "book", "tv"]`.
- **Banned Object Detection**: Using the object detection results, we identify banned objects, i.e., "laptop", "remote", "cell phone", "book", and "tv". If these objects are present in the video for 250 consecutive frames, a log will be created.
- **Face Detection**: This model detects if there are multiple faces in the frame. Similar to Banned Object Detection, if a face is detected for more than 250 consecutive frames, a log will be created.
- **Face Verification**: Using the picture taken at the beginning of the test, the tool verifies if the person writing or attending the exam is the same throughout the session.
- **Head Pose Detection**: This model estimates the head pose of the attendee. It checks if the attendee is looking at the screen or not. If not, a log is created for irregular behavior.
- **Eye Tracking**: Using head pose detection and facial landmarks, the eyes of the attendee are tracked. If the attendee is looking away from the screen, a log is created for irregular behavior.

## Setting Up the Proctoring Environment

1. Create a new conda virtual environment:
   ```
   conda create -n ai-proctor python==3.8.0 -y
   ```
2. Activate the created virtual environment:
   ```
   conda activate ai-proctor
   ```
3. Install the dlib library from conda:
   ```
   conda install -c conda-forge dlib -y
   ```
4. Install the required packages:
   ```
   pip install --no-cache-dir -r requirements/core.txt
   pip install --no-cache-dir -r requirements/dev.txt
   pip install --no-cache-dir -r requirements/api.txt
   ```
   or run this single command
   ```
   for req in requirements/*.txt; do pip install --no-cache-dir -r $req; done
   ```
5. And run the following command
   ```
   uvicorn src.main:app --reload
   ```

## Next Action Item
The next action item is to detect any anomalies in the recorded audio.