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

class VideoStream:
    def __init__(self, video_path, frame_skip=5, queue_size=128):
        self.cap = cv2.VideoCapture(video_path)
        self.stopped = False
        self.frame_skip = frame_skip
        self.Q = Queue(maxsize=queue_size)

        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.aspect_ratio = self.width / self.height if self.height != 0 else None

        self.frame_count = 0  # Initialize frame counter

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.cap.release()
                return

            grabbed, frame = self.cap.read()
            if not grabbed:
                self.stop()
                return
            
            self.frame_count += 1

            # Only enqueue every 'frame_skip' frames
            if self.frame_count % self.frame_skip == 0:
                if not self.Q.full():
                    self.Q.put(frame)

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0

    def stop(self):
        self.stopped = True

    def get_info(self):
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio
        }

def parse_video_path(value):
    try:
        # Try to convert the value to an integer
        return int(value)
    except ValueError:
        # If conversion fails, return the value as a string
        return value