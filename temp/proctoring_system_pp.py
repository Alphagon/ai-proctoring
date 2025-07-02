import os
import cv2
import dlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import face_recognition
from detection.headpose_estimation import load_hp_model
from detection.face_detection import get_face_detector, find_faces
from detection.custom_detection import get_objects_count, get_objects_count_exception, people_detection, banned_object_detection, face_detection_online, \
                              comparing_faces, face_verification, get_facial_landmarks, head_pose_detection, eye_tracker

# Setup
# Attendee Face Encodings
l = os.listdir('attendee_db')
known_face_encodings = []
known_face_names = []

for image in l:
    attendee_image = face_recognition.load_image_file('attendee_db/'+image)
    attendee_face_encoding = face_recognition.face_encodings(attendee_image)[0]

    known_face_encodings.append(attendee_face_encoding)
    known_face_names.append(image.split('.')[0])

# Headpose Model
h_model = load_hp_model('models/Headpose_customARC_ZoomShiftNoise.hdf5')

# Face Detection Model
face_model = get_face_detector(modelFile='models/res10_300x300_ssd_iter_140000.caffemodel', configFile='models/deploy.prototxt')

# Face Landmarks Model
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Others
video_path = '/home/yravi/test_2.mp4'
video_capture = cv2.VideoCapture(video_path)
fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_count = 0
video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
aspect_ratio = video_width / video_height

people_detection_frames = 0
banned_object_frames = 0
face_verification_frames = 0
headpose_detection_frames = 0
eye_tracking_frames = 0
face_detection_frames = 0
flag = True
DEBUG = False

# Desired output window size
output_width = 620
output_height = int(output_width / aspect_ratio)

print(frame_count)

def process_objects_and_people(frame):
    try:
        count_items = get_objects_count(frame.copy())
    except Exception as error:
        count_items = get_objects_count_exception()
        print(error)
        
    people_frames = people_detection(count_items, people_detection_frames, frame_count, fps, None, debug=DEBUG)
    banned_frames = banned_object_detection(count_items, banned_object_frames, frame_count, fps, None, debug=DEBUG)
    
    return count_items, people_frames, banned_frames

def process_faces(frame, face_model):
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
    faces = find_faces(small_frame, face_model)
    return faces

def process_face_detection(faces):
    return face_detection_online(faces, face_detection_frames, frame_count, fps, None, debug=DEBUG)

def process_face_verification(small_frame, face):
    global flag
    if flag:
        name = comparing_faces(small_frame, face, known_face_names, known_face_encodings)
        flag = False
        return name
    return None

def process_landmarks_and_pose(frame, face, name):
    facial_landmarks = get_facial_landmarks(predictor, face, frame)
    pose_frames, frame2, headpose_condition = head_pose_detection(
        h_model, frame, frame.copy(), face, headpose_detection_frames, frame_count, fps, None, debug=DEBUG)
    eye_frames = eye_tracker(frame, facial_landmarks, eye_tracking_frames, headpose_condition, frame_count, fps, None, debug=DEBUG)
    
    return facial_landmarks, pose_frames, eye_frames

#################################################### MAIN #####################################################
with ThreadPoolExecutor(max_workers=6) as executor:
    while True:
        # Grabbing a frame of video
        ret, frame = video_capture.read()
        frame_count += 1

        if not ret:
            print("End of video")
            break

        frame = cv2.resize(frame, (output_width, output_height))
        frame2 = frame.copy()

        # Frame-Skipping to save time
        if frame_count % 5 == 0:
            print(frame_count)
            futures = {}

            # Parallel execution
            futures['objects_people'] = executor.submit(process_objects_and_people, frame2)
            futures['faces'] = executor.submit(process_faces, frame2, face_model)

            # Wait for the object and people detection to complete
            count_items, people_detection_frames, banned_object_frames = futures['objects_people'].result()

            if count_items['person'] > 0:
                faces = futures['faces'].result()

                if len(faces) == 1:
                    face = faces[0]
                    small_frame = cv2.resize(frame2, (0, 0), fx=0.2, fy=0.2)

                    # Proceed with face verification and facial landmarks in parallel
                    futures['face_verification'] = executor.submit(process_face_verification, small_frame, face)
                    futures['landmarks_pose'] = executor.submit(process_landmarks_and_pose, frame2, face, futures['face_verification'].result())

                    # Extract results
                    name = futures['face_verification'].result()
                    facial_landmarks, headpose_detection_frames, eye_tracking_frames = futures['landmarks_pose'].result()

                    # Perform face verification if necessary
                    if name:
                        face_verification_frames = face_verification(name, face_verification_frames, frame_count, fps, None, debug=DEBUG)
                else:
                    face_detection_frames = process_face_detection(faces)

            if DEBUG:
                horizontalAppendedImg = np.hstack((frame2, np.zeros((frame2.shape[0], frame2.shape[1], 3), np.uint8)))
                cv2.imshow("Proctoring_Window", horizontalAppendedImg)

        # Display the resulting image
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("closing window...")
            break

print(frame_count)

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()