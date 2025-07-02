import cv2
import dlib
import logging
import numpy as np
from math import ceil
from .misc import alert
import face_recognition
from datetime import datetime
from collections import Counter

from .landmark_models import get_gaze_ratio
from .object_detection import yoloV8Detect
from .headpose_estimation import headpose_inference, displayHeadpose

TO_DETECT = ['person', 'laptop', 'remote', 'cell phone', 'book', 'tv']

# DEBUGGING Pre-Req
FONT = cv2.FONT_HERSHEY_PLAIN
Y_POSITION_1 = 20
Y_POSITION_2 = 60
ALERT_POSITION = (120, 190)

# ALERT Pre-Req
ALERT_THRESHOLD = 30
MULTIPLE_PEOPLE = 0
BANNED_OBJECTS = 0
FACE_VERIFICATION = True
HEADPOSE_DETECTION = False
EYE_TRACKING = False


def get_objects_count(frame):
    fclasses = yoloV8Detect(frame)

    temp = []
    for i in range(len(fclasses)):
        if(fclasses[i] in TO_DETECT):
            temp.append(fclasses[i])

    return Counter(temp)

def get_objects_count_exception():
    count_items = {}
    for obj in TO_DETECT:
        count_items[obj] = 0

    return count_items

def people_detection(count_items, no_of_frames, frame_count, fps, report, debug=False):
    global MULTIPLE_PEOPLE
    
    condition = (count_items['person'] != 1)
    no_of_frames = alert(condition, no_of_frames)

    if(no_of_frames > ALERT_THRESHOLD):
        MULTIPLE_PEOPLE += 1
        no_of_frames = 0

    if debug:
        cv2.putText(report, f"Number of people detected: {str(count_items['person'])}", (1, Y_POSITION_1), FONT, 1.1, (0, 255, 0), 2)
        
        if(no_of_frames > ALERT_THRESHOLD):
            cv2.putText(report, f"Number of people detected: {str(count_items['person'])}", (1, Y_POSITION_1), FONT, 1.1, (0, 0, 255), 2)
            cv2.putText(report, "ALERT", ALERT_POSITION, FONT, 4, (0, 0, 255), 2)

    return no_of_frames, MULTIPLE_PEOPLE

def banned_object_detection(count_items, no_of_frames, frame_count, fps, report, debug=False):
    global BANNED_OBJECTS

    condition = (count_items['laptop']>=1 or count_items['cell phone']>=1 or 
                 count_items['book']>=1 or count_items['tv']>=1)
    no_of_frames = alert(condition, no_of_frames)

    if(no_of_frames > 30):
        BANNED_OBJECTS += 1
        no_of_frames = -100

    if debug:
        cv2.putText(report, f"Banned objects detected: {str(condition)}", (1, Y_POSITION_1+20), FONT, 1.1, (0, 255, 0), 2)

        if(no_of_frames > 30):
            cv2.putText(report, f"Banned objects detected: {str(condition)}", (1, Y_POSITION_1+20), FONT, 1.1, (0, 0, 255), 2)
            cv2.putText(report, "ALERT", ALERT_POSITION, FONT, 4, (0, 0, 255), 2)

    return no_of_frames, BANNED_OBJECTS

def face_detection_offline(faces, no_of_frames, frame_count, fps, report, debug=False):
    condition = (len(faces) != 1)
    no_of_frames = alert(condition, no_of_frames)

    if(no_of_frames > ALERT_THRESHOLD):
        # log_alert(frame_count, fps)
        no_of_frames = 0

    if debug:
        cv2.putText(report, f"Number of face detected: {str(len(faces))}", (1, Y_POSITION_2), FONT, 1.1, (0, 255, 0), 2)

        if(no_of_frames > ALERT_THRESHOLD):
            cv2.putText(report, f"Number of face detected: {str(len(faces))}", (1, Y_POSITION_2), FONT, 1.1, (0, 0, 255), 2)
            cv2.putText(report, "ALERT", ALERT_POSITION, FONT, 4, (0, 0, 255), 2)        

    return no_of_frames

def comparing_faces(frame, face, attendee_name, attendee_face_encodings):
    (left, top,right,bottom) = face
    face_locations = [[top, right, bottom, left]]

    # Convert BGR image to RGB image
    rgb_small_frame = frame[:, :, ::-1]

    # get CNN feature vector
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # get similarity
    face_encoding = face_encodings[0]
    matches = face_recognition.compare_faces(attendee_face_encodings, face_encoding)
    face_distances = face_recognition.face_distance(attendee_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = attendee_name[best_match_index]
    else:
        name = "Unknown"
    return name

def face_verification(name, no_of_frames, frame_count, fps, report, debug=True):
    global FACE_VERIFICATION

    condition = (name=="Unknown")
    no_of_frames = alert(condition, no_of_frames)

    if(no_of_frames > 100):
        FACE_VERIFICATION = False
        no_of_frames = 0

    if debug:
        cv2.putText(report, f"Face Recognized: {str(name)}", (1, Y_POSITION_2+20), FONT, 1.1, (0, 255, 0), 2)

        if(no_of_frames > 100):
            cv2.putText(report, f"Face Recognized: {str(name)}", (1, Y_POSITION_2+20), FONT, 1.1, (0, 0, 255), 2)
            cv2.putText(report, "ALERT", ALERT_POSITION, FONT, 4, (0, 0, 255), 2)

    return no_of_frames, FACE_VERIFICATION

def get_facial_landmarks(predictor, face, frame):
    left, top, right, bottom = face[0]*4, face[1]*4, face[2]*4, face[3]*4
    face_dlib = dlib.rectangle(left, top, right, bottom)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facial_landmarks = predictor(gray, face_dlib)
    return facial_landmarks

def head_pose_detection(h_model, frame, display_frame, face, no_of_frames,  frame_count, fps, report, debug=False):
    global HEADPOSE_DETECTION

    oAnglesNp, _ = headpose_inference(h_model, frame, face)
    condition = (round(oAnglesNp[0],1) not in [0.0,-1.0,-1.1,-1.2,-1.3,-1.4,-1.5,-1.6,-1.7] and 
                 round(oAnglesNp[1],0) not in [0.0,1.0,2.0,3.0,4.0,5.0])

    no_of_frames = alert(condition, no_of_frames)

    if(no_of_frames > ALERT_THRESHOLD):
        HEADPOSE_DETECTION = True
        no_of_frames = 0

    if debug:
        display_frame = displayHeadpose(display_frame, oAnglesNp, oOffset = 0)

        if(condition):
            cv2.putText(report, "Head Pose: Looking away from the screen", (1, Y_POSITION_2+40), FONT, 1.1, (0, 255, 0), 2)
        else:
            cv2.putText(report, "Head Pose: Looking at the screen", (1, Y_POSITION_2+40), FONT, 1.1, (0, 255, 0), 2)

        if(no_of_frames > ALERT_THRESHOLD):
            cv2.putText(report, "Head Pose: Looking away from the screen", (1, Y_POSITION_2+40), FONT, 1.1, (0, 0, 255), 2)
            cv2.putText(report, "ALERT", ALERT_POSITION, FONT, 4, (0, 0, 255), 2)
        
    return no_of_frames, display_frame, condition, HEADPOSE_DETECTION

def eye_tracker(frame, facial_landmarks, no_of_frames, headpose_condition, frame_count, fps, report, debug=False):
    global EYE_TRACKING

    gaze_ratio1_left_eye, _ = get_gaze_ratio([36, 37, 38, 39, 40, 41], frame, facial_landmarks)
    gaze_ratio1_right_eye, _ = get_gaze_ratio([42, 43, 44, 45, 46, 47], frame, facial_landmarks)
    gaze_ratio1 = (gaze_ratio1_right_eye + gaze_ratio1_left_eye) / 2

    condition = (gaze_ratio1 <= 0.35 or gaze_ratio1>=4 or headpose_condition==True)
    no_of_frames = alert(condition, no_of_frames)

    if(no_of_frames > ALERT_THRESHOLD):
        EYE_TRACKING = True
        no_of_frames = 0

    if debug:
        if(condition):
            cv2.putText(report, "Eye Tracking: Looking away from screen", (1, Y_POSITION_2+60), FONT, 1.1, (0, 255, 0), 2)
        else:
            cv2.putText(report, "Eye Tracking: Looking at screen", (1, Y_POSITION_2+60), FONT, 1.1, (0, 255, 0), 2)

        if(no_of_frames > ALERT_THRESHOLD):
            cv2.putText(report, "Eye Tracking: Looking away from screen", (1, Y_POSITION_2+60), FONT, 1.1, (0, 0, 255), 2)
            cv2.putText(report, "ALERT", ALERT_POSITION, FONT, 4, (0, 0, 255), 2)
    
    return no_of_frames, EYE_TRACKING