"""
This script demonstrates how to use the Mediapipe Pose model to detect human poses in real-time
"""

import cv2
import mediapipe as mp
from xgb import normalize_landmarks_np
import time
from xgb import get_model
from collections import deque
import threading
import copy
import keyboard

LAST_PRESS_TIME = 0
LAST_PRESS_KEY = None
TESTING = False


def press_key(key):
    """
    Press and release a key
    :param key: Key to press
    :return: None
    """
    global LAST_PRESS_TIME
    global LAST_PRESS_KEY
    # Prevent multiple key presses in quick succession(most are a and d)
    if time.time() - LAST_PRESS_TIME < 0.3 and LAST_PRESS_KEY == key:
        return
    else:
        if TESTING:
            print(f'pressed {key}')
        keyboard.press_and_release(key)
        LAST_PRESS_TIME = time.time()
        LAST_PRESS_KEY = key


POSE = {'pushup': 0, 'stand': 1}
POSE_R = {0: 'pushup', 1: 'stand'}

import numpy as np
import math


def get_pos(landmarks, name):
    """
    Get the average position of a body part
    :param landmarks: Pose landmarks
    :param name: Name of the body part
    :return: Average position of the body part
    """
    mp = {'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3, 'right_eye_inner': 4, 'right_eye': 5,
          'right_eye_outer': 6, 'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10, 'left_shoulder': 11,
          'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
          'left_pinky': 17,
          'right_pinky': 18, 'left_index': 19, 'right_index': 20, 'left_thumb': 21, 'right_thumb': 22, 'left_hip': 23,
          'right_hip': 24, 'left_knee': 25, 'right_knee': 26, 'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29,
          'right_heel': 30, 'left_foot_index': 31, 'right_foot_index': 32}
    sum_value = (0, 0)
    sum_times = 0
    for key, value in mp.items():
        if name in key:
            sum_value = (sum_value[0] + landmarks[value, 0], sum_value[1] + landmarks[value, 1])
            sum_times += 1
    return sum_value[0] / sum_times, sum_value[1] / sum_times


def calculate_angle2(a, b):
    """
    Calculate the angle between two points A and B
    :param a: Coordinates of point A (x, y)
    :param b: Coordinates of point B (x, y)
    :return: angle in degrees
    """
    # Vector AB
    ab = np.array(a) - np.array(b)
    # Calculate the angle in radians and then convert to degrees
    angle = np.arctan2(ab[1], ab[0])

    # we have no need the direction of the vector
    if angle > np.pi / 2:
        angle -= np.pi
    elif angle <= -np.pi / 2:
        angle += np.pi

    angle = np.degrees(angle)  # Convert to degrees

    return angle


def calculate_angle(a, b, c):
    """
    Calculate the angle between three points A, B, C
    where B is the vertex of the angle.
    :param a: Coordinates of point A (x, y)
    :param b: Coordinates of point B (x, y)
    :param c: Coordinates of point C (x, y)
    :return: angle in degrees
    """
    # Vector AB and BC
    ab = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    # Dot product and magnitude of vectors
    dot_product = np.dot(ab, bc)
    magnitude_ab = np.linalg.norm(ab)
    magnitude_bc = np.linalg.norm(bc)

    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude_ab * magnitude_bc)

    # Ensure the cosine value is within the valid range for acos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Calculate the angle in radians and then convert to degrees
    angle = np.arccos(cos_angle)
    angle = np.degrees(angle)  # Convert to degrees

    return angle


class controller:
    """
    Class to control the key press based on the pose detected
    """

    def __init__(self):
        self.model = get_model(r'datas/mldata2')
        self.queue = deque(maxlen=15)
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,  # Set to False for video processing (real-time detection)
            min_detection_confidence=0.5,  # Minimum confidence value for the detection to be considered successful
            model_complexity=1)
        self.elbow_angle_queue = deque(maxlen=7)  # for pushup
        self.arm_angle_queue = deque(maxlen=7)  # for clapping
        self.hip_angle_queue = deque(maxlen=7)  # for situp, head-hip-foot
        self.knee_angle_queue = deque(maxlen=7)  # for squat, foot-knee-hip
        self.status_angle_queue = [deque(maxlen=15), deque(maxlen=15), deque(maxlen=15)]
        self.shoulder_height_pushup = deque(maxlen=7)
        self.shoulder_height_situp = deque(maxlen=7)
        self.hip_height_squat = deque(maxlen=7)
        self.hip_width_stand = deque(maxlen=15)
        self.pushup_state = None
        self.clap_state = None
        self.situp_state = None
        self.squat_state = None
        self.clapping_state = None

    @staticmethod
    def get_data(landmarks):
        """
        Get the pose landmarks data
        :param landmarks: Pose landmarks
        :return: Pose landmarks data
        """
        data = np.zeros((33, 3))
        if results.pose_landmarks:
            for idx, landmark in enumerate(landmarks.landmark):
                data[idx, 0] = landmark.x
                data[idx, 1] = landmark.y
                data[idx, 2] = landmark.visibility
        return data

    def predict(self, data):
        """
        Predict the pose based on the pose landmarks data using xgboost
        :param data: Pose landmarks data
        :return: Predicted pose
        """
        predict = self.model.predict(data.flatten().reshape(1, -1))[0]
        return POSE_R[predict]

    def check_arm_state(self, data_normalized):
        """
        Check the arm state and trigger key press(including clap and arm out)
        :param data_normalized: Pose landmarks data
        :return: None
        """
        left_wrist = (data_normalized[15, 0], data_normalized[15, 1])
        left_shoulder = (data_normalized[11, 0], data_normalized[11, 1])
        left_hip = (data_normalized[23, 0], data_normalized[23, 1])
        left_elbow = (data_normalized[13, 0], data_normalized[13, 1])

        right_wrist = (data_normalized[16, 0], data_normalized[16, 1])
        right_shoulder = (data_normalized[12, 0], data_normalized[12, 1])
        right_hip = (data_normalized[24, 0], data_normalized[24, 1])
        right_elbow = (data_normalized[14, 0], data_normalized[14, 1])

        left_arm_angle = calculate_angle(left_hip, left_shoulder, left_wrist)
        right_arm_angle = calculate_angle(right_hip, right_shoulder, right_wrist)
        self.arm_angle_queue.append((left_arm_angle, right_arm_angle))

        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_arm_angle2 = calculate_angle(left_hip, left_shoulder, left_elbow)
        right_arm_angle2 = calculate_angle(right_hip, right_shoulder, right_elbow)

        if 60 < left_elbow_angle < 120 and 60 < left_arm_angle2 < 120:
            key_thread = threading.Thread(target=press_key, args=('a',))
            key_thread.start()
        # right arm out
        elif 60 < right_elbow_angle < 120 and 60 < right_arm_angle2 < 120:
            key_thread = threading.Thread(target=press_key, args=('d',))
            key_thread.start()
        # check if hand clapping
        else:
            if len(self.arm_angle_queue) == 7:
                if all(angle[0] > 150 and angle[1] > 150 for angle in self.arm_angle_queue):
                    # from down to up, trigger key press
                    if self.clapping_state == 'down':
                        # print('pressed w!')
                        key_thread = threading.Thread(target=press_key, args=('t',))
                        key_thread.start()
                        self.clapping_state = 'up'

                if all(angle[0] < 150 and angle[1] < 150 for angle in self.arm_angle_queue):
                    self.clapping_state = 'down'

    def check_pushup_state2(self, data_normalized, thresh_height=0.2):
        """
        Check the pushup state and trigger key press
        :param data_normalized: Pose landmarks data
        :param thresh_height: Threshold height for pushup
        :return: None
        """
        shoulder = get_pos(data_normalized, 'shoulder')
        shoulder_y = shoulder[1]
        wrist = get_pos(data_normalized, 'wrist')
        wrist_y = wrist[1]
        delta_h = abs(shoulder_y - wrist_y)

        if TESTING:
            print(f'pushup: {shoulder_y}, {wrist_y}, {delta_h}')

        self.shoulder_height_pushup.append(delta_h)

        if len(self.shoulder_height_pushup) == 7:
            if all(h > thresh_height for h in self.shoulder_height_pushup):
                # from down to up, trigger key press
                if self.pushup_state == 'down':
                    # check the relative position of the head and foot
                    if data_normalized[0, 0] < data_normalized[31, 0]:
                        key_thread = threading.Thread(target=press_key, args=('j',))
                    else:
                        key_thread = threading.Thread(target=press_key, args=('l',))
                    key_thread.start()
                self.pushup_state = 'up'

            if all(h < thresh_height for h in self.shoulder_height_pushup):
                self.pushup_state = 'down'

    def check_squat_state2(self, data_normalized, thresh_height=0.25):
        """
        Check the squat state and trigger key press
        :param data_normalized: Pose landmarks data
        :param thresh_height: Threshold height for squat
        :return: None
        """
        hip = get_pos(data_normalized, 'hip')
        hip_y = hip[1]
        ankle = get_pos(data_normalized, 'ankle')
        ankle_y = ankle[1]
        delta_h = abs(hip_y - ankle_y)
        knee = ((data_normalized[25, 0] + data_normalized[26, 0]) / 2,
                (data_normalized[25, 1] + data_normalized[26, 1]) / 2)
        if TESTING:
            print(f'squat: {hip_y}, {ankle_y}, {delta_h}')

        self.hip_height_squat.append(delta_h)

        if len(self.hip_height_squat) == 7:
            if all(h > thresh_height for h in self.hip_height_squat):
                # from down to up, trigger key press
                if self.squat_state == 'down':
                    self.squat_state = 'up'
                    # check the relative position of the head and knee\
                    if TESTING:
                        print(f'ankle: {ankle}, knee: {knee}')
                    if ankle[0] < knee[0]:
                        key_thread = threading.Thread(target=press_key, args=('z',))
                    else:
                        key_thread = threading.Thread(target=press_key, args=('s',))
                    key_thread.start()

            if all(h < thresh_height for h in self.hip_height_squat):
                self.squat_state = 'down'

    def check_pushup_state(self, data_normalized):
        """
        Check the pushup state and trigger key press
        :param data_normalized: Pose landmarks data
        :return: None
        """
        left_shoulder = (data_normalized[11, 0], data_normalized[11, 1])
        left_elbow = (data_normalized[13, 0], data_normalized[13, 1])
        left_wrist = (data_normalized[15, 0], data_normalized[15, 1])
        angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        self.elbow_angle_queue.append(angle)
        # check if angles are larger than 90
        if TESTING:
            print(f'pushup: {angle}')
        if len(self.elbow_angle_queue) == 7:
            if all(angle > 130 for angle in self.elbow_angle_queue):
                # from down to up, trigger key press
                if self.pushup_state == 'down':
                    self.pushup_state = 'up'
                    # check the relative position of the head and foot
                    if data_normalized[0, 0] < data_normalized[31, 0]:
                        key_thread = threading.Thread(target=press_key, args=('j',))
                    else:
                        key_thread = threading.Thread(target=press_key, args=('l',))
                    key_thread.start()

            if all(angle < 130 for angle in self.elbow_angle_queue):
                self.pushup_state = 'down'

    def check_situp_state2(self, data_normalized, thresh_height=0.1):
        """
        Check the situp state and trigger key press
        :param data_normalized: Pose landmarks data
        :param thresh_height: Threshold height for situp
        :return: None
        """
        shoulder = get_pos(data_normalized, 'shoulder')
        shoulder_y = shoulder[1]
        hip = get_pos(data_normalized, 'hip')
        hip_y = hip[1]
        delta_h = abs(shoulder_y - hip_y)

        if TESTING:
            print(f'situp: {shoulder_y}, {hip_y}, {delta_h}')

        self.shoulder_height_situp.append(delta_h)

        if len(self.shoulder_height_situp) == 7:
            if all(h < thresh_height for h in self.shoulder_height_situp):
                # from down to up, trigger key press
                if self.situp_state == 'down':
                    # check the relative position of the head and foot
                    key_thread = threading.Thread(target=press_key, args=('o',))
                    key_thread.start()
                self.situp_state = 'up'

            if all(h > thresh_height for h in self.shoulder_height_situp):
                self.situp_state = 'down'

    def check_situp_state(self, data_normalized):
        """
        Check the situp state and trigger key press
        :param data_normalized: Pose landmarks data
        :return: None
        """
        head = (data_normalized[0, 0], data_normalized[0, 1])
        hip = ((data_normalized[23, 0] + data_normalized[24, 0]) / 2,
               (data_normalized[23, 1] + data_normalized[24, 1]) / 2)
        foot = ((data_normalized[31, 0] + data_normalized[32, 0]) / 2,
                (data_normalized[31, 1] + data_normalized[32, 1]) / 2)
        angle = calculate_angle(head, hip, foot)
        self.hip_angle_queue.append(angle)
        # check if angles are larger than 150
        if TESTING:
            print(f'situp: {angle}')
        if len(self.hip_angle_queue) == 7:
            if all(angle < 135 for angle in self.hip_angle_queue):
                # from down to up, trigger key press
                if self.situp_state == 'down':
                    key_thread = threading.Thread(target=press_key, args=('o',))
                    key_thread.start()
                    self.situp_state = 'up'

            if all(angle > 135 for angle in self.hip_angle_queue):
                self.situp_state = 'down'

    def check_squat_state(self, data_normalized):
        """
        Check the squat state and trigger key press
        :param data_normalized: Pose landmarks data
        :return: None
        """
        foot = ((data_normalized[31, 0] + data_normalized[32, 0]) / 2,
                (data_normalized[31, 1] + data_normalized[32, 1]) / 2)
        knee = ((data_normalized[25, 0] + data_normalized[26, 0]) / 2,
                (data_normalized[25, 1] + data_normalized[26, 1]) / 2)
        hip = ((data_normalized[23, 0] + data_normalized[24, 0]) / 2,
               (data_normalized[23, 1] + data_normalized[24, 1]) / 2)
        ankle = get_pos(data_normalized, 'ankle')
        angle = calculate_angle(foot, knee, hip)
        self.knee_angle_queue.append(angle)
        if TESTING:
            print(f'squat: {angle}')
        # check if angles are larger than 150
        if len(self.knee_angle_queue) == 7:
            if all(angle < 135 for angle in self.knee_angle_queue):
                # from down to up, trigger key press
                if self.squat_state == 'down':
                    self.squat_state = 'up'
                    # check the relative position of the head and knee\
                    if TESTING:
                        print(f'ankle: {ankle}, knee: {knee}')
                    if ankle[0] < knee[0]:
                        key_thread = threading.Thread(target=press_key, args=('z',))
                    else:
                        key_thread = threading.Thread(target=press_key, args=('s',))
                    key_thread.start()

            if all(angle > 135 for angle in self.knee_angle_queue):
                self.squat_state = 'down'

    def update(self, data):
        """
        main function to update the pose and trigger key press
        :param data: Pose landmarks data
        :return: Predicted pose
        """

        data_normalized = normalize_landmarks_np(copy.deepcopy(data))  # Normalize the landmarks
        predict = self.predict(data_normalized)  # Predict the pose(I think this can be deleted)
        data_normalized = data  # I think it's a bug here, but after code use this, so I keep it

        # self.queue.append(angle)
        self.queue.append(predict)

        # only when the queue is full, we can start to check the pose
        if len(self.queue) < 15:
            return "Not enough data"

        # pushup

        hip = get_pos(data_normalized, 'hip')
        foot = get_pos(data_normalized, 'foot')
        self.hip_width_stand.append(abs(hip[0] - foot[0])) # calculate the distance of the hip and foot(on x axis)

        angle_for_stand = calculate_angle2(get_pos(data_normalized, 'foot'), get_pos(data_normalized, 'hip'))
        angle_for_pushup = calculate_angle(get_pos(data_normalized, 'foot'), get_pos(data_normalized, 'knee'),
                                           get_pos(data_normalized, 'hip'))
        self.status_angle_queue[0].append(abs(angle_for_stand))
        self.status_angle_queue[1].append(angle_for_pushup)

        if TESTING:
            print(f'stand: {abs(hip[0] - foot[0])}, pushup: {angle_for_pushup}')

        # check stand or not
        if all(i < 0.15 for i in self.hip_width_stand):
            self.check_arm_state(data_normalized)
            self.check_squat_state2(data_normalized)

            # clear data of not stand
            self.elbow_angle_queue.clear()
            self.hip_angle_queue.clear()
            self.shoulder_height_pushup.clear()
            self.shoulder_height_situp.clear()
            self.situp_state = None
            self.pushup_state = None
        else:
            # check if the person is doing pushup or situp
            if all(i < 135 for i in self.status_angle_queue[1]):
                self.check_situp_state2(data_normalized)
            if all(i > 135 for i in self.status_angle_queue[1]):
                self.check_pushup_state2(data_normalized)

            # clear data of stand
            self.knee_angle_queue.clear()
            self.arm_angle_queue.clear()
            self.hip_height_squat.clear()
            self.squat_state = None

        return predict


if __name__ == '__main__':
    # Initialize webcam capturea
    vid = cv2.VideoCapture(1)  # Open the default camera (usually the built-in webcam)
    vid.set(3, 640)  # Set the width of the video frame to 640 pixels
    vid.set(4, 480)  # Set the height of the video frame to 480 pixels
    video_path = r'D:\python\BN6206\videos\pushup\datas\2.mp4'  # Replace with your video file path
    # vid = cv2.VideoCapture(video_path)  # Open the video file
    # Initialize Mediapipe Pose solution and drawing utilities
    mp_pose = mp.solutions.pose  # Mediapipe Pose model
    mp_drawing = mp.solutions.drawing_utils  # Utility for drawing landmarks
    mp_drawing_styles = mp.solutions.drawing_styles  # Pre-defined drawing styles for pose landmarks

    controller = controller()

    with mp_pose.Pose(
            static_image_mode=False,  # Set to False for video processing (real-time detection)
            min_detection_confidence=0.5,  # Minimum confidence value for the detection to be considered successful
            model_complexity=1) as pose:  # Model complexity: 0 (Light), 1 (Full), 2 (Heavy). Higher complexity increases accuracy but requires more computation

        prev_frame_time = time.time()
        while True:
            # Capture frame-by-frame from the webcam
            ret, frame = vid.read()  # Read a single frame from the video capture
            if not ret:  # If the frame is not captured correctly, break the loop
                break

            # Convert the captured BGR frame to RGB format
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Mediapipe requires the image to be in RGB format

            # Process the frame with the Pose model to detect pose landmarks
            results = pose.process(image_rgb)  # Get pose landmarks from the image
            # print(results)
            # If pose landmarks are detected, draw them on the original frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # The original frame (in BGR) where the landmarks will be drawn
                    results.pose_landmarks,  # The detected pose landmarks
                    mp_pose.POSE_CONNECTIONS,  # The connections between landmarks (like joints connecting bones)
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())  # Use default landmark drawing styles

            # push data to deque
            if results.pose_landmarks:
                now_pose = controller.update(controller.get_data(results.pose_landmarks))
                cv2.putText(frame, f'Pose: {now_pose}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

            # Calculate and display the frame rate
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the resulting frame with landmarks in a window named 'Pose Estimation'
            cv2.imshow('Pose Estimation', frame)  # Show the frame with overlaid landmarks

            # Wait for 1 millisecond to check if the 'q' key is pressed to exit
            if cv2.waitKey(1) & 0xFF == ord(
                    'q'):  # 0xFF is a bitwise AND operation to ensure compatibility with different OS
                break  # Exit the loop if 'q' is pressed

    # After the loop is finished (when 'q' is pressed), release the video capture object
    vid.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows
