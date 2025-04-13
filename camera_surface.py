# -*- coding: UTF-8 -*-
import cv2
import pygame
import mediapipe as mp

from key_controller import Controller

mp_pose = mp.solutions.pose  # Mediapipe Pose model
mp_drawing = mp.solutions.drawing_utils  # Utility for drawing landmarks
mp_drawing_styles = mp.solutions.drawing_styles  # Pre-defined drawing styles for pose landmarks

class CameraSurface:
    def __init__(
            self,
            camera_index = 0,
            camera_size = (200, 300),
            pose_model = mp_pose.Pose(),
            controller = Controller(),
    ):
        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise Exception("Could not open video device")

        # Set camera resolution
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_size[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_size[1])
        # print(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH), self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize pose model
        self.pose_model = pose_model
        self.controller = controller
        self.camera_size = camera_size
        self.data = None


    def update(self):
        ret, frame = self.capture.read()
        if not ret:
            return None

        # BGR(OpenCV) -> RGB(Pygame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with the Pose model to detect pose landmarks
        results = self.pose_model.process(frame)  # Get pose landmarks from the image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # The original frame (in BGR) where the landmarks will be drawn
                results.pose_landmarks,  # The detected pose landmarks
                mp_pose.POSE_CONNECTIONS,  # The connections between landmarks (like joints connecting bones)
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())  # Use default landmark drawing styles
            self.data = self.controller.get_data(results.pose_landmarks)
            self.controller.update(self.data)

        frame = cv2.flip(frame, 1)
        # Convert the frame to a Pygame surface
        frame = cv2.resize(frame, self.camera_size)
        frame_surface = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], 'RGB')


        return frame_surface

    def release(self):
        self.capture.release()
        self.pose_model.close()
