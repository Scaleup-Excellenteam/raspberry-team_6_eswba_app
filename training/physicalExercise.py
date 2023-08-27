import os
from abc import ABC, abstractmethod
from functools import wraps
from typing import Tuple

import numpy

from constants import VIDEO_EXTENSIONS
from poseDetector import PoseDetector
from constants import *

import cv2

CAMERA = 0


def handle_errors_and_check_source_capture(func: callable) -> callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            print(args)
            source_capture = kwargs.get('source_capture')
            if not source_capture is None:
                print(source_capture)
                # check if is exists and is a video
                if not os.path.exists(source_capture):
                    raise FileNotFoundError(f"Path {source_capture} doesn't exist")
                if source_capture.split('.')[-1] not in VIDEO_EXTENSIONS:
                    raise TypeError(f"{source_capture} is not a video file")
                if cv2.VideoCapture(source_capture).isOpened():
                    raise ValueError(f"Can't open {source_capture}")
        except Exception as e:
            print(e)
            return

        return func(*args, **kwargs)

    return wrapper

    return source_capture(*args, **kwargs)


def check_source_capture(source_capture: str) -> None:
    pass


class PhysicalExercise(ABC):
    def __init__(self, poseDetector: PoseDetector) -> None:
        self.correct_reps: float = 0
        self.pose_detector: PoseDetector = poseDetector
        self.feedback: str = None
        self.is_correct_posture: bool = False
        self.direction: int = 0

    @handle_errors_and_check_source_capture
    @abstractmethod
    def run(self, poseDetector: PoseDetector, source_capture: str = None) -> None:
        pass

    def get_landmark_list(self, pose_detector: PoseDetector, cap: cv2.VideoCapture) -> Tuple[list,numpy.ndarray, bool]:
        success, image = cap.read()
        if not success:
            print(EMPTY_CAMERA_FRAME)
            return [], None, False
        width, height = image.shape[1], image.shape[0]
        image = cv2.resize(image, (800, 600))

        result = pose_detector.detect_pose(image, is_draw=False)
        landmark_list = pose_detector.get_position(image, result.pose_landmarks, is_draw=False, landmark_list=[])
        return landmark_list, image, True

    def prepare_cap(self, source_capture: str = None) -> cv2.VideoCapture:
        return cv2.VideoCapture(source_capture or CAMERA)

# if __name__ == '__main__':
#     detector = PoseDetector()
#     exercise = PhysicalExercise(detector)
#     exercise.run(detector, source_capture='test')
