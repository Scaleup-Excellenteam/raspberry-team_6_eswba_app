from enum import Enum

import mediapipe as mp

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
OUTPUT_DIR_NAME = 'results'
# IMAGE_EXTENSIONS = ['png', 'jpg']
IMAGE_EXTENSIONS = ['mp4']
DEFAULT_BASE_DIR = 'videos'

VIDEO_PATH = 'videos/squat.mp4'

MP_POSE = mp.solutions.pose
POSE = mp.solutions.pose.Pose(
    static_image_mode=False,  # change to False of was video stream
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
EXIT_KEY = ord('q')


# class Color(Enum):
#     RED = (0, 0, 255)
#     GREEN = (0, 255, 0)
#     WHITE = (255, 255, 255)


COLORS: dict[str, tuple[int, int, int]] = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'white': (255, 255, 255),
}

LANDMARK_COLOR = COLORS['white']  # red color for keypoint drawing
CONNECTION_COLOR = COLORS['red']  # white

DRAWING = mp.solutions.drawing_utils
LANDMARK_DRAWING_SPEC = DRAWING.DrawingSpec(
    color=LANDMARK_COLOR,
    thickness=7,
    circle_radius=4
)
# for edges and lines that connect the landmarks
CONNECTION_DRAWING_SPEC = DRAWING.DrawingSpec(
    color=CONNECTION_COLOR,
    thickness=11,
    circle_radius=3,
)

# Degrees
ONE_EIGHTY_DEGREES = 180.0
THREE_SIXTY_DEGREES = 360.0

BODY_PARTS: dict[str, tuple[int, int, int]] = {
        'left_shoulder': (11, 13, 15),
        'right_shoulder': (12, 14, 16),
        'left_elbow': (13, 15, 17),
        'right_elbow': (14, 16, 18),
        'left_hip': (23, 25, 27),
        'right_hip': (24, 26, 28),
        'left_knee': (25, 27, 29),
        'right_knee': (26, 28, 30),
        'left_ankle': (27, 29, 31),
        'right_ankle': (28, 30, 32),
    }
