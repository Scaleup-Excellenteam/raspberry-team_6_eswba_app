from enum import Enum

import mediapipe as mp

MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
OUTPUT_DIR_NAME = 'results'
IMAGE_EXTENSIONS = ['png', 'jpg']
VIDEO_EXTENSIONS = ['mp4']
DEFAULT_BASE_DIR = 'underwater'
BAR_SCALES = (300, 100)
BAR_SIZE = (150, BAR_SCALES[0])
EMPTY_CAMERA_FRAME = 'Ignoring empty camera frame'

VIDEO_PATH = 'videos/squat.mp4'

MP_POSE = mp.solutions.pose
EXIT_KEY = ord('q')


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
FPS = 30
# Degrees
ONE_EIGHTY_DEGREES = 180.0
THREE_SIXTY_DEGREES = 360.0
POINTS_OFFSET = 15

BODY_PARTS: dict[str, tuple[int, int, int]] = {
    'left elbow': (
        MP_POSE.PoseLandmark.LEFT_SHOULDER, MP_POSE.PoseLandmark.LEFT_ELBOW, MP_POSE.PoseLandmark.LEFT_WRIST),
    'right elbow': (
        MP_POSE.PoseLandmark.RIGHT_SHOULDER, MP_POSE.PoseLandmark.RIGHT_ELBOW, MP_POSE.PoseLandmark.RIGHT_WRIST),
    'left hip': (MP_POSE.PoseLandmark.LEFT_SHOULDER, MP_POSE.PoseLandmark.LEFT_HIP, MP_POSE.PoseLandmark.LEFT_KNEE),
    'right hip': (MP_POSE.PoseLandmark.RIGHT_SHOULDER, MP_POSE.PoseLandmark.RIGHT_HIP, MP_POSE.PoseLandmark.RIGHT_KNEE),
    'left knee': (MP_POSE.PoseLandmark.LEFT_HIP, MP_POSE.PoseLandmark.LEFT_KNEE, MP_POSE.PoseLandmark.LEFT_ANKLE),
    'right knee': (MP_POSE.PoseLandmark.RIGHT_HIP, MP_POSE.PoseLandmark.RIGHT_KNEE, MP_POSE.PoseLandmark.RIGHT_ANKLE)
}

INNER_CIRCLE_RADIUS = 5
OUTER_CIRCLE_RADIUS = 15

LANDMARK_VARIABLES = list[int, int, int]
POINT_COORDINATES = list[int, int]

# pose landmarks
POSE_LANDMARKS = MP_POSE.PoseLandmark
LEFT_HIP = POSE_LANDMARKS.LEFT_HIP.value
LEFT_KNEE = POSE_LANDMARKS.LEFT_KNEE.value
LEFT_ANKLE = POSE_LANDMARKS.LEFT_ANKLE.value
RIGHT_HIP = POSE_LANDMARKS.RIGHT_HIP.value
RIGHT_KNEE = POSE_LANDMARKS.RIGHT_KNEE.value
RIGHT_ANKLE = POSE_LANDMARKS.RIGHT_ANKLE.value
LEFT_SHOULDER = POSE_LANDMARKS.LEFT_SHOULDER.value
# squat correct angles
SQUAT_ANGLES = {
    'left-knee': {
        'posture_angle': 160,
        'down_angle': 80,
        'up_angle': 160
    },
    'right-knee': {
        'posture_angle': 160,
        'down_angle': 80,
        'up_angle': 160
    },
    'lef_hip': {
        'posture_angle': 170,
        'down_angle': 75,
        'up_angle': 160
    }
}