import math
import time

import cv2
import numpy as np

from constants import *

LANDMARK_VARIABLES = list[int, int, int]
POINT_COORDINATES = list[int, int]


# Usable cv2 functions for drawing
def draw_line(image, point_a, point_b, color, thickness=2, line_type=cv2.LINE_AA):
    cv2.line(image, point_a, point_b, color, thickness, line_type)


def draw_circle(image, center, radius, color, thickness=cv2.FILLED, line_type=cv2.LINE_AA):
    cv2.circle(image, center, radius, color, thickness, line_type)


def put_text(image, text, org, font_face, font_scale, color, thickness=2, line_type=cv2.LINE_AA):
    cv2.putText(image, text, org, font_face, font_scale, color, thickness, line_type)


# class PoseDetector(mp.solutions.pose.Pose):
#     def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5,
#                  min_tracking_confidence=0.5):
#         super().__init__(static_image_mode, model_complexity, smooth_landmarks, min_detection_confidence,
#                          min_tracking_confidence)
class PoseDetector:
    def __init__(self, static_image_mode: bool = False, model_complexity: int = 1, smooth_landmarks: bool = True,
                 detection_con: float = MIN_DETECTION_CONFIDENCE, tracking_con: float = MIN_TRACKING_CONFIDENCE):
        """
        Initialize the pose detector
        :param static_image_mode: if True, detect the pose from a static image instead of a video stream
        :param model_complexity: 0, 1 or 2
        :param smooth_landmarks: if True, smooth the landmarks
        :param detection_con: minimum confidence value for detection
        :param tracking_con: minimum confidence value for tracking
        """
        # self.pose = POSE
        self.pose = MP_POSE.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=detection_con,
            min_tracking_confidence=tracking_con
        )
        # Storing the result of the pose detection
        self.result = None
        # Storing the list of landmarks
        self.landmark_list: list[LANDMARK_VARIABLES] = []

    def detect_pose(self, image: np.ndarray, is_draw: bool = True) -> None:
        """
        Detect the pose of the person in the image
        :param image: to detect the pose from
        :param is_draw: if True, draw the pose detection result on the image
        :return: image with the pose detection result
        """

        # Convert to RGB and start processing image
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(image_RGB)

        # Draw the pose detection result on the image
        if is_draw and self.result.pose_landmarks:
            DRAWING.draw_landmarks(image, self.result.pose_landmarks, MP_POSE.POSE_CONNECTIONS)

    def get_position(self, image: np.ndarray, is_draw: bool = True) -> list[LANDMARK_VARIABLES]:
        """
        Get the position of the all the landmarks list
        :param image: image to draw the landmarks on
        :param is_draw: if True, draw the landmarks on the image
        :return: list of landmarks with their id and coordinates
        """
        self.landmark_list = []
        if self.result.pose_landmarks:
            for id, landmark in enumerate(self.result.pose_landmarks.landmark):
                height, width, channels = image.shape
                center_x, center_y = int(landmark.x * width), int(landmark.y * height)
                self.landmark_list.append([id, center_x, center_y])
                if is_draw:
                    draw_circle(image, (center_x, center_y), 5, COLORS['red'], cv2.FILLED)

        return self.landmark_list

    @staticmethod
    def calculate_angle_between_points(a: POINT_COORDINATES, b: POINT_COORDINATES, c: POINT_COORDINATES) -> float:
        # Split the points into x and y coordinates
        [ax, ay], [bx, by], [cx, cy] = a[1:], b[1:], c[1:]

        # Calculate the angle between the points
        # Get angle in radians
        radians = math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx)
        # Get angle in degrees
        angle = math.degrees(radians)
        # Set angle in range 0-360
        if angle > ONE_EIGHTY_DEGREES:
            angle = THREE_SIXTY_DEGREES - angle

        elif angle < 0:
            angle = THREE_SIXTY_DEGREES + angle

        return angle

    def get_angle(self, image: np.ndarray, point_a: int, point_b: int, point_c: int, is_draw: bool = True):
        """
        Get the angle between three points
        :param image:
        :param point_a:
        :param point_b:
        :param point_c:
        :param is_draw:
        :return:
        """
        # Get the landmarks based on the landmark list index(a, b, c)
        angle = self.calculate_angle_between_points(a=self.landmark_list[point_a],
                                                    b=self.landmark_list[point_b],
                                                    c=self.landmark_list[point_c])
        point_a = self.landmark_list[point_a][1:]
        point_b = self.landmark_list[point_b][1:]
        point_c = self.landmark_list[point_c][1:]
        if is_draw:
            white_color = COLORS['white']
            red_color = COLORS['red']

            draw_line(image, point_a, point_b, white_color, 3)
            draw_line(image, point_c, point_b, white_color, 3)

            for point in [point_a, point_b, point_c]:
                draw_circle(image, point, 5, red_color, cv2.FILLED)
                draw_circle(image, point, 15, white_color, 2)

            offset = 50
            cv2.putText(image, str(int(angle)), (point_b[0] - offset, point_b[1] - offset), cv2.FONT_HERSHEY_PLAIN, 2,
                        red_color, 2)

        return angle


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    detector = PoseDetector()
    previous_time = 0

    while cap.isOpened():
        success, image = cap.read()
        detector.detect_pose(image)
        landmark_list = detector.get_position(image)

        if len(landmark_list) != 0:
            # test left knee
            left_knee = landmark_list[14]
            print(left_knee)
            draw_circle(image, (left_knee[1], left_knee[2]), 10, COLORS['green'], cv2.FILLED)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)

        previous_time = current_time
        put_text(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['red'], 2)

        cv2.imshow('Image', image)

        if cv2.waitKey(1) & 0xFF == EXIT_KEY:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
