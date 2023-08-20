import math
import time

import cv2
import numpy as np

from constants import *


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
    """
    A class that responsible for detecting the pose of a person in an image or a video stream
    """

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
        if self.result.pose_landmarks and is_draw:
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
                    draw_circle(image, (center_x, center_y), INNER_CIRCLE_RADIUS, COLORS['red'], cv2.FILLED)

        return self.landmark_list

    @staticmethod
    def get_180_degree_angle(angle: float) -> float:

        if angle < 0:
            angle = THREE_SIXTY_DEGREES + angle

        if angle > ONE_EIGHTY_DEGREES:
            angle = THREE_SIXTY_DEGREES - angle

        return angle

    @staticmethod
    def calculate_angle_between_points(a: POINT_COORDINATES, b: POINT_COORDINATES, c: POINT_COORDINATES) -> float:
        """
        Calculate the angle between three points
        :param a: a point as (x, y) coordinates
        :param b: a point as (x, y) coordinates
        :param c: a point as (x, y) coordinates
        :return: angle between the points in range 0-180
        """
        # Split the points into x and y coordinates
        [ax, ay], [bx, by], [cx, cy] = a, b, c

        # Calculate the angle between the points
        # 1.Get angle in radians
        radians = math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx)
        # 2.Get angle in degrees
        angle = math.degrees(radians)
        # 3.Set angle in range 0-180 then return it
        return PoseDetector.get_180_degree_angle(angle)

    def get_angle(self, image: np.ndarray, point_a: int, point_b: int, point_c: int, is_draw: bool = True) -> float:
        """
        Get the angle between three points
        :param image: image to draw the points on
        :param point_a: first point as index from the landmark list
        :param point_b: second point as index from the landmark list
        :param point_c: third point as index from the landmark list
        :param is_draw: if True, draw the points and the angle on the image
        :return: angle between the points
        """
        # Get the landmarks based on the landmark list index(a, b, c)
        point_a = self.landmark_list[point_a][1:]
        point_b = self.landmark_list[point_b][1:]
        point_c = self.landmark_list[point_c][1:]
        angle = self.calculate_angle_between_points(a=point_a,
                                                    b=point_b,
                                                    c=point_c)
        # Get angles as coordinates
        if is_draw:
            white_color = COLORS['white']
            red_color = COLORS['red']

            draw_line(image, point_a, point_b, white_color, 3)
            draw_line(image, point_b, point_c, white_color, 3)

            for point in [point_a, point_b, point_c]:
                draw_circle(image, point, INNER_CIRCLE_RADIUS, red_color, cv2.FILLED)
                draw_circle(image, point, OUTER_CIRCLE_RADIUS, white_color, 2)

            cv2.putText(image, str(int(angle)), (point_b[0] - POINTS_OFFSET, point_b[1] - POINTS_OFFSET),
                        cv2.FONT_HERSHEY_PLAIN, 2,
                        red_color, 2)

        return angle

    @staticmethod
    def draw_bar(image: np.ndarray, angle: float, percentage: float
                 , bar_scales: tuple[int, int] = BAR_SCALES, bar_size: tuple[int, int] = BAR_SIZE) -> None:
        """
        Draw a bar with a percentage on the image
        :param angle: the angle between the points
        :param percentage: progress percentage between min and max angle values
        :param bar_scales: scales of the bar
        :param bar_size: size of the bar
        :return:
        """
        bar = np.interp(angle, (210, 310), bar_scales)
        print(angle, percentage)
        print(bar)
        # Bar with percentage
        cv2.rectangle(image, (bar_scales[1], 100), bar_size, COLORS['red'], cv2.FILLED)
        cv2.rectangle(image, (bar_scales[1], 100), bar_size, COLORS['green'], 3)
        cv2.rectangle(image, (bar_scales[1], int(bar)), bar_size, COLORS['green'], cv2.FILLED)
        cv2.putText(image, f'{int(percentage)}%', (bar_scales[1], 80), cv2.FONT_HERSHEY_PLAIN, 2, COLORS['green'],
                    2)


def main():
    cap = cv2.VideoCapture('videos/curl.mp4')
    detector = PoseDetector()
    previous_time = 0
    # get 30 frames per second exactly
    cap.set(cv2.CAP_PROP_FPS, FPS)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print(EMPTY_CAMERA_FRAME)
            break
        detector.detect_pose(image, is_draw=False)
        landmark_list = detector.get_position(image, is_draw=False)

        if len(landmark_list) != 0:
            # test left knee
            angle = detector.get_angle(image, 11, 13, 15)
            percentage = np.interp(angle, (210, 310), (0, 100))
            PoseDetector.draw_bar(image, angle=angle, percentage=percentage)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)

        previous_time = current_time
        put_text(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['red'], 2)
        width, height = 1280, 720
        if image.shape[0] != height or image.shape[1] != width:
            image = cv2.resize(image, (width, height))

        cv2.imshow('Test pose detection', image)
        # Wait for the user to press EXIT_KEY to exit the program
        if cv2.waitKey(1) & 0xFF == EXIT_KEY:  # change to 0 for pause
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
