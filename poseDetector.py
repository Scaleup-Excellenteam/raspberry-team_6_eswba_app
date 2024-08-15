import math
import time
from typing import NamedTuple

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


class PoseDetector:
    """
    A class that responsible for detecting the pose of a person in an image or a video stream
    """

    def __init__(self, static_image_mode: bool = False, model_complexity: int = 1, smooth_landmarks: bool = True,
                 enable_segmentation: bool = False, detection_con: float = MIN_DETECTION_CONFIDENCE,
                 tracking_con: float = MIN_TRACKING_CONFIDENCE):
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
            enable_segmentation=enable_segmentation,
            min_detection_confidence=detection_con,
            min_tracking_confidence=tracking_con
        )
        # Storing the result of the pose detection
        # self.result = None
        # # Storing the list of landmarks
        # self.landmark_list: list[LANDMARK_VARIABLES] = []

    def detect_pose(self, image: np.ndarray, is_draw: bool = True) -> NamedTuple:
        """
        Detect the pose of the person in the image
        :param image: to detect the pose from
        :param is_draw: if True, draw the pose detection result on the image
        :return: image with the pose detection result
        """

        # Convert to RGB and start processing image
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.pose.process(image_RGB)

        # Draw the pose detection result on the image
        if result.pose_landmarks and is_draw:
            DRAWING.draw_landmarks(image, result.pose_landmarks, MP_POSE.POSE_CONNECTIONS)

        return result

    def get_position(self, image: np.ndarray, result_pose_landmarks, is_draw: bool = True, landmark_list=None,
                     ) -> list[
        LANDMARK_VARIABLES]:
        """
        Get the position of the all the landmarks list
        :param result:
        :param result_pose_landmarks:
        :param image: image to draw the landmarks on
        :param is_draw: if True, draw the landmarks on the image
        :return: list of landmarks with their id and coordinates
        """
        landmark_list = []
        if result_pose_landmarks:
            for id, landmark in enumerate(result_pose_landmarks.landmark):
                height, width, channels = image.shape
                center_x, center_y = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append([id, center_x, center_y])
                if is_draw:
                    draw_circle(image, (center_x, center_y), INNER_CIRCLE_RADIUS, COLORS['red'], cv2.FILLED)

        return landmark_list

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
        # 3.Set angle in range 0-360 then return it
        # normalized_angle = angle % THREE_SIXTY_DEGREES
        if angle < 0:
            angle += THREE_SIXTY_DEGREES
        if angle > ONE_EIGHTY_DEGREES:
            angle = THREE_SIXTY_DEGREES - angle
        return angle

    def get_angle(self, image: np.ndarray, point_a: int, point_b: int, point_c: int, landmark_list,
                  is_draw: bool = True) -> float:
        """
        Get the angle between three points
        :param landmark_list:
        :param image: image to draw the points on
        :param point_a: first point as index from the landmark list
        :param point_b: second point as index from the landmark list
        :param point_c: third point as index from the landmark list
        :param is_draw: if True, draw the points and the angle on the image
        :return: angle between the points
        """
        # Get the landmarks based on the landmark list index(a, b, c)
        point_a = landmark_list[point_a][1:]
        point_b = landmark_list[point_b][1:]
        point_c = landmark_list[point_c][1:]
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

            put_text(image, str(int(angle)), (point_b[0] - POINTS_OFFSET, point_b[1] - POINTS_OFFSET),
                     cv2.FONT_HERSHEY_PLAIN, 2,
                     red_color, 2)

        return angle

    @staticmethod
    def draw_bar(image: np.ndarray, angle: float, percentage: float
                 , bar_scales: tuple[int, int] = BAR_SCALES, bar_size: tuple[int, int] = BAR_SIZE) -> None:
        """
        Draw a bar with a percentage on the image
        :param image: image to draw the bar on
        :param angle: the angle between the points
        :param percentage: progress percentage between min and max angle values
        :param bar_scales: scales of the bar
        :param bar_size: size of the bar
        :return:
        """
        print("angle ", angle, "percentage ", percentage)
        bar = np.interp(angle, (210, 310), bar_scales)
        print(angle, percentage)
        print(bar)
        # Bar with percentage
        cv2.rectangle(image, (bar_scales[1], 100), bar_size, COLORS['green'], 3)
        cv2.rectangle(image, (bar_scales[1], int(bar)), bar_size, COLORS['green'], cv2.FILLED)
        cv2.putText(image, f'{int(percentage)}%', (bar_scales[1], 80), cv2.FONT_HERSHEY_PLAIN, 2, COLORS['green'],
                    2)


# def check_progress(percentage: float, correct_reps: float, direction: int) -> tuple[float, int]:
#     if percentage == 100:
#         if direction == 0:
#             correct_reps += 0.5
#             direction = 1
#     elif percentage == 0:
#         if direction == 1:
#             correct_reps += 0.5
#             direction = 0
#     return correct_reps, direction
#
#
# def display_feedback(image: np.ndarray, percentage: float, correct_reps: float, direction: int) -> tuple[float, int]:
#     correct_reps, direction = check_progress(percentage, correct_reps, direction)
#     # give feedback
#     cv2.putText(image, f'{int(correct_reps)}', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, COLORS['green'], 2)
#     #     give guide fidback to correct the pose
#     if percentage == 0:
#         cv2.putText(image, 'Move your arm up', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, COLORS['red'], 2)
#     elif percentage == 100:
#         cv2.putText(image, 'Move your arm down', (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, COLORS['red'], 2)
#
#     return correct_reps, direction
#
#
# def process_frame(image: np.ndarray, detector: PoseDetector, landmark_list: list[LANDMARK_VARIABLES],
#                   previous_time: float, correct_reps: float
#                   , direction: int, is_display_feedback=True) -> tuple[float, int, float]:
#     result = detector.detect_pose(image, is_draw=False)
#     landmark_list = detector.get_position(image, result.pose_landmarks, is_draw=False, landmark_list=landmark_list)
#
#     if len(landmark_list) != 0:
#         # test left knee
#         angle = detector.get_angle(image, 11, 13, 15, landmark_list)
#         percentage = np.interp(angle, (210, 310), (0, 100))
#         PoseDetector.draw_bar(image, angle=angle, percentage=percentage)
#         if is_display_feedback:
#             correct_reps, direction = display_feedback(image, percentage, correct_reps, direction)
#         else:
#             put_text(image, str(int(percentage)) + "%", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, COLORS['red'], 2)
#
#     current_time = time.time()
#     fps = 1 / (current_time - previous_time)
#
#     previous_time = current_time
#     # put_text(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['red'], 2)
#
#     return correct_reps, direction, previous_time
#
#
# # test function
# def process_video(video_path: str) -> None:
#     cv2.namedWindow(f'Test {video_path}', cv2.WINDOW_NORMAL)
#     cap = cv2.VideoCapture(video_path)
#     detector = PoseDetector()
#     previous_time = 0
#     # todo: get 30 frames per second more accurately
#     cap.set(cv2.CAP_PROP_FPS, FPS)
#     correct_reps, incorrect_reps = 0, 0
#     direction = 0  # 0 - down, 1 - up
#     landmark_list = []  # list of landmarks
#
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             print(EMPTY_CAMERA_FRAME)
#             break
#         correct_reps, direction, previous_time = process_frame(image, detector, landmark_list, previous_time,
#                                                                correct_reps, direction)
#         cv2.resizeWindow(f'Test {video_path}', 800, 800)
#
#         cv2.imshow(f'Test {video_path}', image)
#         # Wait for the user to press EXIT_KEY to exit the program
#         if cv2.waitKey(1) & 0xFF == EXIT_KEY:  # change to 0 for pause
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()

#
# def main():
#     video_path = 'videos/test.mp4'
#     process_video(video_path)
#
#
# if __name__ == '__main__':
#     main()
