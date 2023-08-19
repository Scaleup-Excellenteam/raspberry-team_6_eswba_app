import time

import cv2
from constants import *


# usable cv2 functions for drawing
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

    def detect_pose(self, image, is_draw=True):
        # Convert to RGB and start processing image
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(image_RGB)

        # Draw the pose detection result on the image
        if is_draw and self.result.pose_landmarks:
            DRAWING.draw_landmarks(image, self.result.pose_landmarks, MP_POSE.POSE_CONNECTIONS)

        return image

    def get_position(self, image, is_draw=True):
        landmark_list = []
        if self.result.pose_landmarks:
            for id, landmark in enumerate(self.result.pose_landmarks.landmark):
                height, width, channels = image.shape
                center_x, center_y = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append([id, center_x, center_y])
                if is_draw:
                    draw_circle(image, (center_x, center_y), 5, COLORS['red'], cv2.FILLED)

        return landmark_list


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    detector = PoseDetector()
    previous_time = 0

    while cap.isOpened():
        success, image = cap.read()
        image = detector.detect_pose(image)
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
