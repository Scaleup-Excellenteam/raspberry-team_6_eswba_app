import cv2

from physicalExercise import PhysicalExercise
from poseDetector import PoseDetector
from constants import *

UP_DIRECTION = 0
DOWN_DIRECTION = 1
GO_DOWN = "Go Down"
GO_UP = "Go Up"
TEST_VIDEO = "squats/squat6.mp4"


class Squat(PhysicalExercise):
    def __init__(self, pose_detector: PoseDetector, source_capture: str = None) -> None:
        super().__init__(pose_detector)

    @staticmethod
    def check_posture_correctness(left_knee_angle: float, right_knee_angle: float, lef_hip_angle: float) -> bool:
        """
        Check if the posture is correct
        :param left_knee_angle: angle of the left knee
        :param right_knee_angle: angle of the right knee
        :param lef_hip_angle: angle of the left hip
        :return: True if the posture is correct, False otherwise
        """
        return lef_hip_angle > SQUAT_ANGLES['lef_hip']['posture_angle'] \
            and left_knee_angle > SQUAT_ANGLES['left-knee']['posture_angle'] \
            and right_knee_angle > SQUAT_ANGLES['right-knee']['posture_angle']

    def interment_correct_reps(self, next_direction: int) -> None:
        """
        Increment the correct reps
        :return: None
        """
        self.correct_reps += 0.5
        self.direction = next_direction

    def check_if_should_up(self, left_knee_angle: float, right_knee_angle: float, lef_hip_angle: float) -> None:
        """
        Check if the user should go up
        :param self: instance of the class
        :param left_knee_angle: angle of the left knee
        :param right_knee_angle: angle of the right knee
        :param lef_hip_angle: angle of the left hip
        :return: None
        """
        if lef_hip_angle <= SQUAT_ANGLES['lef_hip']['down_angle'] \
                and left_knee_angle <= SQUAT_ANGLES['left-knee']['down_angle'] \
                and right_knee_angle <= SQUAT_ANGLES['right-knee']['down_angle']:
            self.feedback = GO_UP
            if self.direction == UP_DIRECTION:
                # next direction is down
                self.interment_correct_reps(DOWN_DIRECTION)

    def check_if_should_down(self, left_knee_angle: float, right_knee_angle: float, lef_hip_angle: float) -> None:
        """
        Check if the user should go down
        :param self: instance of the class
        :param left_knee_angle: angle of the left knee
        :param right_knee_angle: angle of the right knee
        :param lef_hip_angle: angle of the left hip
        :return: None
        """
        if lef_hip_angle > SQUAT_ANGLES['lef_hip']['up_angle'] \
                and left_knee_angle > SQUAT_ANGLES['left-knee']['up_angle'] \
                and right_knee_angle > SQUAT_ANGLES['right-knee']['up_angle']:
            self.feedback = GO_DOWN
            if self.direction == DOWN_DIRECTION:
                # next direction is up
                self.interment_correct_reps(UP_DIRECTION)

    def run(self, pose_detector: PoseDetector, source_capture: str = None) -> None:
        cap = super().prepare_cap(source_capture)
        while cap.isOpened():
            if source_capture:
                success, image = cap.read()
                if not success:
                    print("Can't read the video")
                    break
            landmark_list, image, success = self.get_landmark_list(pose_detector, cap)
            if len(landmark_list) != 0:
                left_knee_angle = pose_detector.get_angle(image, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, landmark_list)
                right_knee_angle = pose_detector.get_angle(image, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, landmark_list)
                lef_hip_angle = pose_detector.get_angle(image, LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, landmark_list)
                # print(left_knee_angle, right_knee_angle, lef_hip_angle)
                # Check if the posture is correct
                if self.check_posture_correctness(left_knee_angle, right_knee_angle, lef_hip_angle):
                    self.is_correct_posture = True

                if self.is_correct_posture:
                    self.check_if_should_up(left_knee_angle, right_knee_angle, lef_hip_angle)
                    self.check_if_should_down(left_knee_angle, right_knee_angle, lef_hip_angle)
            # add counter and feedback text
            cv2.rectangle(image, (580, 50), (600, 380), COLORS['green'], 3)
            cv2.putText(image, str(int(self.correct_reps)), (25, 455), cv2.FONT_HERSHEY_PLAIN, 5, COLORS['red'], 5)
            cv2.putText(image, self.feedback
                        , (500, 40), cv2.FONT_HERSHEY_PLAIN, 2,
                        COLORS['red'] if self.feedback not in [UP_DIRECTION, DOWN_DIRECTION] else COLORS['green'], 2)

            cv2.imshow("Image", image)

            # percentage = np.interp(angle, (210, 310), (0, 100))
            # PoseDetector.draw_bar(image, angle=angle, percentage=percentage)
            # if is_display_feedback:
            #     correct_reps, direction = display_feedback(image, percentage, correct_reps, direction)
            # else:
            # put_text(image, str(int(percentage)) + "%", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, COLORS['red'], 2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()


if __name__ == '__main__':
    pose_detector = PoseDetector()
    squat = Squat(pose_detector)
    squat.run(pose_detector, source_capture=TEST_VIDEO)
    cv2.destroyAllWindows()
