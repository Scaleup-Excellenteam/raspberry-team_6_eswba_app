import argparse
from pathlib import Path
import cv2
from constants import *
import poseDetector

pose_detector = poseDetector.PoseDetector()
def detect_pose(frame, pose):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)  # start detecting the keypoints
    return result


def draw_pose_on_frame(frame, detected_frame, drawing):
    if detected_frame.pose_landmarks:
        DRAWING.draw_landmarks(
            image=frame,
            landmark_list=detected_frame.pose_landmarks,
            connections=mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=LANDMARK_DRAWING_SPEC,
            connection_drawing_spec=CONNECTION_DRAWING_SPEC
        )
        # print(detected_frame.pose_landmarks)


def detect_human_pose_from_video(video_path: str):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        # frame_to_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # result = detect_pose(frame, POSE_INIT)
        pose_detector.detect_pose(frame)
        # draw_pose_on_frame(frame, result, DRAWING)
        cv2.imshow('Video After detecting the pose', frame)
        if cv2.waitKey(1) & 0xFF == EXIT_KEY:
            break
    cap.release()
    cv2.destroyAllWindows()


def main(argv=None) -> None:
    """
    It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually examine the results.
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module.

    :param argv: In case you want to programmatically run this.
    """

    parser = argparse.ArgumentParser("Test Pose Detection Mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)

    # If you entered a custom dir to run from or the default dir exists in your project, then:
    directory_path: Path = Path(args.dir or DEFAULT_BASE_DIR)
    if directory_path.exists():
        results_dir = Path(OUTPUT_DIR_NAME)
        results_dir.mkdir(exist_ok=True)

        for video in directory_path.iterdir():
            if video.is_file() and video.name.lower().split('.')[-1] in IMAGE_EXTENSIONS:
                video_path = str(video)
                detect_human_pose_from_video(video_path)


if __name__ == '__main__':
    main()

