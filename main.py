import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np

from constants import *
import poseDetector

pose_detector = poseDetector.PoseDetector()


# todo: add logging for better debugging


def check_if_path_exists(path: str) -> None:
    """
    Check if the path exists
    :param path: to check
    :throws: FileNotFoundError if the path doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} doesn't exist")


def image_process(image: np.ndarray) -> None:
    pose_detector.detect_pose(image, is_draw=False)
    landmarks_list = pose_detector.get_position(image, is_draw=False)
    if landmarks_list:
        for part_name in BODY_PARTS.keys():
            try:
                angles_to_calculate = BODY_PARTS[part_name]
                print("angles_to_calculate: ", angles_to_calculate)
                # print(angles_to_calculate)
                angle = pose_detector.get_angle(image, *angles_to_calculate)

                print("get_angle: ", angle)
                # print("name: ", part_name + ", angle:", angle)
                percentage = np.interp(angle, (210, 310), (0, 100))
                print("percentage: ", percentage)
                # cv2.putText(image, str(int(percentage)) + "%", 10, 50, cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            except IndexError:
                continue
    cv2.imshow('After detecting the pose', image)


def detect_human_pose_from_video(video_path: str) -> None:
    """
    Detect human pose from video or image according to the extension of the file
    :param video_path: path to the video or image
    :return: None
    """
    try:
        check_if_path_exists(video_path)
    except FileNotFoundError as e:
        sys.stderr.write(f"{e}\n")
        return

    if video_path.split('.')[-1] in IMAGE_EXTENSIONS:
        image = cv2.imread(video_path)
        image_process(image)
        cv2.waitKey(0)
        return

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break
        image_process(image)
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
        # extend extensions to include all cases of VIDEO_EXTENSIONS and IMAGE_EXTENSIONS
        extention = list(set(VIDEO_EXTENSIONS + IMAGE_EXTENSIONS))
        for video in directory_path.iterdir():
            if video.is_file() and video.name.lower().split('.')[-1] in extention:
                video_path = str(video)
                detect_human_pose_from_video(video_path)


if __name__ == '__main__':
    main()
    # detect_human_pose_from_video('videos/curl2.mp4')  # test
