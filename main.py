import argparse
import os
from pathlib import Path
from typing import List
import mediapipe as mp
import cv2

IMAGE_EXTENSIONS = ['png', 'jpg']
DEFAULT_BASE_DIR = 'images'
POSE_INIT = mp.solutions.pose.Pose(
    static_image_mode=True, # change to False of was video stream
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

DRAWING = mp.solutions.drawing_utils


def detect_human_pose_from_image(image_path: str):
    pose = POSE_INIT
    img = cv2.imread(image_path, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = pose.process(imgRGB) # start detecting the keypoints

    if result.pose_landmarks:
        DRAWING.draw_landmarks(
            image=img,
            landmark_list=result.pose_landmarks,
            connections=mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=DRAWING.DrawingSpec(
                color=(250, 250, 250)
                ,
                thickness=7,
                circle_radius=4
            ),
            connection_drawing_spec=DRAWING.DrawingSpec(
                color=(0, 0, 255),
                thickness=11,
                circle_radius=3,
            )
        )
    print(result.pose_landmarks)

    filename = Path(image_path).stem
    output_filename = Path('results') / f"{filename}_pose_detection_res.png"
    cv2.imwrite(filename=str(output_filename), img=img)


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
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        file_list: List[Path] = []
        # for subdirectory in directory_path.iterdir():
        #     if subdirectory.is_dir():
        #         for entry in os.scandir(subdirectory):
        #             if entry.is_file() and entry.name.lower().split('.')[-1] in IMAGE_EXTENSIONS:

        # for image in file_list:
        for image in directory_path.iterdir():
            if image.is_file() and image.name.lower().split('.')[-1] in IMAGE_EXTENSIONS:
                image_path = str(image)
                detect_human_pose_from_image(image_path)


if __name__ == '__main__':
    main()
