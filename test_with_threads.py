import queue
import threading
import time

import numpy as np

from constants import *
import cv2
from poseDetector import PoseDetector, process_frame

stop_threads = threading.Event()


def process_video(video_path: str, images_queue: queue.Queue) -> None:
    cv2.namedWindow(f'Test {video_path}', cv2.WINDOW_NORMAL)

    print(f'Processing video {video_path}')
    cap = cv2.VideoCapture(video_path)
    detector = PoseDetector()
    previous_time = 0
    # todo: get 30 frames per second more accurately
    cap.set(cv2.CAP_PROP_FPS, FPS)
    correct_reps, incorrect_reps = 0, 0
    direction = 0  # 0 - down, 1 - up
    landmark_list = []  # list of landmarks

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print(EMPTY_CAMERA_FRAME)
            break

        correct_reps, direction, previous_time = process_frame(image, detector, landmark_list, previous_time,
                                                               correct_reps, direction, is_display_feedback=False)

        # previous_time = current_time
        # cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['red'], 2)
        # width, height = 800, 800
        # if image.shape[0] != height or image.shape[1] != width:
        #     image = cv2.resize(image, (width, height))
        if cv2.waitKey(1) & 0xFF == EXIT_KEY:
            stop_threads.set()
            images_queue.put(None)
            break
        images_queue.put(image)

    cap.release()
    print(f'Finished processing video {video_path} with {correct_reps} correct reps')


class VideoDisplayThread(threading.Thread):
    def __init__(self, window_name: str, images_queue: queue.Queue):
        super().__init__()
        self.window_name = window_name
        self.images_queue = images_queue
        self.should_stop = False

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        while not self.should_stop:
            if not self.images_queue.empty():
                image = self.images_queue.get()
                if image is None:
                    break
                cv2.imshow(self.window_name, image)
                if cv2.waitKey(1) & 0xFF == EXIT_KEY:
                    self.should_stop = True
                    break


def main():
    correct_video_path = 'videos/curl.mp4'
    incorrect_video_path = 'videos/curl2.mp4'

    correct_images_queue = queue.Queue()
    incorrect_images_queue = queue.Queue()

    video_display_incorrect_thread = VideoDisplayThread('Incorrect Video', incorrect_images_queue)
    video_display_correct_thread = VideoDisplayThread('Correct Video', correct_images_queue)

    video_display_incorrect_thread.start()
    video_display_correct_thread.start()

    incorrect_video_thread = threading.Thread(target=process_video, args=(incorrect_video_path, incorrect_images_queue))
    correct_video_thread = threading.Thread(target=process_video, args=(correct_video_path, correct_images_queue))

    incorrect_video_thread.start()
    correct_video_thread.start()

    incorrect_video_thread.join()
    correct_video_thread.join()

    video_display_incorrect_thread.should_stop = True
    video_display_correct_thread.should_stop = True

    video_display_incorrect_thread.join()
    video_display_correct_thread.join()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
