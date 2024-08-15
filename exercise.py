from abc import ABC, abstractmethod

import numpy as np

from constants import *

class Exercise(ABC):
    """
    Abstract class for exercises that will be used in the program
    """
    def __init__(self):
        self.name = None
        self.body_parts = None
        self.angle = None
        self.percentage = None

    def __str__(self):
        return f"{self.name} exercise"

    @abstractmethod
    def run(self, image: np.ndarray) -> None:
        """
        Run the exercise
        :param image: image to run the exercise on
        :return: None
        """
        pass


    def __del__(self):
        self.cap.release()