import itertools
import os

import cv2
import numpy
import pygame.camera
import pygame.image
import pygame.pixelarray


class PyGameVideoCapture:
    def __init__(self, camera="/dev/video0"):
        self._snapshot = None
        pygame.camera.init()
        self._cam = pygame.camera.Camera(camera, (1280, 720))
        self._cam.start()

    def read(self) -> pygame.Surface | None:
        if self._cam.query_image():
            self._snapshot = self._cam.get_image()
        return self._snapshot

    def as_numpy(self) -> numpy.ndarray | None:
        surface = self.read()
        if surface is None:
            return None
        frame = pygame.surfarray.array3d(surface)
        frame = frame.swapaxes(0, 1)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def release(self):
        self._cam.stop()


class Cv2VideoCapture:
    def __init__(self, camera=0):
        self._cam = cv2.VideoCapture(camera)

    def read(self) -> pygame.Surface | None:
        frame = self.as_numpy()
        if frame is None:
            return None
        frame = frame.swapaxes(0, 1)
        return pygame.pixelcopy.make_surface(frame)

    def as_numpy(self) -> numpy.ndarray | None:
        ret_val, frame = self._cam.read()
        if not ret_val:
            return None

        return frame

    def release(self):
        self._cam.release()


class RecordedVideoCapture:
    """
    Reads video frames from a directory and reproduces them in a loop.
    The first frame can be provided directly as a parameter.
    """

    def __init__(self, video_frames: str, first_frame: str | None = None):
        self.path = video_frames
        all_frames = sorted(os.listdir(video_frames))
        self.infinite_frames = itertools.cycle(all_frames)
        if first_frame is not None:
            self.infinite_frames = itertools.chain([first_frame], self.infinite_frames)

    def read(self) -> pygame.Surface | None:
        img_path = next(self.infinite_frames)
        return pygame.image.load(os.path.join(self.path, img_path))

    def release(self):
        pass

    def as_numpy(self) -> numpy.ndarray | None:
        surface = self.read()
        if surface is None:
            return None
        frame = pygame.surfarray.array3d(surface)
        frame = frame.swapaxes(0, 1)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
