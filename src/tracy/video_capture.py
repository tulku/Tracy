import cv2
import pygame
import pygame.camera


class PyGameVideoCapture:
    def __init__(self, camera="/dev/video0"):
        self._snapshot = None
        pygame.camera.init()
        self._cam = pygame.camera.Camera(camera, (1280, 720))
        self._cam.start()

    def read(self):
        if self._cam.query_image():
            self._snapshot = self._cam.get_image()
        return self._snapshot

    def release(self):
        self._cam.stop()


class Cv2VideoCapture:
    def __init__(self, camera=0):
        self._cam = cv2.VideoCapture(camera)

    def read(self):
        return self._cam.read()

    def release(self):
        self._cam.release()


class SingleVideoCapture:
    def __init__(self, name):
        self.name = name

    def read(self):
        cap = cv2.VideoCapture(self.name)
        ret_val, frame = cap.read()
        cap.release()
        if not ret_val:
            frame = None
        return ret_val, frame
