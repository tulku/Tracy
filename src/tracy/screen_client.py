import cv2
import pygame
import zmq

from .custom_types import ScreenConfig


class ZmqScreenClient:
    def __init__(self, screen_config: ScreenConfig, host="localhost", port="1337"):
        context = zmq.Context()
        self._socket = context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")
        self._screen_config = screen_config

    def getConfig(self):
        return self._screen_config

    def show(self, surface):
        surface_array = pygame.surfarray.array3d(surface)
        pygame_image = cv2.cvtColor(surface_array, cv2.COLOR_RGB2BGR)
        screen_image = cv2.resize(
            pygame_image,
            self._screen_config.getScreenSize(),
            interpolation=cv2.INTER_NEAREST,
        )
        self._socket.send(screen_image.data.tobytes())
        self._socket.recv()


class Cv2ScreenClient:
    def __init__(self, screen_config: ScreenConfig):
        self._screen_config = screen_config

    def getConfig(self):
        return self._screen_config

    def show(self, surface):
        surface_array = pygame.surfarray.array3d(surface)
        surface_array = cv2.cvtColor(surface_array, cv2.COLOR_RGB2BGR)
        cv2.imshow("frame", surface_array)
        cv2.waitKey(1)
