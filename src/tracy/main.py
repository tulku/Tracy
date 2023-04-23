import os

import cv2
import numpy
import pygame

from .calibration import calibrate, transform_frame
from .detector import YoloDetector
from .screen_client import ScreenConfig, ZmqScreenClient
from .video_capture import (
    Cv2VideoCapture,
    PyGameVideoCapture,
    RecordedVideoCapture,
    image_from_file,
)
from .visualizations import draw_blobs_as_circles


def init_pygame_display(screen_config: ScreenConfig):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    pygame.display.set_mode(screen_config.get_target_size(), 0, 24)
    return pygame.display.get_surface()


class MainApp:
    def __init__(self, screen_client, video_capture, detector):
        self.detector = detector
        self.camera = video_capture
        self.screen = screen_client
        self.buffer = init_pygame_display(screen_client.get_config())
        self.clock = pygame.time.Clock()
        self.fps = 60

        self._dt = 0
        self._running = False
        self._transform = None

    def setup(self):
        self.buffer.fill("black")
        self._running = True

    def show_calibration_frame(self):
        frame = image_from_file("tag16_05.png", self.screen.get_config())
        print("Showing calibration frame")
        if frame is not None:
            self.frame_to_buffer(frame)

    def try_calibrate(self):
        self._transform = calibrate(
            self.camera, self.screen.get_config().get_target_size()
        )

    def transform_frame(self, frame: numpy.ndarray):
        return transform_frame(
            frame, self._transform, self.screen.get_config().get_target_size()
        )

    def frame_to_buffer(self, frame: numpy.ndarray):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_surface = pygame.surfarray.make_surface(frame)
        self.buffer.blit(image_surface, (0, 0))

    def logic_loop(self):
        frame = self.camera.as_numpy()
        if frame is not None:
            cv2.imshow("Raw Frame", frame)
            frame = self.transform_frame(frame)
            cv2.imshow("Transformed Frame", frame)
            blobs = self.detector.detect(frame)
            frame.fill(0)
            draw_blobs_as_circles(frame, blobs)
            self.frame_to_buffer(frame)

    def calibration_loop(self):
        self.show_calibration_frame()
        self.try_calibrate()

    def graphics_loop(self):
        pygame.display.flip()

    def run(self):
        self.setup()
        while self._running:
            self.graphics_loop()
            if self._transform is None:
                self.calibration_loop()
            else:
                self.logic_loop()

            self.screen.show(self.buffer)
            self._dt = self.clock.tick(self.fps) / 1000
            cv2.waitKey(1)


def mocked():
    screen_config = ScreenConfig(64, 64, 10)
    screen_client = ZmqScreenClient(screen_config, host="localhost")
    camera = RecordedVideoCapture(
        "/mnt/hdd/TracyDataset/videos/dados_cuadrado",
        "/mnt/hdd/TracyDataset/videos/april_tag/tag_2.png",
    )
    detector = YoloDetector()
    app = MainApp(screen_client, camera, detector)
    app.run()
    pygame.quit()


def tracy():
    screen_config = ScreenConfig(64, 64, 10)
    screen_client = ZmqScreenClient(screen_config, host="192.168.1.178")
    camera = Cv2VideoCapture("/dev/video0")
    detector = YoloDetector()
    app = MainApp(screen_client, camera, detector)
    app.run()
    pygame.quit()


if __name__ == "__main__":
    tracy()
