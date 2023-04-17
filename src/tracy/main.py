import os

import cv2
import pygame

from .calibration import calibrate, transform_frame
from .detector import BlobDetector
from .screen_client import ScreenConfig, ZmqScreenClient
from .video_capture import RecordedVideoCapture


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
        self._running = True
        self.buffer.fill("black")

    def calibrate(self):
        if self._transform is None:
            self._transform = calibrate(
                self.camera, self.screen.get_config().get_target_size()
            )

    def transform_frame(self, frame: pygame.Surface):
        return transform_frame(
            frame, self._transform, self.screen.get_config().get_target_size()
        )

    def logic_loop(self):
        image = self.camera.read()
        if image is not None:
            frame = pygame.surfarray.array3d(image)
            _ = self.detector.detect(frame)
            self.buffer.blit(image, (0, 0))

    def graphics_loop(self):
        pygame.display.flip()

    def run(self):
        self.calibrate()
        self.setup()
        while self._running:
            self.logic_loop()
            self.graphics_loop()
            self.screen.show(self.buffer)
            self._dt = self.clock.tick(self.fps) / 1000
            cv2.waitKey(1)


def main():
    screen_config = ScreenConfig(64, 64, 10)
    screen_client = ZmqScreenClient(screen_config)
    camera = RecordedVideoCapture(
        "/mnt/hdd/TracyDataset/videos/dados_barras_sonido",
        "/mnt/hdd/TracyDataset/videos/april_tag/tag_10.png",
    )
    detector = BlobDetector()
    app = MainApp(screen_client, camera, detector)
    app.run()
    pygame.quit()


if __name__ == "__main__":
    main()
