import os

import pygame

from .detector import BlobDetector
from .screen_client import ScreenConfig, ZmqScreenClient
from .video_capture import PyGameVideoCapture


def init_pygame_display(screen_config: ScreenConfig):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()
    pygame.display.set_mode(screen_config.getTargetSize(), 0, 24)
    return pygame.display.get_surface()


class MainApp:
    def __init__(self, screen_client, video_capture, detector):
        self.detector = detector
        self.camera = video_capture
        self.screen = screen_client
        self.buffer = init_pygame_display(screen_client.getConfig())
        self.clock = pygame.time.Clock()
        self.fps = 60

    def setup(self):
        self.buffer.fill("black")
        self._player_pos = pygame.Vector2(
            self.buffer.get_width() / 2, self.buffer.get_height() / 2
        )
        self._dt = 0
        self._running = True

    def logic_loop(self):
        image = self.camera.read()
        if image is not None:
            frame = pygame.surfarray.array3d(image)
            detections = self.detector.detect(frame)
            print(len(detections))

            self.buffer.blit(image, (0, 0))
        # self.buffer.fill("purple")

    def graphics_loop(self):
        pygame.display.flip()

    def run(self):
        self.setup()
        while self._running:
            self.logic_loop()
            self.graphics_loop()
            self.screen.show(self.buffer)
            self._dt = self.clock.tick(self.fps) / 1000


def main():
    screen_config = ScreenConfig(64, 64, 10)
    screen_client = ZmqScreenClient(screen_config)
    camera = PyGameVideoCapture()
    detector = BlobDetector()
    app = MainApp(screen_client, camera, detector)
    app.run()
    pygame.quit()


if __name__ == "__main__":
    main()
