"""
Tracy simulation.
It listens to a ZMQ socket and displays the received image on the screen.
"""

import cv2
import numpy
import zmq

from tracy.custom_types import ScreenConfig


def draw_grid(img, grid_shape, color=(0, 0, 0), thickness=3):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in numpy.linspace(start=dx, stop=w - dx, num=cols - 1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in numpy.linspace(start=dy, stop=h - dy, num=rows - 1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img


class TracySim:
    def __init__(self, screen_config):
        self.screen_config = screen_config
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind("tcp://*:1337")
        screen_size = screen_config.width * screen_config.height
        self.message = numpy.ones(screen_size * 3, dtype=numpy.int8) * 50

    def simulate_screen(self, image):
        image = cv2.resize(
            image, self.screen_config.get_target_size(), interpolation=cv2.INTER_NEAREST
        )
        image = draw_grid(image, self.screen_config.get_screen_size())
        return image

    def wait_for_message(self):
        #  Wait for next request from client
        try:
            self.message = self.socket.recv(flags=zmq.NOBLOCK)
            self.socket.send("OK".encode())
        except zmq.ZMQError:
            pass

        #  Send reply back to client
        return self.message

    def to_numpy(self, message):
        image = numpy.frombuffer(message, dtype=numpy.uint8)
        # Reshaping here to the screen size ensures that the image sent
        # has the correct size.
        width, height = self.screen_config.get_screen_size()
        return image.reshape((width, height, 3))

    def run(self):
        while True:
            message = self.wait_for_message()
            image = self.to_numpy(message)
            simulated = self.simulate_screen(image)
            cv2.imshow("TRACY SIM", simulated)
            cv2.waitKey(50)


def main():
    screen_config = ScreenConfig(64, 64, 10)
    sim = TracySim(screen_config)
    sim.run()


if __name__ == "__main__":
    main()
