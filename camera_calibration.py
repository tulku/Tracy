#!/usr/bin/env python
import queue
import sys
import threading

import apriltag
import cv2
import numpy
import zmq
from ultralytics import YOLO

host = "192.168.1.178"
port = 1337

screen_size = 64
target_size = screen_size * 10


def connect_to_screen(host, port):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{host}:{port}")
    return socket


def open_image(img_path="tag16_05.png"):
    img = cv2.imread(img_path)
    # cv2.imshow("QR",img)
    return img


def send_img(socket, img):
    socket.send(img.data.tobytes())
    socket.recv()


def screen_to_black(socket, screen_size):
    black_image = numpy.zeros(screen_size * screen_size * 3, dtype=numpy.int8)
    send_img(socket, black_image)


def load_yolo_model():
    return YOLO("models/best.pt")


def infer_dice_positions(model, frame):
    results = model(source=frame)
    blobs = []
    for result in results:
        for dice_number in range(0, result.boxes.xywh.shape[0]):
            dice = result.boxes.xywh[dice_number]
            blob = cv2.KeyPoint()
            blob.pt = (float(dice[0]), float(dice[1]))
            blob.size = float(dice[2] / 2)
            blobs.append(blob)

    return blobs


def find_screen_corners(frame):

    options = apriltag.DetectorOptions(
        families="tag16h5",
        border=1,
        nthreads=4,
        quad_decimate=1.0,
        quad_blur=0.0,
        refine_edges=True,
        refine_decode=False,
        refine_pose=False,
        debug=False,
        quad_contours=True,
    )
    detector = apriltag.Detector(options)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)
    # cv2.imshow("filtrado", gray)
    results = detector.detect(gray)
    if results:
        r = results[0]
        (A, B, C, D) = r.corners
        A = (int(A[0]), int(A[1]))
        B = (int(B[0]), int(B[1]))
        C = (int(C[0]), int(C[1]))
        D = (int(D[0]), int(D[1]))
        corners = [A, B, C, D]
        image_with_lines = draw_detection(frame, corners)
        # cv2.imshow("Cuadrado", image_with_lines)
        return corners

    return None


def calculate_transform(corners, target_size):
    pts1 = numpy.float32(corners)
    pts2 = numpy.float32(
        [[0, 0], [target_size, 0], [target_size, target_size], [0, target_size]]
    )
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return M


def draw_detection(frame, corners):
    A, B, C, D = corners
    frame_lines = cv2.line(frame, A, B, (0, 255, 0), 9)
    frame_lines = cv2.line(frame_lines, B, C, (0, 255, 0), 9)
    frame_lines = cv2.line(frame_lines, C, D, (0, 255, 0), 9)
    frame_lines = cv2.line(frame_lines, D, A, (0, 255, 0), 9)
    return frame_lines


def transform_frame(frame, M, target_size):
    # from here https://pypi.org/project/apriltag/
    return cv2.warpPerspective(frame, M, (target_size, target_size))


def calibrate(cam, target_size, socket):
    corners = None
    frame = None
    while corners is None:
        ret_val, frame = cam.read()
        if frame is None:
            continue
        corners = find_screen_corners(frame)
        cv2.imshow("original", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    frame_and_lines = draw_detection(frame, corners)
    cv2.imshow("Cuadrado", frame_and_lines)
    return calculate_transform(corners, target_size)


def brightness_threshold_HSV(img):
    img_HSV = cv2.cvtColor(
        img, cv2.COLOR_BGR2HSV
    )  # convert from RGB to HSV (hue, saturation, value)
    lower = numpy.array([0, 0, 120])  # H,S,V
    higher = numpy.array([180, 255, 255])
    mask = cv2.inRange(img_HSV, lower, higher)
    filtered_img = cv2.bitwise_and(img, img, mask=mask)
    return filtered_img


def brightness_threshold(img):
    img_gray = cv2.cvtColor(
        img, cv2.COLOR_BGR2GRAY
    )  # convert from RGB to HSV (hue, saturation, value)
    _, mask = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    filtered_img = cv2.bitwise_and(img, mask_color)
    return filtered_img


# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.image_queue = queue.Queue(maxsize=1)
        self.t = threading.Thread(target=self._reader)
        self._running = True
        self.t.daemon = True
        self.t.start()

    # grab frames as soon as they are available
    def _reader(self):
        while self._running:
            ret, image = self.cap.read()
            if not ret:
                continue
            self.image_queue.put(image)

    def stop(self):
        self._running = False
        self.cap.release()

    # retrieve latest frame
    def read(self):
        if self.image_queue.empty():
            return False, None
        return True, self.image_queue.get()


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


def blur_find_circles(frame):
    cv2.imwrite(f"Pictures/01_frame.png", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 25)
    cv2.imwrite(f"Pictures/02_gray.png", gray)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        15,
        10,
        param1=100,
        param2=30,
        minRadius=30,
        maxRadius=100,
    )

    return circles


def find_blobs(frame):
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.filterByColor = False
    params.minThreshold = 0
    params.maxThreshold = 255
    params.thresholdStep = 3

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500
    params.maxArea = 2600

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.05

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.2

    # Min distance between blobs
    params.minDistBetweenBlobs = 20

    # Create a detector with the parameters
    # OLD: detector = cv2.SimpleBlobDetector(params)
    detector = cv2.SimpleBlobDetector_create(params)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (25, 25), 0)

    blobs = detector.detect(frame)
    return blobs


def draw_blobs(frame, blobs):
    frame_with_blob = cv2.drawKeypoints(
        frame,
        blobs,
        numpy.array([]),
        (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DEFAULT,
    )
    return frame_with_blob


def draw_circles(frame, circles):
    if circles is not None:
        circles = numpy.uint16(numpy.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(frame, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(frame, center, radius, (255, 0, 255), 3)


def draw_blobs_as_circles(frame, blobs):
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
    ]
    for blob_number, blob in enumerate(blobs):
        center = numpy.uint16(numpy.around(blob.pt))
        # circle center
        radius = int(blob.size)
        cv2.circle(frame, center, radius, colors[blob_number % len(colors)], -1)


def main(use_yolo=False):
    # cam = VideoCapture(4)
    # cam = SingleVideoCapture(4)
    cam = cv2.VideoCapture(4)
    socket = connect_to_screen(host, port)
    calibration_image = open_image("tag16_05.png")
    send_img(socket, calibration_image)
    M = calibrate(cam, target_size, socket)
    screen_to_black(socket, screen_size)

    if use_yolo:
        model = load_yolo_model()

    while True:
        ret_val, frame = cam.read()

        if frame is None:
            continue

        dst = transform_frame(frame, M, target_size)

        # Detect blobs.
        if use_yolo:
            blobs = infer_dice_positions(model, dst)
        else:
            blobs = find_blobs(dst)

        # frame_with_blob = draw_blobs(dst, blobs)
        black_image = numpy.zeros((target_size, target_size, 3), dtype=numpy.int8)
        cv2.imshow("image before", black_image)
        draw_blobs_as_circles(black_image, blobs)
        screen_image = cv2.resize(
            black_image, (screen_size, screen_size), interpolation=cv2.INTER_NEAREST
        )
        # cv2.imshow("im_with_keypoints", frame_with_blob)
        send_img(socket, screen_image)
        cv2.imshow("image", dst)
        cv2.imshow("image with blobs", black_image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Turn screen black
    screen_to_black(socket, screen_size)
    # After the loop release the cap object
    cam.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def yolo_main():
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    model = load_yolo_model()
    blobs = infer_dice_positions(model, image)
    draw_blobs_as_circles(image, blobs)
    cv2.imshow("image with blobs", image)

    if cv2.waitKey(0) & 0xFF == ord("q"):
        pass


if __name__ == "__main__":
    main(use_yolo=True)
