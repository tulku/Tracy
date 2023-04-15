import cv2
from ultralytics import YOLO


class YoloDetector:
    def __init__(self, model="models/best.pt"):
        self.model = YOLO(model)

    def detect(self, frame):
        results = self.model(source=frame)
        blobs = []
        for result in results:
            for dice_number in range(0, result.boxes.xywh.shape[0]):
                dice = result.boxes.xywh[dice_number]
                blob = cv2.KeyPoint()
                blob.pt = (float(dice[0]), float(dice[1]))
                blob.size = float(dice[2] / 2)
                blobs.append(blob)

        return blobs


class BlobDetector:
    def __init__(self):
        # Setup SimpleBlobDetector parameters.
        self.params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        self.params.filterByColor = False
        self.params.minThreshold = 0
        self.params.maxThreshold = 255
        self.params.thresholdStep = 3

        # Filter by Area.
        self.params.filterByArea = True
        self.params.minArea = 1500
        self.params.maxArea = 2600

        # Filter by Circularity
        self.params.filterByCircularity = False
        self.params.minCircularity = 0.05

        # Filter by Convexity
        self.params.filterByConvexity = False
        self.params.minConvexity = 0.5

        # Filter by Inertia
        self.params.filterByInertia = True
        self.params.minInertiaRatio = 0.2

        # Min distance between blobs
        self.params.minDistBetweenBlobs = 20

    def detect(self, frame):
        detector = cv2.SimpleBlobDetector_create(self.params)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (25, 25), 0)
        return detector.detect(frame)
