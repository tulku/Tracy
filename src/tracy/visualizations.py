import cv2
import numpy


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
        cv2.circle(frame, center, 20, colors[blob_number % len(colors)], -1)
