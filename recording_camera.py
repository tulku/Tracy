import sys

import cv2
import numpy as np


def blur_find_circles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, 1, 1, param1=100, param2=30, minRadius=5, maxRadius=50
    )

    return circles


def draw_circles(frame, circles):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(frame, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(frame, center, radius, (255, 0, 255), 3)


cam = cv2.VideoCapture(4)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)


while True:
    ret_val, frame = cam.read()

    circles = blur_find_circles(frame)

    draw_circles(frame, circles)

    cv2.imshow("detected circles", frame)

    # cv2.imshow("frame", frame)
    # cv2.imwrite(f"videos/dados_cambiando_brillo/dados_{i}.png", frame)
    # i = i + 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# After the loop release the cap object
cam.stop()
# Destroy all the windows
cv2.destroyAllWindows()
