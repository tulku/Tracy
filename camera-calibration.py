#!/usr/bin/env python

import cv2
import apriltag
import numpy
# import numpy as np


def find_screen_corners(frame):

    options = apriltag.DetectorOptions(families='tag16h5',
                                 border=1,
                                 nthreads=4,
                                 quad_decimate=1.0,
                                 quad_blur=0.0,
                                 refine_edges=True,
                                 refine_decode=False,
                                 refine_pose=False,
                                 debug=False,
                                 quad_contours=True)
    detector = apriltag.Detector(options)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(9,9),0)
    results = detector.detect(gray)
    #print(result)
    if results:
        r=results[0]
        (A,B,C,D) = r.corners
        A=(int(A[0]),int(A[1]))
        B=(int(B[0]),int(B[1]))
        C=(int(C[0]),int(C[1]))
        D=(int(D[0]),int(D[1]))
        corners=[A,B,C,D]
        image_with_lines = draw_detection(frame,corners)
        cv2.imshow("Cuadrado", image_with_lines)
        return corners

    return None


def calculate_transform(corners,target_size):
    pts1 = numpy.float32(corners)
    pts2 = numpy.float32([[0,0],[target_size,0],[target_size,target_size],[0,target_size]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
#    dst = cv2.warpPerspective(frame,M,(target_size,target_size))
    return M

def draw_detection(frame,corners):
    A,B,C,D=corners
    print(f"{A}, {B}, {C}, {D}")
    frame_lines = cv2.line(frame, A,B, (0,255,0), 9)
    frame_lines = cv2.line(frame_lines, B,C, (0,255,0), 9)
    frame_lines = cv2.line(frame_lines, C,D, (0,255,0), 9)
    frame_lines = cv2.line(frame_lines, D,A, (0,255,0), 9)
    return frame_lines

def transform_frame(frame,M,target_size=64*4):
    return  cv2.warpPerspective(frame,M,(target_size,target_size))

def calibrate(cam,target_size):
    corners = None
    frame = None
    while corners is None:
        ret_val, frame = cam.read()
        corners = find_screen_corners(frame)
        cv2.imshow("original", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frame_and_lines = draw_detection(frame,corners)
    cv2.imshow("Cuadrado", frame_and_lines)
    return calculate_transform(corners,target_size)

target_size = 64*8

cam = cv2.VideoCapture(4)


M = calibrate(cam, target_size)

while True:
    ret_val, frame = cam.read()

    dst = transform_frame(frame,M,target_size)

    cv2.imshow("Transformed", dst)
    cv2.imshow("original", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()


