#!/usr/bin/env python

import cv2
import apriltag


cam = cv2.VideoCapture(4)

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

while True:
    ret_val, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(9,9),0)
    result = detector.detect(gray)
    #print(result)
    for r in result:
        (A,B,C,D) = r.corners
        A=(int(A[0]),int(A[1]))
        B=(int(B[0]),int(B[1]))
        C=(int(C[0]),int(C[1]))
        D=(int(D[0]),int(D[1]))
        
        #print(points)
        frame = cv2.line(frame, A,B, (0,255,0), 9)
        frame = cv2.line(frame, B,C, (0,255,0), 9)
        frame = cv2.line(frame, C,D, (0,255,0), 9)
        frame = cv2.line(frame, D,A, (0,255,0), 9)
    cv2.imshow("Tracy", gray)
    cv2.imshow("Cuadrado", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()


