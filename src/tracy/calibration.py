import apriltag
import cv2
import numpy


def find_screen_corners(frame: numpy.ndarray):
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
    results = detector.detect(gray)
    if results:
        r = results[0]
        (A, B, C, D) = r.corners
        A = (int(A[0]), int(A[1]))
        B = (int(B[0]), int(B[1]))
        C = (int(C[0]), int(C[1]))
        D = (int(D[0]), int(D[1]))
        corners = [A, B, C, D]
        return corners

    return None


def calculate_transform(corners, target_size):
    pts1 = numpy.float32(corners)
    pts2 = numpy.float32(
        [[0, 0], [target_size[0], 0], target_size, [0, target_size[1]]]  # type: ignore
    )
    M = cv2.getPerspectiveTransform(pts1, pts2)  # type: ignore
    return M


def draw_detection(frame, corners):
    A, B, C, D = corners
    frame_lines = cv2.line(frame, A, B, (0, 255, 0), 9)
    frame_lines = cv2.line(frame_lines, B, C, (0, 255, 0), 9)
    frame_lines = cv2.line(frame_lines, C, D, (0, 255, 0), 9)
    frame_lines = cv2.line(frame_lines, D, A, (0, 255, 0), 9)
    return frame_lines


def transform_frame(frame: numpy.ndarray, M, target_size):
    # from here https://pypi.org/project/apriltag/
    return cv2.warpPerspective(frame, M, target_size)


def calibrate(cam, target_size):
    corners = None
    frame = None
    while corners is None:
        frame = cam.as_numpy()
        if frame is None:
            continue
        corners = find_screen_corners(frame)
    frame_and_lines = draw_detection(frame, corners)
    cv2.imshow("Cuadrado", frame_and_lines)
    cv2.waitKey(1)

    return calculate_transform(corners, target_size)
