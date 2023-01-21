#!/usr/bin/env python3
import time
import sys
import numpy as np
import cv2

from rgbmatrix import RGBMatrix, RGBMatrixOptions
from PIL import Image

def cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

cam = cv2.VideoCapture(0)

# Configuration for the matrix
options = RGBMatrixOptions()
options.rows = 64
options.cols = 64
options.chain_length = 1
options.parallel = 1
options.hardware_mapping = 'adafruit-hat' # If you have an Adafruit HAT: 'adafruit-hat'

matrix = RGBMatrix(options = options)

backSub = cv2.createBackgroundSubtractorKNN()

kernel = np.ones((3,3), np.uint8)

black= Image.new(mode="RGB",size=(64,64))
image=black
try:
    print("Press CTRL-C to stop.")
    while True:
        #time.sleep(100)
        #matrix.SetImage(black)
        ret_val, img = cam.read()
        print(img.shape)
        img = img[:,80:560]
        print(img.shape)
        matrix.SetImage(image.convert('RGB'))
        img = cv2.resize(img,(matrix.width, matrix.height))
        #fgMask = backSub.apply(img)
        #fgMask = cv2.erode(fgMask, kernel, iterations=1)
        #fgMask = cv2.dilate(fgMask, kernel, iterations=1)
        #fgMask[np.abs(fgMask)<250] = 0
        #image = cv2_to_pil(fgMask)
        #image.thumbnail((matrix.width, matrix.height), Image.ANTIALIAS)
        image = cv2_to_pil(img)
        #matrix.SetImage(image.convert('RGB'))
except KeyboardInterrupt:
    sys.exit(0)
