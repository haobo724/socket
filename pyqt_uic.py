import numpy as np
import cv2
from tool import Red_seg


def red_area(img):
    background_area = 27.2 * 17
    result = Red_seg(img)
    pixels = sum(result > 1)
    return pixels / background_area, result

CAM = cv2.VideoCapture(0)
while True:
    ret, frame = CAM.read()
    if not ret:
        print('Error')
        break
    area,frame = red_area(frame)
    area = str(area)
    print(area)
    im_show = cv2.putText(frame,text=area)
    cv2.imshow('OCR', im_show)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break


