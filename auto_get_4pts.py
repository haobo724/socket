import glob
import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from paddleocr import PaddleOCR,draw_ocr
import cv2
from tool import Green_seg
from Gui_base import CAMERA_PORT_TOP
print('torch gpu:', torch.cuda.is_available())


def run():
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True )
    CAM = cv2.VideoCapture(0)
    while True:
        ret , frame = CAM.read()
        if not ret :
            print('Error')
            break

        result = ocr.ocr(frame, cls=True)
        for line in result:
            print(line)
        #
        #
        # # draw result
        #
        boxes = [line[0] for line in result]
        # im_show = cv2.rectangle(frame,)
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        im_show = draw_ocr(frame, boxes, txts, scores)
        cv2.imshow('OCR',im_show)
        k =cv2.waitKey(1)
        if k == ord('q'):
            break
def run_classic():
    CAM = cv2.VideoCapture(0)
    while True:
        ret, frame = CAM.read()
        if not ret:
            print('Error')
            break

        img = Green_seg(frame)
        img=cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        blank = np.zeros_like(img)
        rect = cv2.minAreaRect(contours)
        box= cv2.boxPoints(rect).astype(int)
        for p in box:
            p = tuple(p)
            cv2.circle(blank, p, 10, (255, 0, 0), -1)
        # cv2.drawContours(blank, [contours], -1, (255, 255, 255), -1)
        cv2.imshow('green', blank)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break


def split(path):
    video = cv2.VideoCapture(path)
    time = 0
    NAME = os.path.basename(path).split('.mp4')[0]
    print(NAME)
    while True:
        ret, frame = video.read()
        if not ret:
            print('Error')
            break

        if time % 10 ==0:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'pics/{NAME}_{time}.jpg', frame)
        time +=1



if __name__ == '__main__':
    run_classic()
    # mp4 = glob.glob('*.mp4')
    # for i in mp4:
    #     split(i)