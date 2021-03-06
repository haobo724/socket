import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
import socket
import time
from paddleocr import PaddleOCR
import cv2
import numpy as np
from Gui_frame import CAMERA_PORT_BOT
from Gui_frame import host, port
pkl_save_path = 'pkl'
if not os.path.exists(pkl_save_path):
    os.mkdir(pkl_save_path)
template_dir = 'OCR_template'
img_template = []
TEMPLATE_size = (50, 90)
UPPER_NUMMER =200
ocr = PaddleOCR(use_angle_cls=False, lang='en',use_gpu=True,gpu_mem=200, det=False, rec_batch_num=5)
camera_bot = cv2.VideoCapture(CAMERA_PORT_BOT)
camera_bot.set(cv2.CAP_PROP_BRIGHTNESS,100)
camera_bot.set(cv2.CAP_PROP_EXPOSURE,-7)
camera_bot.set(cv2.CAP_PROP_FPS,30)
print(camera_bot.get(cv2.CAP_PROP_EXPOSURE))
print(camera_bot.get(cv2.CAP_PROP_BRIGHTNESS))


def OCR_THIRD(img):
    result = ocr.ocr(img, cls=False,det=False)
    # for line in result:
    #     print(line)
    result = result[0]
    try:
        result = int(result[0])
        if result > UPPER_NUMMER:
            return -1
    except  IndexError :
        result = -1
    except ValueError :
        result = -1

    return result


def get_display():
    s = socket.socket()
    s.connect((host, int(port)))
    v = cv2.VideoCapture(CAMERA_PORT_BOT,cv2.CAP_DSHOW)

    frame_number = 0
    print(os.path.basename(__file__) + ' bind')


    with open(os.path.join(pkl_save_path, 'height.pkl'), 'rb') as file:
        x1, y1, w1, h1 = pickle.load(file)
    with open(os.path.join(pkl_save_path, 'force.pkl'), 'rb') as file:
        x2, y2, w2, h2 = pickle.load(file)
    with open(os.path.join(pkl_save_path, 'display.pkl'), 'rb') as file:
        x3, y3, w3, h3 = pickle.load(file)
    while True:
        t = time.time()
        ret, img = v.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        send_data  = cv2.resize(img[y3:y3+h3,x3:x3+w3,:], (640, 480))
        # send_data = cv2.warpPerspective(img, M, (640, 480))
        # img_gray = cv2.cvtColor(send_data, cv2.COLOR_RGB2GRAY)
        force_block = img[y2:y2+h2,x2:x2+w2,:]
        height_block = img[y1:y1+h1,x1:x1+w1,:]

        height = OCR_THIRD(height_block)
        force = OCR_THIRD(force_block)
        cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),color=(255,0,0),thickness=5)
        cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),color=(255,0,123),thickness=5)
        send_data = np.concatenate((send_data, img), axis=0).tobytes()
        height_b = bytes(height.to_bytes(4, byteorder='little', signed=True))
        force_b = bytes(force.to_bytes(4, byteorder='little', signed=True))
        head =  force_b + height_b
        # print(len(head))
        arrBuf = bytearray(b'\xff\xaa\xff\xaa')
        # if send_data is None:
        #     # s.sendall(arrBuf)
        #     continue
        picBytes = send_data

        # ????????????
        picSize = len(picBytes)

        data_type = b'cam2'
        # ???????????????
        arrBuf += bytearray(picSize.to_bytes(4, byteorder='little'))
        arrBuf += data_type
        arrBuf += head

        arrBuf += picBytes
        s.sendall(arrBuf)
        try:
            rec_data = s.recv(64)
            # print(str(rec_data, encoding='utf-8'))
            # print('\r c2:', frame_number, flush=True)
            frame_number += 1

        except ConnectionResetError or ConnectionAbortedError:
            break
        print(time.time() - t)
    s.close()


if __name__ == '__main__':
    get_display()
    # img = cv2.imread('bot.jpg',0)
    #
    #
    #
    # img=cv2.resize(img, (640, 480))
    # x,y,w,h = cv2.selectROI('roi', img)
    # img = img[y:y+h,x:x+w]
    #
    #
    # v = cv2.VideoCapture(CAMERA_PORT_BOT,cv2.CAP_DSHOW)
    # file_name = 'height.pkl'
    # with open(file_name, 'rb') as file:
    #     x1, y1, w1, h1 = pickle.load(file)
    # file_name = 'force.pkl'
    # with open(file_name, 'rb') as file:
    #     x2, y2, w2, h2 = pickle.load(file)
    # file_name = 'display.pkl'
    # with open(file_name, 'rb') as file:
    #     x3, y3, w3, h3 = pickle.load(file)
    # while True:
    #     t = time.time()
    #     ret, img = v.read()
    #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     send_data = cv2.resize(img[y3:y3 + h3, x3:x3 + w3, :], (640, 480))
    #     force_block = img[y2:y2 + h2, x2:x2 + w2, :]
    #     height_block = img[y1:y1 + h1, x1:x1 + w1, :]
    #     height = OCR_THIRD(height_block)
    #     force = OCR_THIRD(force_block)