import socket
import numpy as np
import cv2
import os
from Gui_base import host ,port
template_dir = 'OCR_template'
img_template = []
if os.path.exists(template_dir):
    for i in range(10):
        img_file_path = os.path.join(template_dir, str(i) + '.jpg')
        if not os.access(img_file_path,os.F_OK):
            img_file_path = os.path.join(template_dir, 'result_'+str(i) + '.tiff')

        t = cv2.imread(img_file_path, 0)
        t = cv2.resize(t, (50, 90))
        img_template.append(t)
        print(f'[INFO] TEMPLATE {i} lOADED')
else:
    raise FileNotFoundError('NO TEMPLATE')
def get_match_score(img, template):
    # print(np.max(template))
    tp = (img == 255) == (template == 255)
    tn = (img == 0) == (template == 0)
    fp = (img == 255) == (template == 0)
    fn = (img == 0) == (template == 255)

    # score =( np.sum(tp) + np.sum(tn)) / (np.sum(tp) + np.sum(tn) + np.sum(fp) + np.sum(fn))
    score =( np.sum(tp) + np.sum(tn)- np.sum(fp) - np.sum(fn))
    return score

def OCR(imfrag):
    # new method of reading digits in the imfrag
    _, imfrag_h = imfrag.shape
    ret2, imfrag = cv2.threshold(imfrag, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    roi_size = (50, 90)
    # detect single digit and detect
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义矩形结构元素

    # convert gray value for contour detection
    cnts, _ = cv2.findContours(imfrag.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digitCnts = []
    xloc = np.array([])
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        # print(h)
        # if height is more than 50, then digit is detected
        if h > 30 / 85 * imfrag_h:
            digitCnts.append(c)
            xloc = np.append(xloc, x)

    # if no connected component is detected, return ''
    if digitCnts == []:
        return -1,-1
    # sort using x direction
    idx = np.argsort(xloc)
    tmp = digitCnts.copy()
    digitCnts = []
    for i in idx:
        digitCnts.append(tmp[i])

    digit = ''
    if len(digitCnts) > 3 or len(digitCnts) < 1:
        print('detect error,Suggested click restart btn')
        return -1,-1
    # print(len(digitCnts))
    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = imfrag[y:y + h, x:x + w]

        if roi is not None:
            roi = cv2.resize(roi, roi_size)
            # cv2.imshow('roi',roi)
            # cv2.waitKey()
            roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)

            acc = np.zeros(10)
            for i in range(10):
                acc[i] = get_match_score(roi, img_template[i])
            if np.max(acc) < 0.8:
                print(acc)
                digit += '-1'
            else:
                digit += str(np.argmax(acc))
        else:
            digit = 0
    try:
        result = int(digit)
    except ValueError:
        result = -1
    if result > 200:
        return -1,-1
    else:
        pass
        # print(result)
    return result , 10

def get_display():

    s = socket.socket()
    s.connect((host, int(port)))
    print(os.path.basename(__file__) + ' bind')
    img = cv2.imread('test2.jpg')
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    frame_number = 0

    while True:

        height , force = OCR(img_gray)

        send_data = cv2.resize(img, (640, 480)).tobytes()

        height_b = bytes(height.to_bytes(4, byteorder='little'))
        force_b = bytes(force.to_bytes(4, byteorder='little',signed=True))
        head = height_b+force_b
        # print(len(head))
        arrBuf = bytearray(b'\xff\xaa\xff\xaa')
        # if send_data is None:
        #     # s.sendall(arrBuf)
        #     continue
        picBytes = send_data

        # 图片大小
        picSize = len(picBytes)

        data_type = b'cam2'
        # 组合数据包
        arrBuf += bytearray(picSize.to_bytes(4, byteorder='little'))
        arrBuf += data_type
        arrBuf += head

        arrBuf += picBytes
        s.sendall(arrBuf)
        rec_data = s.recv(64)
        print(str(rec_data, encoding='utf-8'))
        print('c2:',frame_number)
        frame_number += 1



if __name__ == '__main__':
    get_display()