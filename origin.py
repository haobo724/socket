import os
import socket

import cv2
import numpy as np
from PIL import Image  # 这是一个示例 Python 脚本。

# if os.path.exists(template_dir):
#     for i in range(10):
#         img_file_path = os.path.join(template_dir, str(i) + '.jpg')
#         if not os.access(img_file_path, os.F_OK):
#             img_file_path = os.path.join(template_dir, 'result_' + str(i) + '.tiff')
#
#         t = cv2.imread(img_file_path, 0)
#         t = cv2.resize(t, TEMPLATE_size)
#         img_template.append(t)
#         print(f'[INFO] TEMPLATE {i} lOADED')
# else:
#     raise FileNotFoundError('NO TEMPLATE')

@timer
def OCR(imfrag):
    '''
    :return FORCE , HEIGHT
    '''
    _, imfrag_h = imfrag.shape

    ret2, imfrag = cv2.threshold(imfrag, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    roi_size = TEMPLATE_size
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
        print('no co')
        return -1, -1
    # sort using x direction
    idx = np.argsort(xloc)
    tmp = digitCnts.copy()
    digitCnts = []
    for i in idx:
        digitCnts.append(tmp[i])

    digit = ''
    if len(digitCnts) > 3 or len(digitCnts) < 1:
        print('detect error,Suggested click restart btn')
        return -1, -1
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
    if result > UPPER_NUMMER:
        return -1, -1
    else:
        pass
        # print(result)
    return result, result



def get_match_score(img, template):
    # print(np.max(template))
    tp = (img == 255) == (template == 255)
    tn = (img == 0) == (template == 0)
    fp = (img == 255) == (template == 0)
    fn = (img == 0) == (template == 255)

    # score =( np.sum(tp) + np.sum(tn)) / (np.sum(tp) + np.sum(tn) + np.sum(fp) + np.sum(fn))
    score = (np.sum(tp) + np.sum(tn) - np.sum(fp) - np.sum(fn))
    return score
def get_display():
    host = socket.gethostname()
    port = 1234
    s = socket.socket()
    s.connect((host, int(port)))
    print(os.path.basename(__file__) + 'bind')

    while True:
        # self.mask_type是一个字符串, 用于给服务端判断是返回原图还是预测后的图片
        print('Please type a str origin or pred')
        mask_type = input()
        if not (mask_type == 'origin' or mask_type == 'pred'):
            print('Input valid')
            continue
        s.send(bytes(mask_type, encoding='utf-8'))
        str = s.recv(8)
        # 头部检验
        data = bytearray(str)

        headIndex = data.find(b'\xff\xaa\xff\xaa')
        if headIndex == 0:
            allLen = int.from_bytes(data[headIndex + 4:headIndex + 8], byteorder='little')
            curSize = 0
            allData = b''
            # 通过循环获取完整图片数据
            while curSize < allLen:
                data = s.recv(1024)
                allData += data
                curSize += len(data)
            # 取出图片数据
            imgData = allData[0:]
            # print('len(imgData):',len(imgData))
            if len(imgData) != (640 * 480 * 3):
                print('no return')
                continue
            # bytes转PIL.Image
            img = Image.frombuffer('RGB', (640, 480), imgData)
            # # 传过来的图片被上下镜面了，将其返回来
            # img = img.transpose(Image.FLIP_TOP_BOTTOM)
            # PIL.Image转ndarray
            img_conv = np.array(img)
            cv2.imshow('pic', img_conv)
            cv2.waitKey()
        else:
            print('??')


if __name__ == '__main__':
    get_display()
    # print(bytearray('21321'))
