import os
import socket

import cv2
import numpy as np
from PIL import Image  # 这是一个示例 Python 脚本。


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
                    if len(imgData) != (640*480*3):
                        print('no return')
                        continue
                    # bytes转PIL.Image
                    img = Image.frombuffer('RGB', (640, 480), imgData)
                    # # 传过来的图片被上下镜面了，将其返回来
                    # img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    # PIL.Image转ndarray
                    img_conv = np.array(img)
                    cv2.imshow('pic',img_conv)
                    cv2.waitKey()
            else:
                print('??')

if __name__ == '__main__':
    get_display()
    # print(bytearray('21321'))