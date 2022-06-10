import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import socket
import numpy as np
import cv2
from tool import model_infer



def get_display():
    host = socket.gethostname()
    port = 1234
    s = socket.socket()
    s.connect((host, int(port)))
    print(os.path.basename(__file__) + ' bind')
    img = cv2.imread('test.jpg')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    area_reader  = model_infer(
        r'res34epoch=191-val_Iou=0.78.ckpt')
    # send_data = cv2.resize(img, (640, 480)).tobytes()
    while True:
            # self.mask_type是一个字符串, 用于给服务端判断是返回原图还是预测后的图片
            # print('Please type a str origin or pred')
            # mask_type = input()
            # if not (mask_type == 'origin' or mask_type == 'pred'):
            #     print('Input valid')
            #     continue

            send_data=area_reader.forward(img).astype(
                np.uint8)
            send_data = cv2.resize(send_data, (640, 480)).tobytes()

            arrBuf = bytearray(b'\xff\xaa\xff\xaa')
            # if send_data is None:
            #     # s.sendall(arrBuf)
            #     continue
            picBytes = send_data

            # 图片大小
            picSize = len(picBytes)

            # 数据体长度 = guid大小(固定) + 图片大小
            datalen = picSize
            data_type = b'cam1'
            # 组合数据包
            arrBuf += bytearray(datalen.to_bytes(4, byteorder='little'))
            arrBuf += data_type
            arrBuf += picBytes
            s.sendall(arrBuf)
            rec_data = s.recv(64)
            print(str(rec_data, encoding='utf-8'))




if __name__ == '__main__':
    get_display()