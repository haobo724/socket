import socket

import cv2
import os

def get_display():
    host = socket.gethostname()
    port = 1234
    s = socket.socket()
    s.connect((host, int(port)))
    print(os.path.basename(__file__) + ' bind')
    img = cv2.imread('test2.jpg')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    send_data = cv2.resize(img, (640, 480)).tobytes()
    while True:
            # self.mask_type是一个字符串, 用于给服务端判断是返回原图还是预测后的图片
            # print('Please type a str origin or pred')
            # mask_type = input()
            # if not (mask_type == 'origin' or mask_type == 'pred'):
            #     print('Input valid')
            #     continue


            arrBuf = bytearray(b'\xff\xaa\xff\xaa')
            # if send_data is None:
            #     # s.sendall(arrBuf)
            #     continue
            picBytes = send_data

            # 图片大小
            picSize = len(picBytes)

            # 数据体长度 = guid大小(固定) + 图片大小
            datalen = picSize
            data_type = b'cam2'
            # 组合数据包
            arrBuf += bytearray(datalen.to_bytes(4, byteorder='little'))
            arrBuf += data_type
            arrBuf += picBytes
            s.sendall(arrBuf)
            rec_data = s.recv(64)
            print(str(rec_data, encoding='utf-8'))



if __name__ == '__main__':
    get_display()