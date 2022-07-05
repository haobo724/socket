import os
import socket
import time
import cv2
import numpy as np

from Gui_base import host, port, CAMERA_PORT_TOP
from tool import model_infer


# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def get_display():

    camera_top = cv2.VideoCapture(CAMERA_PORT_TOP)
    camera_top.set(cv2.CAP_PROP_FPS, 30)

    #
    # pipeline = rs.pipeline()
    # config = rs.config()
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    # pipeline.start(config)

    # img = cv2.imread('test.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (640, 480))

    area_reader = model_infer(
        r'u-resnet34_448_256-epoch=53-val_Iou=0.88.ckpt')
    frame_number = 0
    s = socket.socket()
    s.connect((host, int(port)))
    print(os.path.basename(__file__) + ' bind')
    while True:
        # frames = pipeline.wait_for_frames()
        # img = frames.get_color_frame()
        # img = np.asanyarray(img.get_data())
        ret, img = camera_top.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        t = time.time()
        pred = area_reader.forward(img)
        # pred = Red_seg(img).astype(np.uint8)

        pred = cv2.resize(pred, (640, 480)).astype(np.uint8)
        send_data = np.concatenate((img, pred), axis=0).astype(np.uint8)

        # cv2.imshow('red',send_data)
        # cv2.waitKey(1)
        send_data = send_data.tobytes()

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
        try:
            rec_data = s.recv(64)
            # print(str(rec_data, encoding='utf-8'))
            # print('c1:', frame_number)
            frame_number += 1

        except ConnectionAbortedError:
            break
        # print(time.time()-t)
    s.close()


if __name__ == '__main__':
    get_display()
