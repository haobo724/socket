import os
from datetime import datetime
from gui_server import timer
import numpy as np
import serial

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import socket
import time

from Gui_base import host, port

def get_tof(serialport='COM3', data_list=[]):
    # s = socket.socket()
    # s.connect((host, int(port)))
    print(os.path.basename(__file__) + ' bind')
    print("###starting thread###")
    with serial.Serial(
            port=serialport,
            baudrate=921600,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=None) as ser:

        # data storage
        distarray = np.zeros((4, 4, 8))  # 4 sensors, 4 x 8 pixelmatrix
        points = np.zeros((4, 4, 8, 3))  # 4 sensors, 4 x 8 pixelmatrix, 3 coordinates (x, y, z)

        # debug_pose = set_debug_pose()
        count  =0
        while (1):  # <-- insert read flag here
            dataraw = bytearray(ser.read_until(b'\xff\xfa\xff\xfa'))
            data = dataraw[-44:]
            identifier = data[44 - 7]

            # try:
            #     identifier = data[44 - 7]
            # except:
            #     continue

            # print('Sensor ID : ',identifier)
            # status = int.from_bytes(data[44 - 12:44 - 9], 'little')
            # print('Sensorstatus: ', status)

            if (data[44 - 8] == 1):
                for i in range(8):
                    distarray[identifier, 0, i] = int.from_bytes(data[i * 4:i * 4 + 3], 'little')
                # print(np.array_split(data[:32],8))
                # distarray2[identifier, 0, 0:7] = int.from_bytes(np.array_split(data[:32],8), 'little')
                # print(distarray)
                # print(distarray2)
                # input()
                    # print("Reihe 1")
            elif (data[44 - 8] == 2):
                for i in range(8):
                    distarray[identifier, 1, i] = int.from_bytes(data[i * 4:i * 4 + 3], 'little')
                # print("Reihe 2")
            elif (data[44 - 8] == 3):
                for i in range(8):
                    distarray[identifier, 2, i] = int.from_bytes(data[i * 4:i * 4 + 3], 'little')
                # print("Reihe 3")
            elif (data[44 - 8] == 4):
                for i in range(8):
                    distarray[identifier, 3, i] = int.from_bytes(data[i * 4:i * 4 + 3], 'little')
                    # print("Reihe 4")
            list1 = (['timestamp', datetime.now(), 'identifier', identifier, distarray[identifier, :, :]])
            if count % 15 ==0 and count!=0:
                # print(distarray)
                # print(distarray)
                a = distarray.tobytes()
                # tof_signal = np.frombuffer(a)
                # b = np.reshape(tof_signal,((4, 4, 8))).copy()
                # b[0][0][0]=1
                # print(distarray)
                # print(b)
                # assert distarray.all()==b.all()
                # input()

                picSize = len(a)
                arrBuf=''
                data_type = b'tof1'
                # 组合数据包
                arrBuf += bytearray(picSize.to_bytes(4, byteorder='little'))
                arrBuf += data_type

                arrBuf += a
                # s.sendall(arrBuf)
                # try:
                #     rec_data = s.recv(64)
                #     print(str(rec_data, encoding='utf-8'))
                #

                # except ConnectionResetError or ConnectionAbortedError:
                #     break
                data_list.append(list1)

            count+=1
        with open("save_list2.p", 'wb') as f:
            f.dump(data_list)
if __name__ == '__main__':
    a = np.arange(0,32)
    print(a)
    print(np.array_split(a,8))
    get_tof()