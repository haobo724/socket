import argparse
import os
import pickle
from datetime import datetime
import numpy as np
import serial

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import socket

from Gui_frame import host, port

def get_tof(args):
    big_list =[]
    print("###starting thread###")
    # s = socket.socket()
    # s.connect((host, int(port)))
    print(os.path.basename(__file__) + ' bind')
    with serial.Serial(
            port=args.port,
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
            data_list.append(list1)
            # print(list1)
            if count % 15 ==0 and count!=0:
                big_list.append(data_list)
                data_list=[]
            #     # print(distarray)
            #     # print(distarray)
            #     a = distarray.tobytes()
            #     # tof_signal = np.frombuffer(a)
            #     # b = np.reshape(tof_signal,((4, 4, 8))).copy()
            #     # b[0][0][0]=1
            #     # print(distarray)
            #     # print(b)
            #     # assert distarray.all()==b.all()
            #     # input()
            #
            #     picSize = len(a)
            #     print(picSize)
            #     arrBuf = bytearray(b'\xff\xaa\xff\xaa')
            #     data_type = b'tof1'
            #     # ???????????????
            #     arrBuf += bytearray(picSize.to_bytes(4, byteorder='little'))
            #
            #     arrBuf += data_type
            #
            #     arrBuf += a
            #     s.sendall(arrBuf)
            #     try:
            #         rec_data = s.recv(64)
            #         print(str(rec_data, encoding='utf-8'))
            #
            #
            #     except ConnectionResetError or ConnectionAbortedError:
            #         break

            count+=1
        # with open(f"{str(count)}.p", 'wb') as f:
        #     pickle.dump(data_list, f)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, help='', default=r'COM3')

    args = parser.parse_args()
    print(args)
    get_tof(args)