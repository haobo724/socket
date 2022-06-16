import time
import numpy as np


def recv_client_data(c, addr, queue_list, StopEVENT):
    print('connect:', addr)
    while not StopEVENT.is_set():
        time_start = time.time()

        str = c.recv(8)
        save_flag = False

        data = bytearray(str)
        headIndex = data.find(b'\xff\xaa\xff\xaa')
        if headIndex == 0:

            allLen = int.from_bytes(data[4: 8], byteorder='little')
            data_type = c.recv(4)
            if data_type == b'cam1':
                img_conv = get_img(c, allLen)
                if not queue_list[0].full():
                    queue_list[0].put(img_conv)
                    save_flag = True
            elif data_type == b'cam2':
                height_force = c.recv(8)
                height = int.from_bytes(height_force[0:4], byteorder='little', signed=True)
                force = int.from_bytes(height_force[4:8], byteorder='little', signed=True)
                info = (force, height)
                img_conv = get_img(c, allLen)
                if not queue_list[1].full():
                    queue_list[1].put(img_conv)
                    queue_list[4].put(info)
                    save_flag = True

            elif data_type == b'tof1':
                tof_data = get_tof_data(c, allLen)
                if not queue_list[2].full():
                    queue_list[2].put(tof_data)
                    save_flag = True

            elif data_type == b'tof2':
                tof_data = get_tof_data(c, allLen)
                if not queue_list[3].full():
                    queue_list[3].put(tof_data)
                    save_flag = True

            if save_flag:
                queue_list[5].acquire()
                data = bytes('True', encoding='utf-8')
            else:
                print(data_type, 'wrong data')
                data = bytes('False', encoding='utf-8')
            c.sendall(data)
            # print(data_type,time.time() - time_start)

    c.close()
    print('disconnect:', addr)
    return

def get_tof_data(c, allLen):
    curSize = 0
    allData = b''
    # 通过循环获取完整图片数据
    while curSize < allLen:
        data = c.recv(8192 * 4)

        allData += data
        curSize += len(data)
    # 取出图片数据
    imgData = allData[0:]
    # 640 * 480 * 3*2
    if len(imgData) != 1843200:
        print('no return', len(imgData))
        return None
    tof_signal = np.frombuffer(imgData, dtype=np.uint8)

    return tof_signal



def get_img(c, allLen):
    curSize = 0
    allData = b''
    # 通过循环获取完整图片数据
    while curSize < allLen:
        data = c.recv(8192 * 4)

        allData += data
        curSize += len(data)
    # 取出图片数据
    imgData = allData[0:]
    # 640 * 480 * 3*2
    if len(imgData) != 1843200:
        print('no return', len(imgData))
        return None
    img = np.frombuffer(imgData, dtype=np.uint8)
    img = np.reshape(img, (960, 640, 3), dtype=np.uint8)

    return img
