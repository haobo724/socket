import pickle
import socket
import time
from multiprocessing import Process, Queue, Event, Manager
import torch
import cv2
import numpy as np
from PIL import Image, ImageTk

from Gui_base import Gui_base
from Gui_base import host, port, CLIENT_NR
from tool import Buffer

print('torch gpu:',torch.cuda.is_available())

def timer(func):
    def warp(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(time.time() - start)
        return result

    return warp


class Gui(Gui_base):
    def __init__(self, queue_list, StopEVENT):
        super().__init__(queue_list, StopEVENT)
        # self.loop = asyncio.new_event_loop()
        # self.t_update = threading.Thread(target=self.get_loop, args=(self.loop,))
        # self.t_update.start()
        # t = self.update_display()
        # asyncio.run_coroutine_threadsafe(t, self.loop)
        # self.root.update()
        self.Recoding_flag = False
        self.timer = time.time()
        self.force_buffer = Buffer(20)
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        codec2 = cv2.VideoWriter_fourcc(*'mp4v')
        self.patient_idx = 0
        self.recoding_stage = True
        self.out_top_path = f'patient{self.patient_idx}_top.mp4'
        self.out_bot_path = f'patient{self.patient_idx}_bot.mp4'
        self.out_top = cv2.VideoWriter(self.out_top_path, codec, 25, (640, 480))
        self.out_bot = cv2.VideoWriter(self.out_bot_path, codec2, 25, (640, 480))
        self.tl = []
        self.tr = []
        self.bl = []
        self.br = []
        self.pts1 = None
        self.setup()
        while not self.StopEVENT.is_set():
            self.update_display()
            self.root.update()
            self.root.after(1)
        # self.root.mainloop()

    def setup(self):
        with open('001.pkl', 'rb') as look:
            self.pts1 = pickle.load(look)

        # with open('look_upT.pkl', 'rb') as look:
        #     self.tl, self.tr, self.bl, self.br = pickle.load(look)
        # print('[INFO] LOOK UP TABLE FINISHED')

    # @timer
    def update_display(self):
        # while not StopEVENT.is_set():

        if self.queue_list[1].empty():
            return
        bot_img = self.queue_list[1].get()
        info = self.queue_list[4].get()
        self.force_buffer.append(info[1])
        top_img = self.queue_list[0].get()
        self.queue_list[5].release()
        self.queue_list[5].release()
        img, pred = np.split(top_img, 2, axis=0)
        b1, b2 = np.split(bot_img, 2, axis=0)
        pts2 = np.float32([[0, 0], [480, 0], [0, 640], [480, 640]])

        M = cv2.getPerspectiveTransform(np.float32(self.pts1), pts2)

        img = cv2.warpPerspective(img, M, ( 640,480))
        self.height_value.configure(text="{:.1f} mm".format(info[0]))
        self.compression_value.configure(text="{:.1f} N".format(info[1]))
        self.area_value.configure(text="{:.3f} mm^2".format(-1))
        self.Pressure_value.configure(text="{:.3f} N/mm^2".format(99))        # frame0_0 = cv2.resize(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB),
        #                       (int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 2)))
        # frame1_0 = cv2.resize(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB),
        #                       (int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 2)))
        # # frame0_1 = cv2.resize(cv2.cvtColor(self.breast_pred, cv2.COLOR_GRAY2RGB),
        # frame0_1 = cv2.resize(test_img,
        #                       (int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 2)))
        # # frame1_1 = cv2.resize(cv2.imread('thresh.jpg'), (int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 2)))
        # frame1_1 = cv2.resize(test_img, (int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 2)))
        self.frame = np.concatenate((
            np.concatenate((img, pred), axis=0),
            # top_img,
            bot_img
            # np.concatenate((bot_img, bot_img), axis=0)
        ), axis=1)
        self.frame = ImageTk.PhotoImage(Image.fromarray(self.frame))

        self.display_panel.configure(image=self.frame)
        most = int(self.force_buffer.most())

        if most > 5:
            if self.Recoding_flag:
                self.recoding_stage = True
                self.Recoding_btn.configure(text='Recoding ON', bg='red')
                self.Recoding_btn.update()
                self.out_top.write(img)
                self.out_bot.write(b1)
        else:
            if self.Recoding_flag:
                # want recording but Force is smaller than 5, go pause
                if self.recoding_stage :
                    self.out_top.release()
                    self.out_bot.release()
                    self.Recoding_btn.configure(text='Recoding PAUSE', bg='SystemButtonFace')
                    self.Recoding_btn.update()
                    self.new_writer()
                    self.recoding_stage = False
                else:
                    pass
        print('update lag =', time.time() - self.timer)
        self.timer = time.time()

        # print('top:', self.queue_list[0].qsize())
        # print('bot:', self.queue_list[1].qsize())

    def recoding(self):
        self.Recoding_flag = not self.Recoding_flag

        if self.Recoding_flag:
            self.Recoding_btn.configure(text='Recoding ON', bg='red')
            self.recoding_stage = False
        else:

            if self.out_top.isOpened():
                self.out_top.release()
                self.out_bot.release()
                self.new_writer()
            self.Recoding_btn.configure(text='Recoding OFF', bg='SystemButtonFace')

        self.Recoding_btn.update()

    def onClose(self):
        self.StopEVENT.set()
        if self.out_bot.isOpened():
            self.out_top.release()
            self.out_bot.release()
        self.root.quit()

    def new_writer(self):
        self.patient_idx += 1
        self.out_top_path = f'patient{self.patient_idx}_top.mp4'
        self.out_bot_path = f'patient{self.patient_idx}_bot.mp4'
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        codec2 = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_top = cv2.VideoWriter(self.out_top_path, codec, 25, (640, 480))
        self.out_bot = cv2.VideoWriter(self.out_bot_path, codec2, 25, (640, 480))




def get_data(c, addr, queue_list, StopEVENT):
    print('connect:', addr)
    while not StopEVENT.is_set():
        time_start = time.time()

        str = c.recv(8)
        save_flag = False
        # try:
        # except BlockingIOError:
        #     print('不完整？')
        #     continue
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
                img_conv = get_img(c, allLen)

                queue_list[2].put(img_conv)
            elif data_type == b'tof2':
                img_conv = get_img(c, allLen)

                queue_list[3].put(img_conv)
                # cv2.imshow('pic', img_conv)

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
    img = np.frombuffer(imgData,dtype=np.uint8)
    img = np.reshape(img,( 960,640,3),dtype=np.uint8)


    return img


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # pool = multiprocessing.Pool(processes=5)
    StopEVENT = Event()
    s = socket.socket()  # 创建 socket 对象
    s.setblocking(False)
    s.bind((host, port))  # 绑定端口
    print("Server ON")
    s.listen(5)

    cam1_que = Queue(maxsize=1)
    cam2_que = Queue(maxsize=1)
    cam2_info_que = Queue(maxsize=1)
    syn_que = Manager().Semaphore(0)
    tof1_que = Queue(maxsize=100)
    tof2_que = Queue(maxsize=100)
    queue_list = [cam1_que, cam2_que, tof1_que, tof2_que, cam2_info_que, syn_que]
    gui_process = Process(target=Gui, args=(queue_list, StopEVENT,))
    gui_process.start()
    process_pool = []
    client_timer = 0
    while True:
        try:
            clientSock, addr = s.accept()

            # clientSock.setblocking(False)
        except BlockingIOError:
            # print('time out')
            continue
        p = Process(target=get_data, args=(clientSock, addr, queue_list, StopEVENT,))
        process_pool.append(p)
        p.start()
        client_timer += 1
        if client_timer == CLIENT_NR:
            print('stop')
            break
    while not StopEVENT.is_set():
        pass
    s.close()
    gui_process.join()
    gui_process.close()

    for p in process_pool:
        p.kill()
    print('done')

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
