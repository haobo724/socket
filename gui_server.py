import asyncio
import os
import queue
import socket
import threading
import time
import tkinter as tk
from multiprocessing import Process, Queue, Event,Manager

import cv2
import numpy as np
from PIL import Image, ImageTk

global pred_frame_bytes
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

eps = 0.0001

WINDOW_HEIGHT = 960
WINDOW_WIDTH = 1280

BTN_SIZE = [200, 100]
FONT_SIZE = 18
FONT_COLOR = 'black'

def timer(func):
    def warp(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(time.time()-start)
        return result
    return warp


class Gui_base:
    def __init__(self, queue_list, StopEVENT):
        # self, breast_camera, param_camera, param_reader, area_reader, area_calculator, args
        # initialize cameras and detection algorithms
        # self.breast_camera = breast_camera
        # self.param_camera = param_camera
        # self.param_reader = param_reader
        # self.area_reader = area_reader
        # self.area_calculator = area_calculator
        self.StopEVENT = StopEVENT

        # containers for images and detected parameters
        self.breast_img = None
        self.param_img = None
        self.param_thresh = None
        self.paddle_height = 0
        self.compression_force = 0
        self.breast_pred = None
        self.breast_area = 0
        self.pressure = 0
        self.frame = None
        self.paddel = 'G'
        # flags to control the workflow
        self.hide_flag = False  # set True when we need to hide the breast image
        # self.show_time = args.showtime
        # black or Noimg
        # self.no_img = cv2.cvtColor(cv2.imread('Noimg.png'), cv2.COLOR_BGR2RGB)

        self.whole_img_flag = 0
        # configuration of layout
        self.root = tk.Tk()
        self.root.title('Pressure measurement')
        self.root.protocol("WM_DELETE_WINDOW", self.onClose)

        # image display
        # self.display_panel = tk.Canvas(self.root, height=WINDOW_HEIGHT, width=WINDOW_WIDTH)
        self.display_panel = tk.Label(self.root, height=WINDOW_HEIGHT, width=WINDOW_WIDTH)
        self.display_panel['bg'] = 'darkgreen'
        self.display_panel.grid(row=0, column=0, sticky=tk.N + tk.W + tk.E + tk.S)

        # original displayed image
        self.no_img = cv2.cvtColor(cv2.imread('test.jpg'), cv2.COLOR_BGR2RGB)
        ori = ImageTk.PhotoImage(Image.fromarray(np.array(self.no_img)))
        self.display_panel.configure(image=ori)

        # parameter display
        # dummy image for size correlation
        self.dummy_img = tk.PhotoImage(width=1, height=1)

        self.param_panel = tk.Frame(self.root)
        self.height_label = tk.Label(self.param_panel,
                                     image=self.dummy_img,
                                     width=BTN_SIZE[0] + 2,
                                     compound="center",
                                     height=int(BTN_SIZE[1] / 2),
                                     text='Paddle Height:',
                                     font=('', FONT_SIZE),
                                     fg=FONT_COLOR)
        self.height_value = tk.Label(self.param_panel,
                                     image=self.dummy_img,
                                     width=BTN_SIZE[0] + 2,
                                     compound="center",
                                     height=int(BTN_SIZE[1] / 2),
                                     text='0',
                                     font=('', FONT_SIZE),
                                     fg=FONT_COLOR)
        self.compression_label = tk.Label(self.param_panel,
                                          image=self.dummy_img,
                                          width=BTN_SIZE[0] + 2,
                                          compound="center",
                                          height=int(BTN_SIZE[1] / 2),
                                          text='Compression:',
                                          font=('', FONT_SIZE),
                                          fg=FONT_COLOR)
        self.compression_value = tk.Label(self.param_panel,
                                          image=self.dummy_img,
                                          width=BTN_SIZE[0] + 2,
                                          compound="center",
                                          height=int(BTN_SIZE[1] / 2),
                                          text='0',
                                          font=('', FONT_SIZE),
                                          fg=FONT_COLOR)
        self.area_label = tk.Label(self.param_panel,
                                   image=self.dummy_img,
                                   width=BTN_SIZE[0] + 2,
                                   compound="center",
                                   height=int(BTN_SIZE[1] / 2),
                                   text='Area:',
                                   font=('', FONT_SIZE),
                                   fg=FONT_COLOR)
        self.area_value = tk.Label(self.param_panel,
                                   image=self.dummy_img,
                                   width=BTN_SIZE[0] + 2,
                                   compound="center",
                                   height=int(BTN_SIZE[1] / 2),
                                   text='0',
                                   font=('', FONT_SIZE),
                                   fg=FONT_COLOR)
        self.Pressure_label = tk.Label(self.param_panel,
                                       image=self.dummy_img,
                                       width=BTN_SIZE[0] + 2,
                                       compound="center",
                                       height=int(BTN_SIZE[1] / 2),
                                       text='Pressure',
                                       font=('', FONT_SIZE),
                                       fg=FONT_COLOR)
        self.Pressure_value = tk.Label(self.param_panel,
                                       image=self.dummy_img,
                                       width=BTN_SIZE[0] + 2,
                                       compound="center",
                                       height=int(BTN_SIZE[1] / 2),
                                       text='0',
                                       font=('', FONT_SIZE),
                                       fg=FONT_COLOR)

        self.Name_label = tk.Label(self.param_panel,
                                   image=self.dummy_img,
                                   width=BTN_SIZE[0] + 2,
                                   compound="center",
                                   height=int(BTN_SIZE[1] / 2),
                                   text='Name',
                                   font=('', FONT_SIZE),
                                   fg=FONT_COLOR)
        self.Name_value = tk.Label(self.param_panel,
                                   image=self.dummy_img,
                                   width=BTN_SIZE[0] + 2,
                                   compound="center",
                                   height=int(BTN_SIZE[1] / 2),
                                   text='None',
                                   font=('', FONT_SIZE),
                                   fg=FONT_COLOR)

        self.hide_btn = tk.Button(self.param_panel,
                                  image=self.dummy_img,
                                  width=BTN_SIZE[0],
                                  height=BTN_SIZE[1] / 2,
                                  compound="center",
                                  text='Hide Breast',
                                  command=self.onHide,
                                  bg='SystemButtonFace')

        self.Reset_btn = tk.Button(self.param_panel,
                                   image=self.dummy_img,
                                   width=BTN_SIZE[0],
                                   height=int(BTN_SIZE[1] / 2),
                                   compound="center",
                                   text='Reset',
                                   command=self.reset,
                                   bg='SystemButtonFace')

        self.Breast_btn = tk.Button(self.param_panel,
                                    image=self.dummy_img,
                                    width=int(BTN_SIZE[0]),
                                    height=int(BTN_SIZE[1] / 2),
                                    compound="center",
                                    text='show whole',
                                    command=self.show_whole_breast_img,
                                    bg='SystemButtonFace')

        self.exit_btn = tk.Button(self.param_panel,
                                  image=self.dummy_img,
                                  width=BTN_SIZE[0],
                                  height=BTN_SIZE[1] / 2,
                                  compound="center",
                                  text='Exit',
                                  command=self.onClose,
                                  font=('', FONT_SIZE))
        self.param_panel.grid(row=0, column=1, padx=5, pady=5, sticky=tk.N)
        self.height_label.grid(row=0)
        self.height_value.grid(row=1)
        self.compression_label.grid(row=2)
        self.compression_value.grid(row=3)
        self.area_label.grid(row=4)
        self.area_value.grid(row=5)
        self.Pressure_label.grid(row=6)
        self.Pressure_value.grid(row=7)
        self.Name_label.grid(row=8)
        self.Name_value.grid(row=9)

        self.hide_btn.grid(row=10)
        self.Breast_btn.grid(row=12)
        self.Reset_btn.grid(row=11)
        self.exit_btn.grid(row=13)
        self.queue_list = queue_list
        # self.loop = asyncio.new_event_loop()
        # self.t_update = threading.Thread(target=self.get_loop, args=(self.loop,))
        # self.t_update.start()
        # t = self.update_display()
        # asyncio.run_coroutine_threadsafe(t, self.loop)
        # self.root.update()
        while not self.StopEVENT.is_set():
            self.update_display()
            self.root.update()
            self.root.after(10)
        # self.root.mainloop()

    def get_loop(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def reset(self):
        pass

    def show_whole_breast_img(self):
        pass

    def get_param_image(self):
        pass

    def get_breast_image(self):
        pass

    def recognize_param_image(self):
        pass

    def measure_breast_area(self):
        pass
    # @timer
    def update_display(self):
        # while not StopEVENT.is_set():

        try:

            test_img = self.queue_list[0].get_nowait()
            test_img2 = self.queue_list[1].get_nowait()
            # frame0_0 = cv2.resize(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB),
            #                       (int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 2)))
            # frame1_0 = cv2.resize(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB),
            #                       (int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 2)))
            # # frame0_1 = cv2.resize(cv2.cvtColor(self.breast_pred, cv2.COLOR_GRAY2RGB),
            # frame0_1 = cv2.resize(test_img,
            #                       (int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 2)))
            # # frame1_1 = cv2.resize(cv2.imread('thresh.jpg'), (int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 2)))
            # frame1_1 = cv2.resize(test_img, (int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 2)))

            self.frame = np.concatenate((
                np.concatenate((test_img2, test_img), axis=0),
                np.concatenate((test_img, test_img2), axis=0)
            ), axis=1)
            self.frame = ImageTk.PhotoImage(Image.fromarray(self.frame))
            # self.display_panel.itemconfigure(self.display_panel_img, image=self.frame)
            # self.display_panel_img = self.frame
            self.display_panel.configure(image=self.frame)
        except queue.Empty:
            print('000000:',self.queue_list[0].qsize())
            print('111111:',self.queue_list[1].qsize())


    def onHide(self):
        pass

    def onClose(self):
        self.StopEVENT.set()
        self.root.quit()


def get_data(c, addr, queue_list, StopEVENT,s):
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
        # headIndex = data.find(b'\xff\xaa\xff\xaa')
        # if headIndex == 0:

        allLen = int.from_bytes(data[4: 8], byteorder='little')
        data_type = c.recv(4)
        # print('data_type',data_type.decode())
        curSize = 0
        allData = b''
        # 通过循环获取完整图片数据
        while curSize < allLen:
            data = c.recv(2048)

            # try:
            # except BlockingIOError:
            #     print('不完整？')
            #     continue
            allData += data
            curSize += len(data)
        # 取出图片数据
        imgData = allData[0:]
        if len(imgData) != (640 * 480 * 3):
            print('no return')
            continue
        # bytes转PIL.Image
        img = Image.frombuffer('RGB', (640, 480), imgData)
        # 传过来的图片被上下镜面了，将其返回来
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # PIL.Image转ndarray
        img_conv = np.array(img)

        if data_type == b'cam1':
            if not queue_list[0].full():
                queue_list[0].put(img_conv)
                save_flag =True
        elif data_type == b'cam2':
            if not queue_list[1].full():
                queue_list[1].put(img_conv)
                save_flag =True

        elif data_type == b'tof1':
            queue_list[2].put(img_conv)
        elif data_type == b'tof2':
            queue_list[3].put(img_conv)
            # cv2.imshow('pic', img_conv)
        if save_flag:
            data = bytes('True', encoding='utf-8')
        else:
            data = bytes('False', encoding='utf-8')
        c.sendall(data)

    c.close()
    print('disconnect:', addr)
    return


# except Exception as e:
#     print("远程主机强制关闭连接")
#     print(e.args)
# s.close()


def server_test(value):
    if value == "origin":
        return None
    else:
        img = cv2.imread('test.jpg')
        pred_frame_bytes = cv2.resize(img, (640, 480)).tobytes()

        return pred_frame_bytes


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # pool = multiprocessing.Pool(processes=5)
    StopEVENT = Event()
    s = socket.socket()  # 创建 socket 对象
    host = socket.gethostname()  # 获取本地主机名

    s.setblocking(False)
    port = 1234  # 设置端口
    s.bind((host, port))  # 绑定端口
    print("Server ON")
    s.listen(5)

    cam1_que = Queue(maxsize=1)
    cam2_que = Queue(maxsize=1)
    tof1_que = Queue(maxsize=100)
    tof2_que = Queue(maxsize=100)

    queue_list = [cam1_que, cam2_que, tof1_que, tof2_que]
    gui_process = Process(target=Gui_base, args=(queue_list, StopEVENT))
    gui_process.start()
    process_pool = []
    timer = 0
    while True:
        try:
            clientSock, addr = s.accept()
            # clientSock.setblocking(False)
        except BlockingIOError :
            # print('time out')
            continue
        p = Process(target=get_data, args=(clientSock, addr, queue_list, StopEVENT,s,))
        process_pool.append(p)
        p.start()
        timer+=1
        if timer==2:
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
