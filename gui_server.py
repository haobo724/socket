import os
import pickle
import socket
import time
from multiprocessing import Process, Queue, Event, Manager

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

from Gui_base import Gui_base
from Gui_base import host, port, CLIENT_NR
from recv import recv_client_data
from tool import Buffer ,Red_seg

print('torch gpu:', torch.cuda.is_available())
video_save_path = 'video'
if not os.path.exists(video_save_path):
    os.mkdir(video_save_path)
pkl_save_path = 'pkl'
if not os.path.exists(pkl_save_path):
    os.mkdir(pkl_save_path)


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
        with open(os.path.join(pkl_save_path, 'M_list.pkl'), 'rb') as f:
            self.M_list = pickle.load(f)
        with open(os.path.join(pkl_save_path, 'Valid_interval.pkl'), 'rb') as f:
            self.high_lv, self.low_lv = pickle.load(f)

        with open(os.path.join(pkl_save_path, 'last_high_first_low.pkl'), 'rb') as f:
            self.last_high, self.first_low = pickle.load(f)

        self.Valid_interval = [0, 0]
        self.M = None
        self.Recoding_flag = False
        self.timer = time.time()
        self.force_buffer = Buffer(20)
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        codec2 = cv2.VideoWriter_fourcc(*'mp4v')
        self.patient_idx = 0
        self.recoding_stage = True
        self.out_top_path = os.path.join(video_save_path, f'patient{self.patient_idx}_top.mp4')
        self.out_bot_path = os.path.join(video_save_path, f'patient{self.patient_idx}_bot.mp4')
        self.out_top = cv2.VideoWriter(self.out_top_path, codec, 25, (640, 480))
        self.out_bot = cv2.VideoWriter(self.out_bot_path, codec2, 25, (640, 480))
        self.tof1_file = []
        self.tof2_file = []
        # self.pts1 = None
        # self.setup()
        while not self.StopEVENT.is_set():
            self.update_display()
            self.root.update()
            self.root.after(1)
        # self.root.mainloop()

    def setup(self):
        pass
        # with open('001.pkl', 'rb') as look:
        #     self.pts1 = pickle.load(look)

        # with open('look_upT.pkl', 'rb') as look:
        #     self.tl, self.tr, self.bl, self.br = pickle.load(look)
        # print('[INFO] LOOK UP TABLE FINISHED')

    # @timer

    def red_area(self,img):
        background_area = 27.2*17
        result = Red_seg(img)
        pixels = np.sum(result>1)
        return pixels/background_area ,result

    def update_display(self):
        # while not StopEVENT.is_set():

        if self.queue_list[1].empty():
            return
        bot_img = self.queue_list[1].get()
        info = self.queue_list[4].get()
        height, force = info
        if CLIENT_NR>2:

            tof1_info=self.queue_list[2].get()
            tof2_info=self.queue_list[3].get()
        else:
            tof1_info = 0
            tof2_info = 0
        self.force_buffer.append(force)
        top_img = self.queue_list[0].get()
        for _ in range(CLIENT_NR):
            self.queue_list[5].release()
        img, pred = np.split(top_img, 2, axis=0)
        b1, b2 = np.split(bot_img, 2, axis=0)
        if not self.Recoding_flag:
            index = round((height - self.low_lv) * (self.last_high - self.first_low) / (self.high_lv - self.low_lv))
            try:
                self.M = self.M_list[index]
                img_after = cv2.warpPerspective(img, self.M, (640, 480))
            except IndexError:
                img_after = cv2.warpPerspective(img, self.M, (640, 480))
        else:
            img_after = img
        # M = cv2.getPerspectiveTransform(np.float32(self.pts1), pts2)
        self.height_value.configure(text="{:.1f} mm".format(height))
        self.compression_value.configure(text="{:.1f} N".format(force))

        area ,red_seg = self.red_area(img_after)

        self.area_value.configure(text="{:.3f} mm^2".format(area))
        self.Pressure_value.configure(
            text="{:.3f} N/mm^2".format(99))  # frame0_0 = cv2.resize(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB),
        #                       (int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 2)))
        # frame1_0 = cv2.resize(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB),
        #                       (int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 2)))
        # # frame0_1 = cv2.resize(cv2.cvtColor(self.breast_pred, cv2.COLOR_GRAY2RGB),
        # frame0_1 = cv2.resize(test_img,
        #                       (int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 2)))
        # # frame1_1 = cv2.resize(cv2.imread('thresh.jpg'), (int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 2)))
        # frame1_1 = cv2.resize(test_img, (int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 2)))
        self.frame = np.concatenate((
            np.concatenate((img_after, pred), axis=0),
            # top_img,
            bot_img
            # np.concatenate((bot_img, bot_img), axis=0)
        ), axis=1)
        self.frame = ImageTk.PhotoImage(Image.fromarray(self.frame))

        self.display_panel.configure(image=self.frame)
        most = int(self.force_buffer.most())

        if most >= self.force_threshold:
            if self.Recoding_flag:
                self.recoding_stage = True
                self.Recoding_btn.configure(text='Recoding ON', bg='red')
                self.Recoding_btn.update()
                self.out_top.write(img)
                self.out_bot.write(b2)
                self.tof1_file.append(tof1_info)
                self.tof2_file.append(tof2_info)
        else:
            if self.Recoding_flag:
                # want recording but Force is smaller than 5, go pause
                if self.recoding_stage:
                    self.out_top.release()
                    self.out_bot.release()
                    self.Recoding_btn.configure(text='Recoding PAUSE', bg='SystemButtonFace')
                    self.Recoding_btn.update()
                    with open(f'Patient_{self.patient_idx}_tof1.p','wb') as f:
                        pickle.dump(self.tof1_file,f)
                    with open(f'Patient_{self.patient_idx}_tof2.p','wb') as f:
                        pickle.dump(self.tof2_file,f)
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
        if self.Pre_Recoding_status and self.Recoding_flag:
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            codec2 = cv2.VideoWriter_fourcc(*'mp4v')
            self.out_top_path = os.path.join(video_save_path, f'lookup_Table(TOP).mp4')
            self.out_bot_path = os.path.join(video_save_path, f'lookup_Table(BOT).mp4')
            self.out_top = cv2.VideoWriter(self.out_top_path, codec, 25, (640, 480))
            self.out_bot = cv2.VideoWriter(self.out_bot_path, codec2, 25, (640, 480))

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

    def Pre_Recoding(self):
        self.Pre_Recoding_status = not self.Pre_Recoding_status
        if self.Pre_Recoding_status:
            self.force_threshold = 0
        else:
            self.force_threshold = 5

    def onClose(self):
        self.StopEVENT.set()
        if self.out_bot.isOpened():
            self.out_top.release()
            self.out_bot.release()
        self.root.quit()

    def new_writer(self):
        if not self.Pre_Recoding_status:
            self.patient_idx += 1
        else:
            pass

        self.out_top_path = os.path.join(video_save_path, f'patient{self.patient_idx}_top.mp4')
        self.out_bot_path = os.path.join(video_save_path, f'patient{self.patient_idx}_bot.mp4')
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        codec2 = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_top = cv2.VideoWriter(self.out_top_path, codec, 25, (640, 480))
        self.out_bot = cv2.VideoWriter(self.out_bot_path, codec2, 25, (640, 480))



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
    tof1_que = Queue(maxsize=1)
    tof2_que = Queue(maxsize=1)
    queue_list = [cam1_que, cam2_que, tof1_que, tof2_que, cam2_info_que, syn_que]
    # gui_process = Process(target=Gui, args=(queue_list, StopEVENT,))
    # gui_process.start()
    process_pool = []
    client_timer = 0
    while True:
        try:
            clientSock, addr = s.accept()

            # clientSock.setblocking(False)
        except BlockingIOError:
            # print('time out')
            continue
        p = Process(target=recv_client_data, args=(clientSock, addr, queue_list, StopEVENT,))
        process_pool.append(p)
        p.start()
        client_timer += 1
        if client_timer == CLIENT_NR:
            print('stop')
            break
    a = Gui(queue_list, StopEVENT)

    s.close()
    # gui_process.join()
    # gui_process.close()

    for p in process_pool:
        p.kill()
    print('done')
