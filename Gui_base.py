import socket
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

eps = 0.0001

WINDOW_HEIGHT = 960
WINDOW_WIDTH = 1280

BTN_SIZE = [200, 100]
FONT_SIZE = 18
FONT_COLOR = 'black'
host = socket.gethostname()
port = 12
CLIENT_NR = 2
CAMERA_PORT_BOT = 1
CAMERA_PORT_TOP = 0


class Gui_base:
    def __init__(self, queue_list, StopEVENT):
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
        # self.show_time = args.showtime
        # black or Noimg
        # self.no_img = cv2.cvtColor(cv2.imread('Noimg.png'), cv2.COLOR_BGR2RGB)

        self.whole_img_flag = 0
        # configuration of layout
        self.root = tk.Tk()
        self.root.title('Pressure measurement V_0.1')
        self.root.protocol("WM_DELETE_WINDOW", self.onClose)

        # image display
        # self.display_panel = tk.Canvas(self.root, height=WINDOW_HEIGHT, width=WINDOW_WIDTH)
        self.display_panel = tk.Label(self.root, height=WINDOW_HEIGHT, width=WINDOW_WIDTH)
        self.display_panel['bg'] = 'darkgreen'
        self.display_panel.grid(row=0, column=0, sticky=tk.N + tk.W + tk.E + tk.S)

        # original displayed image
        self.no_img = cv2.cvtColor(cv2.imread('bot.jpg'), cv2.COLOR_BGR2RGB)
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

        self.Recoding_btn = tk.Button(self.param_panel,
                                      image=self.dummy_img,
                                      width=BTN_SIZE[0],
                                      height=BTN_SIZE[1] / 2,
                                      compound="center",
                                      text='Recoding',
                                      command=self.recoding,
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

        self.Recoding_btn.grid(row=10)
        self.Breast_btn.grid(row=12)
        self.Reset_btn.grid(row=11)
        self.exit_btn.grid(row=13)
        self.queue_list = queue_list

    def get_loop(self, loop):
        pass

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
        pass

    def recoding(self):
        pass

    def onClose(self):
        pass
