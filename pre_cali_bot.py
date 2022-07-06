import pickle
from collections import deque

import cv2
import numpy as np

from Gui_frame import CAMERA_PORT_BOT

global point_List
point_List = deque(maxlen=4)
print(cv2.__version__)
camera_bot = cv2.VideoCapture(CAMERA_PORT_BOT)
# camera_bot = cv2.VideoCapture(r'Top_Cali (1).mp4')
camera_bot.set(cv2.CAP_PROP_BRIGHTNESS,100)
camera_bot.set(cv2.CAP_PROP_EXPOSURE,-7)
print(camera_bot.get(cv2.CAP_PROP_EXPOSURE))
print(camera_bot.get(cv2.CAP_PROP_BRIGHTNESS))
def get_bot_display(img):

    file_name = 'bot.pkl'
    temp_turple = ()
    while True:
        if img is None:

            ret2, frame_bot = camera_bot.read()
        else:
            frame_bot = img
            ret2= True
        if not ret2:
            raise ValueError
            break
        # frame_bot = np.rot90(frame_bot, 2)
        cv2.imshow('frame_bot', frame_bot)
        k = cv2.waitKey(1)

        if k == ord('q'):
            break
        elif k == ord('s'):
            cv2.imwrite('bot_sample.jpg', frame_bot)
            # x, y, w, h = cv2.selectROI('frame_bot', frame_bot, fromCenter=False)
            # temp_turple = x, y, w, h
            break
    cv2.destroyAllWindows()
    if len(temp_turple):
        with open(file_name, 'wb') as file:
            pickle.dump(temp_turple, file)
    else:
        print('[INFO] Not select ROI ON display')
    return frame_bot


def OnMouseAction(event, x, y, flags, param):
    global x1, y1
    img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 20, (10, 255, 10), -1)
        point_List.append((x, y))
        print(point_List)


def get_M(img):
    # file_name = 'bot.pkl'
    #
    # with open(file_name, 'rb') as file:
    #     box = pickle.load(file)
    # x, y, w, h = box
    w, h = 640, 480
    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)  # 设置窗口标题和大小
    # cv2.resizeWindow('image', 1000, 400)
    cv2.setMouseCallback("image", OnMouseAction, img)
    img_copy = img.copy()
    while (1):
        cv2.imshow('image', img)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        elif k == ord('c'):
            img = img_copy.copy()
            cv2.setMouseCallback('image', OnMouseAction, img)
            point_List.clear()

    cv2.destroyAllWindows()

    pts1 = np.float32(point_List)

    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    img_new = cv2.warpPerspective(img_copy, M, (w, h))
    cv2.namedWindow("img_new", cv2.WINDOW_AUTOSIZE)  # 设置窗口标题和大小
    # cv2.resizeWindow('image', 1000, 400)
    cv2.imshow("img_new", img_new)
    cv2.waitKey()
    file_name = 'M.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(M, file)


def get_force_height_area(img):

    file_name = 'force.pkl'
    force_file = cv2.selectROI('force', img)
    with open(file_name, 'wb') as file:
        pickle.dump(force_file, file)

    file_name = 'height.pkl'
    height_file = cv2.selectROI('height', img)
    with open(file_name, 'wb') as file:
        pickle.dump(height_file, file)
def get_whole_display(img):
    file_name = 'display.pkl'

    display_file = cv2.selectROI('whole_display', img)
    with open(file_name, 'wb') as file:
        pickle.dump(display_file, file)
if __name__ == '__main__':
    # img = cv2.imread('bot.jpg')
    # img = cv2.resize(img, (640, 480))
    img = get_bot_display(None)
    get_force_height_area(img)
    get_whole_display(img)
