import pickle,os
from collections import deque
import pyrealsense2 as rs
from tool import get_regression
import cv2
import numpy as np

global point_List
point_List = deque(maxlen=4)




def OnMouseAction(event, x, y, flags, param):
    global x1, y1
    img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 20, (10, 255, 10), -1)
        point_List.append((x, y))
        print(point_List)


def get_4points():

    file_name = ['001.jpg','050.jpg','100.jpg']
    i = 0

    img=cv2.imread(file_name[i])

    cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)  # 设置窗口标题和大小
    # cv2.resizeWindow('image', 1000, 400)
    cv2.setMouseCallback("image", OnMouseAction, img)
    img_copy = img.copy()
    while (1):

        cv2.imshow('image', img)

        k = cv2.waitKey(1)
        if k == ord('q'):
            continue
        elif k == ord('c'):
            img = img_copy.copy()
            cv2.setMouseCallback('image', OnMouseAction, img)
            point_List.clear()
        if k == ord('s'):
            with open(file_name[i].split('.')[0]+'.pkl','wb') as file :
                saved = np.array(point_List)
                print(saved)
                pickle.dump(saved,file)
                point_List.clear()
            i += 1
            if i == 3:
                break
            img = cv2.imread(file_name[i])
            img_copy = img.copy()
            cv2.setMouseCallback('image', OnMouseAction, img)

            continue

    cv2.destroyAllWindows()


def do_regression():
    list1 =[]
    list50 =[]
    list100=[]
    list_all = [list1,list50,list100]
    file_name = ['001.pkl','050.pkl','100.pkl']

    for l,name  in zip(list_all,file_name):
        with open(name,'rb') as file :
            l = pickle.load(file)
            print(l)
    print(list_all)

def get_top_image(img):
    if img is None:
        # camera_bot = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)
    file_name = ['001.jpg','050.jpg','100.jpg']
    i =0
    while True:
        if img is None:
            frames = pipeline.wait_for_frames()
            frame_bot = frames.get_color_frame()
            frame_bot = np.asanyarray(frame_bot.get_data())
            # ret2, frame_bot = camera_bot.read()
        else:
            frame_bot = img
        # frame_bot = np.rot90(frame_bot, 2)

        cv2.imshow('frame_bot', frame_bot)
        k = cv2.waitKey(1)

        if k == ord('s'):
            cv2.imwrite(file_name[i],frame_bot)
            i+=1
        if k == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # img = cv2.imread('bot.jpg')
    # img = cv2.resize(img, (640, 480))
    # get_top_image(None)
    # get_4points()
    do_regression()
