import pickle
from collections import deque

import cv2
import numpy as np

from Gui_base import CAMERA_PORT_TOP

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
    file_name = ['001.jpg', '050.jpg', '100.jpg']
    i = 0

    img = cv2.imread(file_name[i])

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
            with open(file_name[i].split('.')[0] + '.pkl', 'wb') as file:
                saved = np.array(point_List)
                print(saved)
                pickle.dump(saved, file)
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
    list1 = []
    list50 = []
    list100 = []
    list_all = [list1, list50, list100]
    file_name = ['001.pkl', '050.pkl', '100.pkl']

    for l, name in zip(list_all, file_name):
        with open(name, 'rb') as file:
            l = pickle.load(file)
            get_center(l)
            print(l)
    print(list_all)


def order_points_new(pts):
    # sort the points based on their x-coordinates
    try:
        assert len(pts) == 4
    except:
        print(pts)
        # return self.pts1
        pts = pts[1:]
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    if leftMost[0, 1] != leftMost[1, 1]:
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    else:
        leftMost = leftMost[np.argsort(leftMost[:, 0])[::-1], :]
    (tl, bl) = leftMost
    if rightMost[0, 1] != rightMost[1, 1]:
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    else:
        rightMost = rightMost[np.argsort(rightMost[:, 0])[::-1], :]
    (tr, br) = rightMost
    x = min((tr[0] - tl[0]), (br[0] - bl[0]))
    y = min((br[1] - tr[1]), (bl[1] - tl[1]))
    return np.array([tl, tr, br, bl], dtype="float32"),


def get_center(pt):
    print(pt)
    img = np.zeros((480, 640), dtype=np.uint8)

    pt_new = order_points_new(pt)
    print(pt_new)
    img = cv2.fillConvexPoly(img, np.array(pt_new, dtype=np.int32), 255)
    cv2.imshow('hi', img)
    cv2.waitKey()


def get_top_image(img):
    if img is None:
        camera_top = cv2.VideoCapture(CAMERA_PORT_TOP, cv2.CAP_DSHOW)
        # pipeline = rs.pipeline()
        # config = rs.config()
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        # pipeline.start(config)
    file_name = ['001.jpg', '050.jpg', '100.jpg']
    i = 0
    while True:
        if img is None:
            # frames = pipeline.wait_for_frames()
            # frame_bot = frames.get_color_frame()
            # frame_bot = np.asanyarray(frame_bot.get_data())
            ret2, frame_top = camera_top.read()
        else:
            frame_top = img
        # frame_bot = np.rot90(frame_bot, 2)

        cv2.imshow('frame_top', frame_top)
        k = cv2.waitKey(1)

        if k == ord('s'):
            cv2.imwrite(file_name[i], frame_top)
            i += 1
        if k == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # img = cv2.imread('bot.jpg')
    # img = cv2.resize(img, (640, 480))
    get_top_image(None)
    get_4points()
    # do_regression()
