import os
import pickle

import scipy.signal as signal

from BotCamera import OCR_THIRD

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
import cv2
from tool import Green_seg, KalmanFilter, sort_pts

pkl_save_path = 'pkl'
if not os.path.exists(pkl_save_path):
    os.mkdir(pkl_save_path)


def run():
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
    CAM = cv2.VideoCapture(0)
    while True:
        ret, frame = CAM.read()
        if not ret:
            print('Error')
            break

        result = ocr.ocr(frame, cls=True)
        for line in result:
            print(line)
        #
        #
        # # draw result
        #
        boxes = [line[0] for line in result]
        # im_show = cv2.rectangle(frame,)
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        im_show = draw_ocr(frame, boxes, txts, scores)
        cv2.imshow('OCR', im_show)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break


def Rec_Green_Pattern(path=0):
    CAM = cv2.VideoCapture(path)
    Pts_List = []
    pts = []
    while True:
        ret, frame = CAM.read()
        if not ret:
            print('Error')
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Green_seg(frame)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        blank = np.zeros_like(img)
        rect = cv2.minAreaRect(contours)
        box = cv2.boxPoints(rect).astype(int)
        for p in box:
            p = tuple(p)
            cv2.circle(blank, p, 10, (255, 0, 0), -1)
            pts.append(p)
        Pts_List.append(pts)
        pts = []
        # cv2.drawContours(blank, [contours], -1, (255, 255, 255), -1)
        cv2.imshow('green', blank)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    Pts_List = np.array(Pts_List)
    with open(os.path.join(pkl_save_path, 'Pts_List.pkl'), 'wb') as f:
        pickle.dump(Pts_List, f)
    return Pts_List


def smooth_pts(Pts_List):
    if type(Pts_List) is str:
        with open('Pts_List.pkl', 'rb') as f:
            Pts_List = pickle.load(f)
    for i in range(4):
        for j in range(2):
            Pts_List[:, i, j] = signal.medfilt(Pts_List[:, i, j], 17)

    for i in range(Pts_List.shape[0]):

        blank = np.zeros((480, 640))

        for p in Pts_List[i]:
            p = tuple(p)
            cv2.circle(blank, p, 10, (255, 0, 0), -1)

        # cv2.drawContours(blank, [contours], -1, (255, 255, 255), -1)
        cv2.imshow('Smooth', blank)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    with open(os.path.join(pkl_save_path, 'Pts_List_smooth.pkl'), 'wb') as f:
        pickle.dump(Pts_List, f)
    return Pts_List


def point_mid(Pts_List_smooth):
    if type(Pts_List_smooth) is str:
        with open(os.path.join(pkl_save_path, 'Pts_List_smooth.pkl'), 'rb') as f:
            Pts_List_smooth = pickle.load(f)
    mid_list = []

    for i in Pts_List_smooth:
        print(i)
        mu = cv2.moments(i, False)
        blank = np.zeros((480, 640))

        mc = [mu['m10'] / mu['m00'], mu['m01'] / mu['m00']]
        cv2.circle(blank, (int(mc[0]), int(mc[1])), 10, (255, 255, 255), -1)
        cv2.imshow('hi', blank)
        cv2.waitKey(1)
        mid_list.append((int(mc[0]), int(mc[1])))
    with open(os.path.join(pkl_save_path, 'mid_points.pkl'), 'wb') as f:
        pickle.dump(mid_list, f)


def split(path):
    video = cv2.VideoCapture(path)
    time = 0
    NAME = os.path.basename(path).split('.mp4')[0]
    print(NAME)
    while True:
        ret, frame = video.read()
        if not ret:
            print('Error')
            break

        if time % 10 == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f'pics/{NAME}_{time}.jpg', frame)
        time += 1


def rec_bot(path):
    video = cv2.VideoCapture(path)

    file_name = 'height.pkl'
    with open(file_name, 'rb') as file:
        x1, y1, w1, h1 = pickle.load(file)
    file_name = 'force.pkl'
    with open(file_name, 'rb') as file:
        x2, y2, w2, h2 = pickle.load(file)
    Height_list = []
    while True:
        ret, img = video.read()
        if not ret:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        force_block = img[y2:y2 + h2, x2:x2 + w2, :]
        height_block = img[y1:y1 + h1, x1:x1 + w1, :]
        # cv2.imshow('test',height_block)
        # cv2.waitKey()

        height = OCR_THIRD(height_block)
        force = OCR_THIRD(force_block)
        Height_list.append(height)
    num, counts = np.unique(Height_list, return_counts=True)
    print(Height_list)
    num = np.array(num)
    counts_resort = np.argsort(np.array(-counts))
    most_num = num[counts_resort[0]]
    second_most_num = num[counts_resort[1]]
    if most_num > second_most_num:
        high_lv = most_num
        low_lv = second_most_num
    else:
        high_lv = second_most_num
        low_lv = most_num
    with open(os.path.join(pkl_save_path, 'Valid_interval.pkl'), 'wb') as f:
        pickle.dump([high_lv, low_lv], f)
    print(high_lv, low_lv)
    last_high = np.argwhere(Height_list == high_lv)
    first_low = np.argwhere(Height_list == low_lv)
    print(last_high[-1], first_low[0])
    # print(first_low[0] - last_high[-1])
    with open(os.path.join(pkl_save_path, 'last_high_first_low.pkl'), 'wb') as f:
        pickle.dump([last_high[-1][0], first_low[0][0]], f)
    cv2.destroyAllWindows()
    return last_high[-1][0], first_low[0][0]


def Kalman(Pts_List_smooth):
    if type(Pts_List_smooth) is str:
        with open(os.path.join(pkl_save_path, 'Pts_List_smooth.pkl'), 'rb') as f:
            Pts_List_smooth = pickle.load(f)
    Pts_List_smooth = Pts_List_smooth[:, 0, :]
    print(Pts_List_smooth.shape)
    kf = KalmanFilter()
    img = np.zeros((480, 640, 3))

    for pt in Pts_List_smooth:
        pt = tuple(pt)
        cv2.circle(img, pt, 15, (0, 20, 220), -1)
        predicted = kf.predict(pt[0], pt[1])

        cv2.circle(img, predicted, 15, (20, 220, 0), 4)
        cv2.imshow("Img", img)
        cv2.waitKey(0)


def cut_down(Pts_List_smooth, last_high_first_low):
    if type(Pts_List_smooth) is str:
        with open(os.path.join(pkl_save_path, 'Pts_List_smooth.pkl'), 'rb') as f:
            Pts_List_smooth = pickle.load(f)
    if type(last_high_first_low) is str:
        with open(os.path.join(pkl_save_path, 'last_high_first_low.pkl'), 'rb') as f:
            last_high, first_low = pickle.load(f)

    Pts_List_cut = Pts_List_smooth[last_high:first_low, :, :]
    with open(os.path.join(pkl_save_path, 'Pts_List_cut.pkl'), 'wb') as f:
        pickle.dump(Pts_List_cut, f)
    return Pts_List_cut


def Generate_Lookup_table(Pts_List_cut):
    if type(Pts_List_cut) is str:
        with open(Pts_List_cut, 'rb') as f:
            Pts_List_cut = pickle.load(f)

    M_list = []
    for p in Pts_List_cut:
        pts1 = sort_pts(p)
        # pts1 = np.float32(p)
        print(pts1)
        pts2 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        M_list.append(M)

    with open(os.path.join(pkl_save_path, 'M_list.pkl'), 'wb') as f:
        pickle.dump(M_list, f)
    return M_list


def Test_warp(M_list, video, last_high_first_low):
    if type(M_list) is str:
        with open(os.path.join(pkl_save_path, 'M_list.pkl'), 'rb') as f:
            M_list = pickle.load(f)
    if type(last_high_first_low) is str:
        with open(os.path.join(pkl_save_path, 'last_high_first_low.pkl'), 'rb') as f:
            last_high, first_low = pickle.load(f)
    video = cv2.VideoCapture(video)
    frames = []

    out_bot_path = f'after_M.mp4'
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out_top = cv2.VideoWriter(out_bot_path, codec, 25, (640, 480))
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # cv2.imshow('temp',frame)
        # cv2.waitKey()
        frames.append(frame)
    # if not len(M_list)>(first_low - last_high):
    #     frames = frames[last_high:first_low]

    print(len(M_list), len(frames))
    assert len(M_list) == len(frames)
    for i in range(len(frames)):
        img_new = cv2.warpPerspective(frames[i], M_list[i], (640, 480))
        out_top.write(img_new)

        cv2.namedWindow("img_new", cv2.WINDOW_AUTOSIZE)  # 设置窗口标题和大小

        cv2.imshow("img_new", img_new)
        cv2.waitKey(1)
    out_top.release()


def run_pall(path):
    video = cv2.VideoCapture(path[0])
    video2 = cv2.VideoCapture(path[1])
    while True:
        ret, frame = video.read()
        ret2, frame2 = video2.read()
        if not ret:
            print('Error')
            break
        print(frame.shape)
        print(frame2.shape)
        f = np.concatenate((frame, frame2), axis=0)
        cv2.imshow('hi', f)
        cv2.waitKey(100)


if __name__ == '__main__':
    # run_classic()
    # mp4 = glob.glob('*.mp4')
    # for i in mp4:
    # #     split(i)
    #     video = cv2.VideoCapture(i)
    #     print(video.get(cv2.CAP_PROP_FRAME_COUNT))
    path = 'lookup_Table(TOP).mp4'
    Pts_List = Rec_Green_Pattern(path) #BGR INPUT
    Pts_List_smooth = smooth_pts(Pts_List)
    # # Mid = point_mid('')
    # # smooth = 'pkl/Pts_List_smooth.pkl'
    # # Kalman(smooth)
    path = 'lookup_Table(BOT).mp4'
    last_high_first_low = rec_bot(path)
    Pts_List_cut = cut_down(os.path.join(pkl_save_path, 'Pts_List_smooth.pkl'),
                            os.path.join(pkl_save_path, 'last_high_first_low.pkl'))
    M_list = Generate_Lookup_table(os.path.join(pkl_save_path, 'Pts_List_cut.pkl'))
    # Test_warp(os.path.join(pkl_save_path, 'M_list.pkl'),'Top_Cali (2).mp4',os.path.join(pkl_save_path, 'last_high_first_low.pkl'))
    # run_pall(['Top_Cali (2).mp4','after_M.mp4'])
