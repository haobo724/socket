import os
import pickle
import cv2
from Gui_frame import CAMERA_PORT_BOT

pkl_save_path = 'pkl'
if not os.path.exists(pkl_save_path):
    os.mkdir(pkl_save_path)

camera_bot = cv2.VideoCapture(CAMERA_PORT_BOT)
camera_bot.set(cv2.CAP_PROP_BRIGHTNESS, 100)
camera_bot.set(cv2.CAP_PROP_EXPOSURE, -7)
print(camera_bot.get(cv2.CAP_PROP_EXPOSURE))
print(camera_bot.get(cv2.CAP_PROP_BRIGHTNESS))


def get_bot_display(img):
    temp_turple = ()
    while True:
        if img is None:

            ret2, frame_bot = camera_bot.read()
        else:
            frame_bot = img
            ret2 = True
        if not ret2:
            raise ValueError
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
        with open(os.path.join(pkl_save_path, 'bot.pkl'), 'wb') as file:
            pickle.dump(temp_turple, file)
    else:
        print('[INFO] Not select ROI ON display')
    return frame_bot


def get_force_height_area(img):
    force_file = cv2.selectROI('force', img)
    with open(os.path.join(pkl_save_path, 'force.pkl'), 'wb') as file:
        pickle.dump(force_file, file)
    cv2.destroyAllWindows()


    height_file = cv2.selectROI('height', img)
    with open(os.path.join(pkl_save_path, 'height.pkl'), 'wb') as file:
        pickle.dump(height_file, file)
    cv2.destroyAllWindows()


def get_whole_display(img):
    display_file = cv2.selectROI('whole_display', img)
    with open(os.path.join(pkl_save_path, 'display.pkl'), 'wb') as file:
        pickle.dump(display_file, file)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # img = cv2.imread('bot.jpg')
    # img = cv2.resize(img, (640, 480))
    img = get_bot_display(None)
    get_force_height_area(img)
    get_whole_display(img)
