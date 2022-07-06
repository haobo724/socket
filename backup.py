global point_List
point_List = deque(maxlen=4)


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