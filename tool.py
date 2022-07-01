import glob
import os
import queue

import imageio
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from torch.utils.data import DataLoader, Dataset

TRAIN_IMG_DIR = 'pics/'
IMAGE_HEIGHT = 256  # 1096 originally  0.25
IMAGE_WIDTH = 448  # 1936 originall


class LeafData(Dataset):

    def __init__(self,
                 data,
                 transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # import
        # path = os.path.join(self.directory, self.data.iloc[idx]['image_id'])
        img = imageio.imread(self.data[idx])
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image = np.array(imageio.imread(self.data[idx]))
        # augmentations
        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image


class Buffer():
    def __init__(self, size):
        self.q = queue.Queue(size)
        self.list = []

    def __getitem__(self, num):
        if num > len(self.list):
            raise IndexError('list index out of range')
        return self.list[num]

    def append(self, num):
        if not self.q.full():
            self.q.put(num)
            self.list.append(num)
        else:
            self.q.get()
            self.q.put(num)
            del self.list[0]
            self.list.append(num)

    def mean(self):
        if self.q.empty():
            return 0
        return np.mean(self.list)

    def most(self):
        vals, counts = np.unique(self.list, return_counts=True)
        index = np.argmax(counts)
        return vals[index]


def cal_std_mean(TRAIN_IMG_DIR, IMAGE_HEIGHT, IMAGE_WIDTH):
    augs = A.Compose([A.Resize(height=IMAGE_HEIGHT,
                               width=IMAGE_WIDTH),
                      A.Normalize(mean=(0, 0, 0),
                                  std=(1, 1, 1)),
                      ToTensorV2()])
    imgs = glob.glob(TRAIN_IMG_DIR + '*.jpg')
    print(imgs)
    image_dataset = LeafData(data=imgs,
                             transform=augs)
    image_loader = DataLoader(image_dataset,
                              batch_size=4,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=True)

    # placeholders
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs in tqdm(image_loader):
        psum += inputs.sum(axis=[0, 2, 3])

        psum_sq += (inputs ** 2).sum(axis=[0, 2, 3])
    # pixel count
    count = len(glob.glob(TRAIN_IMG_DIR + '/*.jpg')) * IMAGE_HEIGHT * IMAGE_WIDTH
    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)
    torch.cuda.empty_cache()

    # mean: tensor([0.2292, 0.2355, 0.3064])
    # std:  tensor([0.2448, 0.2450, 0.2833])

    # output
    print('mean: ' + str(tuple(total_mean)))
    print('std:  ' + str(tuple(total_std)))
    return total_mean, total_std


def mapping_color_tensor(img):
    '''
    自己写的，速度快不少，但要自己规定colormap，也可以把制定colormap拿出来单独用randint做，
    但是不能保证一个series里每次运行生成的colormap都一样，或许可以用种子点？
    反正类少还是可以考虑用这个
            '''
    # img = torch.unsqueeze(img, dim=-1)

    img = torch.stack([img, img, img], dim=-1)

    color_map = [[247, 251, 255], [171, 207, 209], [55, 135, 192]]
    for label in range(3):
        cord_1 = torch.where(img[..., 0] == label)
        img[cord_1[0], cord_1[1], 0] = color_map[label][0]
        img[cord_1[0], cord_1[1], 1] = color_map[label][1]
        img[cord_1[0], cord_1[1], 2] = color_map[label][2]
    if torch.is_tensor(img):
        return img
    return img.astype(int)


class model_infer():
    def __init__(self, models):
        # self.model = unet_train.load_from_checkpoint(models)
        self.model_CKPT = torch.load(models, map_location='cpu')

        if torch.cuda.is_available():
            self.DEVICE = torch.device('cuda')
            #
            # self.model_no = Resnet_Unet().to(self.DEVICE)

            self.model = smp.Unet(
                # encoder_depth=4,
                # decoder_channels=[512,256, 128, 64,32],
                in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=3,  # model output channels (number of classes in your dataset)
            ).cuda()
            # total = sum([param.nelement() for param in self.model.parameters()])
            # print("Number of parameter: %.2fM" % (total / 1e6))


        #     # self.model = UNET_S(in_channels=3, out_channels=1,features=[16,32,64,128]).to( self.DEVICE)
        else:
            self.DEVICE = torch.device('cpu')
        self.error_msg = ''
        loaded_dict = self.model_CKPT['state_dict']
        prefix = 'model.'
        n_clip = len(prefix)
        adapted_dict = {k[n_clip:]: v for k, v in loaded_dict.items()
                        if k.startswith(prefix)}
        self.model.load_state_dict(adapted_dict)
        # self.mean , self.std =cal_std_mean(TRAIN_IMG_DIR, IMAGE_HEIGHT, IMAGE_WIDTH)
        self.infer_xform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    mean=[0.6206, 0.6091, 0.6004],
                    std=[(0.1495), (0.1587), (0.1720)],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

    @torch.no_grad()
    def forward(self, image):

        if type(image) == str:
            input = np.array(Image.open(image), dtype=np.uint8)
        else:
            input = image

        input = self.infer_xform(image=input)
        x = input["image"].cuda()

        x = torch.unsqueeze(x, dim=0)
        self.model.eval()
        y_hat = self.model(x)
        preds = torch.softmax(y_hat, dim=1)

        preds = preds.argmax(dim=1).float()
        preds = preds.squeeze()
        # preds = resize_xform(image=preds.cpu().numpy())
        # preds = preds["image"].numpy() * 1

        # preds = resize_xform(image=preds.cpu().numpy())
        # preds = preds["image"]
        # print('breastt:',end-start)
        img_colored = mapping_color_tensor(preds)
        # img_post = self.post_processing(preds.cpu().numpy())
        # return preds.cpu().numpy()
        return img_colored.cpu().numpy()

    def post_processing(self, image):
        contours, hierarchy = cv2.findContours(image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            if not self.error_msg == 'No breast':
                self.error_msg = 'No breast'
                print(self.error_msg)
            return image
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        temp = np.zeros_like(image)

        thresh = cv2.fillPoly(temp, [contours], (255, 255, 255))
        # plt.figure()
        # plt.imshow(thresh * 255, cmap='gray')
        #
        # plt.show()

        return thresh


class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y


def Red_seg(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    result = cv2.inRange(img, lower_red, upper_red).astype(np.uint8)

    result = np.dstack([result for _ in range(3)])

    return result


def Green_seg(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_Green = np.array([70, 20, 20])
    upper_Green = np.array([120, 255, 255])
    result = cv2.inRange(img, lower_Green, upper_Green).astype(np.uint8)

    result = np.dstack([result for _ in range(3)])
    return result


def get_regression(x, y):
    # 将 x，y 分别增加一个轴，以满足 sklearn 中回归模型认可的数据
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    poly = PolynomialFeatures(degree=5)
    X_poly = poly.fit_transform(x)
    poly.fit(X_poly, y)

    lin2 = LinearRegression()
    lin2.fit(X_poly, y)

    return poly, lin2

def sort_pts( pts):
    # sort the points based on their x-coordinates
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
    # print(tl, tr, bl, br)
    x = tr[0] - tl[0]
    y = br[1] - tr[1]
    return np.array([tl, tr, bl, br], dtype="float32")
