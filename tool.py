import os
import queue

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

IMAGE_HEIGHT = 256  # 1096 originally  0.25
IMAGE_WIDTH = 448  # 1936 originall


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
        self.model_CKPT = torch.load(models)

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

        self.infer_xform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
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


def Red_seg(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    result = cv2.inRange(img, lower_red, upper_red).astype(np.uint8)

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
