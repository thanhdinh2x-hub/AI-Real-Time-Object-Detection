import torch

from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset, DataLoader
from pprintpp import pprint
import numpy as np
import cv2
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine, ColorJitter
from torchvision.transforms import ToPILImage

class VOCdataset(VOCDetection):
    def __init__(self, root, year, image_set, download, transform):
        super().__init__(root, year, image_set, download,
                         transform)  # do kế thừa 1 số chỗ vẫn dùng lại hàm của hàm cho nên phải truyền tham số cho nó
        self.categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                           'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                           'train', 'tvmonitor']  # để sau match với số int trong model

    def __getitem__(self, index):
        image, targets = super().__getitem__(index)  # image sẽ là ảnh pil và targets sẽ là dictionary

        #FIXME lấy trong file annotation ra width cà height cũ của ảnh
        old_h= int(targets["annotation"]["size"]["height"])
        old_w = int(targets["annotation"]["size"]["width"])
        # print("old_h:",old_h)
        # print("old_w:",old_w)
        #FIXME kích thước mới của ảnh sau khi đi qua transform của hàm cha
        _, new_h, new_w = image.shape
        # print("new_h:",new_h)
        # print("new_w:",new_w)

        targets = targets["annotation"]["object"]  # trong này sẽ chứa từng bndbox và name
        all_labels = []  # tạo 1 mảng chưa tuần tự các label
        all_boxes = []  # tạo 1 mảng chưa tuần tự các list bbox
        outputs={}
        for obj in targets:
            # print(obj)
            label = obj["name"]
            all_labels.append(self.categories.index(label)) #FIXME match chỉ số index dựa theo label

            '''
            # kích thước bbox cũ
            xtl= obj["bndbox"]["xmin"]
            ytl= obj["bndbox"]["ymin"]
            xbr= obj["bndbox"]["xmax"]
            ybr= obj["bndbox"]["ymax"]
            print(xtl,ytl,xbr,ybr)
            '''
            #FIXME biến đổi bnbox khi ảnh gốc đc resize
            xmin = int((float(obj["bndbox"]["xmin"])/old_w) *new_w )
            ymin = int((float(obj["bndbox"]["ymin"])/old_h) *new_h )
            xmax = int((float(obj["bndbox"]["xmax"])/old_w) *new_w )
            ymax =  int((float(obj["bndbox"]["ymax"])/old_h) *new_h )
            # print("bnbox khi ảnh gốc đc resize:",xmin,ymin,xmax,ymax)
            all_boxes.append([xmin, ymin, xmax, ymax])
        #FIXME  ép thành kiểu của fasterRCNN yêu cầu
        outputs["boxes"] = torch.FloatTensor(all_boxes)
        outputs["labels"] = torch.LongTensor(all_labels)
        # print("----------ra khỏi dataset-------------")
        return image, outputs


if __name__ == '__main__':
    train_transform = Compose([

        Resize((416, 416)),
        ToTensor(),
        # Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     # đây là chuẩn hóa trên từng kênh màu của toàn bộ bức ảnh trong epoch các mô hình transfer thường theo tiêu chuẩn này
        #     std=[0.229, 0.224,  0.225]),
    ])
    dataset = VOCdataset(root="../datasets", year="2007", image_set="trainval", download=False,
                         transform=train_transform)
    print("dataset:",dataset)
    image, targets = dataset.__getitem__(2233)
    print("kết quả của targets:",targets)
    print("kết quả của targets[boxes]:",targets["boxes"])
    print("kết quả của targets[labels]:",targets["labels"])

    #FIXME Convert the tensor to a NumPy array
    bboxes = targets["boxes"].numpy()
    labels=targets["labels"].numpy()

    print("image sau khi lấy từ dataset:",image.shape)
    print("type của image:",type(image))
    print("bboxes:",bboxes)
    print("lables:",labels)

    # FIXME biến tensor -> PIL
    to_pil = ToPILImage()
    pil_image = to_pil(image)
    # FIXME biến PIL -> CV2 NumPy array
    image_cv2 = np.array(pil_image)
    print("image_cv2.shape:",image_cv2.shape)
    # FIXME PIL dùng RGB, OpenCV dùng BGR
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)
    # cv2.imshow("image_cv2",image_cv2)
    # cv2.waitKey(0)
    # vẽ box
    for bbox in bboxes:
        print("bbox:",bbox)
        xmin, ymin, xmax, ymax = bbox
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        cv2.rectangle(
            image_cv2,
            (xmin, ymin),  # góc trái trên
            (xmax, ymax),  # góc phải dưới
            (0, 255, 0),  # màu BGR (xanh lá)
            1  # độ dày đường
        )
    # cái này để vẽ ảnh và lưu trong file
    cv2.imwrite("image1.png", image_cv2)
    # cái này để vẽ ảnh và show thôi
    cv2.imshow("image", image_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
