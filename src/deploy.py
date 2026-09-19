import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights,FastRCNNPredictor,FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from dataset import VOCdataset
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine, ColorJitter
from argparse import ArgumentParser
from tqdm.autonotebook import tqdm  #autonotebook thì chạy đc cả trong notebook cả trong bình thường
from torch.utils.tensorboard import SummaryWriter
import shutil # TODO : Nếu muốn xóa cả folder log cũ thay os.rmdir: cái này chỉ xóa file rỗng đc thôi
import os
from torchmetrics.detection import MeanAveragePrecision
from pprint import pprint
import cv2
import time


def get_args():
    parser = ArgumentParser(description='CNN Argument Parser')
    parser.add_argument('--root','-r', help='root path', default="../datasets", type=str)

    parser.add_argument('--num_epochs','-ne', help='number of epochs', default=100, type=int)
    parser.add_argument('--batch_size','-b', help='batch size', default=3, type=int)
    parser.add_argument('--image_size','-ims', help='image size', default=416, type=int)
    parser.add_argument('--conf_threadhold','-conf', help='conf_threadhold', default=0.5, type=int)

    parser.add_argument('--logging','-l', help='logging tesorboard', default="tensorboard", type=str)
    parser.add_argument('--checkpoint','-c', help='checkpoint is models saved params', default="../trained_models/best_model.pt", type=str)
    parser.add_argument('--save_path','-sp', help='save_path is path where models saved params', default="trained_models", type=str)
    parser.add_argument('--image_path','-imp', help='image path', default="../demo/img_3.png", type=str)

    args = parser.parse_args()
    return args

def deploy(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO-Step 2: Define a Conv Neural Network
    categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                  'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                  'train', 'tvmonitor']  # để sau match với số int trong model

    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
     # region print roi_head details
    # print(model)
    # print("---------")
    # print(model.backbone)
    # print("---------")
    # print(model.roi_heads)
    # print("---------")
    # print(model.roi_heads.box_predictor)
    # print("---------")
    # print(model.roi_heads.box_predictor.cls_score)
    # print("---------")
    # print(model.roi_heads.box_predictor.cls_score.in_features)
    print("----Thay box_predictor -----")
    # endregion
    # FIXME transfer learning nên thay dầu ra của model cũ
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=model.roi_heads.box_predictor.cls_score.in_features,
                                                      num_classes=len(categories))
    model.to(device)
    # print(model.roi_heads.box_predictor)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)  # load đường dẫn
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print('No checkpoint provided.')
        exit(0)

    # TODO-Step 3:Đọc ảnh để inference
    model.eval()

    origin_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape #kích thước gốc
    image = cv2.resize(image, (args.image_size, args.image_size))
    # image = ToTensor()(image) # dùng cái hàm này cũng đc nhưng tôi thích khó nên chọn cách dưới
    image = image/255.0
    #FIXME chuẩn hóa thứ tự kênh màu
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    image = np.transpose(image, (2, 0, 1))  # hàm transpose đổi thứ tự kênh màu
    print(image.dtype) #ddang là float64

    image = torch.from_numpy(image).to(device).float()  # from_numpy(image)hàm này biến numpy thành tensor
    print(image.dtype) # tại vì layer trong của conv yêu cầu float32
    image = [image] # cho ảnh tensor vào list
    print(image)
    print(type(image))
    print(len(image))

    # warm up + benchmark test xem toc do xu ly anh cua model
    # num_trials = 105
    # times = []
    #
    # with torch.no_grad():
    #     for i in range(num_trials):
    #         print(i)
    #
    #         start = time.time()
    #
    #         predictions = model(image)
    #
    #         if i >= 5:
    #             times.append(time.time() - start)
    #
    # print( "Average processing time is {}".format(  sum(times) / (num_trials - 5)   ) )
    # print(torch.cuda.get_device_name(0))

    # FIXME-bắt đầu forward nhưng ko lưu trữ thông tin gradient
    with torch.no_grad():
        prediction = model(image)
    print(prediction)
    for box, score, label in zip(prediction[0]["boxes"], prediction[0]["scores"], prediction[0]["labels"]):
        if score > args.conf_threadhold:
            print(box)
            xmin, ymin, xmax, ymax = box # các toạn độ này mô hình fit với ảnh input 416x416
            xmin= xmin/args.image_size * width
            ymin= ymin/args.image_size * height
            xmax= xmax/args.image_size * width
            ymax= ymax/args.image_size * height
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            cv2.rectangle(
                origin_image,
                (xmin, ymin),  # góc trái trên
                (xmax, ymax),  # góc phải dưới
                (0, 255, 0),  # màu BGR (xanh lá)
                2  # độ dày đường
            )
            cv2.putText(origin_image,categories[label] +"{:0.2f}".format(score),(xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)

     # cái này để vẽ ảnh và lưu trong file
    cv2.imwrite("../demo/camera_demo.jpg", origin_image)

if __name__ == '__main__':
    args = get_args()
    deploy(args)